from data import *
from _utils import *
from train_candi import CANDI_LOADER

from SAGA import write_bed, SoftMultiAssayHMM, write_posteriors_to_tsv
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from sklearn.metrics import  mean_absolute_error
from scipy.stats import pearsonr, spearmanr, poisson, rankdata
from sklearn.metrics import mean_squared_error, r2_score, auc
from sklearn.metrics import roc_auc_score


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.svm import SVR
from scipy.stats import norm, nbinom
from datetime import datetime

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import pyBigWig, sys, argparse

import scipy.stats as stats
from scipy.optimize import minimize

from hmmlearn.hmm import GaussianHMM

import warnings, os, sys
from typing import Literal
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             confusion_matrix as contingency_matrix)
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, Tuple

def compare_hard_clusterings(labels1: np.ndarray, labels2: np.ndarray) -> Dict[str, Any]:
    """
    Compares two hard clustering assignments (MAP state sequences).

    Args:
        labels1: A 1D NumPy array of state assignments from the first model.
        labels2: A 1D NumPy array of state assignments from the second model.

    Returns:
        A dictionary containing the comparison metrics:
        - 'ari': Adjusted Rand Index.
        - 'nmi': Normalized Mutual Information.
        - 'contingency_matrix': A DataFrame showing the overlap between states.
    """
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    # Create a labeled DataFrame for the contingency matrix
    matrix = contingency_matrix(labels1, labels2)
    cont_df = pd.DataFrame(
        matrix,
        index=[f"Model1_State_{i}" for i in np.unique(labels1)],
        columns=[f"Model2_State_{j}" for j in np.unique(labels2)]
    )

    return {
        "ari": ari,
        "nmi": nmi,
        "contingency_matrix": cont_df
    }

def compare_soft_clusterings(posteriors1: np.ndarray, posteriors2: np.ndarray) -> Dict[str, Any]:
    """
    Compares two soft clustering assignments (posterior probability matrices).

    Args:
        posteriors1: A 2D NumPy array (L, K1) of posteriors from the first model.
        posteriors2: A 2D NumPy array (L, K2) of posteriors from the second model.

    Returns:
        A dictionary containing the comparison metrics:
        - 'avg_jsd': Average Jensen-Shannon Divergence between posteriors.
        - 'posterior_correlation': A DataFrame showing the Pearson correlation
                                   between the posterior probabilities of each pair of states.
    """
    # 1. Average Jensen-Shannon Divergence
    # Clip values to avoid errors with log(0) in JSD calculation
    p1 = np.clip(posteriors1, 1e-10, 1)
    p2 = np.clip(posteriors2, 1e-10, 1)
    jsd_scores = [jensenshannon(p1_row, p2_row, base=2) for p1_row, p2_row in zip(p1, p2)]
    avg_jsd = np.mean(jsd_scores)
    
    # 2. State-wise Posterior Correlation Matrix
    # np.corrcoef calculates the full matrix between all pairs of columns
    # from both inputs stacked together.
    full_corr_matrix = np.corrcoef(p1.T, p2.T)
    
    # We only want the cross-correlation block between Model 1 and Model 2
    num_states1 = p1.shape[1]
    cross_corr = full_corr_matrix[:num_states1, num_states1:]
    
    corr_df = pd.DataFrame(
        cross_corr,
        index=[f"Model1_State_{i}" for i in range(num_states1)],
        columns=[f"Model2_State_{j}" for j in range(p2.shape[1])]
    )
    
    return {
        "avg_jsd": avg_jsd,
        "posterior_correlation": corr_df
    }

def bin_gaussian_predictions(mus_hat: torch.Tensor, sigmas_hat_sq: torch.Tensor, bin_size: int, strategy: Literal['average', 'sum'] = 'average') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bins sequences of Gaussian distribution parameters to a coarser resolution using PyTorch.

    Args:
        mus_hat: PyTorch tensor of mean predictions, shape (L, d).
        sigmas_hat_sq: PyTorch tensor of variance predictions, shape (L, d).
        bin_size: The number of consecutive positions to merge (T).
        strategy: The binning method ('average' or 'sum').

    Returns:
        A tuple of PyTorch tensors containing:
        - binned_mus: The new means, shape (L/T, d).
        - binned_sigmas_sq: The new variances, shape (L/T, d).
    """
    # Validate the strategy input
    if strategy not in ['average', 'sum']:
        raise ValueError("strategy must be either 'average' or 'sum'")

    # Ensure the input arrays have a 2D shape (L, d)
    if mus_hat.ndim == 1:
        mus_hat = mus_hat.view(-1, 1)
        sigmas_hat_sq = sigmas_hat_sq.view(-1, 1)

    # Reshape to group data into bins of size T using .view()
    # Shape becomes (num_bins, bin_size, num_assays)
    mus_in_bins = mus_hat.view(-1, bin_size, mus_hat.shape[1])
    sigmas_in_bins = sigmas_hat_sq.view(-1, bin_size, sigmas_hat_sq.shape[1])

    if strategy == 'average':
        # New mean is the average of the means in the bin (dim=1)
        binned_mus = torch.mean(mus_in_bins, dim=1)
        # New variance is the average of variances, scaled by the bin size
        binned_sigmas_sq = torch.mean(sigmas_in_bins, dim=1) / bin_size
    
    elif strategy == 'sum':
        # New mean is the sum of the means in the bin (dim=1)
        binned_mus = torch.sum(mus_in_bins, dim=1)
        # New variance is the sum of the variances
        binned_sigmas_sq = torch.sum(sigmas_in_bins, dim=1)

    return binned_mus, binned_sigmas_sq

def binarize_nbinom(data, threshold=0.0001):
    """
    Fits a Negative Binomial distribution to the data, binarizes the data based on a threshold.

    Parameters:
    data (numpy array): The input data array to analyze.
    threshold (float): The probability threshold for binarization. Default is 0.0001.
    """
    # Ensure the data is a NumPy array
    data = np.asarray(data)

    # Step 1: Fit Negative Binomial Distribution
    def negbinom_loglik(params, data):
        r, p = params
        return -np.sum(stats.nbinom.logpmf(data, r, p))

    initial_params = [1, 0.5]  # Initial guess for r and p
    result_nbinom = minimize(negbinom_loglik, initial_params, args=(data), bounds=[(1e-5, None), (1e-5, 1-1e-5)])
    r, p = result_nbinom.x

    # Step 2: Calculate Probabilities under the Negative Binomial distribution
    probabilities = stats.nbinom.pmf(data, r, p)

    # Step 3: Binarize the Data
    binarized_data = (probabilities < threshold).astype(int)

    return binarized_data

class VISUALS_CANDI(object):
    def __init__(self, resolution=25, savedir="models/evals/"):
        self.metrics = METRICS()
        self.resolution = resolution
        self.savedir = savedir

    def clear_pallete(self):
        sns.reset_orig
        plt.close("all")
        plt.style.use('default')
        plt.clf()

    def count_track(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        example_gene_coord = (33481539//self.resolution, 33588914//self.resolution) # GART
        example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            example_gene_coord, example_gene_coord2, example_gene_coord3,
            example_gene_coord4, example_gene_coord5]

        # Define the size of the figure
        plt.figure(figsize=(6 * len(example_gene_coords), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])
                imputed_values = eval_res[j]["pred_count"][gene_coord[0]:gene_coord[1]]

                # Plot the lines
                if "obs_count" in eval_res[j].keys():
                    observed_values = eval_res[j]["obs_count"][gene_coord[0]:gene_coord[1]]
                    ax.plot(x_values, observed_values, color="blue", alpha=0.7, label="Observed", linewidth=0.1)
                    ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")

                ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, label="Imputed", linewidth=0.1)
                ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)

                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Count")

                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                                mlines.Line2D([], [], color='red',  label='Imputed')]
                ax.legend(handles=custom_lines)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_tracks.png", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_tracks.svg", format="svg")
    
    def signal_track(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        example_gene_coord = (33481539//self.resolution, 33588914//self.resolution) # GART
        example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            example_gene_coord, example_gene_coord2, example_gene_coord3,
            example_gene_coord4, example_gene_coord5]

        # Define the size of the figure
        plt.figure(figsize=(6 * len(example_gene_coords), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])
                imputed_values = eval_res[j]["pred_pval"][gene_coord[0]:gene_coord[1]]

                # Plot the lines
                if "obs_pval" in eval_res[j].keys():
                    observed_values = eval_res[j]["obs_pval"][gene_coord[0]:gene_coord[1]]
                    ax.plot(x_values, observed_values, color="blue", alpha=0.7, label="Observed", linewidth=0.1)
                    ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")

                ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, label="Imputed", linewidth=0.1)
                ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)

                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Signal")

                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                                mlines.Line2D([], [], color='red',  label='Imputed')]
                ax.legend(handles=custom_lines)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.png", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.svg", format="svg")

    def count_confidence(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution) # ITSN1
            ]

        # Define the size of the figure
        plt.figure(figsize=(8 * len(example_gene_coords), len(eval_res) * 2))

        for j, result in enumerate(eval_res):
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])

                # Fill between for confidence intervals
                ax.fill_between(
                    x_values, result['count_lower_95'][gene_coord[0]:gene_coord[1]], result['count_upper_95'][gene_coord[0]:gene_coord[1]], 
                    color='coral', alpha=0.4, label='95% Confidence')

                # Plot the median predictions
                ax.plot(x_values, result['pred_count'][gene_coord[0]:gene_coord[1]], label='Mean', color='red', linewidth=0.5)

                if "obs_count" in result.keys():
                    # Plot the actual observations
                    ax.plot(
                        x_values, result['obs_count'][gene_coord[0]:gene_coord[1]], 
                        label='Observed', color='royalblue', linewidth=0.4, alpha=0.8)


                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution

                # Set plot titles and labels
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Count")
                ax.set_yscale('log') 
                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                # Only show legend in the first subplot to avoid redundancy
                if i == 0 and j ==0:
                    ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_95CI.pdf", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_95CI.svg", format='svg')
    
    def signal_confidence(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution)  # ITSN1
            ]

        # Define the size of the figure
        plt.figure(figsize=(8 * len(example_gene_coords), len(eval_res) * 2))

        for j, result in enumerate(eval_res):
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])

                # Fill between for confidence intervals
                ax.fill_between(
                    x_values, result['pval_lower_95'][gene_coord[0]:gene_coord[1]], result['pval_upper_95'][gene_coord[0]:gene_coord[1]], 
                    color='coral', alpha=0.4, label='95% Confidence')

                # Plot the median predictions
                ax.plot(x_values, result['pred_pval'][gene_coord[0]:gene_coord[1]], label='Mean', color='red', linewidth=0.5)

                if "obs_pval" in result.keys():
                    # Plot the actual observations
                    ax.plot(
                        x_values, result['obs_pval'][gene_coord[0]:gene_coord[1]], 
                        label='Observed', color='royalblue', linewidth=0.4, alpha=0.8)


                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution

                # Set plot titles and labels
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Signal")
                ax.set_yscale('log') 
                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                # Only show legend in the first subplot to avoid redundancy
                if i == 0 and j ==0:
                    ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_95CI.pdf", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_95CI.svg", format="svg")

    def quantile_hist(self, eval_res, b=20):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs_count" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs_count"], eval_res[j]["pred_quantile"]

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"], bin_size=self.resolution)

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"])

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"])

                ax.hist(ys, bins=b, color='blue', alpha=0.7, density=True)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}")
                ax.set_xlabel("Predicted CDF Quantile")
                ax.set_ylabel("Density")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_quantile_hist.png", dpi=150)

    def quantile_heatmap(self, eval_res, b=20):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs_count" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs_count"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['C_Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['C_Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['C_Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['C_Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_count"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['C_Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(np.asarray(xs), np.asarray(ys), bins=b, density=True)
                h = h.T  # Transpose to correct the orientation
                h = h / h.sum(axis=0, keepdims=True)  # Normalize cols

                im = ax.imshow(
                    h, interpolation='nearest', origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    aspect='auto', cmap='viridis', norm=LogNorm())
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted Quantiles")
                plt.colorbar(im, ax=ax, orientation='vertical')
                

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_heatmap.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_heatmap.svg", format="svg")

    def count_error_std_hexbin(self, eval_res):
        save_path = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        num_plots = len(eval_res) * 3  # Each evaluation will have 3 subplots
        plt.figure(figsize=(15, len(eval_res) * 5))  # Adjust width for 3 columns

        for j in range(len(eval_res)):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pcc = f"PCC_GW: {eval_res[j]['C_Pearson-GW']:.2f}"
            error = np.abs(observed - pred_mean)

            # Calculate the percentiles for x-axis limits
            x_90 = np.percentile(error, 90)
            x_99 = np.percentile(error, 99)

            # Define the ranges for subsetting
            # ranges = [(0, x_90), (0, x_99), (0, error.max())]
            ranges = [(0, x_90), (x_90, x_99), (x_99, error.max())]

            for i, (x_min, x_max) in enumerate(ranges):
                # Subset the data for the current range
                mask = (error >= x_min) & (error <= x_max)
                subset_error = error[mask]
                subset_pred_std = pred_std[mask]
                
                ax = plt.subplot(len(eval_res), 3, j * 3 + i + 1)

                # Hexbin plot for the subset data
                # hb = ax.hexbin(subset_error, subset_pred_std, gridsize=50, cmap='viridis', mincnt=1, norm=LogNorm())
                hb = ax.hexbin(subset_error, subset_pred_std, gridsize=50, cmap='Purples', mincnt=0, bins='log', vmin=1e-1, reduce_C_function=lambda x: np.where(x == 0, 1, x))

                max_val = subset_error.max()
                ax.set_xlim(x_min, max_val)
                ax.set_ylim(0, max_val)
                
                # Add diagonal line
                ax.plot([0, max_val], [0, max_val], '--', color='yellow', alpha=0.8)
                
                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Predicted Std Dev')
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc} (Range: {x_min:.2f}-{x_max:.2f})")

                # Add color bar
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Log10(Counts)')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/count_error_std_hexbin.png", dpi=150)
        plt.savefig(f"{save_path}/count_error_std_hexbin.svg", format="svg")

    def signal_error_std_hexbin(self, eval_res):
        save_path = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        num_plots = len(eval_res) * 3  # Each evaluation will have 3 subplots
        plt.figure(figsize=(15, len(eval_res) * 5))  # Adjust width for 3 columns

        for j in range(len(eval_res)):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pcc = f"PCC_GW: {eval_res[j]['P_Pearson-GW']:.2f}"
            error = np.abs(observed - pred_mean)

            # Calculate the percentiles for x-axis limits
            x_90 = np.percentile(error, 90)
            x_99 = np.percentile(error, 99)

            # Define the ranges for subsetting
            # ranges = [(0, x_90), (0, x_99), (0, error.max())]
            ranges = [(0, x_90), (0, x_99), (0, error.max())]

            for i, (x_min, x_max) in enumerate(ranges):
                # Subset the data for the current range
                mask = (error >= x_min) & (error <= x_max)
                subset_error = error[mask]
                subset_pred_std = pred_std[mask]
                
                ax = plt.subplot(len(eval_res), 3, j * 3 + i + 1)

                # Hexbin plot for the subset data
                # hb = ax.hexbin(subset_error, subset_pred_std, gridsize=50, cmap='viridis', mincnt=1, norm=LogNorm())
                hb = ax.hexbin(subset_error, subset_pred_std, gridsize=50, cmap='Purples', mincnt=0, bins='log', vmin=1e-1, reduce_C_function=lambda x: np.where(x == 0, 1, x))

                max_val = subset_error.max()
                ax.set_xlim(x_min, max_val)
                ax.set_ylim(0, max_val)
                
                # Add diagonal line
                ax.plot([0, max_val], [0, max_val], '--', color='yellow', alpha=0.8)

                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Predicted Std Dev')
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc} (Range: {x_min:.2f}-{x_max:.2f})")

                # Add color bar
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Log10(signal)')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/signal_error_std_hexbin.png", dpi=150)
        plt.savefig(f"{save_path}/signal_error_std_hexbin.svg", format="svg")

    def count_mean_std_hexbin(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        plt.figure(figsize=(5, len(eval_res) * 5))  # one column with len(eval_res) rows

        for j in range(len(eval_res)):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(len(eval_res), 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pcc = f"PCC_GW: {eval_res[j]['C_Pearson-GW']:.2f}"

            hb = ax.hexbin(observed, pred_mean, C=pred_std, gridsize=30, cmap='viridis', reduce_C_function=np.mean)
            plt.colorbar(hb, ax=ax, label='Predicted std')
            ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted Mean')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc}")
            # plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_mean_std_hexbin.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_mean_std_hexbin.svg", format="svg")
    
    def signal_mean_std_hexbin(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        plt.figure(figsize=(5, len(eval_res) * 5))  # one column with len(eval_res) rows

        for j in range(len(eval_res)):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(len(eval_res), 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pcc = f"PCC_GW: {eval_res[j]['P_Pearson-GW']:.2f}"

            hb = ax.hexbin(observed, pred_mean, C=pred_std, gridsize=30, cmap='viridis', reduce_C_function=np.mean)
            plt.colorbar(hb, ax=ax, label='Predicted std')
            ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted Mean')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc}")
            # plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_mean_std_hexbin.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_mean_std_hexbin.svg", format="svg")

    def count_scatter_with_marginals(self, eval_res, share_axes=True, percentile_cutoff=99):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]
        num_rows = len(eval_res)
        num_cols = len(cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        for j, result in enumerate(eval_res):
            if "obs_count" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = axes[j, i] if num_rows > 1 else axes[i]

                if c == "GW":
                    xs, ys = eval_res[j]["obs_count"], eval_res[j]["pred_count"]
                    pcc = f"PCC_GW: {eval_res[j]['C_Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['C_Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['C_Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"])
                    pcc = f"PCC_1obs: {eval_res[j]['C_Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"])
                    pcc = f"PCC_1imp: {eval_res[j]['C_Pearson_1imp']:.2f}"
                    
                sns.scatterplot(x=xs, y=ys, ax=ax, color="#4CB391", s=3, alpha=0.9)

                # Calculate percentile cutoffs for both axes
                x_upper = np.percentile(xs, percentile_cutoff)
                y_upper = np.percentile(ys, percentile_cutoff)
                
                # Use the same upper bound for both axes to maintain aspect ratio
                upper_bound = min(x_upper, y_upper)
                
                # Filter points within bounds
                mask = (xs <= upper_bound) & (ys <= upper_bound)
                xs_filtered = xs[mask]
                ys_filtered = ys[mask]

                # Update bin range for histograms using filtered data
                bin_range = np.linspace(0, upper_bound, 50)

                ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
                ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
                
                ax_histx.hist(xs_filtered, bins=bin_range, alpha=0.9, color="#f25a64")
                ax_histy.hist(ys_filtered, bins=bin_range, orientation='horizontal', alpha=0.9, color="#f25a64")
                
                ax_histx.set_xticklabels([])
                ax_histx.set_yticklabels([])
                ax_histy.set_xticklabels([])
                ax_histy.set_yticklabels([])

                # Set title, labels, and range if share_axes is True
                ax.set_title(f"{result['feature']}_{c}_{result['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

                if share_axes:
                    # Set axis limits
                    ax.set_xlim(0, upper_bound)
                    ax.set_ylim(0, upper_bound)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_scatters_with_marginals.png", dpi=150)
        # plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_scatters_with_marginals.svg", format="svg")

    def signal_scatter_with_marginals(self, eval_res, share_axes=True, percentile_cutoff=99):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]
        num_rows = len(eval_res)
        num_cols = len(cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        for j, result in enumerate(eval_res):
            if "obs_pval" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = axes[j, i] if num_rows > 1 else axes[i]

                if c == "GW":
                    xs, ys = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"]
                    pcc = f"PCC_GW: {eval_res[j]['P_Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['P_Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['P_Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"])
                    pcc = f"PCC_1obs: {eval_res[j]['P_Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"])
                    pcc = f"PCC_1imp: {eval_res[j]['P_Pearson_1imp']:.2f}"
                    
                sns.scatterplot(x=xs, y=ys, ax=ax, color="#4CB391", s=3, alpha=0.9)

                # Calculate percentile cutoffs for both axes
                x_upper = np.percentile(xs, percentile_cutoff)
                y_upper = np.percentile(ys, percentile_cutoff)
                
                # Use the same upper bound for both axes to maintain aspect ratio
                upper_bound = min(x_upper, y_upper)
                
                # Filter points within bounds
                mask = (xs <= upper_bound) & (ys <= upper_bound)
                xs_filtered = xs[mask]
                ys_filtered = ys[mask]

                # Update bin range for histograms using filtered data
                bin_range = np.linspace(0, upper_bound, 50)

                ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
                ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
                
                ax_histx.hist(xs_filtered, bins=bin_range, alpha=0.9, color="#f25a64")
                ax_histy.hist(ys_filtered, bins=bin_range, orientation='horizontal', alpha=0.9, color="#f25a64")
                
                ax_histx.set_xticklabels([])
                ax_histx.set_yticklabels([])
                ax_histy.set_xticklabels([])
                ax_histy.set_yticklabels([])

                # Set title, labels, and range if share_axes is True
                ax.set_title(f"{result['feature']}_{c}_{result['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

                if share_axes:
                    # Set axis limits
                    ax.set_xlim(0, upper_bound)
                    ax.set_ylim(0, upper_bound)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.png", dpi=150)
        # plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.svg", format="svg")

    def count_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs_count" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs_count"], eval_res[j]["pred_count"]
                    title_suffix = f"PCC_GW: {eval_res[j]['C_Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"], bin_size=self.resolution)
                    title_suffix = f"PCC_Gene: {eval_res[j]['C_Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"], bin_size=self.resolution)
                    title_suffix = f"PCC_TSS: {eval_res[j]['C_Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"])
                    title_suffix = f"PCC_1obs: {eval_res[j]['C_Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"])
                    title_suffix = f"PCC_1imp: {eval_res[j]['C_Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=True)
                h = h.T  # Transpose to correct the orientation
                ax.imshow(h, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', norm=LogNorm())

                if share_axes:
                    common_min = min(min(xs), min(ys))
                    common_max = max(max(xs), max(ys))
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{title_suffix}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_heatmaps.svg", format="svg")

    def signal_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs_pval" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"]
                    title_suffix = f"PCC_GW: {eval_res[j]['P_Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], bin_size=self.resolution)
                    title_suffix = f"PCC_Gene: {eval_res[j]['P_Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], bin_size=self.resolution)
                    title_suffix = f"PCC_TSS: {eval_res[j]['P_Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"])
                    title_suffix = f"PCC_1obs: {eval_res[j]['P_Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"])
                    title_suffix = f"PCC_1imp: {eval_res[j]['P_Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=True)
                h = h.T  # Transpose to correct the orientation
                ax.imshow(h, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', norm=LogNorm())

                if share_axes:
                    common_min = min(min(xs), min(ys))
                    common_max = max(max(xs), max(ys))
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{title_suffix}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.svg", format="svg")

    def count_rank_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs_count" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs_count"], eval_res[j]["pred_count"]
                    scc = f"SRCC_GW: {eval_res[j]['C_Spearman-GW']:.2f}"
                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"], bin_size=self.resolution)
                    scc = f"SRCC_Gene: {eval_res[j]['C_Spearman_gene']:.2f}"
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"], bin_size=self.resolution)
                    scc = f"SRCC_TSS: {eval_res[j]['C_Spearman_prom']:.2f}"
                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"])
                    scc = f"SRCC_1obs: {eval_res[j]['C_Spearman_1obs']:.2f}"
                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_count"], eval_res[j]["pred_count"])
                    scc = f"SRCC_1imp: {eval_res[j]['C_Spearman_1imp']:.2f}"

                # Rank the values and handle ties
                xs = rankdata(xs, method="ordinal")  # Use average ranks for ties
                ys = rankdata(ys, method="ordinal")

                # Create a 2D histogram
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins)#, density=True)
                h = np.where(np.isnan(h), 1e-6, h)  # Replace NaN values with the minimum value of h
                h = h.T
                h = h + 1 

                ax.imshow(
                    h, interpolation='nearest', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap="Purples", norm=LogNorm())

                if share_axes:
                    common_min = min(xedges[0], yedges[0])
                    common_max = max(xedges[-1], yedges[-1])
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{scc}")
                ax.set_xlabel("Observed | rank")
                ax.set_ylabel("Predicted | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_rank_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_rank_heatmaps.svg", format="svg")

    def signal_rank_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs_pval" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"]
                    scc = f"SRCC_GW: {eval_res[j]['P_Spearman-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], bin_size=self.resolution)
                    scc = f"SRCC_Gene: {eval_res[j]['P_Spearman_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], bin_size=self.resolution)
                    scc = f"SRCC_TSS: {eval_res[j]['P_Spearman_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"])
                    scc = f"SRCC_1obs: {eval_res[j]['P_Spearman_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs_pval"], eval_res[j]["pred_pval"])
                    scc = f"SRCC_1imp: {eval_res[j]['P_Spearman_1imp']:.2f}"

                # Rank the values and handle ties
                xs = rankdata(xs, method="ordinal")  # Use average ranks for ties
                ys = rankdata(ys, method="ordinal")

                # Create a 2D histogram
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins)#, density=True)
                h = np.where(np.isnan(h), 1e-6, h) 
                h = h.T
                h = h + 1

                ax.imshow(
                    h, interpolation='nearest', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap='Purples', norm=LogNorm())

                if share_axes:
                    common_min = min(xedges[0], yedges[0])
                    common_max = max(xedges[-1], yedges[-1])
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{scc}")
                ax.set_xlabel("Observed | rank")
                ax.set_ylabel("Predicted | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_heatmaps.svg", format="svg")
        
    def count_context_length_specific_performance(self, eval_res, context_length, bins=10):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        list_of_metrics = ['MSE-GW', 'Pearson-GW', 'Spearman-GW']

        # Define the size of the figure
        plt.figure(figsize=(6 * len(list_of_metrics), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs_count" not in eval_res[j]:
                continue

            observed_values = eval_res[j]["obs_count"]
            imputed_values = eval_res[j]["pred_count"]

            bin_size = context_length // bins

            observed_values = observed_values.reshape(-1, context_length)
            imputed_values = imputed_values.reshape(-1, context_length)

            observed_values = observed_values.reshape(observed_values.shape[0]*bin_size, bins)
            imputed_values = imputed_values.reshape(imputed_values.shape[0]*bin_size, bins)

            for i, m in enumerate(list_of_metrics):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(list_of_metrics), j * len(list_of_metrics) + i + 1)
                
                xs = [float(xt)/bins for xt in range(bins)]
                # Calculate x_values based on the current gene's coordinates
                ys = []
                for b in range(bins):
                    
                    obs, imp = observed_values[:,b].flatten(), imputed_values[:,b].flatten()
                    if m == 'MSE-GW':
                        ys.append(self.metrics.mse(obs, imp))

                    elif m == 'Pearson-GW':
                        ys.append(self.metrics.pearson(obs, imp))

                    elif m == 'Spearman-GW':
                        ys.append(self.metrics.spearman(obs, imp))
                
                ax.plot(xs, ys, color="grey", linewidth=3)
                # ax.fill_between(xs, 0, ys, alpha=0.7, color="grey")
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_xlabel("position in context")
                ax.set_ylabel(m)
        
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_context.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_context.svg", format="svg")
    
    def signal_context_length_specific_performance(self, eval_res, context_length, bins=10):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        list_of_metrics = ['MSE-GW', 'Pearson-GW', 'Spearman-GW']

        # Define the size of the figure
        plt.figure(figsize=(6 * len(list_of_metrics), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs_pval" not in eval_res[j]:
                continue

            observed_values = eval_res[j]["obs_pval"]
            imputed_values = eval_res[j]["pred_pval"]

            bin_size = context_length // bins

            observed_values = observed_values.reshape(-1, context_length)
            imputed_values = imputed_values.reshape(-1, context_length)

            observed_values = observed_values.reshape(observed_values.shape[0]*bin_size, bins)
            imputed_values = imputed_values.reshape(imputed_values.shape[0]*bin_size, bins)

            for i, m in enumerate(list_of_metrics):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(list_of_metrics), j * len(list_of_metrics) + i + 1)
                
                xs = [float(xt)/bins for xt in range(bins)]
                # Calculate x_values based on the current gene's coordinates
                ys = []
                for b in range(bins):
                    
                    obs, imp = observed_values[:,b].flatten(), imputed_values[:,b].flatten()
                    if m == 'MSE-GW':
                        ys.append(self.metrics.mse(obs, imp))

                    elif m == 'Pearson-GW':
                        ys.append(self.metrics.pearson(obs, imp))

                    elif m == 'Spearman-GW':
                        ys.append(self.metrics.spearman(obs, imp))
                
                ax.plot(xs, ys, color="grey", linewidth=3)
                # ax.fill_between(xs, 0, ys, alpha=0.7, color="grey")
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_xlabel("position in context")
                ax.set_ylabel(m)
        
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_context.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_context.svg", format="svg")

    def count_TSS_confidence_boxplot(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        tss_coords = self.metrics.get_prom_positions("chr21", 25).reset_index(drop=True)

        isTSS = np.zeros(len(eval_res[0]["obs_count"]), dtype=bool)        
        for t in range(len(tss_coords)):
            isTSS[tss_coords["start"][t]:tss_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isTSS': isTSS
            })

            # Create categories
            # df['isTSS'] = np.where(df['isTSS'] == 1, 'TSS', 'NonTSS')
            df = df[df['isTSS'] == True].copy()

            # Create 4 signal intensity categories based on quantiles
            q25 = df['mu'].quantile(0.25)
            q50 = df['mu'].quantile(0.50)
            q75 = df['mu'].quantile(0.75)

            conditions = [
                (df['mu'] < q25),
                (df['mu'] >= q25) & (df['mu'] < q50),
                (df['mu'] >= q50) & (df['mu'] < q75),
                (df['mu'] >= q75)
            ]
            choices = ['[0-p25)', '[p25-p50)', '[p50-p75)', '[p75-p100]']
            df['SigIntensityCategory'] = np.select(conditions, choices, default='[0-p25)')

            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Define category order based on choices
            cat_order = [
                'HighConf_[0-p25)', 'LowConf_[0-p25)',
                'HighConf_[p25-p50)', 'LowConf_[p25-p50)',
                'HighConf_[p50-p75)', 'LowConf_[p50-p75)',
                'HighConf_[p75-p100]', 'LowConf_[p75-p100]'
            ]

            # Plot with observed values on y-axis
            sns.set_style("whitegrid")
            sns.boxplot(
                data=df,
                x='conf_pred_cat',
                y='obs',  # Changed from 'mu' to 'obs'
                # hue='isTSS',
                order=cat_order,
                # palette={'TSS': 'salmon', 'NonTSS': 'grey'},
                color='salmon',  # Changed color to red
                ax=ax,
                showfliers=False
            )  # End of Selection

            ax.set_xlabel("Predicted Count")
            ax.set_ylabel("Observed Count")  # Updated label
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            # ax.legend(title="Region")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_TSS_confidence_boxplot.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_TSS_confidence_boxplot.svg", format="svg")

    def signal_TSS_confidence_boxplot(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        tss_coords = self.metrics.get_prom_positions("chr21", 25).reset_index(drop=True)

        isTSS = np.zeros(len(eval_res[0]["obs_pval"]), dtype=bool)        
        for t in range(len(tss_coords)):
            isTSS[tss_coords["start"][t]:tss_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isTSS': isTSS
            })

            # Create categories
            # df['isTSS'] = np.where(df['isTSS'] == 1, 'TSS', 'NonTSS')
            df = df[df['isTSS'] == True].copy()

            # Create 4 signal intensity categories based on quantiles
            q25 = df['mu'].quantile(0.25)
            q50 = df['mu'].quantile(0.50)
            q75 = df['mu'].quantile(0.75)

            conditions = [
                (df['mu'] < q25),
                (df['mu'] >= q25) & (df['mu'] < q50),
                (df['mu'] >= q50) & (df['mu'] < q75),
                (df['mu'] >= q75)
            ]
            choices = ['[0-p25)', '[p25-p50)', '[p50-p75)', '[p75-p100]']
            df['SigIntensityCategory'] = np.select(conditions, choices, default='[0-p25)')

            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Define category order based on choices
            cat_order = [
                'HighConf_[0-p25)', 'LowConf_[0-p25)',
                'HighConf_[p25-p50)', 'LowConf_[p25-p50)',
                'HighConf_[p50-p75)', 'LowConf_[p50-p75)',
                'HighConf_[p75-p100]', 'LowConf_[p75-p100]'
            ]

            # Plot with observed values on y-axis
            sns.set_style("whitegrid")
            sns.boxplot(
                data=df,
                x='conf_pred_cat',
                y='obs',  # Changed from 'mu' to 'obs'
                # hue='isTSS',
                order=cat_order,
                color='salmon',
                # palette={'TSS': 'salmon', 'NonTSS': 'grey'},
                ax=ax,
                showfliers=False
            )

            ax.set_xlabel("Predicted Signal")
            ax.set_ylabel("Observed Signal")  # Updated label
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            # ax.legend(title="Region")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_TSS_confidence_boxplot.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_TSS_confidence_boxplot.svg", format="svg")

    def count_GeneBody_confidence_boxplot(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        gb_coords = self.metrics.get_gene_positions("chr21", 25).reset_index(drop=True)

        isGB = np.zeros(len(eval_res[0]["obs_count"]), dtype=bool)        
        for t in range(len(gb_coords)):
            isGB[gb_coords["start"][t]:gb_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isGB': isGB
            })

            # Create categories
            # df['isGB'] = np.where(df['isGB'] == 1, 'GeneBody', 'NonGeneBody')
            df = df[df['isGB'] == True].copy()

            # Create 4 signal intensity categories based on quantiles
            q25 = df['mu'].quantile(0.25)
            q50 = df['mu'].quantile(0.50)
            q75 = df['mu'].quantile(0.75)

            conditions = [
                (df['mu'] < q25),
                (df['mu'] >= q25) & (df['mu'] < q50),
                (df['mu'] >= q50) & (df['mu'] < q75),
                (df['mu'] >= q75)
            ]
            choices = ['[0-p25)', '[p25-p50)', '[p50-p75)', '[p75-p100]']
            df['SigIntensityCategory'] = np.select(conditions, choices, default='[0-p25)')

            

            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Define category order based on choices
            cat_order = [
                'HighConf_[0-p25)', 'LowConf_[0-p25)',
                'HighConf_[p25-p50)', 'LowConf_[p25-p50)',
                'HighConf_[p50-p75)', 'LowConf_[p50-p75)',
                'HighConf_[p75-p100]', 'LowConf_[p75-p100]'
            ]

            # Plot with observed values on y-axis
            sns.set_style("whitegrid")
            sns.boxplot(
                data=df,
                x='conf_pred_cat',
                y='obs',  # Changed from 'mu' to 'obs'
                # hue='isGB',
                order=cat_order,
                color='yellowgreen',  # Changed color to green
                ax=ax,
                showfliers=False
            )

            ax.set_xlabel("Predicted Count")
            ax.set_ylabel("Observed Count")  # Updated label
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            # ax.legend(title="Region")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_GeneBody_confidence_boxplot.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_Genebody_confidence_boxplot.svg", format="svg")

    def signal_GeneBody_confidence_boxplot(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        gb_coords = self.metrics.get_gene_positions("chr21", 25).reset_index(drop=True)

        isGB = np.zeros(len(eval_res[0]["obs_pval"]), dtype=bool)        
        for t in range(len(gb_coords)):
            isGB[gb_coords["start"][t]:gb_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isGB': isGB
            })

            # Create categories
            # df['isGB'] = np.where(df['isGB'] == 1, 'GeneBody', 'NonGeneBody')
            df = df[df['isGB'] == True].copy()

            # Create 4 signal intensity categories based on quantiles
            q25 = df['mu'].quantile(0.25)
            q50 = df['mu'].quantile(0.50)
            q75 = df['mu'].quantile(0.75)

            conditions = [
                (df['mu'] < q25),
                (df['mu'] >= q25) & (df['mu'] < q50),
                (df['mu'] >= q50) & (df['mu'] < q75),
                (df['mu'] >= q75)
            ]
            choices = ['[0-p25)', '[p25-p50)', '[p50-p75)', '[p75-p100]']
            df['SigIntensityCategory'] = np.select(conditions, choices, default='[0-p25)')

            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Define category order based on choices
            cat_order = [
                'HighConf_[0-p25)', 'LowConf_[0-p25)',
                'HighConf_[p25-p50)', 'LowConf_[p25-p50)',
                'HighConf_[p50-p75)', 'LowConf_[p50-p75)',
                'HighConf_[p75-p100]', 'LowConf_[p75-p100]'
            ]

            # Plot with observed values on y-axis
            sns.set_style("whitegrid")
            sns.boxplot(
                data=df,
                x='conf_pred_cat',
                y='obs',  # Changed from 'mu' to 'obs'
                # hue='isGB',
                color='yellowgreen',
                order=cat_order,
                # palette={'GeneBody': 'yellowgreen', 'NonGeneBody': 'grey'},
                ax=ax,
                showfliers=False
            )

            ax.set_xlabel("Predicted Signal")
            ax.set_ylabel("Observed Signal")  # Updated label
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            # ax.legend(title="Region")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_GeneBody_confidence_boxplot.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_GeneBody_confidence_boxplot.svg", format="svg")

    def count_obs_vs_confidence(self, eval_res, n_bins=100):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        for j in range(num_plots):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pred_CV = pred_std / pred_mean

            # Create 100 bins based on observed values
            # n_bins = 100
            bins = np.linspace(np.min(observed), np.max(observed), n_bins + 1)
            digitized = np.digitize(observed, bins)
            
            # Calculate statistics for each bin
            bin_means = []
            bin_cv_means = []
            bin_cv_stds = []
            bin_centers = []
            
            # Loop through each bin to calculate statistics
            for i in range(1, n_bins + 1):
                # Check if there are any data points in the current bin
                if len(pred_CV[digitized == i]) > 0:  # Only include bins with data
                    # Calculate the center of the current bin
                    bin_centers.append((bins[i-1] + bins[i])/2)
                    # Calculate the mean of observed values for the current bin
                    bin_means.append(np.mean(observed[digitized == i]))
                    # Calculate the mean of the coefficient of variation for the current bin
                    bin_cv_means.append(np.mean(pred_CV[digitized == i]))
                    # Calculate the standard deviation of the coefficient of variation for the current bin
                    bin_cv_stds.append(np.std(pred_CV[digitized == i]))
            
            bin_centers = np.array(bin_centers)
            bin_cv_means = np.array(bin_cv_means)
            bin_cv_stds = np.array(bin_cv_stds)
            
            # Plot the binned data with error bars
            ax.errorbar(bin_centers, bin_cv_means, yerr=bin_cv_stds, 
                    fmt='o', color='#4CB391', markersize=4, 
                    ecolor='grey', capsize=2, alpha=0.7)
            
            # Fit and plot trend line
            # z = np.polyfit(bin_centers, bin_cv_means, 2)
            # p = np.poly1d(z)
            # x_trend = np.linspace(min(bin_centers), max(bin_centers), 100)
            # ax.plot(x_trend, p(x_trend), '--', color='#f25a64', alpha=0.8)
            
            ax.set_xlabel('Observed Count')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            
            # Set log scale for x-axis if data spans multiple orders of magnitude
            # if np.max(observed)/np.min(observed) > 100:
            #     ax.set_xscale('log')
            
            # Add grid
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_obs_vs_confidence.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_obs_vs_confidence.svg", format="svg")

    def signal_obs_vs_confidence(self, eval_res, n_bins=100):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        for j in range(num_plots):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pred_CV = pred_std / pred_mean

            # Create 100 bins based on observed values
            # n_bins = 100
            bins = np.linspace(np.min(observed), np.max(observed), n_bins + 1)
            digitized = np.digitize(observed, bins)
            
            # Calculate statistics for each bin
            bin_means = []
            bin_cv_means = []
            bin_cv_stds = []
            bin_centers = []
            
            # Loop through each bin to calculate statistics
            for i in range(1, n_bins + 1):
                # Check if there are any data points in the current bin
                if len(pred_CV[digitized == i]) > 0:  # Only include bins with data
                    # Calculate the center of the current bin
                    bin_centers.append((bins[i-1] + bins[i])/2)
                    # Calculate the mean of observed values for the current bin
                    bin_means.append(np.mean(observed[digitized == i]))
                    # Calculate the mean of the coefficient of variation for the current bin
                    bin_cv_means.append(np.mean(pred_CV[digitized == i]))
                    # Calculate the standard deviation of the coefficient of variation for the current bin
                    bin_cv_stds.append(np.std(pred_CV[digitized == i]))
            
            bin_centers = np.array(bin_centers)
            bin_cv_means = np.array(bin_cv_means)
            bin_cv_stds = np.array(bin_cv_stds)
            
            # Plot the binned data with error bars
            ax.errorbar(bin_centers, bin_cv_means, yerr=bin_cv_stds, 
                    fmt='o', color='#4CB391', markersize=4, 
                    ecolor='grey', capsize=2, alpha=0.7)
            
            # Fit and plot trend line
            # z = np.polyfit(bin_centers, bin_cv_means, 2)
            # p = np.poly1d(z)
            # x_trend = np.linspace(min(bin_centers), max(bin_centers), 100)
            # ax.plot(x_trend, p(x_trend), '--', color='#f25a64', alpha=0.8)
            
            ax.set_xlabel('Observed signal')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            
            # Set log scale for x-axis if data spans multiple orders of magnitude
            # if np.max(observed)/np.min(observed) > 100:
            #     ax.set_xscale('log')
            
            # Add grid
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_obs_vs_confidence.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_obs_vs_confidence.svg", format="svg")

    def count_TSS_enrichment_v_confidence(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        tss_coords = self.metrics.get_prom_positions("chr21", 25).reset_index(drop=True)

        isTSS = np.zeros(len(eval_res[0]["obs_count"]), dtype=bool)        
        for t in range(len(tss_coords)):
            isTSS[tss_coords["start"][t]:tss_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isTSS': isTSS
            })

            df['isTSS'] = np.where(df['isTSS'] == 1, 'TSS', 'NonTSS')
            deciles = np.arange(0, 110, 10)  # 0,10,20,...,100
            bins = np.percentile(df['mu'], deciles)
            df['SigIntensityCategory'] = pd.qcut(df['mu'], q=10, labels=[f'[p{i}-p{i+10})' for i in range(0, 100, 10)])

            # For each signal intensity bin, calculate confidence threshold separately
            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Convert 'confidenceCategory' to string type to avoid TypeError
            df['confidenceCategory'] = df['confidenceCategory'].astype(str)

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'].astype(str) + '_' + df['SigIntensityCategory'].astype(str)

            # Calculate fraction of TSS in each category
            tss_fractions = df.groupby('conf_pred_cat')['isTSS'].apply(lambda x: (x == 'TSS').mean()).reset_index()
            tss_fractions.columns = ['conf_pred_cat', 'tss_fraction']

            # Define category order
            cat_order = []
            for sig_cat in [f'[p{i}-p{i+10})' for i in range(0, 100, 10)]:
                cat_order.extend([f'HighConf_{sig_cat}', f'LowConf_{sig_cat}'])

            # Create bar plot
            sns.set_style("whitegrid")
            sns.barplot(
                data=tss_fractions,
                x='conf_pred_cat',
                y='tss_fraction',
                order=cat_order,
                color='salmon',
                alpha=0.7,
                ax=ax
            )

            ax.set_xlabel("Predicted Signal Bins")
            ax.set_ylabel("Fraction of TSS")
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_TSS_enrichment_v_confidence.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_TSS_enrichment_v_confidence.svg", format="svg")

    def signal_TSS_enrichment_v_confidence(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        tss_coords = self.metrics.get_prom_positions("chr21", 25).reset_index(drop=True)

        isTSS = np.zeros(len(eval_res[0]["obs_pval"]), dtype=bool)        
        for t in range(len(tss_coords)):
            isTSS[tss_coords["start"][t]:tss_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isTSS': isTSS
            })

            df['isTSS'] = np.where(df['isTSS'] == 1, 'TSS', 'NonTSS')
            deciles = np.arange(0, 110, 10)  # 0,10,20,...,100
            bins = np.percentile(df['mu'], deciles)
            df['SigIntensityCategory'] = pd.qcut(df['mu'], q=10, labels=[f'[p{i}-p{i+10})' for i in range(0, 100, 10)])

            # For each signal intensity bin, calculate confidence threshold separately
            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Convert to string to avoid TypeError when concatenating
            df['confidenceCategory'] = df['confidenceCategory'].astype(str)
            df['SigIntensityCategory'] = df['SigIntensityCategory'].astype(str)

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Calculate fraction of TSS in each category
            tss_fractions = df.groupby('conf_pred_cat')['isTSS'].apply(lambda x: (x == 'TSS').mean()).reset_index()
            tss_fractions.columns = ['conf_pred_cat', 'tss_fraction']

            # Define category order
            cat_order = []
            for sig_cat in [f'[p{i}-p{i+10})' for i in range(0, 100, 10)]:
                cat_order.extend([f'HighConf_{sig_cat}', f'LowConf_{sig_cat}'])

            # Create bar plot
            sns.set_style("whitegrid")
            sns.barplot(
                data=tss_fractions,
                x='conf_pred_cat',
                y='tss_fraction',
                order=cat_order,
                color='salmon',
                alpha=0.7,
                ax=ax
            )

            ax.set_xlabel("Predicted Signal Bins")
            ax.set_ylabel("Fraction of TSS")
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_TSS_enrichment_v_confidence.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_TSS_enrichment_v_confidence.svg", format="svg")

    def count_GeneBody_enrichment_v_confidence(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        gb_coords = self.metrics.get_gene_positions("chr21", 25).reset_index(drop=True)

        isGB = np.zeros(len(eval_res[0]["obs_count"]), dtype=bool)        
        for t in range(len(gb_coords)):
            isGB[gb_coords["start"][t]:gb_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_count" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_count"], eval_res[j]["pred_count"], eval_res[j]["pred_count_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isGB': isGB
            })

            df['isGB'] = np.where(df['isGB'] == 1, 'GeneBody', 'NonGeneBody')
            deciles = np.arange(0, 110, 10)  # 0,10,20,...,100
            bins = np.percentile(df['mu'], deciles)
            df['SigIntensityCategory'] = pd.qcut(df['mu'], q=10, labels=[f'[p{i}-p{i+10})' for i in range(0, 100, 10)])

            # For each signal intensity bin, calculate confidence threshold separately
            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Convert to string to avoid TypeError when concatenating
            df['confidenceCategory'] = df['confidenceCategory'].astype(str)
            df['SigIntensityCategory'] = df['SigIntensityCategory'].astype(str)

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Calculate fraction of TSS in each category
            gb_fraction = df.groupby('conf_pred_cat')['isGB'].apply(lambda x: (x == 'GeneBody').mean()).reset_index()
            gb_fraction.columns = ['conf_pred_cat', 'gb_fraction']

            # Define category order
            cat_order = []
            for sig_cat in [f'[p{i}-p{i+10})' for i in range(0, 100, 10)]:
                cat_order.extend([f'HighConf_{sig_cat}', f'LowConf_{sig_cat}'])

            # Create bar plot
            sns.set_style("whitegrid")
            sns.barplot(
                data=gb_fraction,
                x='conf_pred_cat',
                y='gb_fraction',
                order=cat_order,
                color='yellowgreen',
                alpha=0.7,
                ax=ax
            )

            ax.set_xlabel("Predicted Signal Bins")
            ax.set_ylabel("Fraction of GeneBody")
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_GeneBody_enrichment_v_confidence.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/count_GeneBody_enrichment_v_confidence.svg", format="svg")

    def signal_GeneBody_enrichment_v_confidence(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        num_plots = len(eval_res)
        plt.figure(figsize=(10, 5 * num_plots))  # Adjusted for multipanel figure

        gb_coords = self.metrics.get_gene_positions("chr21", 25).reset_index(drop=True)

        isGB = np.zeros(len(eval_res[0]["obs_pval"]), dtype=bool)        
        for t in range(len(gb_coords)):
            isGB[gb_coords["start"][t]:gb_coords["end"][t]] = True

        for j in range(num_plots):
            if "obs_pval" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(num_plots, 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs_pval"], eval_res[j]["pred_pval"], eval_res[j]["pred_pval_std"]
            pred_CV = pred_std / pred_mean

            df = pd.DataFrame({
                'obs': observed,
                'mu': pred_mean,
                'sigma': pred_std,
                'cv': pred_CV,
                'isGB': isGB
            })

            # Create categories
            df['isGB'] = np.where(df['isGB'] == 1, 'GeneBody', 'NonGeneBody')
            deciles = np.arange(0, 110, 10)  # 0,10,20,...,100
            bins = np.percentile(df['mu'], deciles)
            df['SigIntensityCategory'] = pd.qcut(df['mu'], q=10, labels=[f'[p{i}-p{i+10})' for i in range(0, 100, 10)])

            # For each signal intensity bin, calculate confidence threshold separately
            df['confidenceCategory'] = 'LowConf'  # initialize all as low confidence
            for sig_cat in df['SigIntensityCategory'].unique():
                mask = df['SigIntensityCategory'] == sig_cat
                cv_median = df[mask]['cv'].quantile(0.5)
                df.loc[mask & (df['cv'] < cv_median), 'confidenceCategory'] = 'HighConf'

            # Convert to string to avoid TypeError when concatenating
            df['confidenceCategory'] = df['confidenceCategory'].astype(str)
            df['SigIntensityCategory'] = df['SigIntensityCategory'].astype(str)

            # Combine confidence and signal intensity
            df['conf_pred_cat'] = df['confidenceCategory'] + '_' + df['SigIntensityCategory']

            # Calculate fraction of TSS in each category
            gb_fraction = df.groupby('conf_pred_cat')['isGB'].apply(lambda x: (x == 'GeneBody').mean()).reset_index()
            gb_fraction.columns = ['conf_pred_cat', 'gb_fraction']

            # Define category order
            cat_order = []
            for sig_cat in [f'[p{i}-p{i+10})' for i in range(0, 100, 10)]:
                cat_order.extend([f'HighConf_{sig_cat}', f'LowConf_{sig_cat}'])

            # Create bar plot
            sns.set_style("whitegrid")
            sns.barplot(
                data=gb_fraction,
                x='conf_pred_cat',
                y='gb_fraction',
                order=cat_order,
                color='yellowgreen',
                alpha=0.7,
                ax=ax
            )

            ax.set_xlabel("Predicted Signal Bins")
            ax.set_ylabel("Fraction of GeneBody")
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_GeneBody_enrichment_v_confidence.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_GeneBody_enrichment_v_confidence.svg", format="svg")

    def count_metagene(self, eval_res, flank_bp: int = 1000, gene_body_bins: int = 100):
        """
        Meta-gene count profiles, one subplot per assay.
        Red = predicted; Blue = observed.
        """

        # make output directory
        outdir = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        os.makedirs(outdir, exist_ok=True)

        flank_bins = flank_bp // self.resolution
        n = len(eval_res)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
        if n == 1: axes = [axes]

        # length of each track
        L = len(eval_res[0]['pred_count'])
        # use gene coordinates (start,end,strand in bin units)
        gene_df = self.metrics.gene_df

        for ax, res in zip(axes, eval_res):
            pred = res['pred_count']
            obs  = res.get('obs_count', None)

            for _, g in gene_df.iterrows():
                # 1) determine TSS/TES based on strand
                # if g['strand'] == '+':
                tss, tes = int(g.start), int(g.end)
                # else:
                #     tss, tes = int(g.end),   int(g.start)

                # 2) upstream indices = [tss-flank_bins, ..., tss-1]
                up_idx = np.arange(tss - flank_bins, tss)
                up_vals = np.full(flank_bins, np.nan)
                m_up = (up_idx >= 0) & (up_idx < L)
                up_vals[m_up] = pred[up_idx[m_up]]
                if obs is not None:
                    up_obs = np.full(flank_bins, np.nan)
                    up_obs[m_up] = obs[up_idx[m_up]]

                # 3) gene-body → fixed # bins via averaging
                body_idx = np.arange(min(tss, tes), max(tss, tes))
                if len(body_idx) > 0:
                    chunks = np.array_split(body_idx, gene_body_bins)
                    b_pred = np.array([pred[c].mean() for c in chunks])
                    b_obs  = np.array([obs[c].mean()  for c in chunks]) if obs is not None else None
                else:
                    b_pred = np.full(gene_body_bins, np.nan)
                    b_obs  = np.full(gene_body_bins, np.nan) if obs is not None else None

                # 4) downstream indices = [tes, tes+1, ..., tes+flank_bins-1]
                dn_idx = np.arange(tes, tes + flank_bins)
                dn_vals = np.full(flank_bins, np.nan)
                m_dn = (dn_idx >= 0) & (dn_idx < L)
                dn_vals[m_dn] = pred[dn_idx[m_dn]]
                if obs is not None:
                    dn_obs = np.full(flank_bins, np.nan)
                    dn_obs[m_dn] = obs[dn_idx[m_dn]]

                # 5) stitch into one profile
                prof_pred = np.concatenate([up_vals, b_pred, dn_vals])
                prof_obs  = (np.concatenate([up_obs, b_obs, dn_vals])
                             if obs is not None else None)

                # 6) if on the minus strand, flip so upstream is always left
                if g['strand'] == '-':
                    prof_pred = prof_pred[::-1]
                    if prof_obs is not None:
                        prof_obs = prof_obs[::-1]

                # 7) x-axis from -flank_bins → gene_body_bins+flank_bins-1
                x = np.arange(-flank_bins, gene_body_bins + flank_bins)
                ax.plot(x, prof_pred, color='red',   alpha=0.3, lw=0.8)
                if prof_obs is not None:
                    ax.plot(x, prof_obs,  color='blue',  alpha=0.3, lw=0.8)

            # 8) mark TSS (0) & TES (gene_body_bins)
            ax.axvline(0,               color='k', ls='--')
            ax.axvline(gene_body_bins, color='k', ls='--')
            ax.set_xlim(-flank_bins, gene_body_bins + flank_bins)
            ax.set_xticks([-flank_bins, 0, gene_body_bins, gene_body_bins+flank_bins])
            ax.set_xticklabels([f"-{flank_bp//1000} kb", "TSS", "TES", f"+{flank_bp//1000} kb"])
            ax.set_title(f"{res['feature']} | {res['comparison']}")
            ax.set_xlabel("Position rel. to gene")

        axes[0].set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir + "count_metagene.png", dpi=150)
        plt.savefig(outdir + "count_metagene.svg", format="svg")

    def signal_metagene(self, eval_res, flank_bp: int = 1000, gene_body_bins: int = 100):
        """
        Meta-gene signal (p-value) profiles, one subplot per assay.
        Red = predicted; Blue = observed.
        """

        outdir = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        os.makedirs(outdir, exist_ok=True)

        flank_bins = flank_bp // self.resolution
        n = len(eval_res)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
        if n == 1: axes = [axes]

        L = len(eval_res[0]['pred_pval'])
        gene_df = self.metrics.gene_df

        for ax, res in zip(axes, eval_res):
            pred = res['pred_pval']
            obs  = res.get('obs_pval', None)

            for _, g in gene_df.iterrows():
                # if g['strand'] == '+':
                tss, tes = int(g.start), int(g.end)
                # else:
                    # tss, tes = int(g.end),   int(g.start)

                up_idx = np.arange(tss - flank_bins, tss)
                up_vals = np.full(flank_bins, np.nan)
                m_up = (up_idx >= 0) & (up_idx < L)
                up_vals[m_up] = pred[up_idx[m_up]]
                if obs is not None:
                    up_obs = np.full(flank_bins, np.nan)
                    up_obs[m_up] = obs[up_idx[m_up]]

                body_idx = np.arange(min(tss,tes), max(tss,tes))
                if len(body_idx) > 0:
                    chunks = np.array_split(body_idx, gene_body_bins)
                    b_pred = np.array([pred[c].mean() for c in chunks])
                    b_obs  = np.array([obs[c].mean()  for c in chunks]) if obs is not None else None
                else:
                    b_pred = np.full(gene_body_bins, np.nan)
                    b_obs  = np.full(gene_body_bins, np.nan) if obs is not None else None

                dn_idx = np.arange(tes, tes + flank_bins)
                dn_vals = np.full(flank_bins, np.nan)
                m_dn = (dn_idx >= 0) & (dn_idx < L)
                dn_vals[m_dn] = pred[dn_idx[m_dn]]
                if obs is not None:
                    dn_obs = np.full(flank_bins, np.nan)
                    dn_obs[m_dn] = obs[dn_idx[m_dn]]

                prof_pred = np.concatenate([up_vals, b_pred, dn_vals])
                prof_obs  = (np.concatenate([up_obs, b_obs, dn_vals])
                             if obs is not None else None)

                if g['strand'] == '-':
                    prof_pred = prof_pred[::-1]
                    if prof_obs is not None:
                        prof_obs = prof_obs[::-1]

                x = np.arange(-flank_bins, gene_body_bins + flank_bins)
                ax.plot(x, prof_pred, color='red',   alpha=0.4, lw=0.8)
                if prof_obs is not None:
                    ax.plot(x, prof_obs,  color='blue',  alpha=0.4, lw=0.8)

            ax.axvline(0,               color='k', ls='--')
            ax.axvline(gene_body_bins, color='k', ls='--')
            ax.set_xlim(-flank_bins, gene_body_bins + flank_bins)
            ax.set_xticks([-flank_bins, 0, gene_body_bins, gene_body_bins+flank_bins])
            ax.set_xticklabels([f"-{flank_bp//1000} kb", "TSS", "TES", f"+{flank_bp//1000} kb"])
            ax.set_title(f"{res['feature']} | {res['comparison']}")
            ax.set_xlabel("Position rel. to gene")

        axes[0].set_ylabel("Signal (p-value)")
        plt.tight_layout()
        plt.savefig(outdir + "signal_metagene.png", dpi=150)
        plt.savefig(outdir + "signal_metagene.svg", format="svg")

    def metagene2_count(self, eval_res, flank_bp: int = 1000, gene_body_bins: int = 100):
        """
        Multipanel meta-gene plot for counts:
         - one subplot per assay in eval_res
         - light-grey hairball for every gene
         - red median curve + 25–75% ribbon for predictions
         - blue median curve + ribbon for observations (if present)
         - shaded rectangle for gene body, dashed lines at TSS/TES
        """
        import os, numpy as np, matplotlib.pyplot as plt

        # 1) Prepare output dir
        outdir = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        os.makedirs(outdir, exist_ok=True)

        # 2) bins
        flank_bins = flank_bp // self.resolution
        total_bins = flank_bins + gene_body_bins + flank_bins
        x = np.arange(-flank_bins, gene_body_bins + flank_bins)

        # 3) get gene coordinates (start,end,strand) in bin units
        genes = self.metrics.gene_df[['start','end','strand']]

        # 4) make one row with one column per assay
        n = len(eval_res)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
        if n == 1: axes = [axes]

        for ax, res in zip(axes, eval_res):
            pred = res['pred_count']
            obs  = res.get('obs_count', None)
            P, M = [], []

            L = len(pred)
            # 5) build per-gene profiles
            for _, g in genes.iterrows():
                # TSS/TES depending on strand
                # if g.strand == '+':
                tss, tes = int(g.start), int(g.end)
                # else:
                #     tss, tes = int(g.end),   int(g.start)

                # upstream
                up_idx = np.arange(tss - flank_bins, tss)
                up_vals = np.full(flank_bins, np.nan)
                m_up = (up_idx>=0)&(up_idx<L)
                up_vals[m_up] = pred[up_idx[m_up]]

                # gene body → split into fixed bins
                body_idx = np.arange(min(tss,tes), max(tss,tes))
                if len(body_idx)>0:
                    chunks = np.array_split(body_idx, gene_body_bins)
                    b_vals = np.array([pred[c].mean() for c in chunks])
                else:
                    b_vals = np.full(gene_body_bins, np.nan)

                # downstream
                dn_idx = np.arange(tes, tes + flank_bins)
                dn_vals = np.full(flank_bins, np.nan)
                m_dn = (dn_idx>=0)&(dn_idx<L)
                dn_vals[m_dn] = pred[dn_idx[m_dn]]

                prof = np.concatenate([up_vals, b_vals, dn_vals])
                if g.strand=='-':
                    prof = prof[::-1]
                P.append(prof)

                # same for observed if present
                if obs is not None:
                    up_o = np.full(flank_bins, np.nan)
                    up_o[m_up] = obs[up_idx[m_up]]
                    if len(body_idx)>0:
                        b_o = np.array([obs[c].mean() for c in chunks])
                    else:
                        b_o = np.full(gene_body_bins, np.nan)
                    dn_o = np.full(flank_bins, np.nan)
                    dn_o[m_dn] = obs[dn_idx[m_dn]]
                    prof_o = np.concatenate([up_o, b_o, dn_o])
                    if g.strand=='-':
                        prof_o = prof_o[::-1]
                    M.append(prof_o)

            P = np.vstack(P)
            M = np.vstack(M) if obs is not None else None

            # 6) plot hairball in grey
            for row in P:
                ax.plot(x, row, color='grey', alpha=0.1, lw=0.5)
            if M is not None:
                for row in M:
                    ax.plot(x, row, color='grey', alpha=0.1, lw=0.5)

            # 7) overlay median + IQR ribbon
            medP = np.nanmedian(P, axis=0)
            q1P, q3P = np.nanpercentile(P, [25,75], axis=0)
            ax.fill_between(x, q1P, q3P, color='red',   alpha=0.3)
            ax.plot(x, medP,          color='red',   lw=2, label='Pred median')

            if M is not None:
                medM = np.nanmedian(M, axis=0)
                q1M, q3M = np.nanpercentile(M, [25,75], axis=0)
                ax.fill_between(x, q1M, q3M, color='blue',  alpha=0.3)
                ax.plot(x, medM,          color='blue',  lw=2, label='Obs median')

            # 8) shade gene body region
            ax.axvspan(0, gene_body_bins, color='lightgrey', alpha=0.2)

            # 9) dashed TSS/TES
            ax.axvline(0,               ls='--', color='k')
            ax.axvline(gene_body_bins, ls='--', color='k')

            # 10) axis labels & title
            ax.set_xlim(-flank_bins, gene_body_bins + flank_bins)
            ax.set_xticks([-flank_bins, 0, gene_body_bins, gene_body_bins+flank_bins])
            ax.set_xticklabels([f"-{flank_bp//1000} kb", "TSS", "TES", f"+{flank_bp//1000} kb"])
            ax.set_xlabel("Position rel. to gene")
            ax.set_title(f"{res['feature']} | {res['comparison']}")
            if obs is not None:
                ax.legend(frameon=False, fontsize='small')

        axes[0].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(outdir + "count_metagene2_panel.png", dpi=150)
        fig.savefig(outdir + "count_metagene2_panel.svg", format="svg")
        plt.close(fig)

    def metagene2_signal(self, eval_res, flank_bp: int = 1000, gene_body_bins: int = 100):
        """
        Same as metagene2_count_panel but for signal (p-value).
        """

        outdir = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        os.makedirs(outdir, exist_ok=True)

        flank_bins = flank_bp // self.resolution
        total_bins = flank_bins + gene_body_bins + flank_bins
        x = np.arange(-flank_bins, gene_body_bins + flank_bins)

        genes = self.metrics.gene_df[['start','end','strand']]

        n = len(eval_res)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
        if n == 1: axes = [axes]

        for ax, res in zip(axes, eval_res):
            pred = res['pred_pval']
            obs  = res.get('obs_pval', None)
            P, M = [], []

            L = len(pred)
            for _, g in genes.iterrows():
                # if g.strand == '+':
                tss, tes = int(g.start), int(g.end)
                # else:
                #     tss, tes = int(g.end),   int(g.start)

                up_idx = np.arange(tss - flank_bins, tss)
                up_vals = np.full(flank_bins, np.nan)
                m_up = (up_idx>=0)&(up_idx<L)
                up_vals[m_up] = pred[up_idx[m_up]]

                body_idx = np.arange(min(tss,tes), max(tss,tes))
                if len(body_idx)>0:
                    chunks = np.array_split(body_idx, gene_body_bins)
                    b_vals = np.array([pred[c].mean() for c in chunks])
                else:
                    b_vals = np.full(gene_body_bins, np.nan)

                dn_idx = np.arange(tes, tes + flank_bins)
                dn_vals = np.full(flank_bins, np.nan)
                m_dn = (dn_idx>=0)&(dn_idx<L)
                dn_vals[m_dn] = pred[dn_idx[m_dn]]

                prof = np.concatenate([up_vals, b_vals, dn_vals])
                if g.strand=='-':
                    prof = prof[::-1]
                P.append(prof)

                if obs is not None:
                    up_o = np.full(flank_bins, np.nan)
                    up_o[m_up] = obs[up_idx[m_up]]
                    if len(body_idx)>0:
                        b_o = np.array([obs[c].mean() for c in chunks])
                    else:
                        b_o = np.full(gene_body_bins, np.nan)
                    dn_o = np.full(flank_bins, np.nan)
                    dn_o[m_dn] = obs[dn_idx[m_dn]]
                    prof_o = np.concatenate([up_o, b_o, dn_o])
                    if g.strand=='-':
                        prof_o = prof_o[::-1]
                    M.append(prof_o)

            P = np.vstack(P)
            M = np.vstack(M) if obs is not None else None

            for row in P:
                ax.plot(x, row, color='grey', alpha=0.1, lw=0.5)
            if M is not None:
                for row in M:
                    ax.plot(x, row, color='grey', alpha=0.1, lw=0.5)

            medP = np.nanmedian(P, axis=0)
            q1P, q3P = np.nanpercentile(P, [25,75], axis=0)
            ax.fill_between(x, q1P, q3P, color='red',   alpha=0.3)
            ax.plot(x, medP,          color='red',   lw=2, label='Pred median')

            if M is not None:
                medM = np.nanmedian(M, axis=0)
                q1M, q3M = np.nanpercentile(M, [25,75], axis=0)
                ax.fill_between(x, q1M, q3M, color='blue',  alpha=0.3)
                ax.plot(x, medM,          color='blue',  lw=2, label='Obs median')

            ax.axvspan(0, gene_body_bins, color='lightgrey', alpha=0.2)
            ax.axvline(0,               ls='--', color='k')
            ax.axvline(gene_body_bins, ls='--', color='k')

            ax.set_xlim(-flank_bins, gene_body_bins + flank_bins)
            ax.set_xticks([-flank_bins, 0, gene_body_bins, gene_body_bins+flank_bins])
            ax.set_xticklabels([f"-{flank_bp//1000} kb", "TSS", "TES", f"+{flank_bp//1000} kb"])
            ax.set_xlabel("Position rel. to gene")
            ax.set_title(f"{res['feature']} | {res['comparison']}")
            if obs is not None:
                ax.legend(frameon=False, fontsize='small')

        axes[0].set_ylabel("Signal (p-value)")
        fig.tight_layout()
        fig.savefig(outdir + "signal_metagene2_panel.png", dpi=150)
        fig.savefig(outdir + "signal_metagene2_panel.svg", format="svg")
        plt.close(fig)

def auc_rec(y_true, y_pred):
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Sort the errors
    sorted_errors = np.sort(errors)
    
    # Calculate cumulative proportion of points within error tolerance
    cumulative_proportion = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # Calculate AUC-REC
    auc_rec = auc(sorted_errors, cumulative_proportion)

    # Determine the maximum possible area under the curve
    max_error = sorted_errors[-1]  # Maximum error tolerance observed
    max_area = max_error * 1  # Since the cumulative proportion ranges from 0 to 1
    
    # Normalize AUC-REC
    normalized_auc_rec = auc_rec / max_area if max_area > 0 else 0

    return normalized_auc_rec

def k_fold_cross_validation(data, k=10, target='TPM', logscale=True, model_type='linear'):
    """
    Perform k-fold cross-validation for linear regression on the provided data.
    
    Args:
        data (pd.DataFrame): The DataFrame containing features and labels.
        k (int): The number of folds for cross-validation.
        target (str): The label to predict ('TPM' or 'FPKM'). Default is 'TPM'.
        
    Returns:
        avg_mse (float): The average Mean Squared Error across all folds.
        avg_r2 (float): The average R-squared value across all folds.
    """
    # Get unique gene IDs
    unique_gene_ids = data["geneID"].unique()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    if logscale:
        data["TPM"] = np.log(data["TPM"]+1)
    
    mse_scores = []
    r2_scores = []
    auc_recs = []
    pearsonrs = []
    spearmanrs = []

    all_errors = []

    # Perform K-Fold Cross Validation
    for train_index, test_index in kf.split(unique_gene_ids):
        train_gene_ids = unique_gene_ids[train_index]
        test_gene_ids = unique_gene_ids[test_index]

        # Split the data into training and testing sets
        train_data = data[data["geneID"].isin(train_gene_ids)]
        test_data = data[data["geneID"].isin(test_gene_ids)]
    
        # Extract features and labels
        X_train = train_data[["promoter_signal", "gene_body_signal", "TES_signal"]]
        y_train = train_data[target]
        
        X_test = test_data[["promoter_signal", "gene_body_signal", "TES_signal"]]
        y_test = test_data[target]

        # Initialize the model based on the model type
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        elif model_type == 'loess':
            # LOESS (Local Regression) model
            lowess_predictions = lowess(y_train, X_train.mean(axis=1), frac=0.3)
            # For LOESS, we use an averaging of predictions as there is no direct fit/predict method
            y_pred = lowess_predictions[:, 1]
        
        # Collect errors from this fold
        errors = np.abs(y_test - y_pred)
        all_errors.extend(errors)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        aucrec = auc_rec(y_test, y_pred)
        pcc = pearsonr(y_pred, y_test)[0]
        scrr = spearmanr(y_pred, y_test)[0]
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        auc_recs.append(aucrec)
        pearsonrs.append(pcc)
        spearmanrs.append(scrr)

    # Compute average metrics
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_aucrec = np.mean(auc_recs)
    avg_pcc = np.mean(pearsonrs)
    avg_srcc = np.mean(spearmanrs)

    all_errors = np.array(all_errors)

    return {
        'avg_mse': avg_mse,
        'avg_r2': avg_r2,
        'avg_aucrec': avg_aucrec,
        'avg_pcc': avg_pcc,
        'avg_srcc': avg_srcc,
        "errors": all_errors
    }

class EVAL_CANDI(object):
    def __init__(
        self, model, data_path, context_length, batch_size, hyper_parameters_path="",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", resolution=25, must_have_chr_access=True,
        savedir="models/evals/", mode="eval", split="test", eic=False, DNA=False,
        DINO=False, ENC_CKP=None, DEC_CKP=None):

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.mkdir(self.savedir)

        self.data_path = data_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution
        self.split = split
        
        self.eic = eic
        self.DNA = DNA
        self.DINO = DINO

        self.model = model
        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        
        self.dataset.init_eval(
            self.context_length, check_completeness=True, split=split, 
            bios_min_exp_avail_threshold=3, eic=eic, merge_ct=True, 
            must_have_chr_access=must_have_chr_access)

        self.gene_coords = load_gene_coords("data/parsed_genecode_data_hg38_release42.csv")
        self.gene_coords = self.gene_coords[self.gene_coords["chr"] == "chr21"].reset_index(drop=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.token_dict = {
                    "missing_mask": -1, 
                    "cloze_mask": -2,
                    "pad": -3
                }

        self.chr_sizes = {}
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.train_data = {}
        self.eval_data = {}
        self.metrics = METRICS()
        self.viz = VISUALS_CANDI(resolution=self.resolution, savedir=self.savedir)

        if mode == "dev":
            return

        if type(self.model) == str:
            if self.DINO:
                with open(hyper_parameters_path, 'rb') as f:
                    self.hyper_parameters = pickle.load(f)
                    self.hyper_parameters["signal_dim"] = self.dataset.signal_dim
                    self.hyper_parameters["metadata_embedding_dim"] = self.dataset.signal_dim*4

                modelpath = self.model

                # print(self.hyper_parameters)
                self.model = MergedDINO(
                    encoder_ckpt_path=ENC_CKP,
                    decoder_ckpt_path=DEC_CKP,
                    signal_dim=            self.hyper_parameters["signal_dim"],
                    metadata_embedding_dim=self.hyper_parameters["metadata_dim"],
                    conv_kernel_size     = self.hyper_parameters["conv_kernel"],
                    n_cnn_layers         = self.hyper_parameters["ncnn"],
                    nhead                = self.hyper_parameters["nhead"],
                    n_sab_layers         = self.hyper_parameters["nsab"],
                    pool_size            = self.hyper_parameters["pool_size"],
                    dropout              = self.hyper_parameters["dropout"],
                    context_length       = self.hyper_parameters["ctx_len"],
                    pos_enc              = self.hyper_parameters["pos_enc"],
                    expansion_factor     = self.hyper_parameters["exp_factor"],
                    pooling_type         = "attention"
                )

                self.model.load_state_dict(torch.load(modelpath))
                print("Loaded DINO pretrained merged encoder and decoder!")

            else:
                with open(hyper_parameters_path, 'rb') as f:
                    self.hyper_parameters = pickle.load(f)
                    self.hyper_parameters["signal_dim"] = self.dataset.signal_dim
                    self.hyper_parameters["metadata_embedding_dim"] = self.dataset.signal_dim*4
                loader = CANDI_LOADER(model, self.hyper_parameters, DNA=self.DNA)
                self.model = loader.load_CANDI()
        
        

        self.expnames = list(self.dataset.aliases["experiment_aliases"].keys())
        self.mark_dict = {i: self.expnames[i] for i in range(len(self.expnames))}

        self.model = self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode
        summary(self.model)
        print(f"# model_parameters: {count_parameters(self.model)}")

    def eval_rnaseq(
        self, bios_name, y_pred, y_true, 
        availability, plot_REC=True,
        k_folds: int = 5,        
        random_state: int = 42,
        observed="pval"):        
        """
        k-Fold evaluation over all genes (no single train/test split):
        train on k-1 folds, test on the held-out fold, repeat.
        Supports: linear, lasso, ridge, elasticnet, svr.
        """

        # 1) load full-genome RNA-seq table 
        rna_seq_data = self.dataset.load_rna_seq_data(bios_name, self.gene_coords)

        # build gene_info lookup 
        gene_info = (
            rna_seq_data[['geneID','chr','TPM','FPKM']]
            .drop_duplicates(subset='geneID')
            .set_index('geneID')
        )

        # 2) build long-format
        long_rows_true = []
        long_rows_pred = []
        for _, row in rna_seq_data.iterrows():
            gene, start, end, strand = row['geneID'], row['start'], row['end'], row['strand']
            for a in range(y_pred.shape[1]):
                assay = self.expnames[a]
                # true only if available
                if a in availability:
                    feats = signal_feature_extraction(start, end, strand, y_true[:, a].numpy())
                    for suffix, val in feats.items():
                        long_rows_true.append({
                            'geneID': gene,
                            'feature': f"{assay}_{suffix}",
                            'signal': val
                        })
                # pred for all
                feats = signal_feature_extraction(start, end, strand, y_pred[:, a].numpy())
                for suffix, val in feats.items():
                    long_rows_pred.append({
                        'geneID': gene,
                        'feature': f"{assay}_{suffix}",
                        'signal': val
                    })

        df_true_long = pd.DataFrame(long_rows_true)
        df_pred_long = pd.DataFrame(long_rows_pred)

        # 3) pivot to wide (unchanged)
        df_true_wide = df_true_long.pivot_table(
            index='geneID', columns='feature', values='signal', aggfunc='mean').fillna(0)
        df_pred_wide_all = df_pred_long.pivot_table(
            index='geneID', columns='feature', values='signal', aggfunc='mean').fillna(0)

        available_assays = { self.expnames[a] for a in availability }
        mask = df_pred_long['feature'].str.split('_').str[0].isin(available_assays)
        df_pred_wide_avail = df_pred_long[mask].pivot_table(
            index='geneID', columns='feature', values='signal', aggfunc='mean').fillna(0)

        # join chr/TPM back (unchanged)
        for df in (df_true_wide, df_pred_wide_all, df_pred_wide_avail):
            df[['chr','TPM','FPKM']] = gene_info[['chr','TPM','FPKM']]

        # 4) assemble feature matrices for each version ⬅ changed
        def build_xy(df):
            feat_cols = [c for c in df.columns if c not in ['chr','TPM','FPKM']]
            X = df[feat_cols].values       # ⬅ changed
            y = np.log1p(df['TPM'].values) # ⬅ changed
            return X, y

        X_true,  y_true_log  = build_xy(df_true_wide)       # ⬅ changed
        X_all,   y_all_log   = build_xy(df_pred_wide_all)   # ⬅ changed
        X_avail, y_avail_log = build_xy(df_pred_wide_avail) # ⬅ changed

        # 5) set up CV splitter ⬅ changed
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # 6) helper to run one fold for a given dataset & method ⬅ changed
        from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
        from sklearn.svm import SVR

        def one_fold(X, y, train_idx, test_idx, algo):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            if algo == 'linear':
                m = LinearRegression()
            elif algo == 'lasso':
                m = Lasso(max_iter=10000)
            elif algo == 'ridge':
                m = Ridge()
            elif algo == 'elasticnet':
                m = ElasticNet(max_iter=10000)
            else:
                m = SVR(kernel='rbf', C=1.0, epsilon=0.2)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            err  = np.abs(yte - pred)
            return {
                'mse':    mean_squared_error(yte, pred),
                'r2':     r2_score(yte, pred),
                'aucrec': auc_rec(yte, pred),
                'pcc':    pearsonr(pred, yte)[0],
                'scrr':   spearmanr(pred, yte)[0],
                'errors': err
            }

        # 7) run CV for each version & method ⬅ changed
        methods = [
            'linear', 'svr',
            'lasso', 'ridge', 'elasticnet']

        report = { f'{ver}_{m}': {'folds': []} 
                   for ver in ['true','denimp','denav'] for m in methods }

        for fold, (tr_idx, te_idx) in enumerate(kf.split(X_true)):  # ⬅ changed: iterate folds
            for m in methods:
                report[f'true_{m}']['folds'].append(
                    one_fold(X_true, y_true_log, tr_idx, te_idx, m)
                )
                report[f'denimp_{m}']['folds'].append(
                    one_fold(X_all, y_all_log, tr_idx, te_idx, m)
                )
                report[f'denav_{m}']['folds'].append(
                    one_fold(X_avail, y_avail_log, tr_idx, te_idx, m)
                )

        # 8) aggregate across folds ⬅ changed
        for key in report:
            # average scalar metrics over folds
            mses = [f['mse'] for f in report[key]['folds']]
            r2s  = [f['r2']  for f in report[key]['folds']]
            aucs = [f['aucrec'] for f in report[key]['folds']]
            pccs = [f['pcc']    for f in report[key]['folds']]
            scrs = [f['scrr']   for f in report[key]['folds']]
            # concatenate all errors for REC
            all_errs = np.concatenate([f['errors'] for f in report[key]['folds']])
            report[key].update({
                'avg_mse':   np.mean(mses),
                'avg_r2':    np.mean(r2s),
                'avg_aucrec':np.mean(aucs),
                'avg_pcc':   np.mean(pccs),
                'avg_scrr':  np.mean(scrs),
                'errors':    all_errs
            })

        # 9) REC plot (unchanged logic, but now multiple folds) 
        if plot_REC:
            fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5), sharey=True)
            for ax, m in zip(axes, methods):
                for key, color, label in [
                    (f'true_{m}', 'blue','Observed'),
                    (f'denav_{m}','orange','Denoised'),
                    (f'denimp_{m}','green','Denoised+Imputed')
                ]:
                    errs = np.sort(report[key]['errors'])
                    ax.plot(errs,
                            np.arange(1,len(errs)+1)/len(errs),
                            label=label, color=color, alpha=0.7)
                ax.set_title(m)
                ax.set_xlabel('Error tol')
                if m=='linear': ax.set_ylabel('Coverage')
                ax.legend(fontsize='small')
                ax.grid(True)
            plt.tight_layout()
            # out = os.path.join(self.savedir, f"{bios_name}_kfold{ k_folds }")
            out = os.path.join(self.savedir, bios_name+f"_{len(available_assays)}")
            os.makedirs(out, exist_ok=True)
            fig.savefig(f"{out}/RNAseq_REC_{observed}.png", format="png")
            fig.savefig(f"{out}/RNAseq_REC_{observed}.svg", format="svg")

        # 10) save scalar metrics to CSV ⬅ changed: use new avg_*
        rows = []
        for key, stats in report.items():
            rows.append({
                'version': key,
                'mse':     stats['avg_mse'],     # ⬅ changed
                'r2':      stats['avg_r2'],      # ⬅ changed
                'aucrec':  stats['avg_aucrec'],  # ⬅ changed
                'pcc':     stats['avg_pcc'],     # ⬅ changed
                'scrr':    stats['avg_scrr']     # ⬅ changed
            })
        pd.DataFrame(rows).to_csv(f"{out}/RNAseq_results_{observed}.csv", index=False)

        return report

    def quick_eval_rnaseq(self, bios_name, y_pred, y_true, availability, k_folds: int = 5, random_state: int = 42, dtype="pval", margin=2000):
        def stats(x):
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if x.size == 0:
                return 0.0, 0.0, 0.0, 0.0
            med = np.nanmedian(x, axis=0)
            q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
            iqr = q75 - q25
            mn = np.nanmin(x, axis=0)
            mx = np.nanmax(x, axis=0)
            return med, iqr, mn, mx

        # 1) load full-genome RNA-seq table 
        rna_seq_data = self.dataset.load_rna_seq_data(bios_name, self.gene_coords)

        # build gene_info lookup 
        gene_info = (rna_seq_data[['geneID','chr','TPM','FPKM']].drop_duplicates(subset='geneID').set_index('geneID')) 

        DF = []
        for _, row in rna_seq_data.iterrows():
            gene, start, end, strand = row['geneID'], row['start'], row['end'], row['strand']

            if dtype.lower() == "z": 
                # in this case, y_pred is actually predicted latent (z_pred) 
                # y_true is only used to find the resolution difference between latent and actual signal
                y2z_resolution_ratio = y_true.shape[0]/y_pred.shape[0]
                bp2z_ratio = 25*y2z_resolution_ratio
                z_start, z_end = int(start//bp2z_ratio), int(end//bp2z_ratio)

                TSS_z = y_pred[z_start-int(margin//bp2z_ratio):z_start+int(margin//bp2z_ratio)]
                gene_z = y_pred[z_start:z_end]
                TTS_z = y_pred[z_end-int(margin//bp2z_ratio):z_end+int(margin//bp2z_ratio)]

                gene_med, gene_iqr, gene_mn, gene_mx = stats(gene_z)
                tss_med, tss_iqr, tss_mn, tss_mx = stats(TSS_z)
                tts_med, tts_iqr, tts_mn, tts_mx = stats(TTS_z)
                
                for j in range(len(tss_med)):
                    DF.append({'geneID': gene, 'feature': f"Pred_Z_gene_med_f{j}", 'signal': gene_med[j]})
                    DF.append({'geneID': gene, 'feature': f"Pred_Z_gene_iqr_f{j}", 'signal': gene_iqr[j]})

                    DF.append({'geneID': gene, 'feature': f"Pred_Z_tss_med_f{j}", 'signal': tss_med[j]})
                    DF.append({'geneID': gene, 'feature': f"Pred_Z_tss_iqr_f{j}", 'signal': tss_iqr[j]})

                    DF.append({'geneID': gene, 'feature': f"Pred_Z_tts_med_f{j}", 'signal': tts_med[j]})
                    DF.append({'geneID': gene, 'feature': f"Pred_Z_tts_iqr_f{j}", 'signal': tts_iqr[j]})
                
            else:
                for a in range(y_pred.shape[1]):
                    assay = self.expnames[a]
                    # true only if available
                    if a in availability:
                        feats = signal_feature_extraction(start, end, strand, y_true[:, a].numpy())
                        for suffix, val in feats.items():
                            DF.append({
                                'geneID': gene,
                                'feature': f"True_{assay}_{suffix}",
                                'signal': val
                            })

                    # pred for all
                    feats = signal_feature_extraction(start, end, strand, y_pred[:, a].numpy())
                    for suffix, val in feats.items():
                        DF.append({
                            'geneID': gene,
                            'feature': f"Pred_{assay}_{suffix}",
                            'signal': val
                        })

        DF = pd.DataFrame(DF)
        DF = DF.pivot(index='geneID', columns='feature', values='signal')
        DF_Pred = DF.loc[:, [c for c in DF.columns if "Pred" in c]]
        
        Y = pd.Series(np.arcsinh(gene_info.loc[DF.index, "TPM"]))
        if dtype.lower() != "z": 
            available_assays = {self.expnames[a] for a in availability}
            avail_cols = []
            for assay in available_assays:
                for c in DF_Pred.columns:
                    if assay in c:
                        avail_cols.append(c)

            DF_True = DF.loc[:, [c for c in DF.columns if "True" in c]]
            DF_Pred_Denoised = DF_Pred.loc[:, avail_cols]

        def evaluate_pipeline(pipe, X, y, k_folds=5):
            cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            pearson_scores, spearman_scores, mse_scores, mae_scores = [], [], [], []

            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                mse_scores.append(mean_squared_error(y_test, y_pred))
                mae_scores.append(mean_absolute_error(y_test, y_pred))
                pearson_scores.append(pearsonr(y_test, y_pred)[0])
                spearman_scores.append(spearmanr(y_test, y_pred)[0])
            
            return {
                "mse": np.mean(mse_scores),
                "mae": np.mean(mae_scores),
                "pearson": np.mean(pearson_scores),
                "spearman": np.mean(spearman_scores)
            }

        regressors = {
            "linear": LinearRegression(), 
            "ridge": Ridge(),
            "lasso": Lasso(),
            "svr": SVR(kernel='rbf'),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "xgb": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        dim_red_options = {"no_pca": None, "pca_80": PCA(n_components=0.8)}
        results = {}

        for dr_name, dim_red in dim_red_options.items():
            for reg_name, reg in regressors.items():
                steps = [('scale', StandardScaler())]
                if dim_red is not None:
                    steps.append(('pca', dim_red))
                steps.append(('reg', reg))

                pipe = Pipeline(steps)
                if dtype.lower() != "z":
                    results[(dtype, "Obs", dr_name, reg_name)] = evaluate_pipeline(pipe, DF_True, Y, k_folds=5)
                    results[(dtype, "Den", dr_name, reg_name)] = evaluate_pipeline(pipe, DF_Pred_Denoised, Y, k_folds=5)
                    results[(dtype, "Den+Imp", dr_name, reg_name)] = evaluate_pipeline(pipe,  DF_Pred, Y, k_folds=5)
                else:
                    results[("latent", dr_name, reg_name)] = evaluate_pipeline(pipe,  DF_Pred, Y, k_folds=5)
                    
        results = pd.DataFrame(results)
        return results

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None):
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        mu = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        var = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            # torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            if self.DNA:
                seq_batch = seq[i:i + self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                # mY_batch_missing_vals = (mY_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]
                # mY_batch[mY_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    if self.DINO:
                        outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch)
                    else:
                        outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch)
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)


            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()
            mu[i:i+outputs_mu.shape[0], :, :] = outputs_mu.cpu()
            var[i:i+outputs_var.shape[0], :, :] = outputs_var.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var  # Free up memory
            # torch.cuda.empty_cache()  # Free up GPU memory

        return n, p, mu, var

    def get_latent_z(self, X, mX, mY, avail, seq=None):
        """
        Compute the latent representation Z for the input batch using the encoder.
        """
        Z_all = []

        for i in range(0, len(X), self.batch_size):
            x_batch = X[i:i + self.batch_size]
            mX_batch = mX[i:i + self.batch_size]
            mY_batch = mY[i:i + self.batch_size]
            avail_batch = avail[i:i + self.batch_size]

            if self.DNA:
                seq_batch = seq[i:i + self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()
                avail_batch = avail_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    Z = self.model.encode(x_batch.float(), seq_batch, mX_batch)
                else:
                    Z = self.model.encode(x_batch.float(), mX_batch)

            Z_all.append(Z.cpu())

            del x_batch, mX_batch, mY_batch, avail_batch, Z
            # torch.cuda.empty_cache()

        return torch.cat(Z_all, dim=0)

    def get_metrics(
        self, 
        imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, 
        Y, P, bios_name, availability, arcsinh=True, quick=False):

        imp_count_mean = imp_count_dist.mean()
        ups_count_mean = ups_count_dist.mean()

        imp_count_std = imp_count_dist.std()
        ups_count_std = ups_count_dist.std()

        imp_pval_mean = imp_pval_dist.mean()
        ups_pval_mean = ups_pval_dist.mean()

        imp_pval_std = imp_pval_dist.std()
        ups_pval_std = ups_pval_dist.std()

        if not quick:
            imp_count_lower_95, imp_count_upper_95 = imp_count_dist.interval(confidence=0.95)
            ups_count_lower_95, ups_count_upper_95 = ups_count_dist.interval(confidence=0.95)

            imp_pval_lower_95, imp_pval_upper_95 = imp_pval_dist.interval(confidence=0.95)
            ups_pval_lower_95, ups_pval_upper_95 = ups_pval_dist.interval(confidence=0.95)
            print("got 0.95 interval conf")

        results = []

        for j in range(Y.shape[1]):
            if j in list(availability):
                assay_name = self.expnames[j]

                C_target = Y[:, j].numpy()
                P_target = P[:, j].numpy()

                if arcsinh:
                    P_target = np.sinh(P_target)

                for comparison in ['imputed', 'upsampled']:
                    
                    if comparison == "imputed":
                        pred_count = imp_count_mean[:, j].numpy()
                        pred_count_std = imp_count_std[:, j].numpy()

                        pred_count_n = imp_count_dist.n[:, j].numpy()
                        pred_count_p = imp_count_dist.p[:, j].numpy()

                        pred_pval = imp_pval_mean[:, j].numpy()
                        pred_pval_std = imp_pval_std[:, j].numpy()

                        if not quick:
                            count_lower_95 = imp_count_lower_95[:, j].numpy()
                            count_upper_95 = imp_count_upper_95[:, j].numpy()

                            pval_lower_95 = imp_pval_lower_95[:, j].numpy()
                            pval_upper_95 = imp_pval_upper_95[:, j].numpy()

                    elif comparison == "upsampled":
                        pred_count = ups_count_mean[:, j].numpy()
                        pred_count_std = ups_count_std[:, j].numpy()

                        pred_count_n = ups_count_dist.n[:, j].numpy()
                        pred_count_p = ups_count_dist.p[:, j].numpy()

                        pred_pval = ups_pval_mean[:, j].numpy()
                        pred_pval_std = ups_pval_std[:, j].numpy()

                        if not quick:
                            count_lower_95 = ups_count_lower_95[:, j].numpy()
                            count_upper_95 = ups_count_upper_95[:, j].numpy()

                            pval_lower_95 = ups_pval_lower_95[:, j].numpy()
                            pval_upper_95 = ups_pval_upper_95[:, j].numpy()

                    if arcsinh:
                        pred_pval = np.sinh(pred_pval)
                        if not quick:
                            pval_lower_95 = np.sinh(pval_lower_95)
                            pval_upper_95 = np.sinh(pval_upper_95)

                    metrics = {
                        'bios':bios_name,
                        'feature': assay_name,
                        'comparison': comparison,
                        'available assays': len(availability),

                        ########################################################################

                        'C_MSE-GW': self.metrics.mse(C_target, pred_count),
                        'C_Pearson-GW': self.metrics.pearson(C_target, pred_count),
                        'C_Spearman-GW': self.metrics.spearman(C_target, pred_count),
                        'C_r2_GW': self.metrics.r2(C_target, pred_count),
                        'C_Cidx_GW':self.metrics.c_index_nbinom(pred_count_n, pred_count_p, C_target),

                        'C_Pearson_1obs': self.metrics.pearson1_obs(C_target, pred_count),
                        'C_MSE-1obs': self.metrics.mse1obs(C_target, pred_count),
                        'C_Spearman_1obs': self.metrics.spearman1_obs(C_target, pred_count),
                        'C_r2_1obs': self.metrics.r2_1obs(C_target, pred_count),

                        'C_MSE-1imp': self.metrics.mse1imp(C_target, pred_count),
                        'C_Pearson_1imp': self.metrics.pearson1_imp(C_target, pred_count),
                        'C_Spearman_1imp': self.metrics.spearman1_imp(C_target, pred_count),
                        'C_r2_1imp': self.metrics.r2_1imp(C_target, pred_count),

                        'C_MSE-gene': self.metrics.mse_gene(C_target, pred_count),
                        'C_Pearson_gene': self.metrics.pearson_gene(C_target, pred_count),
                        'C_Spearman_gene': self.metrics.spearman_gene(C_target, pred_count),
                        'C_r2_gene': self.metrics.r2_gene(C_target, pred_count),
                        
                        'C_MSE-prom': self.metrics.mse_prom(C_target, pred_count),
                        'C_Pearson_prom': self.metrics.pearson_prom(C_target, pred_count),
                        'C_Spearman_prom': self.metrics.spearman_prom(C_target, pred_count),
                        'C_r2_prom': self.metrics.r2_prom(C_target, pred_count),

                        "C_peak_overlap_01thr": self.metrics.peak_overlap(C_target, pred_count, p=0.01),
                        "C_peak_overlap_05thr": self.metrics.peak_overlap(C_target, pred_count, p=0.05),
                        "C_peak_overlap_10thr": self.metrics.peak_overlap(C_target, pred_count, p=0.10),

                        ########################################################################

                        'P_MSE-GW': self.metrics.mse(P_target, pred_pval),
                        'P_Pearson-GW': self.metrics.pearson(P_target, pred_pval),
                        'P_Spearman-GW': self.metrics.spearman(P_target, pred_pval),
                        'P_r2_GW': self.metrics.r2(P_target, pred_pval),
                        'P_Cidx_GW': self.metrics.c_index_gauss(pred_pval, pred_pval_std, P_target),

                        'P_MSE-1obs': self.metrics.mse1obs(P_target, pred_pval),
                        'P_Pearson_1obs': self.metrics.pearson1_obs(P_target, pred_pval),
                        'P_Spearman_1obs': self.metrics.spearman1_obs(P_target, pred_pval),
                        'P_r2_1obs': self.metrics.r2_1obs(P_target, pred_pval),
                        'P_Cidx_1obs': self.metrics.c_index_gauss_1obs(pred_pval, pred_pval_std, P_target, num_pairs=5000),

                        'P_MSE-1imp': self.metrics.mse1imp(P_target, pred_pval),
                        'P_Pearson_1imp': self.metrics.pearson1_imp(P_target, pred_pval),
                        'P_Spearman_1imp': self.metrics.spearman1_imp(P_target, pred_pval),
                        'P_r2_1imp': self.metrics.r2_1imp(P_target, pred_pval),

                        'P_MSE-gene': self.metrics.mse_gene(P_target, pred_pval),
                        'P_Pearson_gene': self.metrics.pearson_gene(P_target, pred_pval),
                        'P_Spearman_gene': self.metrics.spearman_gene(P_target, pred_pval),
                        'P_r2_gene': self.metrics.r2_gene(P_target, pred_pval),
                        'P_Cidx_gene': self.metrics.c_index_gauss_gene(pred_pval, pred_pval_std, P_target, num_pairs=5000),

                        'P_MSE-prom': self.metrics.mse_prom(P_target, pred_pval),
                        'P_Pearson_prom': self.metrics.pearson_prom(P_target, pred_pval),
                        'P_Spearman_prom': self.metrics.spearman_prom(P_target, pred_pval),
                        'P_r2_prom': self.metrics.r2_prom(P_target, pred_pval),
                        'P_Cidx_prom': self.metrics.c_index_gauss_prom(pred_pval, pred_pval_std, P_target, num_pairs=5000),

                        "P_peak_overlap_01thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.01),
                        "P_peak_overlap_05thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.05),
                        "P_peak_overlap_10thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.10),
                    }
                    
                    if not quick:
                        metrics['C_Cidx_gene'] = self.metrics.c_index_nbinom_gene(pred_count_n, pred_count_p, C_target, num_pairs=2000)
                        metrics['C_Cidx_prom'] = self.metrics.c_index_nbinom_prom(pred_count_n, pred_count_p, C_target, num_pairs=2000)
                        
                        metrics["obs_count"] = C_target
                        metrics["obs_pval"] = P_target

                        metrics["pred_count"] = pred_count
                        metrics["pred_count_std"] = pred_count_std

                        metrics["pred_pval"] = pred_pval
                        metrics["pred_pval_std"] = pred_pval_std

                        metrics["count_lower_95"] = count_lower_95
                        metrics["count_upper_95"] = count_upper_95

                        metrics["pval_lower_95"] = pval_lower_95
                        metrics["pval_upper_95"] = pval_upper_95

                        # if self.dataset.has_rnaseq(bios_name):
                        #     print("got rna-seq data")
                        #     rnaseq_res = self.eval_rnaseq(bios_name, ups_count_mean, Y, availability, k_fold=10, plot_REC=True)

                        #     metrics["rnaseq-true-pcc-linear"] = rnaseq_res["true_linear"]["avg_pcc"]
                        #     metrics["rnaseq-true-pcc-svr"] = rnaseq_res["true_svr"]["avg_pcc"]

                        #     metrics["rnaseq-denoised-pcc-linear"] = rnaseq_res["denoised_linear"]["avg_pcc"]
                        #     metrics["rnaseq-denoised-pcc-svr"] = rnaseq_res["denoised_svr"]["avg_pcc"]

                        #     metrics["rnaseq-true-mse-linear"] = rnaseq_res["true_linear"]["avg_mse"]
                        #     metrics["rnaseq-true-mse-svr"] = rnaseq_res["true_svr"]["avg_mse"]
                            
                        #     metrics["rnaseq-denoised-mse-linear"] = rnaseq_res["denoised_linear"]["avg_mse"]
                        #     metrics["rnaseq-denoised-mse-svr"] = rnaseq_res["denoised_svr"]["avg_mse"]

                    results.append(metrics)

            else:
                pred_count = ups_count_mean[:, j].numpy()
                pred_count_std = ups_count_std[:, j].numpy()

                pred_pval = ups_pval_mean[:, j].numpy()
                pred_pval_std = ups_pval_std[:, j].numpy()

                if not quick:
                    count_lower_95 = ups_count_lower_95[:, j].numpy()
                    count_upper_95 = ups_count_upper_95[:, j].numpy()

                    pval_lower_95 = ups_pval_lower_95[:, j].numpy()
                    pval_upper_95 = ups_pval_upper_95[:, j].numpy()

                if arcsinh:
                    pred_pval = np.sinh(pred_pval)
                    if not quick:
                        pval_lower_95 = np.sinh(pval_lower_95)
                        pval_upper_95 = np.sinh(pval_upper_95)

                metrics = {
                    'bios':bios_name,
                    'feature': self.expnames[j],
                    'comparison': "None",
                    'available assays': len(availability),
                    }

                if not quick:
                    metrics["pred_count"] = pred_count
                    metrics["pred_count_std"] = pred_count_std

                    metrics["pred_pval"] = pred_pval
                    metrics["pred_pval_std"] = pred_pval_std

                    metrics["count_lower_95"] =  count_lower_95
                    metrics["count_upper_95"] =  count_upper_95
                    
                    metrics["pval_lower_95"] =  pval_lower_95
                    metrics["pval_upper_95"] =  pval_upper_95

                results.append(metrics)
            
        return results
    
    def get_metric_eic(self, ups_count_dist, ups_pval_dist, Y, X, P, bios_name, available_X_indices, available_Y_indices, arcsinh=True, quick=False):
        ups_count_mean = ups_count_dist.expect()
        ups_count_std = ups_count_dist.std()

        ups_pval_mean = ups_pval_dist.mean()
        ups_pval_std = ups_pval_dist.std()

        if not quick:
            if self.dataset.has_rnaseq(bios_name):
                print("got rna-seq data")
                rnaseq_res = self.eval_rnaseq(bios_name, ups_count_mean, Y, availability, k_fold=10, plot_REC=True)

            print("getting 0.95 interval conf")
            ups_count_lower_95, ups_count_upper_95 = ups_count_dist.interval(confidence=0.95)
            ups_pval_lower_95, ups_pval_upper_95 = ups_pval_dist.interval(confidence=0.95)
        
        results = []
        for j in range(Y.shape[1]):
            pred_count = ups_count_mean[:, j].numpy()
            pred_count_std = ups_count_std[:, j].numpy()

            pred_count_n = ups_count_dist.n[:, j].numpy()
            pred_count_p = ups_count_dist.p[:, j].numpy()   

            pred_pval = ups_pval_mean[:, j].numpy()
            pred_pval_std = ups_pval_std[:, j].numpy()

            if not quick:
                count_lower_95 = ups_count_lower_95[:, j].numpy()
                count_upper_95 = ups_count_upper_95[:, j].numpy()

                pval_lower_95 = ups_pval_lower_95[:, j].numpy()
                pval_upper_95 = ups_pval_upper_95[:, j].numpy()

            if j in list(available_X_indices):
                comparison = "upsampled"
                C_target = X[:, j].numpy()

            elif j in list(available_Y_indices):
                comparison = "imputed"
                C_target = Y[:, j].numpy()

            else:
                continue
                
            P_target = P[:, j].numpy()
            if arcsinh:
                P_target = np.sinh(P_target)
                pred_pval = np.sinh(pred_pval)
                if not quick:
                    pval_lower_95 = np.sinh(pval_lower_95)
                    pval_upper_95 = np.sinh(pval_upper_95)

            metrics = {
                'bios':bios_name,
                'feature': self.expnames[j],
                'comparison': comparison,
                'available assays': len(available_X_indices),

                'C_MSE-GW': self.metrics.mse(C_target, pred_count),
                'C_Pearson-GW': self.metrics.pearson(C_target, pred_count),
                'C_Spearman-GW': self.metrics.spearman(C_target, pred_count),
                'C_r2_GW': self.metrics.r2(C_target, pred_count),
                'C_Cidx_GW':self.metrics.c_index_nbinom(pred_count_n, pred_count_p, C_target),

                'C_Pearson_1obs': self.metrics.pearson1_obs(C_target, pred_count),
                'C_MSE-1obs': self.metrics.mse1obs(C_target, pred_count),
                'C_Spearman_1obs': self.metrics.spearman1_obs(C_target, pred_count),
                'C_r2_1obs': self.metrics.r2_1obs(C_target, pred_count),
                # 'C_Cidx_1obs':self.metrics.c_index_nbinom_1obs(pred_count_n, pred_count_p, C_target),

                'C_MSE-1imp': self.metrics.mse1imp(C_target, pred_count),
                'C_Pearson_1imp': self.metrics.pearson1_imp(C_target, pred_count),
                'C_Spearman_1imp': self.metrics.spearman1_imp(C_target, pred_count),
                'C_r2_1imp': self.metrics.r2_1imp(C_target, pred_count),

                'C_MSE-gene': self.metrics.mse_gene(C_target, pred_count),
                'C_Pearson_gene': self.metrics.pearson_gene(C_target, pred_count),
                'C_Spearman_gene': self.metrics.spearman_gene(C_target, pred_count),
                'C_r2_gene': self.metrics.r2_gene(C_target, pred_count),
                # 'C_Cidx_gene':self.metrics.c_index_nbinom_gene(pred_count_n, pred_count_p, C_target),

                'C_MSE-prom': self.metrics.mse_prom(C_target, pred_count),
                'C_Pearson_prom': self.metrics.pearson_prom(C_target, pred_count),
                'C_Spearman_prom': self.metrics.spearman_prom(C_target, pred_count),
                'C_r2_prom': self.metrics.r2_prom(C_target, pred_count),
                # 'C_Cidx_prom':self.metrics.c_index_nbinom_prom(pred_count_n, pred_count_p, C_target),
                
                "C_peak_overlap_01thr": self.metrics.peak_overlap(C_target, pred_count, p=0.01),
                "C_peak_overlap_05thr": self.metrics.peak_overlap(C_target, pred_count, p=0.05),
                "C_peak_overlap_10thr": self.metrics.peak_overlap(C_target, pred_count, p=0.10),

                ########################################################################

                'P_MSE-GW': self.metrics.mse(P_target, pred_pval),
                'P_Pearson-GW': self.metrics.pearson(P_target, pred_pval),
                'P_Spearman-GW': self.metrics.spearman(P_target, pred_pval),
                'P_r2_GW': self.metrics.r2(P_target, pred_pval),
                'P_Cidx_GW': self.metrics.c_index_gauss(pred_pval, pred_pval_std, P_target),

                'P_MSE-1obs': self.metrics.mse1obs(P_target, pred_pval),
                'P_Pearson_1obs': self.metrics.pearson1_obs(P_target, pred_pval),
                'P_Spearman_1obs': self.metrics.spearman1_obs(P_target, pred_pval),
                'P_r2_1obs': self.metrics.r2_1obs(P_target, pred_pval),
                'P_Cidx_1obs': self.metrics.c_index_gauss_1obs(pred_pval, pred_pval_std, P_target, num_pairs=5000),

                'P_MSE-1imp': self.metrics.mse1imp(P_target, pred_pval),
                'P_Pearson_1imp': self.metrics.pearson1_imp(P_target, pred_pval),
                'P_Spearman_1imp': self.metrics.spearman1_imp(P_target, pred_pval),
                'P_r2_1imp': self.metrics.r2_1imp(P_target, pred_pval),

                'P_MSE-gene': self.metrics.mse_gene(P_target, pred_pval),
                'P_Pearson_gene': self.metrics.pearson_gene(P_target, pred_pval),
                'P_Spearman_gene': self.metrics.spearman_gene(P_target, pred_pval),
                'P_r2_gene': self.metrics.r2_gene(P_target, pred_pval),
                'P_Cidx_gene': self.metrics.c_index_gauss_gene(pred_pval, pred_pval_std, P_target, num_pairs=5000),

                'P_MSE-prom': self.metrics.mse_prom(P_target, pred_pval),
                'P_Pearson_prom': self.metrics.pearson_prom(P_target, pred_pval),
                'P_Spearman_prom': self.metrics.spearman_prom(P_target, pred_pval),
                'P_r2_prom': self.metrics.r2_prom(P_target, pred_pval),
                'P_Cidx_prom': self.metrics.c_index_gauss_prom(pred_pval, pred_pval_std, P_target, num_pairs=5000),

                "P_peak_overlap_01thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.01),
                "P_peak_overlap_05thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.05),
                "P_peak_overlap_10thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.10)
            }

            if not quick:
                metric['C_Cidx_gene'] = self.metrics.c_index_nbinom_gene(pred_count_n, pred_count_p, C_target, num_pairs=2000)
                metric['C_Cidx_prom'] = self.metrics.c_index_nbinom_prom(pred_count_n, pred_count_p, C_target, num_pairs=2000)

                metrics["obs_count"] =  C_target
                metrics["obs_pval"] =  P_target

                metrics["pred_count"] = pred_count
                metrics["pred_count_std"] = pred_count_std

                metrics["pred_pval"] = pred_pval
                metrics["pred_pval_std"] = pred_pval_std

                metrics["count_lower_95"] =  count_lower_95
                metrics["count_upper_95"] =  count_upper_95

                metrics["pval_lower_95"] =  pval_lower_95
                metrics["pval_upper_95"] =  pval_upper_95

            results.append(metrics)

        return results

    def load_bios(self, bios_name, x_dsf, y_dsf=1, fill_in_y_prompt=False):
        print(f"getting bios vals for {bios_name}")

        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
            
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            # Load and process Y (target)
            temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            # Load and process P (probability)
            temp_py = self.dataset.load_bios_BW(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
            if self.split == "test":
                temp_px = self.dataset.load_bios_BW(bios_name.replace("B_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
            elif self.split == "val":
                temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)

            temp_p = {**temp_py, **temp_px}
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            del temp_py, temp_px, temp_p

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_p = self.dataset.load_bios_BW(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            assert (avlP == avY).all(), "avlP and avY do not match"
            del temp_p

        if self.DNA:
            seq = dna_to_onehot(get_DNA_sequence("chr21", 0, self.chr_sizes["chr21"]))

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

        if self.DNA:
            seq = seq[:num_rows*self.resolution, :]
            
        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        P = P.view(-1, self.context_length, P.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        if self.DNA:
            return X, Y, P, seq, mX, mY, avX, avY
        else:
            return X, Y, P, mX, mY, avX, avY

    def bios_pipeline(self, bios_name, x_dsf, quick=False, fill_in_y_prompt=False):
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  
        
        print("loaded input data")

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32)
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32)

        mu_imp = torch.empty_like(X, device="cpu", dtype=torch.float32)
        var_imp = torch.empty_like(X, device="cpu", dtype=torch.float32)

        for leave_one_out in available_indices:
            if self.DNA:
                n, p, mu, var = self.pred(X, mX, mY, avX, seq=seq, imp_target=[leave_one_out])
            else:
                n, p, mu, var = self.pred(X, mX, mY, avX, seq=None, imp_target=[leave_one_out])

            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]

            mu_imp[:, :, leave_one_out] = mu[:, :, leave_one_out]
            var_imp[:, :, leave_one_out] = var[:, :, leave_one_out]

            del n, p, mu, var  # Free up memory

        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        del X, mX, mY, avX, avY  # Free up memory

        print("got predictions")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        mu_imp = mu_imp.view((mu_imp.shape[0] * mu_imp.shape[1]), mu_imp.shape[-1])
        var_imp = var_imp.view((var_imp.shape[0] * var_imp.shape[1]), var_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])

        imp_count_dist = NegativeBinomial(p_imp, n_imp)
        ups_count_dist = NegativeBinomial(p_ups, n_ups)

        imp_pval_dist = Gaussian(mu_imp, var_imp)
        ups_pval_dist = Gaussian(mu_ups, var_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1])
        
        print("evaluating...")
        eval_res = self.get_metrics(imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, available_indices, quick=quick)
        return eval_res
    
    def bios_pipeline_eic(self, bios_name, x_dsf, quick=False, fill_in_y_prompt=False):
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt) 

        print(X.shape, Y.shape, P.shape)

        available_X_indices = torch.where(avX[0, :] == 1)[0]
        available_Y_indices = torch.where(avY[0, :] == 1)[0]

        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])
        
        print(p_ups.shape, n_ups.shape, mu_ups.shape, var_ups.shape)

        ups_count_dist = NegativeBinomial(p_ups, n_ups)
        ups_pval_dist = Gaussian(mu_ups, var_ups)

        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1])
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1])

        print("getting metrics")

        eval_res = self.get_metric_eic(ups_count_dist, ups_pval_dist, Y, X, P, bios_name, available_X_indices, available_Y_indices, quick=quick)
        return eval_res

    def viz_bios(self, eval_res):
        # Define a dictionary mapping function names to corresponding methods
        plot_functions = {
            # "count_track": self.viz.count_track,
            # "signal_track": self.viz.signal_track,

            # "count_confidence": self.viz.count_confidence,
            # "signal_confidence": self.viz.signal_confidence,
            
            # "count_error_std_hexbin": self.viz.count_error_std_hexbin,
            # "signal_error_std_hexbin": self.viz.signal_error_std_hexbin,

            # "count_scatter_with_marginals": self.viz.count_scatter_with_marginals,
            # "signal_scatter_with_marginals": self.viz.signal_scatter_with_marginals,

            # "count_heatmap": self.viz.count_heatmap,
            # "signal_heatmap": self.viz.signal_heatmap,
            
            # "count_rank_heatmap": self.viz.count_rank_heatmap,
            # "signal_rank_heatmap": self.viz.signal_rank_heatmap,

            # "count_TSS_confidence_boxplot": self.viz.count_TSS_confidence_boxplot,
            # "signal_TSS_confidence_boxplot": self.viz.signal_TSS_confidence_boxplot,

            # "count_GeneBody_confidence_boxplot": self.viz.count_GeneBody_confidence_boxplot,
            # "signal_GeneBody_confidence_boxplot": self.viz.signal_GeneBody_confidence_boxplot,

            # "count_obs_vs_confidence": self.viz.count_obs_vs_confidence,
            # "signal_obs_vs_confidence": self.viz.signal_obs_vs_confidence,

            # "count_TSS_enrichment_v_confidence": self.viz.count_TSS_enrichment_v_confidence,
            # "signal_TSS_enrichment_v_confidence": self.viz.signal_TSS_enrichment_v_confidence,

            # "count_GeneBody_enrichment_v_confidence": self.viz.count_GeneBody_enrichment_v_confidence,
            # "signal_GeneBody_enrichment_v_confidence": self.viz.signal_GeneBody_enrichment_v_confidence,

            # "count_context_length_specific_performance": self.viz.count_context_length_specific_performance,
            # "signal_context_length_specific_performance": self.viz.signal_context_length_specific_performance,

            "count_metagene": self.viz.count_metagene,
            "signal_metagene": self.viz.signal_metagene,

            "metagene2_count": self.viz.metagene2_count,
            "metagene2_signal": self.viz.metagene2_signal
        }
        
        for func_name, func in plot_functions.items():
            print(f"plotting {func_name.replace('_', ' ')}")
            # try:
            if "context_length_specific" in func_name:
                func(eval_res, self.context_length)
            else:
                func(eval_res)
            self.viz.clear_pallete()
            # except Exception as e:
            #     print(f"Failed to plot {func_name.replace('_', ' ')}: {e}")

    def filter_res(self, eval_res):
        new_res = []
        to_del = [
            "obs_count", "obs_pval", "pred_count", "pred_count_std", 
            "pred_pval", "pred_pval_std", "count_lower_95", "count_upper_95", 
            "pval_lower_95", "pval_upper_95", "pred_quantile"]
        
        for f in eval_res:
            if f["comparison"] != "None":
                fkeys = list(f.keys())
                for d in to_del:
                    if d in fkeys:
                        del f[d]
                new_res.append(f)
        return new_res

    def viz_all(self, dsf=1):
        self.model_res = []
        print(f"Evaluating {len(list(self.dataset.navigation.keys()))} biosamples...")
        for bios in list(self.dataset.navigation.keys()):
            # if not self.dataset.has_rnaseq(bios):
            #     continue

            try:
                print("evaluating ", bios)
                if self.eic:
                    eval_res_bios = self.bios_pipeline_eic(bios, dsf)
                else:
                    eval_res_bios = self.bios_pipeline(bios, dsf)
                print("got results for ", bios)
                self.viz_bios(eval_res_bios)
                
                to_del = [
                    "obs_count", "obs_pval", "pred_count", "pred_count_std", 
                    "pred_pval", "pred_pval_std", "count_lower_95", "count_upper_95", 
                    "pval_lower_95", "pval_upper_95", "pred_quantile"]
                
                for f in eval_res_bios:
                    fkeys = list(f.keys())
                    for d in to_del:
                        if d in fkeys:
                            del f[d]
                    
                    if f["comparison"] != "None":
                        self.model_res.append(f)
            except:
                pass

        self.model_res = pd.DataFrame(self.model_res)
        self.model_res.to_csv(f"{self.savedir}/model_eval_DSF{dsf}.csv", index=False)

    def bios_rnaseq_eval(self, bios_name, x_dsf, quick=False, fill_in_y_prompt=False):
        if not self.dataset.has_rnaseq(bios_name):
            print(f"{bios_name} doesn't have RNA-seq data!")
            return

        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  

        available_indices = torch.where(avX[0, :] == 1)[0]

        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        if self.DNA:
            Z = self.get_latent_z(X, mX, mY, avX, seq=seq)
        else:
            Z = self.get_latent_z(X, mX, mY, avX, seq=None)

        del X, mX, mY, avX, avY  # Free up memory

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])

        ups_count_dist = NegativeBinomial(p_ups, n_ups)
        ups_pval_dist = Gaussian(mu_ups, var_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1])
        Z = Z.view((Z.shape[0] * Z.shape[1]), Z.shape[-1])

        ups_count_mean = ups_count_dist.expect()
        ups_count_std = ups_count_dist.std()
        ups_pval_mean = ups_pval_dist.mean()
        ups_pval_std = ups_pval_dist.std()

        if quick:
            res = []
            print("rna-seq evaluation on count [QUICK]")
            res.append(
                self.quick_eval_rnaseq(
                bios_name, ups_count_mean, Y, 
                available_indices, dtype="count")
                )

            print("rna-seq evaluation on pval [QUICK]")
            res.append(
            self.quick_eval_rnaseq(
                bios_name, np.sinh(ups_pval_mean), np.sinh(P), 
                available_indices, dtype="pval")
                )

            print("rna-seq evaluation on latent [QUICK]")
            res.append(
            self.quick_eval_rnaseq(
                bios_name, Z, P, available_indices, dtype="Z")
            )
            res = pd.concat(res, axis=1)
            res = res.T
            return res

        else:
            print("rna-seq evaluation on count")
            self.eval_rnaseq(
                bios_name, ups_count_mean, Y, 
                available_indices, plot_REC=True, observed="count")

            print("rna-seq evaluation on pval")
            self.eval_rnaseq(
                bios_name, np.sinh(ups_pval_mean), np.sinh(P),
                available_indices, plot_REC=True, observed="pval")

    def rnaseq_all(self, dsf=1):
        self.model_res = []
        print(f"Evaluating for {len(list(self.dataset.navigation.keys()))} biosamples...")
        for bios in list(self.dataset.navigation.keys()):
            if self.dataset.has_rnaseq(bios):
                print(bios, len(self.dataset.navigation[bios]))
                self.bios_rnaseq_eval(bios, dsf)

    def saga(self, bios_name, x_dsf, fill_in_y_prompt=False, n_components=18, n_iter=100, tol=1e-4, random_state=0, resolution=200):
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf, fill_in_y_prompt=fill_in_y_prompt)  

        available_indices = torch.where(avX[0, :] == 1)[0]

        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        if self.DNA:
            Z = self.get_latent_z(X, mX, mY, avX, seq=seq)
        else:
            Z = self.get_latent_z(X, mX, mY, avX, seq=None)

        del X, mX, mY, avX, avY  # Free up memory

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])

        ups_count_dist = NegativeBinomial(p_ups, n_ups)
        ups_pval_dist = Gaussian(mu_ups, var_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1])
        Z = Z.view((Z.shape[0] * Z.shape[1]), Z.shape[-1])

        denoised_mu = mu_ups[:, available_indices]
        denoised_var = var_ups[:, available_indices]
        obs_data = P[:, available_indices]

        bin_size = int(resolution // 25)

        obs_data =    torch.mean(obs_data.view(-1, bin_size, obs_data.shape[1]), dim=1)
        denimp_data = torch.hstack(bin_gaussian_predictions(mu_ups, var_ups, bin_size))
        den_data =    torch.hstack(bin_gaussian_predictions(denoised_mu, denoised_var, bin_size))

        print(f"\nfitting the SAGA on observed signal (d={len(available_indices)})")
        SAGA_obs = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=random_state, n_iter=n_iter, tol=tol)
        SAGA_obs.fit(obs_data)
        SAGA_obs_MAP = SAGA_obs.predict(obs_data)
        SAGA_obs_posterior = SAGA_obs.predict_proba(obs_data)
        print("obs_MAP", {f"state_#{k}": float(v)/obs_data.shape[0] for k, v in Counter(SAGA_obs_MAP).items()})
    
        print(f"\nfitting the SAGA on denoised signal (d={len(available_indices)})")
        SAGA_den = SoftMultiAssayHMM(n_components=n_components, n_iter=n_iter, tol=tol, init_params="stmc", params="stmc", random_state=random_state)
        SAGA_den.fit(den_data)
        SAGA_den_MAP = SAGA_den.predict(den_data)
        SAGA_den_posterior = SAGA_den.predict_proba(den_data)
        print("den_MAP", {f"state_#{k}": float(v)/obs_data.shape[0] for k, v in Counter(SAGA_den_MAP).items()})

        print(f"\nfitting the SAGA on denoised + imputed signal (d={mu_ups.shape[1]})")
        SAGA_denimp = SoftMultiAssayHMM(n_components=n_components, n_iter=n_iter, tol=tol, init_params="stmc", params="stmc", random_state=random_state)
        SAGA_denimp.fit(denimp_data)
        SAGA_denimp_MAP = SAGA_denimp.predict(denimp_data)
        SAGA_denimp_posterior = SAGA_denimp.predict_proba(denimp_data)
        print("denimp_MAP", {f"state_#{k}": float(v)/obs_data.shape[0] for k, v in Counter(SAGA_denimp_MAP).items()})

        # 1. Initialize and fit PCA
        pca = PCA(n_components=0.9, random_state=random_state)
        Z_reduced = pca.fit_transform(Z)
        print(f"\nfitting the SAGA on latent (d={Z.shape[1]} -pca-> (d'={Z_reduced.shape[1]})")
        SAGA_latent = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=random_state, n_iter=n_iter, tol=tol, verbose=False)
        SAGA_latent.fit(Z_reduced)
        SAGA_latent_MAP = SAGA_latent.predict(Z_reduced)
        SAGA_latent_posterior = SAGA_latent.predict_proba(Z_reduced)
        print("latent_MAP", {f"state_#{k}": float(v)/obs_data.shape[0] for k, v in Counter(SAGA_latent_MAP).items()})

        return {
            "MAP":{
                "obs":SAGA_obs_MAP,
                "den":SAGA_den_MAP,
                "denimp":SAGA_denimp_MAP,
                "latent":SAGA_latent_MAP,
            },
            "posterior":{
                "obs":SAGA_obs_posterior,
                "den":SAGA_den_posterior,
                "denimp":SAGA_denimp_posterior,
                "latent":SAGA_latent_posterior,
            }
        }
         
# def main():
#     pd.set_option('display.max_rows', None)
#     # bios -> "B_DND-41"
#     parser = argparse.ArgumentParser(description="Evaluate CANDI model with specified parameters.")

#     parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model.")
#     parser.add_argument("-hp", "--hyper_parameters_path", type=str, required=True, help="Path to hyperparameters file.")
#     parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the input data.")
#     parser.add_argument("-s", "--savedir", type=str, default="/project/compbio-lab/CANDI_res/", help="Directory to save evaluation results.")
#     parser.add_argument("--enc_ckpt", type=str, default=None, help="If CANDI-DINO, path to encoder checkpoint model.")
#     parser.add_argument("--dec_ckpt", type=str, default=None, help="If CANDI-DINO, path to decoder checkpoint model.")
#     parser.add_argument("-r", "--resolution", type=int, default=25, help="Resolution for evaluation.")
#     parser.add_argument("-cl", "--context_length", type=int, default=1200, help="Context length for evaluation.")
#     parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for evaluation.")
#     parser.add_argument("--eic", action="store_true", help="Flag to enable EIC mode.")
#     parser.add_argument("--rnaonly", action="store_true", help="Flag to evaluate only RNAseq prediction.")
#     parser.add_argument("--dino", action="store_true", help="Flag to enable DINO mode.")
#     parser.add_argument("--dna", action="store_true", default=True, help="Flag to include DNA in the evaluation.")
#     parser.add_argument("--quick", action="store_true", help="Flag to quickly compute eval metrics for one biosample.")
#     parser.add_argument("--list_bios", action="store_true", help="print the list of all available biosamples for evaluating")
#     parser.add_argument("--supertrack", action="store_true", help="supertrack")
#     parser.add_argument("--saga", action="store_true", help="saga")

#     parser.add_argument("--dsf", type=int, default=1, help="Down-sampling factor.")
#     parser.add_argument("bios_name", type=str, help="BIOS argument for the pipeline.")
#     parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split to evaluate on. Options: test, val.")
#     parser.add_argument("--chr_sizes_file", type=str, default="data/hg38.chrom.sizes", help="Path to chromosome sizes file.")


#     args = parser.parse_args()
#     savedir = args.savedir
#     fill_in_y_prompt = args.supertrack

#     if args.dino:
#         savedir = savedir.replace("CANDI", "CANDINO")
        
#     ec = EVAL_CANDI(
#         args.model_path, args.data_path, args.context_length, args.batch_size, args.hyper_parameters_path,
#         chr_sizes_file=args.chr_sizes_file, resolution=args.resolution, savedir=savedir, 
#         mode="eval", split=args.split, eic=args.eic, DNA=args.dna, 
#         DINO=args.dino, ENC_CKP=args.enc_ckpt, DEC_CKP=args.dec_ckpt)

#     if args.list_bios:
#         for k, v in ec.dataset.navigation.items():
#             print(f"{k}: {len(v)} available assays")
#         exit()
        
#     if args.bios_name == "all":
#         if args.rnaonly:
#             if args.quick:
#                 for k, v in ec.dataset.navigation.items():
#                     res = ec.bios_rnaseq_eval(k, args.dsf, args.quick, fill_in_y_prompt)
#                     if res is not None:
#                         num_avail = len(v.keys())-1 if "RNA-seq" in v.keys() else len(v.keys())
#                         if not os.path.exists(f"{ec.savedir}/{k}_{num_avail}/"):
#                             os.mkdir(f"{ec.savedir}/{k}_{num_avail}/")
#                         res.to_csv(os.path.join(f"{ec.savedir}",f"{k}_{num_avail}", "RNA-seq.csv"))

#             else:
#                 ec.rnaseq_all(dsf=args.dsf)
        
#         elif args.saga:
#             for k, v in ec.dataset.navigation.items():
#                 num_avail = len(v.keys())-1 if "RNA-seq" in v.keys() else len(v.keys())
#                 saga_res = ec.saga(k, args.dsf, fill_in_y_prompt, resolution=200, n_components=int(10+(num_avail**(1/2))))

#                 if not os.path.exists(f"{ec.savedir}/{k}_{num_avail}/"):
#                     os.mkdir(f"{ec.savedir}/{k}_{num_avail}/")
#                 write_bed(
#                     saga_res["MAP"]["obs"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_obs.bed"), 
#                     is_posterior=False, track_name="Chromatin State Annotation", 
#                     track_description="observed signals", visibility="dense")

#                 write_posteriors_to_tsv(
#                     saga_res["posterior"]["obs"], "chr21", 0, 200,  
#                     os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_obs.bed"))

#                 write_bed(
#                     saga_res["MAP"]["den"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_den.bed"),
#                     is_posterior=False, track_name="Chromatin State Annotation", 
#                     track_description="candi denoised signals", visibility="dense")

#                 write_posteriors_to_tsv(
#                     saga_res["posterior"]["den"], "chr21", 0, 200, 
#                     os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_den.bed"))

#                 write_bed(
#                     saga_res["MAP"]["denimp"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_denimp.bed"), 
#                     is_posterior=False, track_name="Chromatin State Annotation", 
#                     track_description="candi denoised and imputed signals", visibility="dense")

#                 write_posteriors_to_tsv(
#                     saga_res["posterior"]["denimp"], "chr21", 0, 200, 
#                     os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_denimp.bed"))

#                 write_bed(
#                     saga_res["MAP"]["latent"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_latent.bed"), 
#                     is_posterior=False, track_name="Chromatin State Annotation", 
#                     track_description="candi latent (+ 0.9 PCA)", visibility="dense")

#                 write_posteriors_to_tsv(
#                     saga_res["posterior"]["latent"], "chr21", 0, 200, 
#                     os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_latent.bed"))

#             exit()

#         else:
#             if args.quick:
#                 all_dfs = []
#                 # iterate over each biosample key in your dataset
#                 for bios in ec.dataset.navigation.keys():

#                     # skip ones without any RNaseq if you like
#                     if args.eic:
#                         res = ec.bios_pipeline_eic(bios, args.dsf, args.quick, fill_in_y_prompt=fill_in_y_prompt)
#                     else:
#                         res = ec.bios_pipeline(bios, args.dsf, args.quick, fill_in_y_prompt=fill_in_y_prompt)

#                     # make sure it’s a DataFrame
#                     all_dfs.append(pd.DataFrame(res))

#                 # concatenate, save
#                 report = pd.concat(all_dfs, ignore_index=True)
#                 os.makedirs(args.savedir, exist_ok=True)
#                 report.to_csv(os.path.join(args.savedir, "quick_report.csv"), index=False)
#                 print(report)
#             else:
#                 ec.viz_all(dsf=args.dsf, fill_in_y_prompt=fill_in_y_prompt)

#     else:
#         if args.saga:
#             k, v = args.bios_name, ec.dataset.navigation[args.bios_name]
#             num_avail = len(v.keys())-1 if "RNA-seq" in v.keys() else len(v.keys())
#             if not os.path.exists(f"{ec.savedir}/{k}_{num_avail}/"):
#                 os.mkdir(f"{ec.savedir}/{k}_{num_avail}/")

#             saga_res = ec.saga(args.bios_name, args.dsf, fill_in_y_prompt, resolution=200, n_components=int(10+(num_avail**(1/2))))
            
#             write_bed(
#                 saga_res["MAP"]["obs"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_obs.bed"), 
#                 is_posterior=False, track_name="Chromatin State Annotation", 
#                 track_description="observed signals", visibility="dense")

#             write_posteriors_to_tsv(
#                 saga_res["posterior"]["obs"], "chr21", 0, 200,  
#                 os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_obs.bed"))

#             write_bed(
#                 saga_res["MAP"]["den"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_den.bed"),
#                 is_posterior=False, track_name="Chromatin State Annotation", 
#                 track_description="candi denoised signals", visibility="dense")

#             write_posteriors_to_tsv(
#                 saga_res["posterior"]["den"], "chr21", 0, 200, 
#                 os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_den.bed"))

#             write_bed(
#                 saga_res["MAP"]["denimp"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_denimp.bed"), 
#                 is_posterior=False, track_name="Chromatin State Annotation", 
#                 track_description="candi denoised and imputed signals", visibility="dense")

#             write_posteriors_to_tsv(
#                 saga_res["posterior"]["denimp"], "chr21", 0, 200, 
#                 os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_denimp.bed"))

#             write_bed(
#                 saga_res["MAP"]["latent"], "chr21", 0, 200, os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "MAP_latent.bed"), 
#                 is_posterior=False, track_name="Chromatin State Annotation", 
#                 track_description="candi latent (+ 0.9 PCA)", visibility="dense")

#             write_posteriors_to_tsv(
#                 saga_res["posterior"]["latent"], "chr21", 0, 200, 
#                 os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "posterior_latent.bed"))

#             exit()

#         if args.rnaonly and not args.eic:
#             report = ec.bios_rnaseq_eval(args.bios_name, args.dsf, args.quick, fill_in_y_prompt)
            
#             if report is not None:
#                 k, v = args.bios_name, ec.dataset.navigation[args.bios_name]
#                 num_avail = len(v.keys())-1 if "RNA-seq" in v.keys() else len(v.keys())
#                 if not os.path.exists(f"{ec.savedir}/{k}_{num_avail}/"):
#                     os.mkdir(f"{ec.savedir}/{k}_{num_avail}/")

#                 report.to_csv(os.path.join(f"{ec.savedir}",f"{k}_{num_avail}/", "RNA-seq.csv"))
#             exit()
            
#         if args.eic:
#             t0 = datetime.now()
#             res = ec.bios_pipeline_eic(args.bios_name, args.dsf, args.quick, fill_in_y_prompt=fill_in_y_prompt)
#             elapsed_time = datetime.now() - t0
#             print(f"took {elapsed_time}")

#             if args.quick:
#                 report = pd.DataFrame(res)
#                 print(report[["feature", "comparison"] + [c for c in report.columns if "Cidx" in c]])

#         else:
#             t0 = datetime.now()
#             res = ec.bios_pipeline(args.bios_name, args.dsf, args.quick, fill_in_y_prompt=fill_in_y_prompt)
#             elapsed_time = datetime.now() - t0
#             print(f"took {elapsed_time}")

#             if args.quick:
#                 report = pd.DataFrame(res)
#                 print(report[["feature", "comparison"] + [c for c in report.columns if "Cidx" in c]])

#         if not args.quick:
#             ec.viz_bios(eval_res=res)
#             res = ec.filter_res(res)
#             print(pd.DataFrame(res))

def main():
    pd.set_option('display.max_rows', None)
    
    # --- 1. Main Parser Setup ---
    # This parser handles global arguments common to all sub-commands.
    parser = argparse.ArgumentParser(
        description="Evaluate CANDI model performance and generate annotations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Group arguments for better --help output
    model_args = parser.add_argument_group('Model and Data Arguments')
    model_args.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    model_args.add_argument("-hp", "--hyper_parameters_path", type=str, required=True, help="Path to the model's hyperparameters file (.pkl).")
    model_args.add_argument("-s", "--savedir", type=str, required=True, help="Directory to save evaluation results.")
    model_args.add_argument("-d", "--data_path", type=str, default="/project/compbio-lab/encode_data/", help="Path to the root ENCODE data directory.")
    model_args.add_argument("--chr_sizes_file", type=str, default="data/hg38.chrom.sizes", help="Path to chromosome sizes file.")

    config_args = parser.add_argument_group('Configuration Arguments')
    config_args.add_argument("-r", "--resolution", type=int, default=25, help="Data resolution in base pairs.")
    config_args.add_argument("-cl", "--context_length", type=int, default=1200, help="Model context length.")
    config_args.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for predictions.")
    config_args.add_argument("--dsf", type=int, default=1, help="Down-sampling factor for evaluation data.")
    config_args.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Data split to evaluate on.")
    config_args.add_argument("--dna", action="store_true", default=True, help="Flag to include DNA sequence in the model.")
    config_args.add_argument("--dino", action="store_true", help="Flag to use the CANDI-DINO model architecture.")
    config_args.add_argument("--enc_ckpt", type=str, default=None, help="Path to encoder checkpoint (for CANDI-DINO).")
    config_args.add_argument("--dec_ckpt", type=str, default=None, help="Path to decoder checkpoint (for CANDI-DINO).")

    # --- 2. Sub-parser Setup ---
    # This creates the sub-command structure (eval, rna-seq, etc.)
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # Sub-parser for 'eval' command 
    parser_eval = subparsers.add_parser('eval', help='Run the full evaluation pipeline on one or all biosamples.')
    parser_eval.add_argument('bios_name', type=str, help="Name of the biosample to evaluate, or 'all' for all biosamples.")
    parser_eval.add_argument('--eic', action='store_true', help='Run in Experiment-Input-Control (EIC) mode.')
    parser_eval.add_argument('--quick', action='store_true', help='Run a quicker, summarized evaluation without generating plots.')
    parser_eval.add_argument('--supertrack', action='store_true', help='Enable supertrack mode (fill_in_y_prompt).')

    # Sub-parser for 'rna-seq' command
    parser_rna = subparsers.add_parser('rna-seq', help='Evaluate only the RNA-seq prediction performance.')
    parser_rna.add_argument('bios_name', type=str, help="Name of the biosample to evaluate, or 'all' for all biosamples with RNA-seq data.")
    parser_rna.add_argument('--quick', default=True, action='store_true', help='Generate a summarized CSV report instead of detailed plots.')
    parser_rna.add_argument('--supertrack', action='store_true', help='Enable supertrack mode (fill_in_y_prompt).')

    # Sub-parser for 'saga' command
    parser_saga = subparsers.add_parser('saga', help='Generate chromatin state annotations using SAGA.')
    parser_saga.add_argument('bios_name', type=str, help="Name of the biosample to annotate, or 'all'.")
    parser_saga.add_argument('--n_states', type=int, default=None, help="Number of chromatin states. If None, defaults to 10 + sqrt(num_assays).")
    parser_saga.add_argument('--supertrack', action='store_true', help='Enable supertrack mode (fill_in_y_prompt).')

    # Sub-parser for 'list-bios' command
    subparsers.add_parser('list-bios', help='List all available biosamples in the dataset.')

    args = parser.parse_args()

    # --- 3. Command Execution ---
    savedir = args.savedir.replace("CANDI", "CANDINO") if args.dino else args.savedir

    ec = EVAL_CANDI(
        args.model_path, args.data_path, args.context_length, args.batch_size, args.hyper_parameters_path,
        chr_sizes_file=args.chr_sizes_file, resolution=args.resolution, savedir=savedir,
        mode="eval", split=args.split, eic=getattr(args, 'eic', False), DNA=args.dna,
        DINO=args.dino, ENC_CKP=args.enc_ckpt, DEC_CKP=args.dec_ckpt
    )

    bios_to_process = ec.dataset.navigation.keys() if args.bios_name == 'all' else [args.bios_name]

    # --- Command Logic ---
    if args.command == 'list-bios':
        print("Available biosamples for evaluation:")
        for k, v in ec.dataset.navigation.items():
            print(f"- {k}: {len(v)} available assays")
            
    elif args.command == 'eval':
        all_results = []
        for bios in bios_to_process:
            print(f"\n--- Running Full Evaluation for: {bios} ---")
            try:
                if args.eic:
                    res = ec.bios_pipeline_eic(bios, args.dsf, args.quick, fill_in_y_prompt=args.supertrack)
                else:
                    res = ec.bios_pipeline(bios, args.dsf, args.quick, fill_in_y_prompt=args.supertrack)
                
                if args.quick:
                    all_results.append(pd.DataFrame(res))
                else:
                    ec.viz_bios(eval_res=res)
                    res = ec.filter_res(res)
                    all_results.append(pd.DataFrame(res))

            except Exception as e:
                print(f"Failed to evaluate {bios}. Error: {e}")
        
        if all_results:
            report = pd.concat(all_results, ignore_index=True)
            report_path = os.path.join(args.savedir, "full_report.csv")
            report.to_csv(report_path, index=False)
            print(f"\nEvaluation complete. Full report saved to {report_path}")
            print(report)

    elif args.command == 'rna-seq':
        all_results = []
        for bios in bios_to_process:
            if not ec.dataset.has_rnaseq(bios):
                if args.bios_name != 'all': print(f"Skipping {bios}: No RNA-seq data found.")
                continue
            print(f"\n--- Running RNA-seq Evaluation for: {bios} ---")
            try:
                res = ec.bios_rnaseq_eval(bios, args.dsf, args.quick, fill_in_y_prompt=args.supertrack)
                if res is not None:
                    k, v = bios, ec.dataset.navigation[bios]
                    num_avail = len(v.keys())-1 if "RNA-seq" in v.keys() else len(v.keys())
                    out_dir = os.path.join(ec.savedir, f"{k}_{num_avail}")
                    os.makedirs(out_dir, exist_ok=True)
                    res.to_csv(os.path.join(out_dir, "RNA-seq_quick_report.csv"))
                    all_results.append(res)
            except Exception as e:
                print(f"Failed RNA-seq evaluation for {bios}. Error: {e}")
        
        if all_results:
             print("\nRNA-seq evaluation complete. Results saved in respective biosample directories.")

    elif args.command == 'saga':
        for bios in bios_to_process:
            print(f"\n--- Running SAGA Annotation for: {bios} ---")
            try:
                v = ec.dataset.navigation[bios]
                num_avail = len(v.keys()) - 1 if "RNA-seq" in v.keys() else len(v.keys())
                n_states = args.n_states if args.n_states else int(10 + (num_avail**(1/2)))
                
                out_dir = os.path.join(ec.savedir, f"{bios}_{num_avail}")
                os.makedirs(out_dir, exist_ok=True)
                
                saga_res = ec.saga(bios, args.dsf, fill_in_y_prompt=args.supertrack, resolution=200, n_components=n_states)

                # Save results
                for key in saga_res['MAP']:
                    write_bed(saga_res['MAP'][key], "chr21", 0, 200, os.path.join(out_dir, f"MAP_{key}.bed"))
                    write_posteriors_to_tsv(saga_res['posterior'][key], "chr21", 0, 200, os.path.join(out_dir, f"posterior_{key}.bed"))
                print(f"SAGA annotations for {bios} saved to {out_dir}")

            except Exception as e:
                print(f"Failed SAGA annotation for {bios}. Error: {e}")

if __name__ == "__main__":
    main()

    #python eval.py -m models/CANDIeic_DNA_random_mask_ccre10k_model_checkpoint_epoch5_chr7.pth -hp models/hyper_parameters_CANDIeic_DNA_random_mask_ccre10k_20250515220155_params46300375.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDI_ccre10k/ --eic all
    #python eval.py -m models/CANDIeic_DNA_random_mask_rand10k_model_checkpoint_epoch4_chr8.pth -hp models/hyper_parameters_CANDIeic_DNA_random_mask_rand10k_20250515215719_params46300375.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDI_rand10k/ --eic all
    #python eval.py -m models/CANDINO_ccre10k_merged.pth -hp models/hyper_parameters_DINO_CANDI_ccre10k.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDINO_ccre10k/ --dino --eic all
    #python eval.py -m models/CANDINO_rand10k_merged.pth -hp models/hyper_parameters_DINO_CANDI_rand10k.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDINO_rand10k/ --dino --eic all


    # python eval.py -m models/CANDIeic_DNA_random_mask_ccre10k_model_checkpoint_epoch5_chr9.pth -hp models/hyper_parameters_CANDIeic_DNA_random_mask_ccre10k_20250515220155_params46300375.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDI_ccre10k/ --rnaonly all
    # python eval.py -m models/CANDIeic_DNA_random_mask_rand10k_model_checkpoint_epoch4_chr8.pth -hp models/hyper_parameters_CANDIeic_DNA_random_mask_rand10k_20250515215719_params46300375.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDI_rand10k/ --rnaonly all
    # python eval.py -m models/CANDINO_ccre10k_merged.pth -hp models/hyper_parameters_DINO_CANDI_ccre10k.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDINO_ccre10k/ --dino --rnaonly all
    # python eval.py -m models/CANDINO_rand10k_merged.pth -hp models/hyper_parameters_DINO_CANDI_rand10k.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDINO_rand10k/ --dino --rnaonly all

    # python eval.py -m models/CANDIfull_DNA_random_mask_admx_cos_shdc_model_checkpoint_epoch6.pth -hp models/hyper_parameters_CANDIfull_DNA_random_mask_admx_cos_shdc_20250705123426_params43234025.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDIfull_admx_cos_shdc/ --quick

    # python eval.py -m models/CANDIfull_DNA_random_mask_admx_cos_shdc_model_checkpoint_epoch6.pth -hp models/hyper_parameters_CANDIfull_DNA_random_mask_admx_cos_shdc_20250705123426_params43234025.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDIfull_admx_cos_shdc/ --quick --rnaonly foreskin_keratinocyte_grp1_rep1
    # python eval.py -m models/CANDIeic_DNA_random_mask_jul15_model_checkpoint_epoch3.pth -hp models/hyper_parameters_CANDIeic_DNA_random_mask_jul15_20250715163659_params43234025.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDIeic_jul15/ --quick --rnaonly foreskin_keratinocyte_grp1_rep1
    # python eval.py -m models/CANDIfull_DNA_random_mask_jul15_model_checkpoint_epoch1.pth -hp models/hyper_parameters_CANDIfull_DNA_random_mask_jul15_20250715163654_params43234025.pkl -d /project/compbio-lab/encode_data/ -s /project/compbio-lab/CANDIfull_jul15/ --quick --rnaonly foreskin_keratinocyte_grp1_rep1

    #  python SAGAconf.py /project/compbio-lab/CANDI/CANDIfull_jul15/GM23248_grp1_rep1_13/posterior_obs.bed /project/compbio-lab/CANDI/CANDIfull_jul15/GM23248_grp1_rep1_13/posterior_denimp.bed /project/compbio-lab/CANDI/CANDIfull_jul15/ -v