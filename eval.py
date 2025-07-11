# from CANDI import *
from data import *
from dino_candi import *
# from _utils import *

from scipy.stats import pearsonr, spearmanr, poisson, rankdata
from sklearn.metrics import mean_squared_error, r2_score, auc
from sklearn.metrics import roc_auc_score
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
PROC_GENE_BED_FPATH = "data/gene_bodies.bed"
PROC_PROM_BED_PATH = "data/tss.bed"

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

class METRICS(object):
    def __init__(self, chrom='chr21', bin_size=25):
        self.prom_df = self.get_prom_positions(chrom, bin_size)
        self.gene_df = self.get_gene_positions(chrom, bin_size)

    def get_gene_positions(self, chrom, bin_size):
        gene_df = pd.read_csv(PROC_GENE_BED_FPATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name'])
        chrom_subset = gene_df[gene_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))
        return chrom_subset

    def get_prom_positions(self, chrom, bin_size):
        prom_df = pd.read_csv(PROC_PROM_BED_PATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name', "strand"])
        chrom_subset = prom_df[prom_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))

        return chrom_subset
        
    def get_signals(self, array, df):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in df.iterrows()])
        valid_indices = indices[indices < len(array)]

        signals = array[valid_indices]
        return signals

    ################################################################################

    def get_gene_signals(self, y_true, y_pred, bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return gt_vals, pred_vals
    
    def get_prom_signals(self, y_true, y_pred, bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return gt_vals, pred_vals
    
    def get_1obs_signals(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return y_true[perc_99_pos], y_pred[perc_99_pos]

    def get_1imp_signals(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return y_true[perc_99_pos], y_pred[perc_99_pos]
    
    ################################################################################
    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def r2_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.r2(y_true=gt_vals, y_pred=pred_vals)
    
    def r2_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.r2(y_true=gt_vals, y_pred=pred_vals)

    def r2_1obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def r2_1imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.r2(y_true[perc_99_pos], y_pred[perc_99_pos])

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        return np.mean((np.array(y_true) - np.array(y_pred))**2)
    
    def mse_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def pearson(self, y_true, y_pred):
        """
        Calculate the genome-wide Pearson Correlation. This measures the linear relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return pearsonr(y_pred, y_true)[0]

    def spearman(self, y_true, y_pred):
        """
        Calculate the genome-wide Spearman Correlation. This measures the monotonic relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return spearmanr(y_pred, y_true)[0]

    def mse_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def mse1obs(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by experimental signal (mse1obs). 
        This is a measure of how well predictions match observations at positions with high experimental signal. 
        It's similar to recall.
        """
        top_1_percent = int(0.01 * len(y_true))
        top_1_percent_indices = np.argsort(y_true)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def mse1imp(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by predicted signal (mse1imp). 
        This is a measure of how well predictions match observations at positions with high predicted signal. 
        It's similar to precision.
        """
        top_1_percent = int(0.01 * len(y_pred))
        top_1_percent_indices = np.argsort(y_pred)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def pearson1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def pearson1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def peak_overlap(self, y_true, y_pred, p=0.01):
        if p == 0:
            return 0

        elif p == 1:
            return 1

        top_p_percent = int(p * len(y_true))

        # Get the indices of the top p percent of the observed (true) values
        top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
        
        # Get the indices of the top p percent of the predicted values
        top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

        # Calculate the overlap
        overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

        # Calculate the percentage of overlap
        overlap_percent = overlap / top_p_percent 

        return overlap_percent

    def correspondence_curve(self, y_true, y_pred):
        curve = []
        derivatives = []
        steps = [float(p / 100) for p in range(0, 101, 1)]

        obs_rank = np.argsort(y_true)
        pred_rank = np.argsort(y_pred)

        for p in steps:
            if p == 0 or p == 1:
                overlap_percent = p
            else:
                top_p_percent = int(p * len(y_true))
                top_p_percent_obs_i = obs_rank[-top_p_percent:]
                top_p_percent_pred_i = pred_rank[-top_p_percent:]

                overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))
                overlap_percent = overlap / len(y_true)

            curve.append((p, overlap_percent))

        # Calculate derivatives using finite differences
        for i in range(1, len(curve)):
            dp = curve[i][0] - curve[i-1][0]
            d_overlap_percent = curve[i][1] - curve[i-1][1]
            derivative = d_overlap_percent / dp
            derivatives.append((curve[i][0], derivative))

        return curve, derivatives

    def confidence_quantile(self, nbinom_p, nbinom_n, y_true):
        nbinom_dist = NegativeBinomial(nbinom_p, nbinom_n)
        return nbinom_dist.cdf(y_true)

    def foreground_vs_background(self, nbinom_p, nbinom_n, y_true):
        """
        inputs: 1) nbinom_p, nbinom_n -> two arrays with length L -> negative binomial dist parameters for each position
                2) y_true -> one array of true observed signal

        task:

            - NEW 2: peak vs. background comparison
            - binarize each observed experiment according to some threshold.
            - measure the two following
                - what fraction of positions outside peaks have overlap (confidence interval) with zero
                - for peaks, for what fraction, the confidence interval overlap with 0. this should ideally be low since otherwise the model is not sure about the peak
            - intervals 90 95 99 percent
        """

        nbinom_dist = NegativeBinomial(nbinom_p, nbinom_n)
        binarized_y = binarize_nbinom(y_true)

        pmf_zero = (nbinom_dist.pmf(0))

        analysis = {}

        background_pmf_zero = pmf_zero[binarized_y == 0].mean() 
        peak_pmf_zero = pmf_zero[binarized_y == 1].mean() 

        analysis["p0_bg"] = background_pmf_zero.item()
        analysis["p0_fg"] = peak_pmf_zero.item()

        return analysis

    def c_index_gauss(self, mus, sigmas, y_true, num_pairs: int = 10000):
        """
        Concordance index for Gaussian predictive marginals,
        estimating over `num_pairs` randomly sampled pairs.
        
        Inputs:
          - mus:       array_like, shape (N,) of predicted means μ_i
          - sigmas:    array_like, shape (N,) of predicted stddevs σ_i
          - y_true:    array_like, shape (N,) of true values y_i
          - num_pairs: number of random (i<j) pairs to sample;
                       if -1, use all possible pairs (i<j)
        Returns:
          - c_index: float in [0,1]
        """
        N = len(y_true)
        labels = []
        scores = []

        if num_pairs == -1:
            # exact over all valid pairs
            for i in range(N):
                for j in range(i+1, N):
                    print("Gauss", N, i, j)
                    if y_true[i] == y_true[j]:
                        continue
                    labels.append(int(y_true[i] > y_true[j]))
                    delta = mus[i] - mus[j]
                    sd = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
                    scores.append(norm.cdf(delta / sd))
        else:
            # Monte Carlo sampling of pairs
            rng = np.random.default_rng()
            count = 0
            while count < num_pairs:
                i, j = rng.integers(0, N, size=2)
                if i == j or y_true[i] == y_true[j]:
                    continue
                # Optional: enforce ordering i<j for consistency
                if i > j:
                    i, j = j, i
                labels.append(int(y_true[i] > y_true[j]))
                delta = mus[i] - mus[j]
                sd = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
                scores.append(norm.cdf(delta / sd))
                count += 1

        return roc_auc_score(labels, scores)

    def c_index_gauss_gene(self, mus, sigmas, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.gene_df.iterrows()])
        valid_indices = indices[indices < len(array)]
        
        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[valid_indices], sigmas[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_gauss_prom(self, mus, sigmas, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(array)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[valid_indices], sigmas[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_gauss_1obs(self, mus, sigmas, y_true, num_pairs=10000):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        N = len(perc_99_pos)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[perc_99_pos], sigmas[perc_99_pos], y_true[perc_99_pos], num_pairs)
        return c_idx

    def c_index_nbinom(self,
                       rs, ps, y_true,
                       epsilon: float = 1e-6,
                       num_pairs: int = 10000):
        """
        Concordance index for Negative‐Binomial predictive marginals,
        estimating over `num_pairs` randomly sampled pairs.

        Inputs:
          - rs:        (N,) array of NB 'r' parameters (must be >0)
          - ps:        (N,) array of NB 'p' parameters (0<p<1)
          - y_true:    (N,) array of true values y_i
          - epsilon:   tail‐mass cutoff for truncation (default 1e-6)
          - num_pairs: number of random (i<j) pairs to sample;
                       if -1, use all valid pairs (i<j, y_i≠y_j)
        Returns:
          - c_index: float in [0,1]
        """
        N = len(y_true)
        labels = []
        scores = []

        def compute_score(i, j):
            # clamp p_j into (ε,1−ε)
            p_j = np.clip(ps[j], epsilon, 1 - epsilon)
            r_j = rs[j]
            if r_j <= 0:
                return None
            # find cutoff K
            K = nbinom.ppf(1 - epsilon, r_j, p_j)
            if not np.isfinite(K):
                K = 0
            else:
                K = int(K)
            # PMF/CDF arrays
            k = np.arange(K + 1)
            pmf_j = nbinom.pmf(k, r_j, p_j)
            cdf_i = nbinom.cdf(k, rs[i], ps[i])
            # if anything is nan, bail
            if not (np.isfinite(pmf_j).all() and np.isfinite(cdf_i).all()):
                return None
            return np.sum(pmf_j * (1.0 - cdf_i))

        if num_pairs == -1:
            # exact mode
            for i in range(N):
                for j in range(i+1, N):
                    print("NBinom", N, i, j)
                    if y_true[i] == y_true[j]:
                        continue
                    sc = compute_score(i, j)
                    if sc is None:
                        continue
                    labels.append(int(y_true[i] > y_true[j]))
                    scores.append(sc)
        else:
            # sampling mode
            rng = np.random.default_rng()
            count = 0
            while count < num_pairs:
                i, j = rng.integers(0, N, size=2)
                if i == j or y_true[i] == y_true[j]:
                    continue
                if i > j:
                    i, j = j, i
                sc = compute_score(i, j)
                if sc is None:
                    continue
                labels.append(int(y_true[i] > y_true[j]))
                scores.append(sc)
                count += 1

        if len(labels) == 0:
            # no valid pairs
            return np.nan

        return roc_auc_score(labels, scores)

    def c_index_nbinom_gene(self, rs, ps, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.gene_df.iterrows()])
        valid_indices = indices[indices < len(array)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_nbinom(rs[valid_indices], ps[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_nbinom_prom(self, rs, ps, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(array)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_nbinom(rs[valid_indices], ps[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_nbinom_1obs(self, rs, ps, y_true, num_pairs=10000):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        N = len(perc_99_pos)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1
        
        c_idx = self.c_index_nbinom(rs[perc_99_pos], ps[perc_99_pos], y_true[perc_99_pos], num_pairs)
        return c_idx


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

        # 1) load full-genome RNA-seq table (unchanged)
        rna_seq_data = self.dataset.load_rna_seq_data(bios_name, self.gene_coords)

        # build gene_info lookup (unchanged)
        gene_info = (
            rna_seq_data[['geneID','chr','TPM','FPKM']]
            .drop_duplicates(subset='geneID')
            .set_index('geneID')
        )

        # 2) build long-format (unchanged)
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
            if self.dataset.has_rnaseq(bios_name):
                print("got rna-seq data")
                rnaseq_res = self.eval_rnaseq(bios_name, ups_count_mean, Y, availability, k_fold=10, plot_REC=True)

            print("getting 0.95 interval conf")
            imp_count_lower_95, imp_count_upper_95 = imp_count_dist.interval(confidence=0.95)
            ups_count_lower_95, ups_count_upper_95 = ups_count_dist.interval(confidence=0.95)

            imp_pval_lower_95, imp_pval_upper_95 = imp_pval_dist.interval(confidence=0.95)
            ups_pval_lower_95, ups_pval_upper_95 = ups_pval_dist.interval(confidence=0.95)

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

                    # corresp, corresp_deriv = self.metrics.correspondence_curve(target, pred)
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
                        'C_Cidx_1obs':self.metrics.c_index_nbinom_1obs(pred_count_n, pred_count_p, C_target, num_pairs=-1),

                        'C_MSE-1imp': self.metrics.mse1imp(C_target, pred_count),
                        'C_Pearson_1imp': self.metrics.pearson1_imp(C_target, pred_count),
                        'C_Spearman_1imp': self.metrics.spearman1_imp(C_target, pred_count),
                        'C_r2_1imp': self.metrics.r2_1imp(C_target, pred_count),

                        'C_MSE-gene': self.metrics.mse_gene(C_target, pred_count),
                        'C_Pearson_gene': self.metrics.pearson_gene(C_target, pred_count),
                        'C_Spearman_gene': self.metrics.spearman_gene(C_target, pred_count),
                        'C_r2_gene': self.metrics.r2_gene(C_target, pred_count),
                        # 'C_Cidx_gene':self.metrics.c_index_nbinom_gene(pred_count_n, pred_count_p, C_target, num_pairs=-1),

                        'C_MSE-prom': self.metrics.mse_prom(C_target, pred_count),
                        'C_Pearson_prom': self.metrics.pearson_prom(C_target, pred_count),
                        'C_Spearman_prom': self.metrics.spearman_prom(C_target, pred_count),
                        'C_r2_prom': self.metrics.r2_prom(C_target, pred_count),
                        # 'C_Cidx_prom':self.metrics.c_index_nbinom_prom(pred_count_n, pred_count_p, C_target, num_pairs=-1),

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
                        'P_Cidx_1obs': self.metrics.c_index_gauss_1obs(pred_pval, pred_pval_std, P_target, num_pairs=-1),

                        'P_MSE-1imp': self.metrics.mse1imp(P_target, pred_pval),
                        'P_Pearson_1imp': self.metrics.pearson1_imp(P_target, pred_pval),
                        'P_Spearman_1imp': self.metrics.spearman1_imp(P_target, pred_pval),
                        'P_r2_1imp': self.metrics.r2_1imp(P_target, pred_pval),

                        'P_MSE-gene': self.metrics.mse_gene(P_target, pred_pval),
                        'P_Pearson_gene': self.metrics.pearson_gene(P_target, pred_pval),
                        'P_Spearman_gene': self.metrics.spearman_gene(P_target, pred_pval),
                        'P_r2_gene': self.metrics.r2_gene(P_target, pred_pval),
                        # 'P_Cidx_gene': self.metrics.c_index_gauss_gene(pred_pval, pred_pval_std, P_target, num_pairs=-1),

                        'P_MSE-prom': self.metrics.mse_prom(P_target, pred_pval),
                        'P_Pearson_prom': self.metrics.pearson_prom(P_target, pred_pval),
                        'P_Spearman_prom': self.metrics.spearman_prom(P_target, pred_pval),
                        'P_r2_prom': self.metrics.r2_prom(P_target, pred_pval),
                        # 'P_Cidx_prom': self.metrics.c_index_gauss_prom(pred_pval, pred_pval_std, P_target, num_pairs=-1),

                        "P_peak_overlap_01thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.01),
                        "P_peak_overlap_05thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.05),
                        "P_peak_overlap_10thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.10),
                    }
                    
                    if not quick:
                        metric["obs_count"] = C_target
                        metric["obs_pval"] = P_target

                        metric["pred_count"] = pred_count
                        metric["pred_count_std"] = pred_count_std

                        metric["pred_pval"] = pred_pval
                        metric["pred_pval_std"] = pred_pval_std

                        metric["count_lower_95"] = count_lower_95
                        metric["count_upper_95"] = count_upper_95

                        metric["pval_lower_95"] = pval_lower_95
                        metric["pval_upper_95"] = pval_upper_95

                        if self.dataset.has_rnaseq(bios_name):
                            metrics["rnaseq-true-pcc-linear"] = rnaseq_res["true_linear"]["avg_pcc"]
                            metrics["rnaseq-true-pcc-svr"] = rnaseq_res["true_svr"]["avg_pcc"]

                            metrics["rnaseq-denoised-pcc-linear"] = rnaseq_res["denoised_linear"]["avg_pcc"]
                            metrics["rnaseq-denoised-pcc-svr"] = rnaseq_res["denoised_svr"]["avg_pcc"]

                            metrics["rnaseq-true-mse-linear"] = rnaseq_res["true_linear"]["avg_mse"]
                            metrics["rnaseq-true-mse-svr"] = rnaseq_res["true_svr"]["avg_mse"]
                            
                            metrics["rnaseq-denoised-mse-linear"] = rnaseq_res["denoised_linear"]["avg_mse"]
                            metrics["rnaseq-denoised-mse-svr"] = rnaseq_res["denoised_svr"]["avg_mse"]

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
                    metrics["pval_upper_95"] =  pval_upper_9

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
                'C_Cidx_1obs':self.metrics.c_index_nbinom_1obs(pred_count_n, pred_count_p, C_target),

                'C_MSE-1imp': self.metrics.mse1imp(C_target, pred_count),
                'C_Pearson_1imp': self.metrics.pearson1_imp(C_target, pred_count),
                'C_Spearman_1imp': self.metrics.spearman1_imp(C_target, pred_count),
                'C_r2_1imp': self.metrics.r2_1imp(C_target, pred_count),

                'C_MSE-gene': self.metrics.mse_gene(C_target, pred_count),
                'C_Pearson_gene': self.metrics.pearson_gene(C_target, pred_count),
                'C_Spearman_gene': self.metrics.spearman_gene(C_target, pred_count),
                'C_r2_gene': self.metrics.r2_gene(C_target, pred_count),
                'C_Cidx_gene':self.metrics.c_index_nbinom_gene(pred_count_n, pred_count_p, C_target),

                'C_MSE-prom': self.metrics.mse_prom(C_target, pred_count),
                'C_Pearson_prom': self.metrics.pearson_prom(C_target, pred_count),
                'C_Spearman_prom': self.metrics.spearman_prom(C_target, pred_count),
                'C_r2_prom': self.metrics.r2_prom(C_target, pred_count),
                'C_Cidx_prom':self.metrics.c_index_nbinom_prom(pred_count_n, pred_count_p, C_target),
                
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
                'P_Cidx_1obs': self.metrics.c_index_gauss_1obs(pred_pval, pred_pval_std, P_target),

                'P_MSE-1imp': self.metrics.mse1imp(P_target, pred_pval),
                'P_Pearson_1imp': self.metrics.pearson1_imp(P_target, pred_pval),
                'P_Spearman_1imp': self.metrics.spearman1_imp(P_target, pred_pval),
                'P_r2_1imp': self.metrics.r2_1imp(P_target, pred_pval),

                'P_MSE-gene': self.metrics.mse_gene(P_target, pred_pval),
                'P_Pearson_gene': self.metrics.pearson_gene(P_target, pred_pval),
                'P_Spearman_gene': self.metrics.spearman_gene(P_target, pred_pval),
                'P_r2_gene': self.metrics.r2_gene(P_target, pred_pval),
                'P_Cidx_gene': self.metrics.c_index_gauss_gene(pred_pval, pred_pval_std, P_target),

                'P_MSE-prom': self.metrics.mse_prom(P_target, pred_pval),
                'P_Pearson_prom': self.metrics.pearson_prom(P_target, pred_pval),
                'P_Spearman_prom': self.metrics.spearman_prom(P_target, pred_pval),
                'P_r2_prom': self.metrics.r2_prom(P_target, pred_pval),
                'P_Cidx_prom': self.metrics.c_index_gauss_prom(pred_pval, pred_pval_std, P_target),

                "P_peak_overlap_01thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.01),
                "P_peak_overlap_05thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.05),
                "P_peak_overlap_10thr": self.metrics.peak_overlap(P_target, pred_pval, p=0.10)
            }

            if not quick:
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

    def bios_pipeline(self, bios_name, x_dsf, quick=False):
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf)  
        
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
    
    def bios_pipeline_eic(self, bios_name, x_dsf, quick=False):
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf) 

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
            "count_track": self.viz.count_track,
            "signal_track": self.viz.signal_track,

            "count_confidence": self.viz.count_confidence,
            "signal_confidence": self.viz.signal_confidence,
            
            "count_error_std_hexbin": self.viz.count_error_std_hexbin,
            "signal_error_std_hexbin": self.viz.signal_error_std_hexbin,

            "count_scatter_with_marginals": self.viz.count_scatter_with_marginals,
            "signal_scatter_with_marginals": self.viz.signal_scatter_with_marginals,

            "count_heatmap": self.viz.count_heatmap,
            "signal_heatmap": self.viz.signal_heatmap,
            
            "count_rank_heatmap": self.viz.count_rank_heatmap,
            "signal_rank_heatmap": self.viz.signal_rank_heatmap,

            "count_TSS_confidence_boxplot": self.viz.count_TSS_confidence_boxplot,
            "signal_TSS_confidence_boxplot": self.viz.signal_TSS_confidence_boxplot,

            "count_GeneBody_confidence_boxplot": self.viz.count_GeneBody_confidence_boxplot,
            "signal_GeneBody_confidence_boxplot": self.viz.signal_GeneBody_confidence_boxplot,

            "count_obs_vs_confidence": self.viz.count_obs_vs_confidence,
            "signal_obs_vs_confidence": self.viz.signal_obs_vs_confidence,

            "count_TSS_enrichment_v_confidence": self.viz.count_TSS_enrichment_v_confidence,
            "signal_TSS_enrichment_v_confidence": self.viz.signal_TSS_enrichment_v_confidence,

            "count_GeneBody_enrichment_v_confidence": self.viz.count_GeneBody_enrichment_v_confidence,
            "signal_GeneBody_enrichment_v_confidence": self.viz.signal_GeneBody_enrichment_v_confidence,

            # "quantile_hist": self.viz.quantile_hist,
            # "quantile_heatmap": self.viz.quantile_heatmap,
            # "count_mean_std_hexbin": self.viz.count_mean_std_hexbin,
            # "signal_mean_std_hexbin": self.viz.signal_mean_std_hexbin,

            "count_context_length_specific_performance": self.viz.count_context_length_specific_performance,
            "signal_context_length_specific_performance": self.viz.signal_context_length_specific_performance
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
            if not self.dataset.has_rnaseq(bios):
                continue

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

    def bios_rnaseq_eval(self, bios_name, x_dsf):
        if not self.dataset.has_rnaseq(bios_name):
            return

        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf)  
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf)  

        available_indices = torch.where(avX[0, :] == 1)[0]

        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            n_ups, p_ups, mu_ups, var_ups = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        del X, mX, mY, avX, avY  # Free up memory

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])

        ups_count_dist = NegativeBinomial(p_ups, n_ups)

        ups_pval_dist = Gaussian(mu_ups, var_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1])

        ups_count_mean = ups_count_dist.expect()
        ups_count_std = ups_count_dist.std()
        ups_pval_mean = ups_pval_dist.mean()
        ups_pval_std = ups_pval_dist.std()

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


def main():
    pd.set_option('display.max_rows', None)
    # bios -> "B_DND-41"
    parser = argparse.ArgumentParser(description="Evaluate CANDI model with specified parameters.")

    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("-hp", "--hyper_parameters_path", type=str, required=True, help="Path to hyperparameters file.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the input data.")
    parser.add_argument("-s", "--savedir", type=str, default="/project/compbio-lab/EPD/CANDI_APR2025/", help="Directory to save evaluation results.")
    parser.add_argument("--enc_ckpt", type=str, default=None, help="If CANDI-DINO, path to encoder checkpoint model.")
    parser.add_argument("--dec_ckpt", type=str, default=None, help="If CANDI-DINO, path to decoder checkpoint model.")
    parser.add_argument("-r", "--resolution", type=int, default=25, help="Resolution for evaluation.")
    parser.add_argument("-cl", "--context_length", type=int, default=1200, help="Context length for evaluation.")
    parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for evaluation.")
    parser.add_argument("--eic", action="store_true", help="Flag to enable EIC mode.")
    parser.add_argument("--rnaonly", action="store_true", help="Flag to evaluate only RNAseq prediction.")
    parser.add_argument("--dino", action="store_true", help="Flag to enable DINO mode.")
    parser.add_argument("--dna", action="store_true", default=True, help="Flag to include DNA in the evaluation.")
    parser.add_argument("--quick", action="store_true", help="Flag to quickly compute eval metrics for one biosample.")
    parser.add_argument("--list_bios", action="store_true", help="print the list of all available biosamples for evaluating")

    parser.add_argument("--dsf", type=int, default=1, help="Down-sampling factor.")
    parser.add_argument("bios_name", type=str, help="BIOS argument for the pipeline.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split to evaluate on. Options: test, val.")
    parser.add_argument("--chr_sizes_file", type=str, default="data/hg38.chrom.sizes", help="Path to chromosome sizes file.")


    args = parser.parse_args()
    savedir = args.savedir

    if args.dino:
        savedir = savedir.replace("CANDI", "CANDINO")
        
    ec = EVAL_CANDI(
        args.model_path, args.data_path, args.context_length, args.batch_size, args.hyper_parameters_path,
        chr_sizes_file=args.chr_sizes_file, resolution=args.resolution, savedir=savedir, 
        mode="eval", split=args.split, eic=args.eic, DNA=args.dna, 
        DINO=args.dino, ENC_CKP=args.enc_ckpt, DEC_CKP=args.dec_ckpt)

    if args.list_bios:
        for k, v in ec.dataset.navigation.items():
            print(f"{k}: {len(v)} available assays")
        exit()
        
    if args.bios_name == "all":
        if args.rnaonly:
            ec.rnaseq_all(dsf=1)
        else:
            ec.viz_all(dsf=1)

    else:
        if args.rnaonly and not args.eic:
            ec.bios_rnaseq_eval(args.bios_name, args.dsf)
            exit()
            
        if args.eic:
            t0 = datetime.now()
            res = ec.bios_pipeline_eic(args.bios_name, args.dsf, args.quick)
            elapsed_time = datetime.now() - t0
            print(f"took {elapsed_time}")

            if args.quick:
                report = pd.DataFrame(res)
                print(report[["feature", "comparison"] + [c for c in report.columns if "Cidx" in c]])
        else:
            t0 = datetime.now()
            res = ec.bios_pipeline(args.bios_name, args.dsf, args.quick)
            elapsed_time = datetime.now() - t0
            print(f"took {elapsed_time}")

            if args.quick:
                report = pd.DataFrame(res)
                print(report[["feature", "comparison"] + [c for c in report.columns if "Cidx" in c]])

        if not args.quick:
            ec.viz_bios(eval_res=res)
            res = ec.filter_res(res)
            print(pd.DataFrame(res))

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