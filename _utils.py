import os, pyBigWig, pybedtools, random, datetime, gzip, pickle, psutil, math
from torch.utils.data import Dataset
from io import BytesIO
import pandas as pd
import numpy as np
import multiprocessing as mp
import torch
from scipy.stats import nbinom
import torch.distributions as dist
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import psutil

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr, poisson, rankdata
from sklearn.metrics import auc


PROC_GENE_BED_FPATH = "data/gene_bodies.bed"
PROC_PROM_BED_PATH = "data/tss.bed"


class DataMasker:
    def __init__(self, mask_value, chunk_size=5, prog=False):
        self.mask_value = mask_value
        self.chunk_size = chunk_size

    def mask_assays(self, data, metadata, availability, num_mask):
        B, L, F = data.shape

        # Number of features to mask per sample in the batch
        num_to_mask = []
        num_available = availability.sum(dim=1)
        for b in range(B):
            available_count = int(num_available[b].item())
            if available_count > num_mask:
                num_to_mask.append(num_mask)
            elif available_count > 1:
                # Mask at least 1 but leave at least 1 available
                num_to_mask.append(max(1, available_count - 1))
            else:
                # Can't mask anything if only 1 or 0 features available
                num_to_mask.append(0)

        # Prepare the new tensors
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Mask indices generation and masking operation
        for b in range(B):
            if num_to_mask[b] > 0:
                available_indices = torch.where(availability[b] == 1)[0]  # Find indices where features are available
                if len(available_indices) >= num_to_mask[b]:
                    mask_indices = torch.randperm(available_indices.size(0))[:num_to_mask[b]]  # Randomly select indices to mask
                    actual_indices_to_mask = available_indices[mask_indices]  # Actual indices in the feature dimension

                    data[b, :, actual_indices_to_mask] = self.mask_value  # Mask the features in X
                    new_md[b, :, actual_indices_to_mask] = self.mask_value
                    new_A[b, actual_indices_to_mask] = self.mask_value  # Update the availability tensor to indicate masked features

        return data, new_md, new_A

class METRICS(object):
    def __init__(self, chrom='chr21', bin_size=25):
        self.prom_df = self.get_prom_positions(chrom, bin_size)
        self.gene_df = self.get_gene_positions(chrom, bin_size)

        self.gene_df["strand"] = self.prom_df["strand"]

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
        valid_indices = indices[indices < len(y_true)]
        
        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[valid_indices], sigmas[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_gauss_prom(self, mus, sigmas, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

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

    # def c_index_nbinom(self,
    #                    rs, ps, y_true,
    #                    epsilon: float = 1e-6,
    #                    num_pairs: int = 10000):
    #     """
    #     Concordance index for Negative‐Binomial predictive marginals,
    #     estimating over `num_pairs` randomly sampled pairs.

    #     Inputs:
    #       - rs:        (N,) array of NB 'r' parameters (must be >0)
    #       - ps:        (N,) array of NB 'p' parameters (0<p<1)
    #       - y_true:    (N,) array of true values y_i
    #       - epsilon:   tail‐mass cutoff for truncation (default 1e-6)
    #       - num_pairs: number of random (i<j) pairs to sample;
    #                    if -1, use all valid pairs (i<j, y_i≠y_j)
    #     Returns:
    #       - c_index: float in [0,1]
    #     """
    #     N = len(y_true)
    #     labels = []
    #     scores = []

    #     def compute_score(i, j):
    #         # clamp p_j into (ε,1−ε)
    #         p_j = np.clip(ps[j], epsilon, 1 - epsilon)
    #         r_j = rs[j]
    #         if r_j <= 0:
    #             return None
    #         # find cutoff K
    #         K = nbinom.ppf(1 - epsilon, r_j, p_j)
    #         if not np.isfinite(K):
    #             K = 0
    #         else:
    #             K = int(K)
              
    #         # PMF/CDF arrays
    #         k = np.arange(K + 1)
    #         pmf_j = nbinom.pmf(k, r_j, p_j)
    #         cdf_i = nbinom.cdf(k, rs[i], ps[i])
    #         # if anything is nan, bail
    #         if not (np.isfinite(pmf_j).all() and np.isfinite(cdf_i).all()):
    #             return None
    #         return np.sum(pmf_j * (1.0 - cdf_i))

    #     if num_pairs == -1:
    #         # exact mode
    #         for i in range(N):
    #             for j in range(i+1, N):
    #                 if y_true[i] == y_true[j]:
    #                     continue
    #                 sc = compute_score(i, j)
    #                 if sc is None:
    #                     continue
    #                 labels.append(int(y_true[i] > y_true[j]))
    #                 scores.append(sc)
    #     else:
    #         # sampling mode
    #         rng = np.random.default_rng()
    #         count = 0
    #         while count < num_pairs:
    #             i, j = rng.integers(0, N, size=2)
    #             if i == j or y_true[i] == y_true[j]:
    #                 continue
    #             if i > j:
    #                 i, j = j, i
    #             sc = compute_score(i, j)
    #             if sc is None:
    #                 continue
    #             labels.append(int(y_true[i] > y_true[j]))
    #             scores.append(sc)
    #             count += 1

    #     if len(labels) == 0:
    #         # no valid pairs
    #         return np.nan

    #     return roc_auc_score(labels, scores)

    def c_index_nbinom(self,
                    rs, ps, y_true,
                    M: int = 500,
                    num_pairs: int = 10000,
                    random_state: int = None):
        """
        Monte Carlo Concordance index for Negative‐Binomial predictive marginals.

        For each sampled pair (i,j), draw M samples from NB(ri,pi) and NB(rj,pj),
        then estimate Pr(Y_i>Y_j) by the fraction of draws with u>v.

        Inputs:
          - rs:        (N,) array of NB 'r' parameters
          - ps:        (N,) array of NB 'p' parameters
          - y_true:    (N,) array of true values y_i
          - M:         number of Monte Carlo samples per pair
          - num_pairs: how many random (i<j) pairs to sample;
                       if -1, use all valid pairs (i<j, y_i≠y_j)
          - random_state: seed for reproducibility
        Returns:
          - c_index: float in [0,1]
        """
        rng = np.random.default_rng(random_state)
        N = len(y_true)
        labels = []
        scores = []

        def mc_score(i, j):
            # draw M samples from each distribution
            u = nbinom.rvs(rs[i], ps[i], size=M, random_state=rng)
            v = nbinom.rvs(rs[j], ps[j], size=M, random_state=rng)
            return np.mean(u > v)

        if num_pairs == -1:
            # exact over all pairs
            for i in range(N):
                for j in range(i+1, N):
                    if y_true[i] == y_true[j]:
                        continue
                    labels.append(int(y_true[i] > y_true[j]))
                    scores.append(mc_score(i, j))
        else:
            # sample random pairs
            count = 0
            while count < num_pairs:
                i, j = rng.integers(0, N, size=2)
                if i == j or y_true[i] == y_true[j]:
                    continue
                # ensure i<j for consistency (optional)
                if i > j:
                    i, j = j, i
                labels.append(int(y_true[i] > y_true[j]))
                scores.append(mc_score(i, j))
                count += 1

        if not labels:
            return np.nan
        return roc_auc_score(labels, scores)

    def c_index_nbinom_gene(self, rs, ps, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.gene_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_nbinom(rs[valid_indices], ps[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_nbinom_prom(self, rs, ps, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

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

def get_divisible_heads(dim, target):
    """
    Given a dimension and a target number of heads, returns the largest integer
    <= target that divides dim. If no such number is found, returns 1.
    """
    for n in range(target, 0, -1):
        if dim % n == 0:
            return n
    return 1

def log_resource_usage():
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.memory_stats()
        print(f"GPU Memory Allocated: {gpu_stats['allocated_bytes.all.current'] / (1024 ** 2)} MB")
        print(f"GPU Memory Reserved: {gpu_stats['reserved_bytes.all.current'] / (1024 ** 2)} MB")
        print(f"GPU Active Memory Allocations: {gpu_stats['active.all.current']}")
        print(f"GPU Memory Allocated (peak): {gpu_stats['allocated_bytes.all.peak'] / (1024 ** 2)} MB")
        print(f"GPU Memory Reserved (peak): {gpu_stats['reserved_bytes.all.peak'] / (1024 ** 2)} MB")

def compute_perplexity(probabilities):
    """
    Computes the perplexity given a list of probabilities.

    Parameters:
    probabilities (list or np.array): A list or array of probabilities assigned by the model to each word in the sequence.

    Returns:
    float: The perplexity of the model on the given sequence.
    """
    N = len(probabilities)
    log_prob_sum = torch.sum(torch.log(probabilities))
    perplexity = torch.exp(-log_prob_sum / N)
    
    return perplexity

class Gaussian:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
        self.sigma = self.var ** (1/2)

    def mean(self):
        return self.mu

    def median(self):
        return self.mu

    def mode(self):
        return self.mu

    def var(self):
        return self.var

    def std(self):
        return self.sigma

    def cdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        return torch.tensor(norm.cdf(x, self.mu, self.sigma), dtype=torch.float32)

    def pdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        return torch.tensor(norm.pdf(x, self.mu, self.sigma), dtype=torch.float32)

    def icdf(self, q):
        q = q.detach().cpu().numpy() if torch.is_tensor(q) else q
        return torch.tensor(norm.ppf(q, self.mu, self.sigma), dtype=torch.float32)

    def expect(self):
        return self.mu

    def interval(self, confidence=0.95):
        lower = self.icdf((1 - confidence) / 2)
        upper = self.icdf((1 + confidence) / 2)
        return lower, upper

class NegativeBinomial:
    def __init__(self, p, n):
        self.n = n
        self.p = p

    def mean(self):
        return (self.n * (1 - self.p)) / self.p

    def median(self):
        return self.icdf(torch.tensor(0.5))

    def mode(self):
        mode = torch.floor(((self.n - 1) * (1 - self.p)) / self.p)
        mode[mode < 0] = 0  # Mode is 0 if the computed value is negative
        return mode

    def var(self):
        return self.n * (1 - self.p) / (self.p ** 2)

    def std(self):
        return self.var().sqrt()

    def cdf(self, k):
        return torch.Tensor(nbinom.cdf(k, self.n, self.p))

    # def pmf(self, k):
    #     k = torch.tensor(k, dtype=torch.float32)
    #     comb = torch.lgamma(k + self.n) - torch.lgamma(k + 1) - torch.lgamma(self.n)
    #     return torch.exp(comb) * (self.p ** self.n) * ((1 - self.p) ** k)

    def pmf(self, k):
        return torch.Tensor(nbinom.pmf(k, self.n, self.p))

    def icdf(self, q):
        return torch.Tensor(nbinom.ppf(q, self.n, self.p))

    def expect(self, stat="mean"):
        if stat == "mean":
            return self.mean()
        elif stat == "mode":
            return self.mode()
        else:
            return self.median()

    def interval(self, confidence=0.95):
        lower = self.icdf(q=(1-confidence)/2)
        upper = self.icdf(q=(1+confidence)/2)
        return lower, upper

def negative_binomial_loss(y_true, n_pred, p_pred):
    """
        Negative binomial loss function for PyTorch.
        
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth values of the predicted variable.
        n_pred : torch.Tensor
            Tensor containing n values of the predicted distribution.
        p_pred : torch.Tensor
            Tensor containing p values of the predicted distribution.
            
        Returns
        -------
        nll : torch.Tensor
            Negative log likelihood.
    """
    eps = 1e-6

    # Clamp predictions for numerical stability
    p_pred = torch.clamp(p_pred, min=eps, max=1 - eps)
    n_pred = torch.clamp(n_pred, min=1e-2, max=1e3)

    # Compute NB NLL
    nll = (
        torch.lgamma(n_pred + eps)
        + torch.lgamma(y_true + 1 + eps)
        - torch.lgamma(n_pred + y_true + eps)
        - n_pred * torch.log(p_pred + eps)
        - y_true * torch.log(1 - p_pred + eps)
    )
    
    return nll

random.seed(73)
def get_overlap(tup1, tup2):

    x = range(tup1[0], tup1[1])
    y = range(tup2[0], tup2[1])

    return len( range(max(x[0], y[0]), min(x[-1], y[-1])+1))    

def load_gene_coords(file, drop_negative_strand=True, drop_overlapping=True):
    gene_coords = pd.read_csv(file)
    gene_coords = gene_coords.drop(["Unnamed: 0"], axis=1)

    gene_coords["start"] = gene_coords["start"].astype("int")
    gene_coords["end"] = gene_coords["end"].astype("int")

    if drop_negative_strand:
        gene_coords = gene_coords.loc[gene_coords["strand"]=="+", :].reset_index(drop=True)
    
    if drop_overlapping:
        todrop = []
        for i in range(len(gene_coords)-1):
            if get_overlap((gene_coords["start"][i], gene_coords["end"][i]),(gene_coords["start"][i+1], gene_coords["end"][i+1])) >0:
                if (gene_coords["end"][i] - gene_coords["start"][i]) <= gene_coords["end"][i+1] - gene_coords["start"][i+1]:
                    todrop.append(i)
                else:
                    todrop.append(i+1)
        gene_coords = gene_coords.drop(todrop).reset_index(drop=True)

    return gene_coords

def signal_feature_extraction(start, end, strand, chip_seq_signal,
                              bin_size=25, margin=2000):
    """
    Extracts robust ChIP-seq signal summaries for promoter, gene body, and TES regions.

    Returns, for each region:
      - median signal
      - inter-quartile range (75th – 25th percentile)
      - minimum signal
      - maximum signal
    """

    # 1) Define TSS and TES by strand
    tss = start if strand == '+' else end
    tes = end   if strand == '+' else start

    # 2) Compute bp intervals for each region
    promoter_bp = (tss - margin, tss + margin)
    gene_body_bp = (start, end)
    tes_bp = (tes - margin, tes + margin)

    # 3) Map to bin indices (inclusive of any overlapping bin)
    def to_bins(bp_start, bp_end):
        i0 = max(bp_start // bin_size, 0)
        i1 = min(bp_end   // bin_size + 1, len(chip_seq_signal))
        return i0, i1

    p0, p1 = to_bins(*promoter_bp)
    g0, g1 = to_bins(*gene_body_bp)
    t0, t1 = to_bins(*tes_bp)

    promoter_signal   = chip_seq_signal[p0:p1]
    gene_body_signal  = chip_seq_signal[g0:g1]
    tes_region_signal = chip_seq_signal[t0:t1]

    # 4) Compute robust stats
    def stats(x):
        if x.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        med = np.median(x)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        mn = x.min()
        mx = x.max()
        return med, iqr, mn, mx

    prom_med, prom_iqr, prom_min, prom_max = stats(promoter_signal)
    body_med, body_iqr, body_min, body_max = stats(gene_body_signal)
    tes_med, tes_iqr, tes_min, tes_max = stats(tes_region_signal)

    # 5) Return all 12 features
    return {
        'median_sig_promoter':   prom_med,
        'iqr_sig_promoter':      prom_iqr,
        # 'min_sig_promoter':      prom_min,
        # 'max_sig_promoter':      prom_max,

        'median_sig_gene_body':  body_med,
        'iqr_sig_gene_body':     body_iqr,
        # 'min_sig_gene_body':     body_min,
        # 'max_sig_gene_body':     body_max,

        'median_sig_around_TES': tes_med,
        'iqr_sig_around_TES':    tes_iqr,
        # 'min_sig_around_TES':    tes_min,
        # 'max_sig_around_TES':    tes_max,
    }

def capture_gradients_hook(module, grad_input, grad_output):
    if hasattr(module, 'weight') and module.weight is not None:
        if grad_input[0] is not None:
            module.weight.grad_norm = grad_input[0].norm().item()
        else:
            module.weight.grad_norm = 0  # Assign a default value if grad_input[0] is None
    if hasattr(module, 'bias') and module.bias is not None:
        if len(grad_input) > 1 and grad_input[1] is not None:
            module.bias.grad_norm = grad_input[1].norm().item()
        else:
            module.bias.grad_norm = 0  # Assign a default value if grad_input[1] is None

def register_hooks(model):
    for name, module in model.named_modules():
        module.register_full_backward_hook(capture_gradients_hook)

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)
    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def linear_divisible_linspace(start_size, end_size, layers):
    """Generate channel sizes where each size is divisible by the previous size."""
    sizes = [start_size]
    step = (end_size - start_size) / (layers - 1)

    for i in range(1, layers):
        # Calculate the next size
        next_size = start_size + i * step
        # Ensure the next_size is a multiple of the last size in the list
        last_size = sizes[-1]
        next_size = np.ceil(next_size / last_size) * last_size

        # Ensure not to exceed the end_size on the last step
        if i == layers - 1 and next_size != end_size:
            next_size = end_size

        sizes.append(int(next_size))

    return sizes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_pad(data, max_length, pad_value=-1):
    # Get the original dimensions of the data
    original_size = data.size()
    
    # Create a tensor filled with the pad value with the desired size
    padded_data = torch.full((original_size[0], max_length, original_size[2]), pad_value)
    
    # Copy the original data into the padded data tensor
    padded_data[:, :original_size[1], :] = data
    
    # Create a boolean mask indicating whether each value is padded or not
    pad_mask = padded_data == pad_value
    
    return padded_data, pad_mask

def get_bin_value(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]
    bin_value = bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

    if input_dict["bw_obj"] == False:
        bw.close()

    return bin_value

def get_bin_value_dict(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]
    bin_value = bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

    input_dict["signals"] = bin_value

    if input_dict["bw_obj"] == False:
        bw.close()
        del input_dict["bw"]
        
    return input_dict

def add_noise(data, noise_factor):
    noise = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=data.shape))
    noisy_data = data + noise_factor * noise
    noisy_data = torch.clamp(noisy_data, min=0)
    return noisy_data.to(torch.float32)

def peak_overlap(y_true, y_pred, p=0.01):
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
        
class COORD(object):
    def __init__(self, Meuleman_file="data/Meuleman.tsv", cCRE_file="data/GRCh38-cCREs.bed", 
                resolution=1000, chr_sizes_file="data/hg38.chrom.sizes", outdir="data/"):
        
        self.resolution = resolution
        self.cCRE_file = cCRE_file
        self.Meuleman_file = Meuleman_file
        self.outdir = outdir
        self.chr_sizes_file = chr_sizes_file    

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        main_chrs.remove("chr21") # reserved for validation
        self.chr_sizes = {}

        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def init_bins(self):
        if os.path.exists(f"{self.outdir}/bins_{self.resolution}bp.csv"):
            self.bins = pd.read_csv(f"{self.outdir}/bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)
        else:

            # Create bins
            self.bins = []
            for chr, size in self.chr_sizes.items():
                start_coords = range(0, size, self.resolution)
                end_coords = range(self.resolution, size + self.resolution, self.resolution)
                self.bins.extend([[chr, start, end] for start, end in zip(start_coords, end_coords)][:-1])

            self.bins = pd.DataFrame(self.bins, columns =["chrom", "start", "end"])
            self.bins = self.bins.sort_values(["chrom", "start"]).reset_index(drop=True)
        self.bins.to_csv(f"{self.outdir}/bins_{self.resolution}bp.csv")

    def get_foreground(self):
        if os.path.exists(f'{self.outdir}/foreground_nobin.csv'):
            self.foreground = pd.read_csv(f'{self.outdir}/foreground_nobin.csv').drop("Unnamed: 0", axis=1)
        else:
            ccre = pybedtools.BedTool(self.cCRE_file)
            if self.Meuleman_file == "_":
                self.foreground = ccre.to_dataframe()

            else:
                Meuleman = pd.read_csv(self.Meuleman_file, sep="\t")
                Meuleman.columns = ["chr", "start", "end", "identifier", "mean_signal", "numsamples", "summit", "core_start", "core_end", "component"]
                Meuleman = pybedtools.BedTool.from_dataframe(Meuleman)

                # get the union of ccre and Meuleman
                self.foreground = ccre.cat(Meuleman, postmerge=False)
                self.foreground = self.foreground.to_dataframe()

            self.foreground = self.foreground[["chrom", "start", "end"]]
            self.foreground = self.foreground.sort_values(["chrom", "start"]).reset_index(drop=True)
            self.foreground.to_csv(f'{self.outdir}/foreground_nobin.csv')

    def bin_fg_bg(self):
        self.bins = pybedtools.BedTool.from_dataframe(self.bins)
        self.foreground = pybedtools.BedTool.from_dataframe(self.foreground)

        if os.path.exists(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv") == False:
            # Get the subset of bins that overlap with the foreground
            self.fg_bins = self.bins.intersect(self.foreground, u=True)
            self.fg_bins = self.fg_bins.to_dataframe()
            self.fg_bins.to_csv(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv")
        else:
            self.fg_bins = pd.read_csv(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)

        if os.path.exists(f"{self.outdir}/background_bins_{self.resolution}bp.csv") == False:
            # Get the subset of bins that do not overlap with the foreground
            self.bg_bins = self.bins.intersect(self.foreground, v=True)
            self.bg_bins = self.bg_bins.to_dataframe()
            self.bg_bins.to_csv(f"{self.outdir}/background_bins_{self.resolution}bp.csv")
        else:
            self.bg_bins = pd.read_csv(f"{self.outdir}/background_bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)

        print(f"number of foreground bins: {len(self.fg_bins)} | number of background bins: {len(self.bg_bins)}")

if __name__ == "__main__":

    # solar_path = "/project/compbio-lab/EIC/"
    # sample = "/project/compbio-lab/EIC/training_data/C01_m2000_25bp.pkl.gz"
    # traineic = PROCESS_EIC_DATA(solar_path+"training_data/", stratified=True)
    # m_regions = traineic.load_m_regions(sample)

    # valeic = PROCESS_EIC_DATA(solar_path+"validation_data/", stratified=True)
    # blindeic = PROCESS_EIC_DATA(solar_path+"blind_data/", stratified=True)

    # valeic.generate_m_samples_from_predefined_regions(m_regions=m_regions, multi_p=True, n_p=20)
    # blindeic.generate_m_samples_from_predefined_regions(m_regions=m_regions, multi_p=True, n_p=20)

    # exit()
    # solar_path = "/project/compbio-lab/EIC/training_data/"
    # # solar_path = "data/test/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True)
    # t0 = datetime.datetime.now()
    # # m_regions = eic.load_m_regions("data/test/C02_m500_25bp.pkl.gz")
    # # print(len(m_regions))
    # eic.generate_m_samples(m=2000, multi_p=True)
    # t1 = datetime.datetime.now()
    # # print("generated training datasets in :", t1-t0)
    # exit()


    solar_path = "/project/compbio-lab/EIC/training_data/"
    eic = PROCESS_EIC_DATA(solar_path, stratified=False, resolution=25, max_len=8000)
    t0 = datetime.datetime.now()
    eic.generate_m_samples(m=6000, n_datasets=150, multi_p=True, n_p=2)
    t1 = datetime.datetime.now()
    print("generated training datasets in :", t1-t0)

    # solar_path = "/project/compbio-lab/EIC/training_data/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True, resolution=128, max_len=8000)
    # t0 = datetime.datetime.now()
    # eic.generate_m_samples(m=2000, n_datasets=30, multi_p=True, n_p=10)
    # t1 = datetime.datetime.now()
    # print("generated training datasets in :", t1-t0)



    
    # solar_path = "/project/compbio-lab/EIC/validation_data/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True)
    # t0 = datetime.datetime.now()
    # eic.generate_m_samples(m=2000, multi_p=True)
    # t1 = datetime.datetime.now()
    # print("generated validation datasets in :", t1-t0)

    # solar_path = "/project/compbio-lab/EIC/blind_data/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True)
    # t0 = datetime.datetime.now()
    # eic.generate_m_samples(m=2000, multi_p=True)
    # t1 = datetime.datetime.now()
    # print("generated blind datasets in :", t1-t0)
    exit()
    # c = COORD(resolution=25, Meuleman_file="_")
    # c.init_bins()
    # c.get_foreground()
    # c.bin_fg_bg()

    # c.fg_bins["len"]= c.fg_bins["end"] - c.fg_bins["start"] 
    # c.bg_bins["len"]= c.bg_bins["end"] - c.bg_bins["start"] 

    # print(c.fg_bins["len"].mean())
    # print(c.bg_bins["len"].mean())

    # plt.hist(c.foreground["len"], bins=100)

    # plt.show()

    

    # exit()

    # b = BIOSAMPLE("data/test", "C02", chr_sizes_file="data/hg38.chrom.sizes", resolution=25)
    # print(b.tracks)