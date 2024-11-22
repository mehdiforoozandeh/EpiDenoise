import os, pyBigWig, pybedtools, random, datetime, gzip, pickle, psutil, math
from torch.utils.data import Dataset
from io import BytesIO
import pandas as pd
import numpy as np
import multiprocessing as mp
import torch
from scipy.stats import nbinom
import torch.distributions as dist
from eval import METRICS
from data import ExtendedEncodeDataHandler, get_DNA_sequence, dna_to_onehot
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import matplotlib.pyplot as plt

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

import torch
from scipy.stats import norm

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

    def pmf(self, k):
        k = torch.tensor(k, dtype=torch.float32)
        comb = torch.lgamma(k + self.n) - torch.lgamma(k + 1) - torch.lgamma(self.n)
        return torch.exp(comb) * (self.p ** self.n) * ((1 - self.p) ** k)

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

class MONITOR_VALIDATION(object): # CANDI
    def __init__(
        self, data_path, context_length, batch_size, 
        chr_sizes_file="data/hg38.chrom.sizes", DNA=False, eic=False, resolution=25, split="val", 
        token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}, device=None):

        self.data_path = data_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution
        self.DNA = DNA
        self.eic = eic

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(
            self.context_length, check_completeness=True, split=split, 
            bios_min_exp_avail_threshold=1, eic=eic)

        self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.example_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution), # ITSN1
            (36260000//self.resolution, 36450000//self.resolution), # RUNX1
            (45000000//self.resolution, 45250000//self.resolution), # COL18A1
            (36600000//self.resolution, 36850000//self.resolution), # MX1
            (39500000//self.resolution, 40000000//self.resolution) # Highly Conserved Non-Coding Sequences (HCNS)
            ]

        self.token_dict = token_dict

        self.chr_sizes = {}
        self.metrics = METRICS()
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)


        print(self.mark_dict)
        exit()

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None):
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        mu = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        var = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
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
                    outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch)
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)


            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()
            mu[i:i+outputs_mu.shape[0], :, :] = outputs_mu.cpu()
            var[i:i+outputs_var.shape[0], :, :] = outputs_var.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var  # Free up memory
            torch.cuda.empty_cache()  # Free up GPU memory

        return n, p, mu, var

    def get_bios_frame(self, bios_name, x_dsf=1, y_dsf=1, fill_in_y_prompt=False, fixed_segment=None):
        print(f"getting bios vals for {bios_name}")
        
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

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

        subsets_X = []
        subsets_Y = []
        subsets_P = []

        if self.DNA:
            subsets_seq = []

        if fixed_segment is None:
            # Use example coordinates (similar to get_bios_eic behavior)
            coordinates = self.example_coords
        else:
            # Use fixed segment (similar to get_bios_frame behavior)
            start, end = fixed_segment
            coordinates = [(start, end)]

        for start, end in coordinates:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])
            subsets_P.append(P[start:adjusted_end, :])

            if self.DNA:
                subsets_seq.append(
                    dna_to_onehot(get_DNA_sequence("chr21", start*self.resolution, adjusted_end*self.resolution)))

        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)
        P = torch.cat(subsets_P, dim=0)

        if self.DNA:
            seq = torch.cat(subsets_seq, dim=0)
        
        # print(X.shape, Y.shape, P.shape, seq.shape)
            
        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        P = P.view(-1, self.context_length, P.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

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

        return imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, available_indices

    def get_bios_frame_eic(self, bios_name, x_dsf=1, y_dsf=1, fill_in_y_prompt=False, fixed_segment=None):
        print(f"getting bios vals for {bios_name}")
        
        # Load and process X (input) with "T_" prefix replacement in bios_name
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
        temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        temp_p = {**temp_py, **temp_px}

        P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
        del temp_py, temp_px, temp_p

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

        subsets_X = []
        subsets_Y = []
        subsets_P = []

        if self.DNA:
            subsets_seq = []

        if fixed_segment is None:
            coordinates = self.example_coords
        else:
            start, end = fixed_segment
            coordinates = [(start, end)]

        for start, end in coordinates:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])
            subsets_P.append(P[start:adjusted_end, :])

            if self.DNA:
                subsets_seq.append(
                    dna_to_onehot(get_DNA_sequence("chr21", start*self.resolution, adjusted_end*self.resolution)))

        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)
        P = torch.cat(subsets_P, dim=0)

        if self.DNA:
            seq = torch.cat(subsets_seq, dim=0)
        
        # print(X.shape, Y.shape, P.shape, seq.shape)
            
        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        P = P.view(-1, self.context_length, P.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

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

        ups_count_dist = NegativeBinomial(p_ups, n_ups)
        ups_pval_dist = Gaussian(mu_ups, var_ups)

        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1])
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1])

        return ups_count_dist, ups_pval_dist, Y, X, P, bios_name, available_X_indices, available_Y_indices

    def get_metric(self, imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, availability):
        imp_mean = imp_count_dist.expect()
        ups_mean = ups_count_dist.expect()

        imp_gaussian_mean = imp_pval_dist.mean()
        ups_gaussian_mean = ups_pval_dist.mean()

        results = []
        for j in range(Y.shape[1]):

            if j in list(availability):
                for comparison in ['imputed', 'upsampled']:
                    if comparison == "imputed":
                        pred_count = imp_mean[:, j].numpy()
                        pred_pval = imp_gaussian_mean[:, j].numpy()
                        
                    elif comparison == "upsampled":
                        pred_count = ups_mean[:, j].numpy()
                        pred_pval = ups_gaussian_mean[:, j].numpy()

                    target_count = Y[:, j].numpy()
                    target_pval = P[:, j].numpy()

                    metrics = {
                        'bios':bios_name,
                        'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                        'comparison': comparison,
                        'available assays': len(availability),

                        'MSE_count': self.metrics.mse(target_count, pred_count),
                        'Pearson_count': self.metrics.pearson(target_count, pred_count),
                        'Spearman_count': self.metrics.spearman(target_count, pred_count),
                        'r2_count': self.metrics.r2(target_count, pred_count),
                        
                        'MSE_pval': self.metrics.mse(target_pval, pred_pval),
                        'Pearson_pval': self.metrics.pearson(target_pval, pred_pval),
                        'Spearman_pval': self.metrics.spearman(target_pval, pred_pval),
                        'r2_pval': self.metrics.r2(target_pval, pred_pval)
                    }
                    results.append(metrics)

        return results
    
    def get_metric_eic(self, ups_count_dist, ups_pval_dist, Y, X, P, bios_name, available_X_indices, available_Y_indices):
        ups_mean = ups_count_dist.expect()
        ups_pval = ups_pval_dist.mean()
        
        results = []
        for j in range(Y.shape[1]):
            pred_count = ups_mean[:, j].numpy()
            pred_pval = ups_pval[:, j].numpy()
            target_pval = P[:, j].numpy()

            if j in list(available_X_indices):
                comparison = "upsampled"
                target_count = X[:, j].numpy()

            elif j in list(available_Y_indices):
                comparison = "imputed"
                target_count = Y[:, j].numpy()

            else:
                continue

            metrics = {
                'bios':bios_name,
                'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                'comparison': comparison,
                'available assays': len(available_X_indices),

                'MSE_count': self.metrics.mse(target_count, pred_count),
                'Pearson_count': self.metrics.pearson(target_count, pred_count),
                'Spearman_count': self.metrics.spearman(target_count, pred_count),
                'r2_count': self.metrics.r2(target_count, pred_count),

                'MSE_pval': self.metrics.mse(target_pval, pred_pval),
                'Pearson_pval': self.metrics.pearson(target_pval, pred_pval),
                'Spearman_pval': self.metrics.spearman(target_pval, pred_pval),
                'r2_pval': self.metrics.r2(target_pval, pred_pval)
            }
            results.append(metrics)

        return results

    def get_validation(self, model, x_dsf=1, y_dsf=1):
        t0 = datetime.datetime.now()
        self.model = model

        full_res = []
        bioses = list(self.dataset.navigation.keys())

        for bios_name in bioses:
            if self.eic:
                # try:
                ups_count_dist, ups_pval_dist, Y, X, P, bios_name, available_X_indices, available_Y_indices = self.get_bios_frame_eic(
                    bios_name, x_dsf=x_dsf, y_dsf=y_dsf)
                full_res += self.get_metric_eic(ups_count_dist, ups_pval_dist, Y, X, P, bios_name, available_X_indices, available_Y_indices)
                
                # except:
                    # pass
            else:
                # try:
                imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, availability = self.get_bios_frame(
                    bios_name, x_dsf=x_dsf, y_dsf=y_dsf)

                
                full_res += self.get_metric(imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, availability)
                # except:
                    # pass

        del self.model
        del model
        df = pd.DataFrame(full_res)

        # Separate the data based on comparison type
        imputed_df = df[df['comparison'] == 'imputed']
        upsampled_df = df[df['comparison'] == 'upsampled']

        # Function to calculate mean, min, and max for a given metric
        def calculate_stats(df, metric):
            return df[metric].mean(), df[metric].min(), df[metric].max()

        # Imputed statistics for count metrics
        imp_mse_count_stats = calculate_stats(imputed_df, 'MSE_count')
        imp_pearson_count_stats = calculate_stats(imputed_df, 'Pearson_count')
        imp_spearman_count_stats = calculate_stats(imputed_df, 'Spearman_count')
        imp_r2_count_stats = calculate_stats(imputed_df, 'r2_count')

        # Imputed statistics for p-value metrics
        imp_mse_pval_stats = calculate_stats(imputed_df, 'MSE_pval')
        imp_pearson_pval_stats = calculate_stats(imputed_df, 'Pearson_pval')
        imp_spearman_pval_stats = calculate_stats(imputed_df, 'Spearman_pval')
        imp_r2_pval_stats = calculate_stats(imputed_df, 'r2_pval')

        # Upsampled statistics for count metrics
        ups_mse_count_stats = calculate_stats(upsampled_df, 'MSE_count')
        ups_pearson_count_stats = calculate_stats(upsampled_df, 'Pearson_count')
        ups_spearman_count_stats = calculate_stats(upsampled_df, 'Spearman_count')
        ups_r2_count_stats = calculate_stats(upsampled_df, 'r2_count')

        # Upsampled statistics for p-value metrics
        ups_mse_pval_stats = calculate_stats(upsampled_df, 'MSE_pval')
        ups_pearson_pval_stats = calculate_stats(upsampled_df, 'Pearson_pval')
        ups_spearman_pval_stats = calculate_stats(upsampled_df, 'Spearman_pval')
        ups_r2_pval_stats = calculate_stats(upsampled_df, 'r2_pval')

        elapsed_time = datetime.datetime.now() - t0
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the updated print statement
        print_statement = f"""
        Took {int(minutes)}:{int(seconds)}
        For Imputed Counts:
        - MSE_count: mean={imp_mse_count_stats[0]:.2f}, min={imp_mse_count_stats[1]:.2f}, max={imp_mse_count_stats[2]:.2f}
        - PCC_count: mean={imp_pearson_count_stats[0]:.2f}, min={imp_pearson_count_stats[1]:.2f}, max={imp_pearson_count_stats[2]:.2f}
        - SRCC_count: mean={imp_spearman_count_stats[0]:.2f}, min={imp_spearman_count_stats[1]:.2f}, max={imp_spearman_count_stats[2]:.2f}
        - R2_count: mean={imp_r2_count_stats[0]:.2f}, min={imp_r2_count_stats[1]:.2f}, max={imp_r2_count_stats[2]:.2f}

        For Imputed P-values:
        - MSE_pval: mean={imp_mse_pval_stats[0]:.2f}, min={imp_mse_pval_stats[1]:.2f}, max={imp_mse_pval_stats[2]:.2f}
        - PCC_pval: mean={imp_pearson_pval_stats[0]:.2f}, min={imp_pearson_pval_stats[1]:.2f}, max={imp_pearson_pval_stats[2]:.2f}
        - SRCC_pval: mean={imp_spearman_pval_stats[0]:.2f}, min={imp_spearman_pval_stats[1]:.2f}, max={imp_spearman_pval_stats[2]:.2f}
        - R2_pval: mean={imp_r2_pval_stats[0]:.2f}, min={imp_r2_pval_stats[1]:.2f}, max={imp_r2_pval_stats[2]:.2f}

        For Upsampled Counts:
        - MSE_count: mean={ups_mse_count_stats[0]:.2f}, min={ups_mse_count_stats[1]:.2f}, max={ups_mse_count_stats[2]:.2f}
        - PCC_count: mean={ups_pearson_count_stats[0]:.2f}, min={ups_pearson_count_stats[1]:.2f}, max={ups_pearson_count_stats[2]:.2f}
        - SRCC_count: mean={ups_spearman_count_stats[0]:.2f}, min={ups_spearman_count_stats[1]:.2f}, max={ups_spearman_count_stats[2]:.2f}
        - R2_count: mean={ups_r2_count_stats[0]:.2f}, min={ups_r2_count_stats[1]:.2f}, max={ups_r2_count_stats[2]:.2f}

        For Upsampled P-values:
        - MSE_pval: mean={ups_mse_pval_stats[0]:.2f}, min={ups_mse_pval_stats[1]:.2f}, max={ups_mse_pval_stats[2]:.2f}
        - PCC_pval: mean={ups_pearson_pval_stats[0]:.2f}, min={ups_pearson_pval_stats[1]:.2f}, max={ups_pearson_pval_stats[2]:.2f}
        - SRCC_pval: mean={ups_spearman_pval_stats[0]:.2f}, min={ups_spearman_pval_stats[1]:.2f}, max={ups_spearman_pval_stats[2]:.2f}
        - R2_pval: mean={ups_r2_pval_stats[0]:.2f}, min={ups_r2_pval_stats[1]:.2f}, max={ups_r2_pval_stats[2]:.2f}
        """

        metrics_dict = {
            "imputed_counts": {
                "MSE_count": {"mean": imp_mse_count_stats[0], "min": imp_mse_count_stats[1], "max": imp_mse_count_stats[2]},
                "PCC_count": {"mean": imp_pearson_count_stats[0], "min": imp_pearson_count_stats[1], "max": imp_pearson_count_stats[2]},
                "SRCC_count": {"mean": imp_spearman_count_stats[0], "min": imp_spearman_count_stats[1], "max": imp_spearman_count_stats[2]},
                "R2_count": {"mean": imp_r2_count_stats[0], "min": imp_r2_count_stats[1], "max": imp_r2_count_stats[2]},
            },
            "imputed_pvals": {
                "MSE_pval": {"mean": imp_mse_pval_stats[0], "min": imp_mse_pval_stats[1], "max": imp_mse_pval_stats[2]},
                "PCC_pval": {"mean": imp_pearson_pval_stats[0], "min": imp_pearson_pval_stats[1], "max": imp_pearson_pval_stats[2]},
                "SRCC_pval": {"mean": imp_spearman_pval_stats[0], "min": imp_spearman_pval_stats[1], "max": imp_spearman_pval_stats[2]},
                "R2_pval": {"mean": imp_r2_pval_stats[0], "min": imp_r2_pval_stats[1], "max": imp_r2_pval_stats[2]},
            },
            "upsampled_counts": {
                "MSE_count": {"mean": ups_mse_count_stats[0], "min": ups_mse_count_stats[1], "max": ups_mse_count_stats[2]},
                "PCC_count": {"mean": ups_pearson_count_stats[0], "min": ups_pearson_count_stats[1], "max": ups_pearson_count_stats[2]},
                "SRCC_count": {"mean": ups_spearman_count_stats[0], "min": ups_spearman_count_stats[1], "max": ups_spearman_count_stats[2]},
                "R2_count": {"mean": ups_r2_count_stats[0], "min": ups_r2_count_stats[1], "max": ups_r2_count_stats[2]},
            },
            "upsampled_pvals": {
                "MSE_pval": {"mean": ups_mse_pval_stats[0], "min": ups_mse_pval_stats[1], "max": ups_mse_pval_stats[2]},
                "PCC_pval": {"mean": ups_pearson_pval_stats[0], "min": ups_pearson_pval_stats[1], "max": ups_pearson_pval_stats[2]},
                "SRCC_pval": {"mean": ups_spearman_pval_stats[0], "min": ups_spearman_pval_stats[1], "max": ups_spearman_pval_stats[2]},
                "R2_pval": {"mean": ups_r2_pval_stats[0], "min": ups_r2_pval_stats[1], "max": ups_r2_pval_stats[2]}
            }}


        return print_statement, metrics_dict

    def generate_training_gif_frame(self, model, fig_title):
        def gen_subplt(
            ax, x_values, 
            observed_count, observed_p_value,
            ups11_count, ups21_count,   #ups41_count, 
            ups11_pval, ups21_pval,     #ups41_pval, 
            imp11_count, imp21_count,   #imp41_count, 
            imp11_pval, imp21_pval,     #imp41_pval, 
            col, assname, ytick_fontsize=6, title_fontsize=6):

            # Define the data and labels
            data = [
                (observed_count, "Obs_count", "royalblue", f"{assname}_Obs_Count"),
                (ups11_count, "Count Ups. 1->1", "darkcyan", f"{assname}_Ups1->1"),
                (imp11_count, "Count Imp. 1->1", "salmon", f"{assname}_Imp1->1"),
                (ups21_count, "Count Ups. 2->1", "darkcyan", f"{assname}_Ups2->1"),
                (imp21_count, "Count Imp. 2->1", "salmon", f"{assname}_Imp2->1"),
                # (ups41, "Count Ups. 4->1", "darkcyan", f"{assname}_Ups4->1"),
                # (imp41, "Count Imp. 4->1", "salmon", f"{assname}_Imp4->1"),

                (observed_p_value, "Obs_P", "royalblue", f"{assname}_Obs_P"),
                (ups11_pval, "P-Value Ups 1->1", "darkcyan", f"{assname}_Ups1->1"),
                (imp11_pval, "P-Value Imp 1->1", "salmon", f"{assname}_Imp1->1"),
                (ups21_pval, "P-Value Ups 2->1", "darkcyan", f"{assname}_Ups2->1"),
                (imp21_pval, "P-Value Imp 2->1", "salmon", f"{assname}_Imp2->1"),
                # (ups41, "P-Value Ups 4->1", "darkcyan", f"{assname}_Ups4->1"),
                # (imp41, "P-Value Imp 4->1", "salmon", f"{assname}_Imp4->1"),
            ]
            
            for i, (values, label, color, title) in enumerate(data):
                ax[i, col].plot(x_values, values, "--" if i != 0 else "-", color=color, alpha=0.7, label=label, linewidth=0.01)
                ax[i, col].fill_between(x_values, 0, values, color=color, alpha=0.7)
                # print("done!", values.shape, label, color, title)
                
                if i != len(data)-1:
                    ax[i, col].tick_params(axis='x', labelbottom=False)
                
                ax[i, col].tick_params(axis='y', labelsize=ytick_fontsize)
                ax[i, col].set_xticklabels([])
                ax[i, col].set_title(title, fontsize=title_fontsize)

        self.model = model
        
        bios_name = list(self.dataset.navigation.keys())[1]
        
        # dsf2-1
        imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, available_indices = self.get_bios_frame(
            bios_name, x_dsf=1, y_dsf=1, fixed_segment=(33481539//self.resolution, 33588914//self.resolution))
        
        ups11_count = ups_count_dist.expect()
        imp11_count = imp_count_dist.expect()
        
        ups11_pval = ups_pval_dist.mean()
        imp11_pval = imp_pval_dist.mean() 

        del imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, available_indices
        
        # dsf1-1
        imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, available_indices = self.get_bios_frame(
            bios_name, x_dsf=2, y_dsf=1, fixed_segment=(33481539//self.resolution, 33588914//self.resolution))

        ups21_count = ups_count_dist.expect()
        imp21_count = imp_count_dist.expect()

        ups21_pval = ups_pval_dist.mean()
        imp21_pval = imp_pval_dist.mean()

        del self.model

        selected_assays = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K9me3", "CTCF", "DNase-seq", "ATAC-seq"]
        available_selected = []
        for col, jj in enumerate(available_indices):
            assay = self.mark_dict[f"M{str(jj.item()+1).zfill(len(str(len(self.mark_dict))))}"]
            if assay in selected_assays:
                available_selected.append(jj)

        fig, axes = plt.subplots(10, len(available_selected), figsize=(len(available_selected) * 3, 10), sharex=True, sharey=False)
        
        for col, jj in enumerate(available_selected):
            j = jj.item()
            assay = self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"]
            x_values = list(range(len(Y[:, j])))

            obs_count = Y[:, j].numpy()
            obs_pval =  P[:, j].numpy()

            gen_subplt(axes, x_values, 
                    obs_count, obs_pval,
                    ups11_count[:, j].numpy(), ups21_count[:, j].numpy(),   #ups41_count, 
                    ups11_pval[:, j].numpy(), ups21_pval[:, j].numpy(),     #ups41_pval, 
                    imp11_count[:, j].numpy(), imp21_count[:, j].numpy(),   #imp41_count, 
                    imp11_pval[:, j].numpy(), imp21_pval[:, j].numpy(), 
                    col, assay)

        fig.suptitle(fig_title, fontsize=10)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return buf

    def generate_training_gif_frame_eic(self, model, fig_title):
        def gen_subplt(
            ax, x_values, 
            observed_count, observed_p_value,
            ups11_count, ups21_count,  # ups41_count, 
            ups11_pval, ups21_pval,    # ups41_pval, 
            col, assname, comparison, ytick_fontsize=6, title_fontsize=6):

            # Define the data and labels
            data = [
                (observed_count, f"Obs_count ({comparison})", "royalblue", f"{assname}_Obs_Count"),
                (ups11_count, f"Count Ups. 1->1 ({comparison})", "darkcyan", f"{assname}_Ups1->1 ({comparison})"),
                (ups21_count, f"Count Ups. 2->1 ({comparison})", "darkcyan", f"{assname}_Ups2->1 ({comparison})"),
                (observed_p_value, f"Obs_P ({comparison})", "royalblue", f"{assname}_Obs_P"),
                (ups11_pval, f"P-Value Ups 1->1 ({comparison})", "darkcyan", f"{assname}_Ups1->1 ({comparison})"),
                (ups21_pval, f"P-Value Ups 2->1 ({comparison})", "darkcyan", f"{assname}_Ups2->1 ({comparison})"),
            ]
            
            for i, (values, label, color, title) in enumerate(data):
                ax[i, col].plot(x_values, values, "--" if i != 0 else "-", color=color, alpha=0.7, label=label, linewidth=0.01)
                ax[i, col].fill_between(x_values, 0, values, color=color, alpha=0.7)
                
                if i != len(data)-1:
                    ax[i, col].tick_params(axis='x', labelbottom=False)
                
                ax[i, col].tick_params(axis='y', labelsize=ytick_fontsize)
                ax[i, col].set_xticklabels([])
                ax[i, col].set_title(title, fontsize=title_fontsize)

        self.model = model

        bios_dict_sorted = dict(sorted(self.dataset.navigation.items(), key=lambda item: len(item[1]), reverse=True))
        bios_name = list(bios_dict_sorted.keys())[2]

        # DSF 2->1 (EIC-specific logic)
        ups_count_dist_21, ups_pval_dist_21, Y, X, P, bios_name, available_X_indices, available_Y_indices = self.get_bios_frame_eic(
            bios_name, x_dsf=2, y_dsf=1, fixed_segment=(33481539//self.resolution, 33588914//self.resolution))
        
        ups21_count = ups_count_dist_21.expect()
        ups21_pval = ups_pval_dist_21.mean()

        del ups_count_dist_21, ups_pval_dist_21

        # DSF 1->1 (EIC-specific logic)
        ups_count_dist_11, ups_pval_dist_11, _, _, _, _, _, _ = self.get_bios_frame_eic(
            bios_name, x_dsf=1, y_dsf=1, fixed_segment=(33481539//self.resolution, 33588914//self.resolution))

        ups11_count = ups_count_dist_11.expect()
        ups11_pval = ups_pval_dist_11.mean()

        del ups_count_dist_11, ups_pval_dist_11

        selected_assays = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K9me3", "CTCF", "DNase-seq", "ATAC-seq"]
        available_selected = []
        for col, jj in enumerate(available_X_indices):
            assay = self.mark_dict[f"M{str(jj.item()+1).zfill(len(str(len(self.mark_dict))))}"]
            if assay in selected_assays:
                available_selected.append(jj)

        for col, jj in enumerate(available_Y_indices):
            assay = self.mark_dict[f"M{str(jj.item()+1).zfill(len(str(len(self.mark_dict))))}"]
            if assay in selected_assays:
                available_selected.append(jj)

        fig, axes = plt.subplots(6, len(available_selected), figsize=(len(available_selected) * 3, 9), sharex=True, sharey=False)
        
        for col, jj in enumerate(available_selected):
            j = jj.item()
            assay = self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"]
            x_values = list(range(len(Y[:, j])))

            if j in list(available_X_indices):
                comparison = "upsampled"
                obs_count = X[:, j].numpy()
            elif j in list(available_Y_indices):
                comparison = "imputed"
                obs_count = Y[:, j].numpy()

            obs_pval =  P[:, j].numpy()

            gen_subplt(axes, x_values, 
                    obs_count, obs_pval,
                    ups11_count[:, j].numpy(), ups21_count[:, j].numpy(),  # ups41_count
                    ups11_pval[:, j].numpy(), ups21_pval[:, j].numpy(),   # ups41_pval
                    col, assay, comparison)

        fig.suptitle(fig_title, fontsize=10)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return buf

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

def signal_feature_extraction(start, end, strand, chip_seq_signal, bin_size=25, margin=2e3):
    """
    Extracts mean ChIP-seq signals for defined regions around TSS, TES, and within the gene body.

    Parameters:
    - chr: Chromosome (string)
    - start: Start position of the gene (int)
    - end: End position of the gene (int)
    - strand: Strand information ('+' or '-') (string)
    - chip_seq_signal: A numpy array representing ChIP-seq signal binned at 25-bp resolution (1D array)

    Returns:
    - dict: Dictionary containing mean signals for the specified regions
    """

    # Define TSS and TES based on the strand
    tss = start if strand == '+' else end
    tes = end if strand == '+' else start

    # Define the regions
    promoter_start = tss - int(margin)
    promoter_end = tss + int(margin)
    gene_body_start = start
    gene_body_end = end
    tes_region_start = tes - int(margin)
    tes_region_end = tes + int(margin)

    # Convert regions to bin indices
    promoter_start_bin = max(promoter_start // bin_size, 0)
    promoter_end_bin = min(promoter_end // bin_size, len(chip_seq_signal))
    gene_body_start_bin = max(gene_body_start // bin_size, 0)
    gene_body_end_bin = min(gene_body_end // bin_size, len(chip_seq_signal))
    tes_region_start_bin = max(tes_region_start // bin_size, 0)
    tes_region_end_bin = min(tes_region_end // bin_size, len(chip_seq_signal))

    # Extract signal for each region
    promoter_signal = chip_seq_signal[promoter_start_bin:promoter_end_bin]
    gene_body_signal = chip_seq_signal[gene_body_start_bin:gene_body_end_bin]
    tes_region_signal = chip_seq_signal[tes_region_start_bin:tes_region_end_bin]

    # Calculate mean signal for each region
    mean_signal_promoter = np.mean(promoter_signal) if len(promoter_signal) > 0 else 0
    mean_signal_gene_body = np.mean(gene_body_signal) if len(gene_body_signal) > 0 else 0
    mean_signal_tes_region = np.mean(tes_region_signal) if len(tes_region_signal) > 0 else 0

    # Return the calculated mean signals in a dictionary
    return {
        'mean_sig_promoter': mean_signal_promoter,
        'mean_sig_gene_body': mean_signal_gene_body,
        'mean_sig_around_TES': mean_signal_tes_region
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

class DataMasker:
    def __init__(self, mask_value, mask_percentage, chunk_size=5, prog=False):
        self.mask_value = mask_value
        self.mask_percentage = mask_percentage
        self.chunk_size = chunk_size

    def mask_chunks(self, data):
        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        num_masks = int((L * self.mask_percentage) / self.chunk_size)
        for _ in range(num_masks):
            start = random.randint(0, L - self.chunk_size)
            mask_indicator[:, start:start+self.chunk_size, :] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mask_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)

        if len(available_features) == 1:
            return data, mask_indicator
        else:
            num_features_to_mask = int(len(self.available_features) * self.mask_percentage) + 1

        features_to_mask = random.sample(self.available_features, num_features_to_mask)

        for feature in features_to_mask:
            mask_indicator[:, :, feature] = True

        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mask_feature30(self, data, metadata, availability):
        B, L, F = data.shape

        # Number of features to mask per sample in the batch
        num_to_mask = []
        num_available = availability.sum(dim=1)
        for b in range(B):
            if num_available[b] == 1:
                num_to_mask.append(0)
            else:
                num_to_mask.append(max(1, int(num_available[b] * self.mask_percentage)))

        # Prepare the new availability tensor
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Mask indices generation and masking operation
        for b in range(B):
            if num_to_mask[b] > 0:
                available_indices = torch.where(availability[b] == 1)[0]  # Find indices where features are available
                mask_indices = torch.randperm(available_indices.size(0))[:num_to_mask[b]]  # Randomly select indices to mask
                actual_indices_to_mask = available_indices[mask_indices]  # Actual indices in the feature dimension

                data[b, :, actual_indices_to_mask] = self.mask_value  # Mask the features in X
                new_md[b, :, actual_indices_to_mask] = self.mask_value
                new_A[b, actual_indices_to_mask] = self.mask_value  # Update the availability tensor to indicate masked features

        return data, new_md, new_A

    def progressive(self, data, metadata, availability, num_mask):
        B, L, F = data.shape

        # Number of features to mask per sample in the batch
        num_to_mask = []
        num_available = availability.sum(dim=1)
        for b in range(B):

            if num_available[b] > num_mask:
                num_to_mask.append(num_mask)

            else:
                num_to_mask.append(num_available[b] - 1)

        # Prepare the new availability tensor
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Mask indices generation and masking operation
        for b in range(B):
            if num_to_mask[b] > 0:
                available_indices = torch.where(availability[b] == 1)[0]  # Find indices where features are available
                mask_indices = torch.randperm(available_indices.size(0))[:num_to_mask[b]]  # Randomly select indices to mask
                actual_indices_to_mask = available_indices[mask_indices]  # Actual indices in the feature dimension

                data[b, :, actual_indices_to_mask] = self.mask_value  # Mask the features in X
                new_md[b, :, actual_indices_to_mask] = self.mask_value
                new_A[b, actual_indices_to_mask] = self.mask_value  # Update the availability tensor to indicate masked features

        return data, new_md, new_A
    
    def mask_chunk_features_30(self, data, metadata, availability):
        B, L, F = data.shape

        # Prepare the new availability tensor
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Calculate the total number of signals and chunks to mask per batch sample
        num_all_signals = L * availability.sum(dim=1)
        num_masks = (num_all_signals * self.mask_percentage / self.chunk_size).int()

        # Masking operation for each sample in the batch
        for b in range(B):
            for _ in range(num_masks[b]):
                # Select a random chunk start and feature index
                length_start = random.randint(0, L - self.chunk_size)
                available_indices = torch.where(availability[b] == 1)[0]
                if len(available_indices) == 0:
                    continue
                feature_start = random.choice(available_indices)
                
                # Apply the mask to the data, metadata, and update availability
                data[b, length_start:length_start+self.chunk_size, feature_start] = self.mask_value
                # new_md[b, length_start:length_start+self.chunk_size, feature_start] = self.mask_value
                # new_A[b, feature_start] = 0  # Update the availability to indicate masked feature

        return data, new_md, new_A

    def mask_chunk_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        num_all_signals = L * len(self.available_features)
        num_masks = int((num_all_signals * self.mask_percentage) / self.chunk_size)
        for _ in range(num_masks):
            length_start = random.randint(0, L - self.chunk_size)
            feature_start = random.choice(self.available_features)
            mask_indicator[:, length_start:length_start+self.chunk_size, feature_start] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_mask(self, data, available_features):
        data = data.clone()
        N, L, F = data.size()
        slice_length = int(L * self.mask_percentage)
        start = L // 2 - slice_length // 2
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        mask_indicator[:, start:start+slice_length, available_features] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_mask_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        slice_length = int(L * self.mask_percentage)
        num_features_to_mask = int(len(self.available_features) * self.mask_percentage)
        features_to_mask = random.sample(self.available_features, num_features_to_mask)
        start = L // 2 - slice_length // 2
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        for feature in features_to_mask:
            mask_indicator[:, start:start+slice_length, feature] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_focused_full_feature_mask(self, data, missing_mask_value, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()

        num_features_to_mask = int(len(self.available_features) * self.mask_percentage)
        if num_features_to_mask == 0:
            features_to_mask = available_features
        else:
            features_to_mask = random.sample(self.available_features, num_features_to_mask)
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)

        # Mask features completely
        for feature in features_to_mask:
            data[:, :, feature] = missing_mask_value

        # Mark only the middle part of those masked features
        slice_length = int(L * self.mask_percentage)
        start = L // 2 - slice_length // 2
        mask_indicator[:, start:start+slice_length, features_to_mask] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator


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