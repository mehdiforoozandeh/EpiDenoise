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
from data import ExtendedEncodeDataHandler
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

class MONITOR_VALIDATION(object):
    def __init__(
        self, data_path, context_length, batch_size,
        chr_sizes_file="data/hg38.chrom.sizes", 
        resolution=25, split="val", arch="a"):

        self.data_path = data_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution
        self.arch = arch

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=10)

        self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.token_dict = {
                    "missing_mask": -1, 
                    "cloze_mask": -2,
                    "pad": -3
                }
        
        if self.arch in ["c", "d"]:
            self.token_dict["cloze_mask"] = self.token_dict["missing_mask"]

        self.chr_sizes = {}
        self.metrics = METRICS()
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)
    
    def pred(self, X, mX, mY, avail, imp_target=[]):
        # print("making preds")
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

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
                if self.arch in ["a", "b"]:
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    if self.arch in ["a", "b"]:
                        avail_batch[:, imp_target] = self.token_dict["cloze_mask"]
                    elif self.arch in ["c", "d"]:
                        avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.arch in ["a", "b", "d"]:
                    outputs_p, outputs_n, _, _ = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)
                elif self.arch in ["c"]:
                    outputs_p, outputs_n = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)

            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n  # Free up memory
            torch.cuda.empty_cache()  # Free up GPU memory

        return n, p

    def get_bios(self, bios_name, x_dsf=1, y_dsf=1):
        print(f"getting bios vals for {bios_name}")
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        for start, end in self.example_coords:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            # print(f"got imputations for feature #{leave_one_out+1}")
            del n, p  # Free up memory
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        del X, mX, mY, avX, avY  # Free up memoryrm m
        # print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return imp_dist, ups_dist, Y, bios_name, available_indices
    
    def get_frame(self, bios_name, x_dsf=1, y_dsf=1):
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        start, end = 33481539//self.resolution, 33588914//self.resolution
        segment_length = end - start
        adjusted_length = (segment_length // self.context_length) * self.context_length
        adjusted_end = start + adjusted_length

        subsets_X.append(X[start:adjusted_end, :])
        subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            # print(f"got imputations for feature #{leave_one_out+1}")
            del n, p  # Free up memory
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        del X, mX, mY, avX, avY  # Free up memoryrm m
        # print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return imp_dist, ups_dist, Y, bios_name, available_indices

    def get_metric(self, imp_dist, ups_dist, Y, bios_name, availability):
        # print(f"getting metrics")
        imp_mean = imp_dist.expect()
        ups_mean = ups_dist.expect()

        # print(f"got nbinom stuff")
        # imp_lower_95, imp_upper_95 = imp_dist.interval(confidence=0.95)
        # ups_lower_95, ups_upper_95 = ups_dist.interval(confidence=0.95)
        
        results = []
        # for j in availability:  # for each feature i.e. assay
        for j in range(Y.shape[1]):

            if j in list(availability):
                # j = j.item()
                for comparison in ['imputed', 'upsampled']:
                    if comparison == "imputed":
                        pred = imp_mean[:, j].numpy()
                        # lower_95 = imp_lower_95[:, j].numpy()
                        # upper_95 = imp_upper_95[:, j].numpy()
                        
                    elif comparison == "upsampled":
                        pred = ups_mean[:, j].numpy()
                        # lower_95 = ups_lower_95[:, j].numpy()
                        # upper_95 = ups_upper_95[:, j].numpy()

                    target = Y[:, j].numpy()

                    # Check if the target values fall within the intervals
                    # within_interval = (target >= lower_95) & (target <= upper_95)
                    
                    # Calculate the fraction
                    # print(
                    #     f"adding {bios_name} | {self.mark_dict[f'M{str(j+1).zfill(len(str(len(self.mark_dict))))}']} | {comparison}")
                    metrics = {
                        'bios':bios_name,
                        'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                        'comparison': comparison,
                        'available assays': len(availability),

                        'MSE': self.metrics.mse(target, pred),
                        'Pearson': self.metrics.pearson(target, pred),
                        'Spearman': self.metrics.spearman(target, pred),
                        'r2': self.metrics.r2(target, pred)
                    }
                    results.append(metrics)

        return results
    
    def get_validation(self, model, x_dsf=1, y_dsf=1):
        
        t0 = datetime.datetime.now()
        self.model = model
        # self.model.eval()
        full_res = []
        bioses = list(self.dataset.navigation.keys())
        # bioses = [list(self.dataset.navigation.keys())[0]]

        for bios_name in bioses:
            try:
                imp_dist, ups_dist, Y, _, available_indices = self.get_bios(bios_name, x_dsf=x_dsf, y_dsf=y_dsf)
                full_res += self.get_metric(imp_dist, ups_dist, Y, bios_name, available_indices)
                del imp_dist, ups_dist, Y
            except:
                pass
        del self.model
        
        df = pd.DataFrame(full_res)

        # Separate the data based on comparison type
        imputed_df = df[df['comparison'] == 'imputed']
        upsampled_df = df[df['comparison'] == 'upsampled']

        # Function to calculate mean, min, and max for a given metric
        def calculate_stats(df, metric):
            return df[metric].mean(), df[metric].min(), df[metric].max()

        # Imputed statistics
        imp_mse_stats = calculate_stats(imputed_df, 'MSE')
        imp_pearson_stats = calculate_stats(imputed_df, 'Pearson')
        imp_spearman_stats = calculate_stats(imputed_df, 'Spearman')
        imp_r2_stats = calculate_stats(imputed_df, 'r2')
        # imp_frac95conf_stats = calculate_stats(imputed_df, 'frac_95_confidence')

        # Upsampled statistics
        ups_mse_stats = calculate_stats(upsampled_df, 'MSE')
        ups_pearson_stats = calculate_stats(upsampled_df, 'Pearson')
        ups_spearman_stats = calculate_stats(upsampled_df, 'Spearman')
        ups_r2_stats = calculate_stats(upsampled_df, 'r2')
        # ups_frac95conf_stats = calculate_stats(upsampled_df, 'frac_95_confidence')

        elapsed_time = datetime.datetime.now() - t0
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the compact print statement
        print_statement = f"""
        Took {int(minutes)}:{int(seconds)}
        For Imputed:
        - MSE: mean={imp_mse_stats[0]:.2f}, min={imp_mse_stats[1]:.2f}, max={imp_mse_stats[2]:.2f}
        - PCC: mean={imp_pearson_stats[0]:.2f}, min={imp_pearson_stats[1]:.2f}, max={imp_pearson_stats[2]:.2f}
        - SRCC: mean={imp_spearman_stats[0]:.2f}, min={imp_spearman_stats[1]:.2f}, max={imp_spearman_stats[2]:.2f}
        - R2: mean={imp_r2_stats[0]:.2f}, min={imp_r2_stats[1]:.2f}, max={imp_r2_stats[2]:.2f}

        For Upsampled:
        - MSE: mean={ups_mse_stats[0]:.2f}, min={ups_mse_stats[1]:.2f}, max={ups_mse_stats[2]:.2f}
        - PCC: mean={ups_pearson_stats[0]:.2f}, min={ups_pearson_stats[1]:.2f}, max={ups_pearson_stats[2]:.2f}
        - SRCC: mean={ups_spearman_stats[0]:.2f}, min={ups_spearman_stats[1]:.2f}, max={ups_spearman_stats[2]:.2f}
        - R2: mean={ups_r2_stats[0]:.2f}, min={ups_r2_stats[1]:.2f}, max={ups_r2_stats[2]:.2f}
        """

        return print_statement

    def generate_training_gif_frame(self, model, fig_title):
        def gen_subplt(
            ax, x_values, observed_values, 
            ups11, ups21, ups41, 
            imp11, imp21, imp41, 
            col, assname, ytick_fontsize=6, title_fontsize=6):

            # Define the data and labels
            data = [
                (observed_values, "Observed", "royalblue", f"{assname}_Observed"),
                (ups11, "Upsampled 1->1", "darkcyan", f"{assname}_Ups1->1"),
                (imp11, "Imputed 1->1", "salmon", f"{assname}_Imp1->1"),
                (ups21, "Upsampled 2->1", "darkcyan", f"{assname}_Ups2->1"),
                (imp21, "Imputed 2->1", "salmon", f"{assname}_Imp2->1"),
                (ups41, "Upsampled 4->1", "darkcyan", f"{assname}_Ups4->1"),
                (imp41, "Imputed 4->1", "salmon", f"{assname}_Imp4->1"),
            ]
            
            for i, (values, label, color, title) in enumerate(data):
                ax[i, col].plot(x_values, values, "--" if i != 0 else "-", color=color, alpha=0.5, label=label, linewidth=0.1)
                ax[i, col].fill_between(x_values, 0, values, color=color, alpha=0.5)
                
                if i != len(data)-1:
                    ax[i, col].tick_params(axis='x', labelbottom=False)
                
                ax[i, col].tick_params(axis='y', labelsize=ytick_fontsize)
                ax[i, col].set_xticklabels([])
                ax[i, col].set_title(title, fontsize=title_fontsize)

        self.model = model

        bios = list(self.dataset.navigation.keys())[0]
        # print(bios)

        # dsf4-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=4, y_dsf=1)
        imp_mean41, ups_mean41 = imp_dist.expect(), ups_dist.expect()

        # dsf2-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=2, y_dsf=1)
        imp_mean21, ups_mean21 = imp_dist.expect(), ups_dist.expect()

        # dsf1-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=1, y_dsf=1)
        imp_mean11, ups_mean11 = imp_dist.expect(), ups_dist.expect()

        del self.model

        selected_assays = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K9me3", "CTCF", "DNase-seq", "ATAC-seq"]
        fig, axes = plt.subplots(7, len(available_indices), figsize=(len(selected_assays) * 3, 6), sharex=True, sharey=False)
        
        for col, jj in enumerate(available_indices):
            j = jj.item()
            assay = self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"]
            x_values = list(range(len(Y[:, j])))

            if assay in selected_assays:
                obs = Y[:, j].numpy()

                gen_subplt(axes, x_values, 
                        obs, 
                        ups_mean11[:, j].numpy(), ups_mean21[:, j].numpy(), ups_mean41[:, j].numpy(), 
                        imp_mean11[:, j].numpy(), imp_mean21[:, j].numpy(), imp_mean41[:, j].numpy(), 
                        col, assay)

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
    def __init__(self, mask_value, mask_percentage, chunk_size=6):
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

class PROCESS_EIC_DATA(object):
    def __init__(self, path, max_len=8000, resolution=25, stratified=False):
        self.path = path
        self.stratified = stratified
        self.resolution = resolution
        self.max_len = max_len * self.resolution #converts max_len from #bins to #bp
        self.util = COORD(resolution=self.resolution, Meuleman_file="_", outdir=self.path)
        self.genomesize = sum(list(self.util.chr_sizes.values()))
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in self.biosamples.keys():
                    self.biosamples[f[:3]] = {}
                    
                self.biosamples[f[:3]][f[3:6]] = pyBigWig.open(self.path + "/" + f)

    def pkl_generate_m_samples(self, m, multi_p=True, n_p=20): # m per biosample           
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            for _ in range(m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len


                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))

                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region in the same chromosome
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

    def generate_m_samples(self, m, n_datasets=50, multi_p=True, n_p=10):
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            while len(m_regions) < (m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                mii = 0
                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 
                        break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))
                mii = 0

                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        ds_number = 0  
        print("m2:   ", len(m_regions))
        samples_per_ds = len(m_regions) // n_datasets
        for ds_i in range(0, len(m_regions), samples_per_ds):
            ds_number += 1

            ds_i_regions = m_regions[ds_i : (ds_i + samples_per_ds)]
            ds_i_regions.sort(key=lambda x: x[1]) # sorted based on start coord
            
            all_samples_tensor = []

            for bios in self.biosamples.keys():
                print("     ct:   ", bios)
                bios_data = {}

                for assay in self.all_assays:
                    bios_data[assay] = []

                    if assay in self.biosamples[bios].keys(): # if available
                        print("         assay:   ", assay)
                        bw = self.biosamples[bios][assay]
                        bw_query_dicts = []

                        for r in ds_i_regions:
                            bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})
                        
                        if multi_p:
                            with mp.Pool(n_p) as p:
                                outs = p.map(get_bin_value_dict, bw_query_dicts)
                        else:
                            outs = []
                            for ii in range(len(bw_query_dicts)):
                                outs.append(get_bin_value_dict(bw_query_dicts[ii]))

                        outs.sort(key=lambda x: x['start']) # assert is sorted based on start coord
                        m_signals = [o["signals"] for o in outs]
                        
                        for sample in m_signals:
                            bios_data[assay].append(sample)

                    else: # if missing
                        for r in ds_i_regions:
                            bios_data[assay].append([-1 for _ in range(self.max_len // self.resolution)])
                
                # Convert bios_data to a numpy array
                bios_data_array = np.array([bios_data[assay] for assay in self.all_assays], dtype=np.float32)

                # Add bios_data_array to all_samples
                all_samples_tensor.append(bios_data_array)

            # Convert all_samples to a numpy array
            all_samples_tensor = np.array(all_samples_tensor)

            # Convert all_samples_array to a PyTorch tensor
            all_samples_tensor = torch.from_numpy(all_samples_tensor)

            # Ensure the tensor is of type float
            all_samples_tensor = all_samples_tensor.float()

            all_samples_tensor = torch.permute(all_samples_tensor, (2, 0, 3, 1))
            # Get the shape of the current tensor
            shape = all_samples_tensor.shape

            # Calculate the new dimensions
            new_shape = [shape[0]*shape[1]] + list(shape[2:])

            # Reshape the tensor
            all_samples_tensor = all_samples_tensor.reshape(new_shape)
            
            file_path = f"{self.path}/mixed_dataset{ds_number}_{m//n_datasets}samples_{self.resolution}bp.pt"
            torch.save(all_samples_tensor, file_path)
            print(f"saved DS # {ds_number}, with shape {all_samples_tensor.shape}")

    def load_m_regions(self, file_path):
        # Open the gzip file
        with gzip.open(file_path, 'rb') as f:
            # Load the data using pickle
            bios_data = pickle.load(f)

        # Initialize an empty list to store the m_regions
        m_regions = []

        # Iterate over each biosample and assay
        for sample in bios_data[list(bios_data.keys())[0]]:

            # Append the regions to the m_regions list
            if sample[0] not in m_regions:
                m_regions.append(sample[0])
            
        return m_regions
    
    def generate_m_samples_from_predefined_regions(self, m_regions, multi_p=True, n_p=100):
        m = len(m_regions)
        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")
                
    def generate_wg_samples(self):
        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in biosamples[bios].keys():
                bios_data[assay] = {}

                bw = biosamples[bios][assay]
                for chr, size in self.util.chr_sizes.items():
                    signals = get_bin_value(bw, chr, 0, size, self.resolution)
                    bios_data[assay][chr] = signals
            
            file_path = f"{self.path}/{bios}_WG_25bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

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