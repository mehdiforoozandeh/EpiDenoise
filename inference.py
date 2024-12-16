import random
import torch
import pickle
import os, time, gc, psutil
from CANDI import *
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.decomposition import PCA
import umap, scipy
from difflib import SequenceMatcher

################################################################################

def viz_feature_importance(df, savedir="models/output/"):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    def plot_metric_heatmap(df, metric, title):
        # Calculate mean and standard deviation
        mean_pivot = df.pivot_table(values=metric, index='input', columns='output', aggfunc='mean')
        std_pivot = df.pivot_table(values=metric, index='input', columns='output', aggfunc='std')
        
        plt.figure(figsize=(12, 8))
        
        # Determine the colorbar limits based on the metric
        if "Pearson" in metric or "Spearman" in metric: 
            vmin, vmax = -1, 1
        elif "PP" in metric:
            vmin, vmax = 0, np.ceil(mean_pivot.max().max())
        else:
            vmin, vmax = None, None
        
        # Create heatmap using means for colors
        sns.heatmap(mean_pivot, annot=False, cmap='coolwarm', vmin=vmin, vmax=vmax)
        
        # Add annotations with both mean and std
        for i in range(mean_pivot.shape[0]):
            for j in range(mean_pivot.shape[1]):
                mean_val = mean_pivot.iloc[i, j]
                std_val = std_pivot.iloc[i, j]
                if not np.isnan(mean_val):  # Check if the value exists
                    plt.text(j + 0.5, i + 0.5, f'{mean_val:.2f}\n±{std_val:.2f}',
                            ha='center', va='center',
                            color='white' if mean_val > mean_pivot.mean().mean() else 'black')
        
        plt.title(f'{title} - {metric}\n(mean ± std)')
        plt.tight_layout()
        plt.savefig(f'{savedir}/heatmap_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metric_correlations(df):
        metrics = [
            'PP_pval', 'PP_count', 'Pearson_pval', 
            'Pearson_count', 'Spearman_pval', 'Spearman_count']
        sns.pairplot(df[metrics], diag_kind='kde')
        plt.savefig(f'{savedir}/metric_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

    metrics_to_plot = [
        'Pearson_count', 'Pearson_pval', 
        'PP_count', 'PP_pval', 
        'Spearman_count', 'Spearman_pval'
        ]
        
    for metric in metrics_to_plot:
        plot_metric_heatmap(df, metric, 'Assay Prediction Performance')
    
    plot_metric_correlations(df)

################################################################################

def perplexity(probabilities):
    N = len(probabilities)
    epsilon = 1e-6  # Small constant to prevent log(0)
    log_prob_sum = torch.sum(torch.log(probabilities + epsilon))
    perplexity = torch.exp(-log_prob_sum / N)
    return perplexity

def fraction_within_ci(dist, x, c=0.95):
    lower, upper = dist.interval(c)

    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(lower):
        lower = lower.numpy()

    if torch.is_tensor(upper):
        upper = upper.numpy()

    if torch.is_tensor(x):
        x = x.numpy()

    # Add small epsilon to avoid divide by zero
    # eps = np.finfo(float).eps
    # x = np.asarray(x) + eps
    return np.mean((x >= lower) & (x <= upper))

def confidence_calibration(dist, true, n_bins=20):
    # Generate confidence levels from 0 to 1
    confidence_levels = np.linspace(0, 1, n_bins)
    
    # Calculate empirical coverage for each confidence level
    calibration = []
    for c in confidence_levels:
        empirical = fraction_within_ci(dist, true, c)
        calibration.append([c, empirical])
    
    return calibration

def plot_calibration_grid(calibrations, titles, figsize=(12, 12)):
    """
    Visualize 4 calibration curves in a 2x2 grid.
    
    Parameters:
    - calibrations: list of 4 calibration outputs, where each calibration output
                   is a list of [c, empirical] pairs
    - titles: list of 4 strings for subplot titles
    - figsize: tuple specifying figure size (width, height)
    
    Returns:
    - fig: matplotlib figure object
    """
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Reference line points (perfect calibration)
    ref_line = np.linspace(0, 1, 100)
    
    # Plot each calibration curve
    for idx, (cal, title, ax) in enumerate(zip(calibrations, titles, axes)):
        # Convert calibration data to numpy arrays for easier plotting
        cal_array = np.array(cal)
        c_values = cal_array[:, 0]
        empirical_values = cal_array[:, 1]
        
        # Plot reference line (perfect calibration)
        ax.plot(ref_line, ref_line, '--', color='orange', alpha=0.8, label='Perfect calibration')
        
        # Plot empirical calibration
        ax.plot(c_values, empirical_values, '-', color='grey', linewidth=2, label='Empirical calibration')
        
        # Customize plot
        ax.set_xlabel('c')
        ax.set_ylabel('Fraction within c% confidence interval')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add legend
        if idx == 0:  # Only add legend to first subplot
            ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

################################################################################

class CANDIPredictor:
    def __init__(self, model, hyper_parameters_path, 
        split="test", DNA=False, eic=True, chr="chr21", resolution=25, context_length=1600,
        savedir="models/output/", data_path="/project/compbio-lab/encode_data/"):

        self.model = model
        self.chr = chr
        self.resolution = resolution
        self.savedir = savedir
        self.DNA = DNA
        self.context_length = context_length
        self.data_path = data_path
        self.eic = eic
        self.split = split

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(
            self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5, eic=eic, merge_ct=True)

        if isinstance(self.model, str):
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
                self.hyper_parameters["signal_dim"] = self.dataset.signal_dim
                self.hyper_parameters["metadata_embedding_dim"] = self.dataset.signal_dim
                self.context_length = self.hyper_parameters["context_length"]

            loader = CANDI_LOADER(model, self.hyper_parameters, DNA=self.DNA)
            self.model = loader.load_CANDI()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.chr_sizes = {}
        with open("data/hg38.chrom.sizes", 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                # if chr_name == self.chr:
                self.chr_sizes[chr_name] = int(chr_size)
                    # break

        self.context_length = self.hyper_parameters["context_length"]
        self.batch_size = self.hyper_parameters["batch_size"]
        self.token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
            }

    def load_encoder_input_bios(self, bios_name, x_dsf=1, chr=None, y_dsf=1):
        print("loading encoder inputs for biosample: ", bios_name)
        if chr == None:
            chr = self.chr

        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [chr, 0, self.chr_sizes[chr]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [chr, 0, self.chr_sizes[chr]], x_dsf)

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, [chr, 0, self.chr_sizes[chr]], x_dsf)

        # print(temp_x.keys(), temp_mx.keys())
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        if self.DNA:
            seq = dna_to_onehot(get_DNA_sequence(self.chr, 0, self.chr_sizes[self.chr]))

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X = X[:num_rows, :]

        if self.DNA:
            seq = seq[:num_rows*self.resolution, :]
            
        X = X.view(-1, self.context_length, X.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX= mX.expand(X.shape[0], -1, -1)
        avX = avX.expand(X.shape[0], -1)

        if self.DNA:
            return X, seq, mX
        else:
            return X, mX

    def load_bios(self, bios_name, x_dsf, y_dsf=1, fill_in_y_prompt=False, chr=None, start=None, end=None):
        # Load biosample data
        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            
            # print(temp_x.keys(), temp_mx.keys())
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_py = self.dataset.load_bios_BW(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            if self.split == "test":
                temp_px = self.dataset.load_bios_BW(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            elif self.split == "val":
                temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)

            temp_p = {**temp_py, **temp_px}
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            del temp_py, temp_px, temp_p

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_p = self.dataset.load_bios_BW(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            assert (avlP == avY).all(), "avlP and avY do not match"
            del temp_p

        if self.DNA:
            seq = dna_to_onehot(get_DNA_sequence(self.chr, 0, self.chr_sizes[self.chr]))

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

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None):
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        mu = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        var = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        Z = torch.empty((X.shape[0], self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)

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
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch, return_z=True)
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch, return_z=True)

            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()
            mu[i:i+outputs_mu.shape[0], :, :] = outputs_mu.cpu()
            var[i:i+outputs_var.shape[0], :, :] = outputs_var.cpu()
            Z[i:i+latent.shape[0], :, :] = latent.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, latent
            torch.cuda.empty_cache()
            
        n = n.view(n.shape[0] * n.shape[1], n.shape[-1])
        p = p.view(p.shape[0] * p.shape[1], p.shape[-1])
        mu = mu.view(mu.shape[0] * mu.shape[1], mu.shape[-1])
        var = var.view(var.shape[0] * var.shape[1], var.shape[-1])
        Z = Z.view(Z.shape[0] * Z.shape[1], Z.shape[-1])
        return n, p, mu, var, Z

    def get_latent_representations_cropped(self, X, mX, imp_target=[], seq=None, crop_percent=0.1):
        # Calculate dimensions
        crop_size = int(self.context_length * crop_percent)
        stride = self.context_length - (crop_size * 2)
        num_windows = X.shape[0]
        total_length = num_windows * self.context_length

        Z_crop_size = int(crop_size * (self.model.l2 / self.model.l1))
        
        # Flatten input tensors
        X_flat = X.view(-1, X.shape[-1])
        if self.DNA:
            seq_flat = seq.view(-1, seq.shape[-1])
        
        # Initialize output tensors
        Z = torch.zeros((num_windows * self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)
        z_coverage_mask = torch.zeros(num_windows * self.model.l2, dtype=torch.bool, device="cpu")  # New mask for Z
        
        # Collect all windows and their metadata
        window_data = []
        target_regions = []
        
        # Process sliding windows
        for i in range(0, total_length, stride):
            if i + self.context_length >= total_length:
                i = total_length - self.context_length
                
            window_end = i + self.context_length
            x_window = X_flat[i:window_end].unsqueeze(0)
            
            # Use first row of metadata tensors (verified identical)
            mx_window = mX[0].unsqueeze(0)
            
            if self.DNA:
                seq_start = i * self.resolution
                seq_end = window_end * self.resolution
                seq_window = seq_flat[seq_start:seq_end].unsqueeze(0)
            
            # Determine prediction regions
            if i == 0:  # First window
                start_idx = 0
                end_idx = self.context_length - crop_size
            elif i + self.context_length >= total_length:  # Last window
                start_idx = crop_size
                end_idx = self.context_length
            else:  # Middle windows
                start_idx = crop_size
                end_idx = self.context_length - crop_size
                
            target_start = i + start_idx
            target_end = i + end_idx
            
            # Store window data and target regions
            window_info = {
                'x': x_window,
                'mx': mx_window,
                'seq': seq_window if self.DNA else None
            }
            target_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'target_start': target_start,
                'target_end': target_end
            }
            
            window_data.append(window_info)
            target_regions.append(target_info)
        
        # Process windows in batches
        for i in range(0, len(window_data), self.batch_size):
            batch_windows = window_data[i:i + self.batch_size]
            batch_targets = target_regions[i:i + self.batch_size]
            
            # Prepare batch tensors
            x_batch = torch.cat([w['x'] for w in batch_windows])
            mx_batch = torch.cat([w['mx'] for w in batch_windows])
            
            if self.DNA:
                seq_batch = torch.cat([w['seq'] for w in batch_windows])
            
            # Apply imp_target if specified
            if len(imp_target) > 0:
                x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                mx_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
            
            # Get predictions
            with torch.no_grad():
                if self.DNA:
                    outputs = self.model.encode(
                        x_batch.float().to(self.device),
                        seq_batch.to(self.device),
                        mx_batch.to(self.device)
                    )
                else:
                    outputs = self.model.encode(
                        x_batch.float().to(self.device),
                        mx_batch.to(self.device)
                    )
                
                outputs_Z = outputs
            
            # Update predictions for each window in batch
            for j, (out_Z, target) in enumerate(zip(outputs_Z, batch_targets)):
                start_idx = target['start_idx']
                end_idx = target['end_idx']
                target_start = target['target_start']
                target_end = target['target_end']

                i = target_start - start_idx

                i_z = i * (self.model.l2 / self.model.l1)
                if start_idx == 0:
                    start_z_idx = 0
                elif start_idx == crop_size:
                    start_z_idx = Z_crop_size

                if end_idx == self.context_length - crop_size:
                    end_z_idx = self.model.l2 - Z_crop_size
                elif end_idx == self.context_length:
                    end_z_idx = self.model.l2

                target_z_start = int(i_z + start_z_idx)
                target_z_end = int(i_z + end_z_idx)
                
                Z[target_z_start:target_z_end, :] = out_Z[start_z_idx:end_z_idx, :].cpu()
                
                z_coverage_mask[target_z_start:target_z_end] = True  # Track Z coverage
            
            del outputs
            torch.cuda.empty_cache()

            
        if not z_coverage_mask.all():
            print(f"Missing Z predictions for positions: {torch.where(~z_coverage_mask)[0]}")
            raise ValueError("Missing Z predictions")
        
        return Z

    def pred_cropped(self, X, mX, mY, avail, imp_target=[], seq=None, crop_percent=0.1):
        # Calculate dimensions
        crop_size = int(self.context_length * crop_percent)
        stride = self.context_length - (crop_size * 2)
        num_windows = X.shape[0]
        total_length = num_windows * self.context_length

        Z_crop_size = int(crop_size * (self.model.l2 / self.model.l1))
        
        # Flatten input tensors
        X_flat = X.view(-1, X.shape[-1])
        if self.DNA:
            seq_flat = seq.view(-1, seq.shape[-1])
        
        # Initialize output tensors
        n = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        p = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        mu = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        var = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        Z = torch.zeros((num_windows * self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)
        coverage_mask = torch.zeros(total_length, dtype=torch.bool, device="cpu")
        z_coverage_mask = torch.zeros(num_windows * self.model.l2, dtype=torch.bool, device="cpu")  # New mask for Z
        
        # Collect all windows and their metadata
        window_data = []
        target_regions = []
        
        # Process sliding windows
        for i in range(0, total_length, stride):
            if i + self.context_length >= total_length:
                i = total_length - self.context_length
                
            window_end = i + self.context_length
            x_window = X_flat[i:window_end].unsqueeze(0)
            
            # Use first row of metadata tensors (verified identical)
            mx_window = mX[0].unsqueeze(0)
            my_window = mY[0].unsqueeze(0)
            avail_window = avail[0].unsqueeze(0)
            
            if self.DNA:
                seq_start = i * self.resolution
                seq_end = window_end * self.resolution
                seq_window = seq_flat[seq_start:seq_end].unsqueeze(0)
            
            # Determine prediction regions
            if i == 0:  # First window
                start_idx = 0
                end_idx = self.context_length - crop_size
            elif i + self.context_length >= total_length:  # Last window
                start_idx = crop_size
                end_idx = self.context_length
            else:  # Middle windows
                start_idx = crop_size
                end_idx = self.context_length - crop_size
                
            target_start = i + start_idx
            target_end = i + end_idx
            
            # Store window data and target regions
            window_info = {
                'x': x_window,
                'mx': mx_window,
                'my': my_window,
                'avail': avail_window,
                'seq': seq_window if self.DNA else None
            }
            target_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'target_start': target_start,
                'target_end': target_end
            }
            
            window_data.append(window_info)
            target_regions.append(target_info)
        
        # Process windows in batches
        for i in range(0, len(window_data), self.batch_size):
            batch_windows = window_data[i:i + self.batch_size]
            batch_targets = target_regions[i:i + self.batch_size]
            
            # Prepare batch tensors
            x_batch = torch.cat([w['x'] for w in batch_windows])
            mx_batch = torch.cat([w['mx'] for w in batch_windows])
            my_batch = torch.cat([w['my'] for w in batch_windows])
            avail_batch = torch.cat([w['avail'] for w in batch_windows])
            
            if self.DNA:
                seq_batch = torch.cat([w['seq'] for w in batch_windows])
            
            # Apply imp_target if specified
            if len(imp_target) > 0:
                x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                mx_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                avail_batch[:, imp_target] = 0
            
            # Get predictions
            with torch.no_grad():
                if self.DNA:
                    outputs = self.model(
                        x_batch.float().to(self.device),
                        seq_batch.to(self.device),
                        mx_batch.to(self.device),
                        my_batch.to(self.device),
                        avail_batch.to(self.device),
                        return_z=True
                    )
                else:
                    outputs = self.model(
                        x_batch.float().to(self.device),
                        mx_batch.to(self.device),
                        my_batch.to(self.device),
                        avail_batch.to(self.device),
                        return_z=True
                    )
                
                outputs_p, outputs_n, outputs_mu, outputs_var, outputs_Z = outputs
            
            # Update predictions for each window in batch
            for j, (window_pred, target) in enumerate(zip(zip(outputs_n, outputs_p, outputs_mu, outputs_var, outputs_Z), batch_targets)):
                out_n, out_p, out_mu, out_var, out_Z = window_pred
                start_idx = target['start_idx']
                end_idx = target['end_idx']
                target_start = target['target_start']
                target_end = target['target_end']

                i = target_start - start_idx

                i_z = i * (self.model.l2 / self.model.l1)
                if start_idx == 0:
                    start_z_idx = 0
                elif start_idx == crop_size:
                    start_z_idx = Z_crop_size

                if end_idx == self.context_length - crop_size:
                    end_z_idx = self.model.l2 - Z_crop_size
                elif end_idx == self.context_length:
                    end_z_idx = self.model.l2

                target_z_start = int(i_z + start_z_idx)
                target_z_end = int(i_z + end_z_idx)
                
                n[target_start:target_end, :] = out_n[start_idx:end_idx, :].cpu()
                p[target_start:target_end, :] = out_p[start_idx:end_idx, :].cpu()
                mu[target_start:target_end, :] = out_mu[start_idx:end_idx, :].cpu()
                var[target_start:target_end, :] = out_var[start_idx:end_idx, :].cpu()
                Z[target_z_start:target_z_end, :] = out_Z[start_z_idx:end_z_idx, :].cpu()
                
                coverage_mask[target_start:target_end] = True
                z_coverage_mask[target_z_start:target_z_end] = True  # Track Z coverage
            
            del outputs
            torch.cuda.empty_cache()
        
        # Verify complete coverage for both signal and Z
        if not coverage_mask.all():
            print(f"Missing predictions for positions: {torch.where(~coverage_mask)[0]}")
            raise ValueError("Missing signal predictions")
            
        if not z_coverage_mask.all():
            print(f"Missing Z predictions for positions: {torch.where(~z_coverage_mask)[0]}")
            raise ValueError("Missing Z predictions")
        
        return n, p, mu, var, Z

    def get_latent_representations(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=None, imp_target=[])
        return Z

    def get_decoded_signal(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            p, n, mu, var, _ = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            p, n, mu, var, _ = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        count_dist = NegativeBinomial(p, n)
        pval_dist = Gaussian(mu, var)
        
        return count_dist.mean(), pval_dist.mean()

    def evaluate_leave_one_out(self, X, mX, mY, avX, Y, P, seq=None, crop_edges=True, return_preds=False):
        """
        Performs leave-one-out evaluation and returns metrics for both count and p-value predictions.
        
        Returns:
            Dictionary containing metrics for each feature:
            {feature_idx: {
                'count_metrics': {
                    'imp_pearson': float, 'imp_spearman': float, 'imp_mse': float, 'imp_r2': float,
                    'ups_pearson': float, 'ups_spearman': float, 'ups_mse': float, 'ups_r2': float,
                    'p0_bg': float, 'p0_fg': float  # probability of zero in background/foreground
                },
                'pval_metrics': {
                    'imp_pearson': float, 'imp_spearman': float, 'imp_mse': float, 'imp_r2': float,
                    'ups_pearson': float, 'ups_spearman': float, 'ups_mse': float, 'ups_r2': float
                }
            }}
        """
        available_indices = torch.where(avX[0, :] == 1)[0]

        expnames = list(self.dataset.aliases["experiment_aliases"].keys())
        
        # Initialize tensors for imputation predictions
        n_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        p_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        mu_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        var_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        
        if crop_edges:
            # Get upsampling predictions 
            n_ups, p_ups, mu_ups, var_ups, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[], seq=seq)
        else:
            # Get upsampling predictions 
            n_ups, p_ups, mu_ups, var_ups, _ = self.pred(X, mX, mY, avX, imp_target=[], seq=seq)
        
        # Perform leave-one-out predictions
        for ii, leave_one_out in enumerate(available_indices):
            if crop_edges:
                n, p, mu, var, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            else:
                n, p, mu, var, _ = self.pred(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            n_imp[:, leave_one_out] = n[:, leave_one_out]
            p_imp[:, leave_one_out] = p[:, leave_one_out]
            mu_imp[:, leave_one_out] = mu[:, leave_one_out]
            var_imp[:, leave_one_out] = var[:, leave_one_out]
            print(f"Completed feature {ii+1}/{len(available_indices)}")
        
        # Create distributions and get means
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])
        
        imp_count_dist = NegativeBinomial(p_imp, n_imp)
        ups_count_dist = NegativeBinomial(p_ups, n_ups)
        imp_count_mean = imp_count_dist.mean()
        ups_count_mean = ups_count_dist.mean()
        
        imp_pval_dist = Gaussian(mu_imp, var_imp)
        ups_pval_dist = Gaussian(mu_ups, var_ups)
        imp_pval_mean = imp_pval_dist.mean()
        ups_pval_mean = ups_pval_dist.mean()

        prob_imp_pval = imp_pval_dist.pdf(P)
        prob_imp_count = imp_count_dist.pmf(Y)
        prob_ups_pval = ups_pval_dist.pdf(P)
        prob_ups_count = ups_count_dist.pmf(Y)
        
        if return_preds:
            return imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist
        
        # Calculate metrics for each feature
        metrics = {}
        for idx in available_indices:
            # true 
            count_true = Y[:, idx].numpy()
            pval_true = P[:, idx].numpy()

            # pred
            imp_count = imp_count_mean[:, idx].numpy()
            ups_count = ups_count_mean[:, idx].numpy()

            # perplexity
            # start_time = time.time()
            imp_pp_pval =  perplexity(prob_imp_pval[:, idx])
            imp_pp_count = perplexity(prob_imp_count[:, idx])
            ups_pp_pval =  perplexity(prob_ups_pval[:, idx])
            ups_pp_count = perplexity(prob_ups_count[:, idx])
            # end_time = time.time()
            # print(f"Perplexity calculations took {end_time - start_time:.4f} seconds")

            # assay distributions
            imp_pval_dist_idx = Gaussian(mu_imp[:, idx], var_imp[:, idx])
            ups_pval_dist_idx = Gaussian(mu_ups[:, idx], var_ups[:, idx])
            imp_count_dist_idx = NegativeBinomial(p_imp[:, idx], n_imp[:, idx])
            ups_count_dist_idx = NegativeBinomial(p_ups[:, idx], n_ups[:, idx])

            # calibration curve
            # imp_pval_calibration = confidence_calibration(imp_pval_dist_idx, pval_true)
            # ups_pval_calibration = confidence_calibration(ups_pval_dist_idx, pval_true)
            # imp_count_calibration = confidence_calibration(imp_count_dist_idx, count_true)
            # ups_count_calibration = confidence_calibration(ups_count_dist_idx, count_true)

            # print(imp_pval_calibration[0], ups_pval_calibration[0], imp_count_calibration[0], ups_count_calibration[0])
            # print(imp_pval_calibration[-1], ups_pval_calibration[-1], imp_count_calibration[-1], ups_count_calibration[-1])

            # fig = plot_calibration_grid(
            #     [imp_pval_calibration, ups_pval_calibration, imp_count_calibration, ups_count_calibration],
            #     ["Imputed signal", "Upsampled signal", "Imputed count", "Upsampled count"])
            # fig.savefig(f"output/calibration_curve_{expnames[idx]}.png")
            # plt.close(fig)
            
            # fraction within 95% CI pval
            # start_time = time.time()
            imp_pval_95ci = fraction_within_ci(imp_pval_dist_idx, pval_true, c=0.95)
            ups_pval_95ci = fraction_within_ci(ups_pval_dist_idx, pval_true, c=0.95)

            # fraction within 95% CI count
            imp_count_95ci = fraction_within_ci(imp_count_dist_idx, count_true, c=0.95)
            ups_count_95ci = fraction_within_ci(ups_count_dist_idx, count_true, c=0.95)
            # end_time = time.time()
            # print(f"95% CI calculations took {end_time - start_time:.4f} seconds")
            
            # P-value (apply sinh transformation)
            imp_pval = np.sinh(imp_pval_mean[:, idx].numpy())
            ups_pval = np.sinh(ups_pval_mean[:, idx].numpy())
            pval_true = np.sinh(pval_true)
            # start_time = time.time()
            metrics[idx.item()] = {
                'count_metrics': {
                    'imp_pearson': stats.pearsonr(count_true, imp_count)[0],
                    'imp_spearman': stats.spearmanr(count_true, imp_count)[0],
                    'imp_mse': np.mean((count_true - imp_count) ** 2),
                    'imp_r2': 1 - (np.sum((count_true - imp_count) ** 2) / 
                                np.sum((count_true - np.mean(count_true)) ** 2)),
                    'imp_perplexity': imp_pp_count.item(),  # Add perplexity
                    'imp_95ci': imp_count_95ci,  # Add 95% CI
                    'ups_pearson': stats.pearsonr(count_true, ups_count)[0],
                    'ups_spearman': stats.spearmanr(count_true, ups_count)[0],
                    'ups_mse': np.mean((count_true - ups_count) ** 2),
                    'ups_r2': 1 - (np.sum((count_true - ups_count) ** 2) / np.sum((count_true - np.mean(count_true)) ** 2)),
                    'ups_perplexity': ups_pp_count.item(),  # Add perplexity
                    'ups_95ci': ups_count_95ci  # Add 95% CI
                },
                'pval_metrics': {
                    'imp_pearson': stats.pearsonr(pval_true, imp_pval)[0],
                    'imp_spearman': stats.spearmanr(pval_true, imp_pval)[0],
                    'imp_mse': np.mean((pval_true - imp_pval) ** 2),
                    'imp_r2': 1 - (np.sum((pval_true - imp_pval) ** 2) / 
                                np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'imp_perplexity': imp_pp_pval.item(),  # Add perplexity
                    'imp_95ci': imp_pval_95ci,  # Add 95% CI
                    'ups_pearson': stats.pearsonr(pval_true, ups_pval)[0],
                    'ups_spearman': stats.spearmanr(pval_true, ups_pval)[0],
                    'ups_mse': np.mean((pval_true - ups_pval) ** 2),
                    'ups_r2': 1 - (np.sum((pval_true - ups_pval) ** 2) / np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'ups_perplexity': ups_pp_pval.item(),  # Add perplexity
                    'ups_95ci': ups_pval_95ci  # Add 95% CI
                }
            }
            # end_time = time.time()
            # print(f"Metrics calculations took {end_time - start_time:.4f} seconds")

        # Print summary
        print("\nEvaluation Results:")
        print("\nCount Metrics:")
        print("Feature | Type      | Pearson | Spearman | MSE    | R2     | PP     | 95% CI")
        print("-" * 75)
        for idx in available_indices:
            m = metrics[idx.item()]['count_metrics']
            print(f"{expnames[idx]:10s} | Imputed   | {m['imp_pearson']:7.4f} | {m['imp_spearman']:8.4f} | "
                f"{m['imp_mse']:6.4f} | {m['imp_r2']:6.4f} | {m['imp_perplexity']:6.4f} | {m['imp_95ci']:6.4f}")
            print(f"{' '*10} | Upsampled | {m['ups_pearson']:7.4f} | {m['ups_spearman']:8.4f} | "
                f"{m['ups_mse']:6.4f} | {m['ups_r2']:6.4f} | {m['ups_perplexity']:6.4f} | {m['ups_95ci']:6.4f}")
            print("-" * 75)

        print("\nP-value Metrics:")
        print("Feature | Type      | Pearson | Spearman | MSE    | R2     | PP     | 95% CI")
        print("-" * 75)
        for idx in available_indices:
            m = metrics[idx.item()]['pval_metrics']
            print(f"{expnames[idx]:10s} | Imputed   | {m['imp_pearson']:7.4f} | {m['imp_spearman']:8.4f} | "
                f"{m['imp_mse']:6.4f} | {m['imp_r2']:6.4f} | {m['imp_perplexity']:6.4f} | {m['imp_95ci']:6.4f}")
            print(f"{' '*10} | Upsampled | {m['ups_pearson']:7.4f} | {m['ups_spearman']:8.4f} | "
                f"{m['ups_mse']:6.4f} | {m['ups_r2']:6.4f} | {m['ups_perplexity']:6.4f} | {m['ups_95ci']:6.4f}")
            print("-" * 75)
        
        return metrics

    def evaluate_leave_one_out_eic(self, X, mX, mY, avX, Y, P, avY, seq=None, crop_edges=True, return_preds=False):

        available_X_indices = torch.where(avX[0, :] == 1)[0]
        available_Y_indices = torch.where(avY[0, :] == 1)[0]

        expnames = list(self.dataset.aliases["experiment_aliases"].keys())

        if crop_edges:
            # Get upsampling predictions 
            n, p, mu, var, Z = self.pred_cropped(X, mX, mY, avX, imp_target=[], seq=seq)
        else:
            # Get upsampling predictions 
            n, p, mu, var, Z = self.pred(X, mX, mY, avX, imp_target=[], seq=seq)

        # Create distributions and get means
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])
        X = X.view(-1, X.shape[-1])

        ups_count_dist = NegativeBinomial(p, n)
        ups_count_mean = ups_count_dist.mean()
        
        ups_pval_dist = Gaussian(mu, var)
        ups_pval_mean = ups_pval_dist.mean()

        prob_ups_pval = ups_pval_dist.pdf(P)
        prob_ups_count = ups_count_dist.pmf(Y)

        metrics = {}
        for j in range(Y.shape[1]):
            pred_count = ups_count_mean[:, j].numpy()
            pred_pval = np.sinh(ups_pval_mean[:, j].numpy())
            pval_true = np.sinh(P[:, j].numpy())

            if j in list(available_X_indices):
                comparison = "upsampled"
                count_true = X[:, j].numpy()

            elif j in list(available_Y_indices):
                comparison = "imputed"
                count_true = Y[:, j].numpy()

            else:
                continue
            
            pp_pval  = perplexity(prob_ups_pval[:, j])
            pp_count = perplexity(prob_ups_count[:, j])

            ups_pval_dist_j = Gaussian(mu[:, j], var[:, j])
            ups_count_dist_j = NegativeBinomial(p[:, j], n[:, j])

            pval_95ci =  fraction_within_ci(ups_pval_dist_j, pval_true, c=0.95)
            count_95ci = fraction_within_ci(ups_count_dist_j, count_true, c=0.95)

            metrics[j] = {
                'comparison': comparison,
                'count_metrics': {
                    'pearson': stats.pearsonr(count_true, pred_count)[0],
                    'spearman': stats.spearmanr(count_true, pred_count)[0],
                    'mse': np.mean((count_true - pred_count) ** 2),
                    'r2': 1 - (np.sum((count_true - pred_count) ** 2) / 
                              np.sum((count_true - np.mean(count_true)) ** 2)),
                    'perplexity': pp_count.item(),  # Add perplexity
                    '95ci': count_95ci  # Add 95% CI
                },
                'pval_metrics': {
                    'pearson': stats.pearsonr(pval_true, pred_pval)[0],
                    'spearman': stats.spearmanr(pval_true, pred_pval)[0],
                    'mse': np.mean((pval_true - pred_pval) ** 2),
                    'r2': 1 - (np.sum((pval_true - pred_pval) ** 2) / 
                              np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'perplexity': pp_pval.item(),  # Add perplexity
                    '95ci': pval_95ci  # Add 95% CI
                }
            }

        # Print summary with updated headers and format
        print("\nEvaluation Results:")
        print("\nCount Metrics:")
        print("Feature | Type      | Pearson | Spearman | MSE    | R2     | PP     | 95% CI")
        print("-" * 75)
        
        for idx, m in metrics.items():
            feature_name = expnames[idx]
            comp_type = m['comparison']
            count_m = m['count_metrics']
            print(f"{feature_name:10s} | {comp_type:9s} | {count_m['pearson']:7.4f} | {count_m['spearman']:8.4f} | "
                  f"{count_m['mse']:6.4f} | {count_m['r2']:6.4f} | {count_m['perplexity']:6.4f} | {count_m['95ci']:6.4f}")
        
        print("\nP-value Metrics:")
        print("Feature | Type      | Pearson | Spearman | MSE    | R2     | PP     | 95% CI")
        print("-" * 75)
        
        for idx, m in metrics.items():
            feature_name = expnames[idx]
            comp_type = m['comparison']
            pval_m = m['pval_metrics']
            print(f"{feature_name:10s} | {comp_type:9s} | {pval_m['pearson']:7.4f} | {pval_m['spearman']:8.4f} | "
                  f"{pval_m['mse']:6.4f} | {pval_m['r2']:6.4f} | {pval_m['perplexity']:6.4f} | {pval_m['95ci']:6.4f}")

    def evaluate(self, bios_name):
        X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf=1)
        if self.eic:
            metrics = self.evaluate_leave_one_out_eic(X, mX, mY, avX, Y, P, avY, seq=seq, crop_edges=True, return_preds=False)
        else:
            metrics = self.evaluate_leave_one_out(X, mX, mY, avX, Y, P, seq=seq, crop_edges=True, return_preds=False)
        return metrics

################################################################################

def latent_reproducibility(
    model_path, hyper_parameters_path, 
    repr1_bios, repr2_bios, 
    chr="chr21", dataset_path="/project/compbio-lab/encode_data/"):

    candi = CANDIPredictor(model_path, hyper_parameters_path, data_path=dataset_path, DNA=True, eic=True)
    candi.chr = chr

    # Load latent representations
    X1, seq1, mX1 = candi.load_encoder_input_bios(repr1_bios)
    X2, seq2, mX2 = candi.load_encoder_input_bios(repr2_bios)

    latent_repr1 = candi.get_latent_representations_cropped(X1, mX1, seq=seq1)
    latent_repr2 = candi.get_latent_representations_cropped(X2, mX2, seq=seq2)

    del X1, X2, seq1, seq2, mX1, mX2
    
    # Assume latent_repr1 and latent_repr2 are tensors of shape (L, d)
    assert latent_repr1.shape == latent_repr2.shape, "latent_repr1 and latent_repr2 must have the same shape"

    # Convert cosine similarity to cosine distance
    cosine_distances = 1 - F.cosine_similarity(latent_repr1, latent_repr2, dim=-1)  # Shape: (L,)
    euclidean_distances = torch.sqrt(torch.sum((latent_repr1 - latent_repr2)**2, dim=1))

    # Scale cosine distances by 1/2
    cosine_distances_scaled = cosine_distances / 2

    # Calculate summary statistics
    stats = {
        'euclidean': {
            'mean': euclidean_distances.mean().item(),
            'std': euclidean_distances.std().item(),
            'median': euclidean_distances.median().item(),
            'min': euclidean_distances.min().item(),
            'max': euclidean_distances.max().item()
        },
        'cosine': {
            'mean': cosine_distances_scaled.mean().item(),
            'std': cosine_distances_scaled.std().item(),
            'median': cosine_distances_scaled.median().item(),
            'min': cosine_distances_scaled.min().item(),
            'max': cosine_distances_scaled.max().item()
        }
    }
    
    # Function to plot CDF and calculate AUC
    def plot_cdf(ax, data, title, xlabel, color='blue', is_cosine=False):
        sorted_data = np.sort(data.cpu().numpy())
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Extend the CDF to cover the entire x-axis range
        sorted_data = np.concatenate([[0], sorted_data, [1 if is_cosine else sorted_data[-1]]])
        cumulative = np.concatenate([[0], cumulative, [1]])
        
        # Plot CDF
        ax.plot(sorted_data, cumulative, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Fraction of bins')
        ax.grid(True, alpha=0.3)
        
        # Calculate AUC using trapezoidal rule
        auc = integrate.trapz(cumulative, sorted_data)
        
        # Annotate AUC on the plot
        if is_cosine:
            ax.text(0.8, 0.8, f"AUC: {auc:.4f}", transform=ax.transAxes, color=color, fontsize=10)
        
        return auc
    
    fig = plt.figure(figsize=(12, 8))
    
    # Plot CDFs (top row)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    
    # Plot for cosine distance (scaled)
    stats['cosine']['auc'] = plot_cdf(
        ax1, 
        cosine_distances_scaled,
        'Cosine Distance CDF',
        'Cosine Distance / 2',
        'blue',
        is_cosine=True
    )
    
    # Plot for euclidean distance
    stats['euclidean']['auc'] = plot_cdf(
        ax2,
        euclidean_distances,
        'Euclidean Distance CDF',
        'Euclidean Distance',
        'green',
        is_cosine=False
    )
    
    # Compute PCA
    pca = PCA(n_components=2)
    pca1 = pca.fit_transform(latent_repr1.cpu().numpy())
    pca2 = pca.fit_transform(latent_repr2.cpu().numpy())
    
    # Plot PCA (bottom row)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # PCA plots
    ax3.scatter(pca1[:, 0], pca1[:, 1], alpha=0.5, s=1)
    ax3.set_title(f'PCA of {repr1_bios}')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    
    ax4.scatter(pca2[:, 0], pca2[:, 1], alpha=0.5, s=1)
    ax4.set_title(f'PCA of {repr2_bios}')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'latent_space_comparison_{repr1_bios}_{repr2_bios}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Latent space comparison statistics:")
    print(f"Mean cosine distance: {stats['cosine']['mean']:.4f}")
    print(f"Mean euclidean distance: {stats['euclidean']['mean']:.4f}")
    print(f"AUC cosine: {stats['cosine']['auc']:.4f}")
    return stats, euclidean_distances, cosine_distances_scaled

class ChromatinStateProbe(nn.Module):
    def __init__(self, input_dim, output_dim=18):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.class_to_index = None  # Placeholder for the class-to-index mapping
    
    def forward(self, x, normalize=False):
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)  # Apply log_softmax to get log probabilities
        return x

    def encode_class_indices(self, class_names):
        """
        Convert a list of class names to class indices.
        """
        if self.class_to_index is None:
            unique_classes = sorted(set(class_names))
            self.class_to_index = {name: idx for idx, name in enumerate(unique_classes)}
            self.index_to_class = {idx: name for name, idx in self.class_to_index.items()}
        return [self.class_to_index[name] for name in class_names]

    def decode_class_indices(self, class_indices):
        """
        Convert class indices back to class names.
        """
        if self.class_to_index is None:
            raise ValueError("class_to_index mapping is not defined.")
        return [self.index_to_class[idx] for idx in class_indices]

    def train_batch(self, X, y, optimizer, criterion):
        optimizer.zero_grad()
        output = self(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate(self, X, y):
        self.eval()
        with torch.no_grad():
            output = self(X)
            criterion = nn.NLLLoss()
            val_loss = criterion(output, y)
            
            # Convert log probabilities to probabilities
            probabilities = torch.exp(output)
            
            # Get predicted classes
            _, predicted = torch.max(probabilities, 1)
            
            # Convert tensors to numpy arrays
            y_true = y.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            # Get class names
            class_names = [self.index_to_class[idx] for idx in range(len(self.index_to_class))]
            
            # Compute classification report
            report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
            
            # Calculate overall accuracy
            total = y.size(0)
            correct = (predicted == y).sum().item()
            overall_accuracy = 100 * correct / total
            
            # Print metrics
            print(f'\nOverall Validation Loss: {val_loss:.4f}, Accuracy: {overall_accuracy:.2f}%')
            print('\nClassification Report:')
            print(report)
            
        self.train()  # Set the model back to training mode
        return val_loss.item(), overall_accuracy

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, learning_rate=0.001, batch_size=200):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        # Encode class names to indices
        y_train = torch.tensor(self.encode_class_indices(y_train), dtype=torch.long)
        y_val = torch.tensor(self.encode_class_indices(y_val), dtype=torch.long)

        # Convert inputs to tensors if they aren't already
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

        n_batches = (len(X_train) + batch_size - 1) // batch_size
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            total_loss = 0

            # Shuffle training data
            indices = torch.randperm(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Train in batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                loss = self.train_batch(batch_X, batch_y, optimizer, criterion)
                total_loss += loss

            avg_loss = total_loss / n_batches
            print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss: {avg_loss:.4f}')

            if epoch % 20 == 0:
                # Validate every epoch
                val_loss, val_acc = self.validate(X_val, y_val)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optionally save model weights here
                # torch.save(self.state_dict(), 'best_model.pt')
        
        return val_loss, val_acc

################################################################################

def chromatin_state_dataset_eic_train_test_val_split(solar_data_path="/project/compbio-lab/encode_data/"):
    bios_names = [t for t in os.listdir(solar_data_path) if t.startswith("T_")]
    # print(bios_names)

    cs_names = [t for t in os.listdir(os.path.join(solar_data_path, "chromatin_state_annotations"))]

    # Remove 'T_' prefix from biosample names for comparison
    bios_names_cleaned = [name.replace("T_", "") for name in bios_names]
    
    def similar(a, b, threshold=0.70):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    # Find exact and similar matches
    shared_names = set()
    similar_matches = {}  # Store similar but not exact matches
    
    for bios_name in bios_names_cleaned:
        if bios_name in cs_names:
            shared_names.add(bios_name)
        else:
            # Look for similar names
            for cs_name in cs_names:
                if similar(bios_name, cs_name):
                    similar_matches[bios_name] = cs_name
                    shared_names.add(cs_name)  # Add the CS name as it's the reference

    print(f"\nNumber of shared cell types (including similar matches): {len(shared_names)}")

    # Add 'T_' prefix back to shared names for comparison with original bios_names
    shared_names_with_prefix = [f"T_{name}" for name in shared_names]
    
    # Find unshared biosamples
    unshared_bios = [name for name in bios_names if name not in shared_names_with_prefix]
    
    print("\nBiosamples without matching chromatin states:")
    for name in unshared_bios:
        print(name)
    
    print("\nShared cell types between biosamples and chromatin states:")
    for name in shared_names:
        print(name)
        
    print("\nSimilar name matches found:")
    print(f"Biosample: {bios_name} -> Chromatin State: {cs_name}")


    print("\nAll paired biosamples and chromatin states:")
    print("Format: Biosample -> Chromatin State")
    print("-" * 50)
    
    # Print exact matches (where biosample name without T_ prefix matches CS name)
    for name in shared_names:
        if name in bios_names_cleaned:  # It's an exact match
            print(f"T_{name} -> {name}")
    
    # Print similar matches
    for bios_name, cs_name in similar_matches.items():
        print(f"T_{bios_name} -> {cs_name}")

    # Create a list of all valid pairs
    paired_data = []
    
    # Add exact matches
    for name in shared_names:
        if name in bios_names_cleaned:  # It's an exact match
            paired_data.append({
                'biosample': f"T_{name}",
                'chromatin_state': name
            })
    
    # Add similar matches
    for bios_name, cs_name in similar_matches.items():
        paired_data.append({
            'biosample': f"T_{bios_name}",
            'chromatin_state': cs_name
        })

    # Shuffle the pairs randomly
    random.seed(7)  # For reproducibility
    random.shuffle(paired_data)

    # Calculate split sizes
    total_samples = len(paired_data)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    # test_size will be the remainder

    # Split the data
    train_pairs = paired_data[:train_size]
    val_pairs = paired_data[train_size:train_size + val_size]
    test_pairs = paired_data[train_size + val_size:]

    # Print the splits
    print(f"\nTotal number of paired samples: {total_samples}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")

    print("\nTrain Split:")
    print("-" * 50)
    for pair in train_pairs:
        print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    print("\nValidation Split:")
    print("-" * 50)
    for pair in val_pairs:
        print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    print("\nTest Split:")
    print("-" * 50)
    for pair in test_pairs:
        print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    # Optionally, save the splits to files
    import json
    
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    return splits

def chromatin_state_dataset_merged_train_test_val_split(solar_data_path="/project/compbio-lab/encode_data/"):
    merged_navigation = os.path.join(solar_data_path, "merged_navigation.json")
    
    import json
    with open(merged_navigation, "r") as f:
        navigation = json.load(f)

    # Get original biosample names
    original_bios_names = [t for t in navigation.keys()]

    # Clean biosample names and create mapping
    clean_to_original = {}
    for name in original_bios_names:
        # Remove _grp\d+_rep\d+ pattern
        cleaned_name = '_'.join([part for part in name.split('_') 
                               if not ('grp' in part or 'rep' in part or 'nonrep' in part)])
        if cleaned_name not in clean_to_original:
            clean_to_original[cleaned_name] = []
        clean_to_original[cleaned_name].append(name)

    # Get unique cleaned names
    unique_cleaned_names = list(clean_to_original.keys())
    
    # Get chromatin state names
    cs_names = [t for t in os.listdir(os.path.join(solar_data_path, "chromatin_state_annotations"))]

    # Find intersection between cleaned names and chromatin states
    shared_names = set(unique_cleaned_names) & set(cs_names)
    
    print(f"\nNumber of shared cell types: {len(shared_names)}")
    
    # Convert to list and shuffle
    shared_names = list(shared_names)
    random.seed(7)  # For reproducibility
    random.shuffle(shared_names)

    # Calculate split sizes
    total_samples = len(shared_names)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)

    # Split the cleaned names
    train_names = shared_names[:train_size]
    val_names = shared_names[train_size:train_size + val_size]
    test_names = shared_names[train_size + val_size:]

    # Create final splits with original biosample names
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    # Helper function to create pairs
    def create_pairs(clean_names, split_name):
        pairs = []
        for clean_name in clean_names:
            # Get all original biosample names for this cleaned name
            original_names = clean_to_original[clean_name]
            # Create pairs with corresponding chromatin state
            for orig_name in original_names:
                pairs.append({
                    'biosample': orig_name,
                    'chromatin_state': clean_name  # clean_name is same as cs_name
                })
        return pairs

    # Create final splits
    splits['train'] = create_pairs(train_names, 'train')
    splits['val'] = create_pairs(val_names, 'val')
    splits['test'] = create_pairs(test_names, 'test')

    # Print statistics and results
    print("\nSplit Statistics:")
    print(f"Total unique cell types: {total_samples}")
    print(f"Train cell types: {len(train_names)}")
    print(f"Validation cell types: {len(val_names)}")
    print(f"Test cell types: {len(test_names)}")
    
    print(f"\nTotal biosamples: {len(original_bios_names)}")
    print(f"Train biosamples: {len(splits['train'])}")
    print(f"Validation biosamples: {len(splits['val'])}")
    print(f"Test biosamples: {len(splits['test'])}")

    # print("\nTrain Split:")
    # print("-" * 50)
    # for pair in splits['train']:
    #     print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    # print("\nValidation Split:")
    # print("-" * 50)
    # for pair in splits['val']:
    #     print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    # print("\nTest Split:")
    # print("-" * 50)
    # for pair in splits['test']:
    #     print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    return splits

def train_chromatin_state_probe(
    model_path, hyper_parameters_path, 
    num_train_regions=10000, num_val_regions=3000, num_test_regions=30, 
    train_chrs=["chr19"], val_chrs=["chr21"], test_chrs=["chr21"],
    dataset_path="/project/compbio-lab/encode_data/", resolution=200, eic=True, stratified=True):

    candi = CANDIPredictor(model_path, hyper_parameters_path, data_path=dataset_path, DNA=True, eic=eic, split="all")

    probe = ChromatinStateProbe(candi.model.d_model, output_dim=18)

    if eic:
        splits = chromatin_state_dataset_eic_train_test_val_split(dataset_path)
    else:
        splits = chromatin_state_dataset_merged_train_test_val_split(dataset_path)

    splits["train"] = splits["train"][:50]
    splits["val"] = splits["val"][:15]

    def prepare_data(split, chrs, start_idx, end_idx):
        chromatin_state_data = {}
        # Process each chromosome
        for chr in chrs:  
            candi.chr = chr
            chromatin_state_data[chr] = {}

            # Load chromatin state data for each cell type in training split
            for pair in splits[split][start_idx:end_idx]:
                try:
                    bios_name = pair['biosample']
                    cs_name = pair['chromatin_state']
                    cs_dir = os.path.join(dataset_path, "chromatin_state_annotations", cs_name)
                    parsed_dirs = [d for d in os.listdir(cs_dir) if d.startswith(f'parsed{resolution}_')]

                    X, seq, mX = candi.load_encoder_input_bios(bios_name, x_dsf=1)
                    Z = candi.get_latent_representations_cropped(X, mX, seq=seq)
                    del X, seq, mX
                    Z = Z.cpu()
                except:
                    continue

                chromatin_state_data[chr][cs_name] = (Z, [])
                for idx, parsed_cs in enumerate(parsed_dirs):
                    annot = load_region_chromatin_states(os.path.join(cs_dir, parsed_cs), chr) 
                    context_len = (candi.model.l1 * 25) // resolution
                    target_len = ((len(annot) // context_len) * context_len)
                    annot = annot[:target_len]

                    chromatin_state_data[chr][cs_name][1].append(annot)
                
                del Z 
                gc.collect()

        return chromatin_state_data
    
    def stratify_batch(Z_batch, Y_batch):
        """Helper function to stratify a single batch of data"""
        # Check for empty batch
        if len(Z_batch) == 0 or len(Y_batch) == 0:
            return np.array([]), np.array([])
            
        # Convert lists to numpy arrays if needed
        Z_batch = torch.stack([z for z in Z_batch if z is not None]).numpy()  # Stack non-None tensors
        Y_batch = np.array([y for y in Y_batch if y is not None])
        
        # Get class distribution
        unique_labels, counts = np.unique(Y_batch, return_counts=True)
        min_count = min(counts)
        
        # Stratify the batch
        stratified_indices = []
        for label in unique_labels:
            label_indices = np.where(Y_batch == label)[0]
            # If we have fewer samples than min_count, use all of them
            n_samples = min(min_count, len(label_indices))
            selected_indices = np.random.choice(label_indices, n_samples, replace=False)
            stratified_indices.extend(selected_indices)
        
        # Shuffle the stratified indices
        np.random.shuffle(stratified_indices)
        
        return Z_batch[stratified_indices], Y_batch[stratified_indices]

    Z_train = []
    Y_train = []
    batch_size = len(splits["train"])//10
    for i in range(0, len(splits["train"]), batch_size):
        train_chromatin_state_data = prepare_data("train", train_chrs, i, i+batch_size)
        
        # Collect data for current batch
        Z_batch = []
        Y_batch = []
        
        for chr in train_chromatin_state_data.keys():
            for ct in train_chromatin_state_data[chr].keys():
                z, annots = train_chromatin_state_data[chr][ct]
                for annot in annots:
                    assert len(annot) == len(z), f"annot and Z are not the same length for {ct} on {chr}"
                    for bin in range(len(annot)):
                        label = annot[bin]
                        latent_vector = z[bin]
                        
                        if label is not None:
                            Z_batch.append(latent_vector)
                            Y_batch.append(label)
        
        if stratified:
            # Print batch distribution before stratification
            unique_labels, counts = np.unique(Y_batch, return_counts=True)
            total_samples = len(Y_batch)
            
            print(f"\nBatch {i//batch_size + 1} Distribution Before Stratification:")
            print("Label | Count | Percentage")
            print("-" * 30)
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")
            
            # Stratify the current batch
            Z_batch_stratified, Y_batch_stratified = stratify_batch(Z_batch, Y_batch)
            
            # Print batch distribution after stratification
            unique_labels, counts = np.unique(Y_batch_stratified, return_counts=True)
            total_samples = len(Y_batch_stratified)
            
            print(f"\nBatch {i//batch_size + 1} Distribution After Stratification:")
            print("Label | Count | Percentage")
            print("-" * 30)
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")
            
            # Add stratified batch to training data
            Z_train.extend(Z_batch_stratified)
            Y_train.extend(Y_batch_stratified)
        else:
            # Add unstratified batch to training data
            Z_train.extend(Z_batch)
            Y_train.extend(Y_batch)
        
        del train_chromatin_state_data, Z_batch, Y_batch
        gc.collect()

    # Convert lists to numpy arrays
    Z_train = np.stack(Z_train)
    Y_train = np.array(Y_train)

    # Print final training set statistics
    print("\nFinal Training Dataset Analysis:")
    print(f"Z_train shape: {Z_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    unique_labels, counts = np.unique(Y_train, return_counts=True)
    total_samples = len(Y_train)

    print("\nFinal Class Distribution:")
    print("Label | Count | Percentage")
    print("-" * 30)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")

    #  Analysis and stratification of training data
    print("\nTraining Dataset Analysis:")
    print(f"Z_train shape: {Z_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    # Analyze class distribution
    unique_labels, counts = np.unique(Y_train, return_counts=True)
    total_samples = len(Y_train)

    print("\nClass Distribution:")
    print("Label | Count | Percentage")
    print("-" * 30)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")  # Changed :5d to :5s for label

    Z_val = [] 
    Y_val = []
    batch_size = len(splits["val"])//10
    for i in range(0, len(splits["val"]), batch_size):
        val_chromatin_state_data = prepare_data("val", val_chrs, i, i+batch_size)
        
        for chr in val_chromatin_state_data.keys():
            for ct in val_chromatin_state_data[chr].keys():
                z, annots = val_chromatin_state_data[chr][ct]
                for annot in annots:

                    assert len(annot) == len(z), f"annot and Z are not the same length for {ct} on {chr}"
                    for bin in range(len(annot)):
                        label = annot[bin]
                        latent_vector = z[bin]

                        if label is not None:
                            Z_val.append(latent_vector)
                            Y_val.append(label)
    
        del val_chromatin_state_data
        gc.collect()

    # Convert lists to tensors first since Z contains torch tensors
    Z_val = np.stack(Z_val)
    Y_val = np.array(Y_val)

    # # Use stratified training data for model training
    probe.fit(Z_train, Y_train, Z_val, Y_val, 
        num_epochs=800, learning_rate=0.001, batch_size=100)

################################################################################

def assay_importance(candi, bios_name, crop_edges=True):
    """
    we want to evaluate predictability of different assays as a function of input assays
    we want to see which input assays are most important for predicting the which output assay

    different tested settings:
        - just one input assay
        - top 6 histone mods
        - accessibility
        - accessibility + top 6 histone mods
    """

    X, Y, P, seq, mX, mY, avX, avY = candi.load_bios(bios_name, x_dsf=1)
    available_indices = torch.where(avX[0, :] == 1)[0]
    expnames = list(candi.dataset.aliases["experiment_aliases"].keys())

    available_assays = list(candi.dataset.navigation[bios_name].keys())
    print("available assays: ", available_assays)

    # Create distributions and get means
    Y = Y.view(-1, Y.shape[-1])
    P = P.view(-1, P.shape[-1])

    # keys: list of inputs, values: metrics per output assay | metrics: PP, Pearson, Spearman
    results = {}

    # # pred based on just one input assay
    for ii, keep_only in enumerate(available_indices):
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if idx != keep_only]
        print(f"single input: {expnames[keep_only]}")

        results[expnames[keep_only]] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.mean()
        count_mean = count_dist.mean()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()
            
            # Calculate perplexity
            pp_pval = perplexity(prob_pval[:, jj]).item()
            pp_count = perplexity(prob_count[:, jj]).item()

            # Calculate correlations
            pearson_pval = pearsonr(pval_true, pval_pred)[0]
            spearman_pval = spearmanr(pval_true, pval_pred)[0]
            pearson_count = pearsonr(count_true, count_pred)[0]
            spearman_count = spearmanr(count_true, count_pred)[0]   

            results[expnames[keep_only]][expnames[jj]] = {
                "PP_pval": pp_pval, "PP_count": pp_count,
                "Pearson_pval": pearson_pval, "Spearman_pval": spearman_pval,
                "Pearson_count": pearson_count, "Spearman_count": spearman_count
            }

    accessibility_assays = ["ATAC-seq", "DNase-seq"]
    has_accessibility = all(assay in available_assays for assay in accessibility_assays)
    if has_accessibility:
        print(f"accessibility inputs: {accessibility_assays}")
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if expnames[idx] not in accessibility_assays]
        results["accessibility"] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.mean()
        count_mean = count_dist.mean()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()
            
            # Calculate perplexity
            pp_pval = perplexity(prob_pval[:, jj]).item()
            pp_count = perplexity(prob_count[:, jj]).item()

            # Calculate correlations
            pearson_pval = pearsonr(pval_true, pval_pred)[0]
            spearman_pval = spearmanr(pval_true, pval_pred)[0]
            pearson_count = pearsonr(count_true, count_pred)[0]
            spearman_count = spearmanr(count_true, count_pred)[0]   

            results["accessibility"][expnames[jj]] = {
                "PP_pval": pp_pval, "PP_count": pp_count,
                "Pearson_pval": pearson_pval, "Spearman_pval": spearman_pval,
                "Pearson_count": pearson_count, "Spearman_count": spearman_count
            }

    histone_mods = ["H3K4me3", "H3K4me1", "H3K27ac", "H3K27me3", "H3K9me3", "H3K36me3"]
    has_histone_mods = all(assay in available_assays for assay in histone_mods)
    if has_histone_mods:
        print(f"6 histone mods inputs: {histone_mods}")
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if expnames[idx] not in histone_mods]
        results["histone_mods"] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.mean()
        count_mean = count_dist.mean()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()
            
            # Calculate perplexity
            pp_pval = perplexity(prob_pval[:, jj]).item()
            pp_count = perplexity(prob_count[:, jj]).item()

            # Calculate correlations
            pearson_pval = pearsonr(pval_true, pval_pred)[0]
            spearman_pval = spearmanr(pval_true, pval_pred)[0]
            pearson_count = pearsonr(count_true, count_pred)[0]
            spearman_count = spearmanr(count_true, count_pred)[0]   

            results["histone_mods"][expnames[jj]] = {
                "PP_pval": pp_pval, "PP_count": pp_count,
                "Pearson_pval": pearson_pval, "Spearman_pval": spearman_pval,
                "Pearson_count": pearson_count, "Spearman_count": spearman_count
            }

    if has_accessibility and has_histone_mods and len(available_assays) > len(accessibility_assays) + len(histone_mods):
        print(f"6 histone mods + accessibility inputs: {histone_mods + accessibility_assays}")
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if expnames[idx] not in histone_mods + accessibility_assays]
        results["histone_mods_accessibility"] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.mean()
        count_mean = count_dist.mean()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()
            
            # Calculate perplexity
            pp_pval = perplexity(prob_pval[:, jj]).item()
            pp_count = perplexity(prob_count[:, jj]).item()

            # Calculate correlations
            pearson_pval = pearsonr(pval_true, pval_pred)[0]
            spearman_pval = spearmanr(pval_true, pval_pred)[0]
            pearson_count = pearsonr(count_true, count_pred)[0]
            spearman_count = spearmanr(count_true, count_pred)[0]   

            results["histone_mods_accessibility"][expnames[jj]] = {
                "PP_pval": pp_pval, "PP_count": pp_count,
                "Pearson_pval": pearson_pval, "Spearman_pval": spearman_pval,
                "Pearson_count": pearson_count, "Spearman_count": spearman_count
            }

    return results  

def perplexity_v_pred_error():
    pass

################################################################################

if __name__ == "__main__":
    if sys.argv[1] == "cs_probe":
        model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"
        eic = False
        train_chromatin_state_probe(model_path, hyper_parameters_path, dataset_path="/project/compbio-lab/encode_data/", eic=eic)

    elif sys.argv[1] == "latent_repr":
        # model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"
        eic = True

        ct0_repr1 = "ENCBS706NOO"
        ct0_repr2 = "ENCBS314QQU"
        latent_reproducibility(model_path, hyper_parameters_path, ct0_repr1, ct0_repr2, dataset_path="/project/compbio-lab/encode_data/")

        ct1_repr1 = "ENCBS674MPN"
        ct1_repr2 = "ENCBS639AAA"
        latent_reproducibility(model_path, hyper_parameters_path, ct1_repr1, ct1_repr2, dataset_path="/project/compbio-lab/encode_data/")

        ct2_repr1 = "ENCBS967MVZ"
        ct2_repr2 = "ENCBS789UPK"
        latent_reproducibility(model_path, hyper_parameters_path, ct2_repr1, ct2_repr2, dataset_path="/project/compbio-lab/encode_data/") 

        ct3_repr1 = "ENCBS715VCP"
        ct3_repr2 = "ENCBS830CIQ"
        latent_reproducibility(model_path, hyper_parameters_path, ct3_repr1, ct3_repr2, dataset_path="/project/compbio-lab/encode_data/")  
        
        ct4_repr1 = "ENCBS865RXK"
        ct4_repr2 = "ENCBS188BKX"
        latent_reproducibility(model_path, hyper_parameters_path, ct4_repr1, ct4_repr2, dataset_path="/project/compbio-lab/encode_data/")   

        ct5_repr1 = "ENCBS655ARO"
        ct5_repr2 = "ENCBS075PNA"
        latent_reproducibility(model_path, hyper_parameters_path, ct5_repr1, ct5_repr2, dataset_path="/project/compbio-lab/encode_data/")   

        # Random pairs from different cell types to test cross-cell-type reproducibility
        print("\nTesting cross-cell-type reproducibility:")
        # CT1 vs CT2
        latent_reproducibility(model_path, hyper_parameters_path, ct1_repr1, ct2_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT2 vs CT3 
        latent_reproducibility(model_path, hyper_parameters_path, ct2_repr2, ct3_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT3 vs CT4
        latent_reproducibility(model_path, hyper_parameters_path, ct3_repr2, ct4_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT4 vs CT5
        latent_reproducibility(model_path, hyper_parameters_path, ct4_repr2, ct5_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT5 vs CT1 
        latent_reproducibility(model_path, hyper_parameters_path, ct5_repr2, ct1_repr2, dataset_path="/project/compbio-lab/encode_data/")

        # CT0 vs CT3
        print("\nTesting CT0 vs CT3 reproducibility:")
        latent_reproducibility(model_path, hyper_parameters_path, ct0_repr1, ct3_repr1, dataset_path="/project/compbio-lab/encode_data/")

    elif sys.argv[1] == "perplexity":
        model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"
        eic = False

        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=True)
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"
        bios_name = sys.argv[2]

        # Load latent representations
        X, Y, P, seq, mX, mY, avX, avY = candi.load_bios(bios_name, x_dsf=1)

        n, p, mu, var, Z = candi.pred_cropped(X, mX, mY, avX, seq=seq, crop_percent=0.3)
        
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])

        count_dist = NegativeBinomial(p, n)
        pval_dist = Gaussian(mu, var)

        count_probabilities = count_dist.pmf(Y)
        pval_probabilities = pval_dist.pdf(P)

        for i in range(Y.shape[1]):
            # print(Y[:, i].mean())
            if avY[0, i] == 1:
                print(
                f"Assay: {expnames[i]}, PP_count: {perplexity(count_probabilities[:, i]):.3f}, PP_pval: {perplexity(pval_probabilities[:, i]):.3f}")
        
        # position_PP_count = []
        # position_PP_pval = []
        # for i in range(Y.shape[0]):
        #     # Get probabilities for available assays at each position
        #     p_count = count_probabilities[i, avY[0]==1]
        #     p_pval = pval_probabilities[i, avY[0]==1]
        #     position_PP_count.append(perplexity(p_count))
        #     position_PP_pval.append(perplexity(p_pval))

        # Create mask for available assays
        available_mask = (avY[0] == 1)

        # Calculate perplexity for all positions at once using broadcasting
        # Shape: (n_positions, n_available_assays)
        count_probs_available = count_probabilities[:, available_mask]
        pval_probs_available = pval_probabilities[:, available_mask]

        # Calculate perplexity using vectorized operations
        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        position_PP_count = torch.exp(-torch.sum(torch.log(count_probs_available + epsilon), dim=1) / count_probs_available.shape[1])
        position_PP_pval = torch.exp(-torch.sum(torch.log(pval_probs_available + epsilon), dim=1) / pval_probs_available.shape[1])

        # Convert to numpy for statistics (if needed)
        position_PP_count = position_PP_count.cpu().numpy()
        position_PP_pval = position_PP_pval.cpu().numpy()

        # Print statistics
        print(f"Position PP_count: {np.mean(position_PP_count):.3f}, Position PP_pval: {np.mean(position_PP_pval):.3f}")
        print(f"Position PP_count std: {np.std(position_PP_count):.3f}, Position PP_pval std: {np.std(position_PP_pval):.3f}")
        print(f"Position PP_count 95% CI: {np.percentile(position_PP_count, 2.5):.3f} - {np.percentile(position_PP_count, 97.5):.3f}")
        print(f"Position PP_pval 95% CI: {np.percentile(position_PP_pval, 2.5):.3f} - {np.percentile(position_PP_pval, 97.5):.3f}")
        
        # Reduce resolution of perplexity scores by averaging every 8 values
        def reduce_resolution(arr, factor=8):
            # Ensure the array length is divisible by factor
            pad_length = (factor - (len(arr) % factor)) % factor
            if pad_length > 0:
                arr = np.pad(arr, (0, pad_length), mode='edge')
            
            # Reshape and average
            arr_reshaped = arr.reshape(-1, factor)
            return np.mean(arr_reshaped, axis=1)

        # Reduce resolution of perplexity scores
        position_PP_count_reduced = reduce_resolution(position_PP_count)
        position_PP_pval_reduced = reduce_resolution(position_PP_pval)

        # Get 99th percentile values to clip outliers
        pp_count_99th = np.percentile(position_PP_count_reduced, 99)
        pp_pval_99th = np.percentile(position_PP_pval_reduced, 99)

        # Clip values above 99th percentile
        position_PP_count_reduced = np.clip(position_PP_count_reduced, None, pp_count_99th)
        position_PP_pval_reduced = np.clip(position_PP_pval_reduced, None, pp_pval_99th)

        # Print statistics after reduction
        print("\nAfter resolution reduction:")
        print(f"Position PP_count: {np.mean(position_PP_count_reduced):.3f}, Position PP_pval: {np.mean(position_PP_pval_reduced):.3f}")
        print(f"Position PP_count std: {np.std(position_PP_count_reduced):.3f}, Position PP_pval std: {np.std(position_PP_pval_reduced):.3f}")
        print(f"Position PP_count 99% CI: {np.percentile(position_PP_count_reduced, 0.5):.3f} - {np.percentile(position_PP_count_reduced, 99.5):.3f}")
        print(f"Position PP_pval 99% CI: {np.percentile(position_PP_pval_reduced, 0.5):.3f} - {np.percentile(position_PP_pval_reduced, 99.5):.3f}")
        # Print min and max values
        print("\nMin/Max values:")
        print(f"Position PP_count min: {np.min(position_PP_count_reduced):.3f}, max: {np.max(position_PP_count_reduced):.3f}")
        print(f"Position PP_pval min: {np.min(position_PP_pval_reduced):.3f}, max: {np.max(position_PP_pval_reduced):.3f}")

        # Calculate and print correlations between count and p-value perplexity scores
        pearson_corr, pearson_p = scipy.stats.pearsonr(position_PP_count_reduced, position_PP_pval_reduced)
        spearman_corr, spearman_p = scipy.stats.spearmanr(position_PP_count_reduced, position_PP_pval_reduced)

        print("\nCorrelations between count and p-value perplexity:")
        print(f"Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.3e})")
        print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")
        # exit()

        # Verify lengths match
        assert len(position_PP_count_reduced) == len(Z), f"Length mismatch: {len(position_PP_count_reduced)} vs {len(Z)}"
        
        # Create visualization plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Compute PCA
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Z.cpu().numpy())
        
        # Compute UMAP
        reducer = umap.UMAP(random_state=42)
        Z_umap = reducer.fit_transform(Z.cpu().numpy())
        
        # Plot PCA colored by count perplexity
        scatter1 = axes[0,0].scatter(Z_pca[:, 0], Z_pca[:, 1], 
                                    c=position_PP_count_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[0,0].set_title('PCA - Colored by Count Perplexity')
        axes[0,0].set_xlabel('PC1')
        axes[0,0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # Plot PCA colored by p-value perplexity
        scatter2 = axes[0,1].scatter(Z_pca[:, 0], Z_pca[:, 1], 
                                    c=position_PP_pval_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[0,1].set_title('PCA - Colored by P-value Perplexity')
        axes[0,1].set_xlabel('PC1')
        axes[0,1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # Plot UMAP colored by count perplexity
        scatter3 = axes[1,0].scatter(Z_umap[:, 0], Z_umap[:, 1], 
                                    c=position_PP_count_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[1,0].set_title('UMAP - Colored by Count Perplexity')
        axes[1,0].set_xlabel('UMAP1')
        axes[1,0].set_ylabel('UMAP2')
        plt.colorbar(scatter3, ax=axes[1,0])
        
        # Plot UMAP colored by p-value perplexity
        scatter4 = axes[1,1].scatter(Z_umap[:, 0], Z_umap[:, 1], 
                                    c=position_PP_pval_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[1,1].set_title('UMAP - Colored by P-value Perplexity')
        axes[1,1].set_xlabel('UMAP1')
        axes[1,1].set_ylabel('UMAP2')
        plt.colorbar(scatter4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f'latent_space_perplexity_{bios_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    elif sys.argv[1] == "eval_full":
        model_path = "models/CANDIfull_DNA_random_mask_Dec9_20241209114510_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic)
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"
        for bios_name in random.sample(list(candi.dataset.navigation.keys()), len(candi.dataset.navigation)):
            try:
                print(bios_name)
                start_time = time.time()
                metrics = candi.evaluate(bios_name)
                end_time = time.time()
                print(f"Evaluation took {end_time - start_time:.2f} seconds")
                print("\n\n")

            except Exception as e:
                print(f"Error processing {bios_name}: {e}")
                continue

    elif sys.argv[1] == "eval_eic":
        model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"
        eic = True

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="test")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"
        for bios_name in list(candi.dataset.navigation.keys()):
            try:
                print(bios_name)
                metrics = candi.evaluate(bios_name)
                print("\n\n")

            except Exception as e:
                print(f"Error processing {bios_name}: {e}")
                continue
    
    elif sys.argv[1] == "eval_full_bios":
    
        model_path = "models/CANDIfull_DNA_random_mask_Dec9_20241209114510_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False


        # model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        # hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="test")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"

        if sys.argv[2] == "show_test_bios":
            print(candi.dataset.navigation.keys())
            exit()
        else:
            bios_name = sys.argv[2]

        try:
            print(bios_name)
            metrics = candi.evaluate(bios_name)
            print("\n\n")
            
        except Exception as e:
            print(f"Error processing {bios_name}: {e}")
    
    elif sys.argv[1] == "eval_eic_bios":

        # model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        # hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"

        model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"
        eic = True

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="val")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"

        if sys.argv[2] == "show_test_bios":
            print(candi.dataset.navigation.keys())
            exit()
        else:
            bios_name = sys.argv[2]

        try:
            print(bios_name)
            metrics = candi.evaluate(bios_name)
            print("\n\n")
            
        except Exception as e:
            print(f"Error processing {bios_name}: {e}")

    elif sys.argv[1] == "assay_importance":
        model_path = "models/CANDIfull_DNA_random_mask_Dec9_20241209114510_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="test")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"

        # if sys.argv[2] == "show_test_bios":
        #     print(candi.dataset.navigation.keys())
        #     exit()
        # else:
            # bios_name = sys.argv[2]
        
        metrics = {}
        bios_names = list(candi.dataset.navigation.keys())[4:5]
        for bios_name in bios_names:
            print(bios_name)
            metrics[bios_name] = assay_importance(candi, bios_name)


        results = []
        for bios_name in bios_names:
            for input in metrics[bios_name]:
                for output in metrics[bios_name][input]:
                    results.append({
                        "bios_name": bios_name,
                        "input": input,
                        "output": output,
                        "PP_pval": metrics[bios_name][input][output]["PP_pval"],
                        "PP_count": metrics[bios_name][input][output]["PP_count"],
                        "Pearson_pval": metrics[bios_name][input][output]["Pearson_pval"],
                        "Spearman_pval": metrics[bios_name][input][output]["Spearman_pval"],
                        "Pearson_count": metrics[bios_name][input][output]["Pearson_count"],
                        "Spearman_count": metrics[bios_name][input][output]["Spearman_count"]
                    })

        df = pd.DataFrame(results)
        print(df)

        df.to_csv("models/output/assay_importance.csv", index=False)

        viz_feature_importance(df, savedir="models/output/")
