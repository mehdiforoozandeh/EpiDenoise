import random
import torch
import pickle
import os, time, gc, psutil
from CANDI import *
from scipy import stats
import numpy as np


# sequence clustering
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
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [chr, 0, self.chr_sizes[chr]], x_dsf)

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, [chr, 0, self.chr_sizes[chr]], x_dsf)
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
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
        
        # print(f"getting bios vals for {bios_name}")

        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            
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

    def evaluate_leave_one_out(self, X, mX, mY, avX, Y, P, seq=None, crop_edges=True):
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
        for leave_one_out in available_indices:
            if crop_edges:
                n, p, mu, var, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            else:
                n, p, mu, var, _ = self.pred(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            n_imp[:, leave_one_out] = n[:, leave_one_out]
            p_imp[:, leave_one_out] = p[:, leave_one_out]
            mu_imp[:, leave_one_out] = mu[:, leave_one_out]
            var_imp[:, leave_one_out] = var[:, leave_one_out]
            print(f"Completed feature {leave_one_out+1}/{len(available_indices)}")
        
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
        
        # Calculate metrics for each feature
        metrics = {}
        for idx in available_indices:
            count_true = Y[:, idx].numpy()
            pval_true = P[:, idx].numpy()
            
            # Count predictions
            imp_count = imp_count_mean[:, idx].numpy()
            ups_count = ups_count_mean[:, idx].numpy()
            
            # P-value predictions (apply arcsinh transformation)
            imp_pval = np.sinh(imp_pval_mean[:, idx].numpy())
            ups_pval = np.sinh(ups_pval_mean[:, idx].numpy())
            pval_true = np.sinh(pval_true)
            
            
            metrics[idx.item()] = {
                'count_metrics': {
                    'imp_pearson': stats.pearsonr(count_true, imp_count)[0],
                    'imp_spearman': stats.spearmanr(count_true, imp_count)[0],
                    'imp_mse': np.mean((count_true - imp_count) ** 2),
                    'imp_r2': 1 - (np.sum((count_true - imp_count) ** 2) / 
                                 np.sum((count_true - np.mean(count_true)) ** 2)),
                    'ups_pearson': stats.pearsonr(count_true, ups_count)[0],
                    'ups_spearman': stats.spearmanr(count_true, ups_count)[0],
                    'ups_mse': np.mean((count_true - ups_count) ** 2),
                    'ups_r2': 1 - (np.sum((count_true - ups_count) ** 2) / 
                                 np.sum((count_true - np.mean(count_true)) ** 2)),
                },
                'pval_metrics': {
                    'imp_pearson': stats.pearsonr(pval_true, imp_pval)[0],
                    'imp_spearman': stats.spearmanr(pval_true, imp_pval)[0],
                    'imp_mse': np.mean((pval_true - imp_pval) ** 2),
                    'imp_r2': 1 - (np.sum((pval_true - imp_pval) ** 2) / 
                                 np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'ups_pearson': stats.pearsonr(pval_true, ups_pval)[0],
                    'ups_spearman': stats.spearmanr(pval_true, ups_pval)[0],
                    'ups_mse': np.mean((pval_true - ups_pval) ** 2),
                    'ups_r2': 1 - (np.sum((pval_true - ups_pval) ** 2) / 
                                 np.sum((pval_true - np.mean(pval_true)) ** 2))
                }
            }
        
        # Print summary
        print("\nEvaluation Results:")
        print("\nCount Metrics:")
        print("Feature | Type      | Pearson | Spearman | MSE    | R2")
        print("-" * 55)
        for idx in available_indices:
            m = metrics[idx.item()]['count_metrics']
            print(f"{idx:7d} | Imputed   | {m['imp_pearson']:7.4f} | {m['imp_spearman']:8.4f} | "
                  f"{m['imp_mse']:6.4f} | {m['imp_r2']:6.4f}")
            print(f"        | Upsampled | {m['ups_pearson']:7.4f} | {m['ups_spearman']:8.4f} | "
                  f"{m['ups_mse']:6.4f} | {m['ups_r2']:6.4f}")
            print("-" * 55)
            
        print("\nP-value Metrics:")
        print("Feature | Type      | Pearson | Spearman | MSE    | R2")
        print("-" * 55)
        for idx in available_indices:
            m = metrics[idx.item()]['pval_metrics']
            print(f"{idx:7d} | Imputed   | {m['imp_pearson']:7.4f} | {m['imp_spearman']:8.4f} | "
                  f"{m['imp_mse']:6.4f} | {m['imp_r2']:6.4f}")
            print(f"        | Upsampled | {m['ups_pearson']:7.4f} | {m['ups_spearman']:8.4f} | "
                  f"{m['ups_mse']:6.4f} | {m['ups_r2']:6.4f}")
            print("-" * 55)
            
        return metrics

"""
# given a model, i want to train a linear probe on the latent space
# and evaluate the performance of the linear probe on the linear probe dataset.
# implemented using pytorch
# we want one class per probe
# we want one train function per probe
# we want one evaluate function per probe

the following are different probes that i will implement
1. chromatin_state_classification probe:
    - input: CANDI latent representation
    - output: 18 state classification
    - loss function: cross entropy

2. peak_calling probe:
    - input: CANDI latent representation
    - output: binary peak/no peak
    - loss function: binary cross entropy

3. activity_prediction_probe:
    - input: CANDI latent representation
    - output: binary active/inactive
    - loss function: binary cross entropy

4. conservation_prediction_probe:
    - input: CANDI latent representation
    - output: conservation score
    - loss function: mean squared error

5. expression_prediction_probe:
    - input: CANDI latent representation for a window of sequence
    - output: expression (TPM) prediction
    - loss function: mean squared error
"""

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

    # Calculate Euclidean distances
    euclidean_distances = torch.norm(latent_repr1 - latent_repr2, dim=1)

    # Calculate Cosine distances
    cosine_similarities = F.cosine_similarity(latent_repr1, latent_repr2, dim=1)
    cosine_distances = 1 - cosine_similarities

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
            'mean': cosine_distances.mean().item(),
            'std': cosine_distances.std().item(),
            'median': cosine_distances.median().item(),
            'min': cosine_distances.min().item(),
            'max': cosine_distances.max().item()
        }
    }

    print(f"\nLatent Space Reproducibility Analysis between {repr1_bios} and {repr2_bios}")
    print("-" * 80)
    
    for metric in ['euclidean', 'cosine']:
        print(f"\n{metric.capitalize()} Distance Statistics:")
        print(f"Mean: {stats[metric]['mean']:.4f}")
        print(f"Std:  {stats[metric]['std']:.4f}")
        print(f"Med:  {stats[metric]['median']:.4f}")
        print(f"Min:  {stats[metric]['min']:.4f}")
        print(f"Max:  {stats[metric]['max']:.4f}")

    return stats, euclidean_distances, cosine_distances

    """
    for i in range(Z1.shape[0]):
        get euclidean distance between Z1[i] and Z2[i]
        get cosine distance between Z1[i] and Z2[i]
    """


# class ChromatinStateProbe(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#         self.softmax = nn.Softmax(dim=1)
#         self.class_to_index = None  # Placeholder for the class-to-index mapping

#     def forward(self, x):
#         x = self.linear(x)
#         x = self.softmax(x)
#         return x

#     def encode_one_hot(self, class_names):
#         """
#         One-hot encode a list of class names.
#         """
#         if self.class_to_index is None:
#             # Create mapping if not already defined
#             unique_classes = sorted(set(class_names))
#             self.class_to_index = {name: idx for idx, name in enumerate(unique_classes)}
        
#         class_indices = torch.tensor([self.class_to_index[name] for name in class_names])
#         num_classes = len(self.class_to_index)
#         return torch.nn.functional.one_hot(class_indices, num_classes=num_classes)
    
#     def decode_one_hot(self, one_hot_tensor):
#         """
#         Decode a one-hot encoded tensor back to the list of class names.
#         """
#         if self.class_to_index is None:
#             raise ValueError("class_to_index mapping is not defined.")
        
#         # Invert the class_to_index dictionary
#         index_to_class = {idx: name for name, idx in self.class_to_index.items()}
#         class_indices = torch.argmax(one_hot_tensor, dim=1)
#         return [index_to_class[idx.item()] for idx in class_indices]

#     def train_batch(self, X, y, optimizer, criterion):
#         optimizer.zero_grad()
#         output = self(X)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()
#         return loss.item()
    
#     def validate(self, X, y):
#         self.eval()
#         with torch.no_grad():
#             output = self(X)
#             criterion = nn.BCELoss()  # Changed to BCE since we're using one-hot
#             val_loss = criterion(output, y)
            
#             # Calculate accuracy
#             _, predicted = torch.max(output.data, 1)
#             total = y.size(0)
#             correct = (predicted == y).sum().item()
#             accuracy = 100 * correct / total
            
#             print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
#         self.train()
#         return val_loss.item(), accuracy
    
#     def train_loop(self, X_train, y_train, X_val, y_val, num_epochs=10, learning_rate=0.01, batch_size=200):
#         optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         criterion = nn.CrossEntropyLoss() 
        
#         y_train = self.encode_one_hot(y_train)
#         y_val = self.encode_one_hot(y_val)

#         # Convert inputs to tensors if they aren't already
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.long)
#         X_val = torch.tensor(X_val, dtype=torch.float32)
#         y_val = torch.tensor(y_val, dtype=torch.long)
        
#         n_batches = (len(X_train) + batch_size - 1) // batch_size
#         best_val_loss = float('inf')
        
#         for epoch in range(num_epochs):
#             total_loss = 0
#             # Shuffle training data
#             indices = torch.randperm(len(X_train))
#             X_train = X_train[indices]
#             y_train = y_train[indices]
            
#             # Train in batches
#             for i in range(0, len(X_train), batch_size):
#                 batch_X = X_train[i:i + batch_size]
#                 batch_y = y_train[i:i + batch_size]
                
#                 loss = self.train_batch(batch_X, batch_y, optimizer, criterion)
#                 total_loss += loss
            
#             avg_loss = total_loss / n_batches
#             print(f'Epoch {epoch}/{num_epochs-1}:   |   Training Loss: {avg_loss:.4f}')

#             # Validate every 5 epochs or on the last epoch
#             if epoch % 5 == 0 or epoch == num_epochs - 1:
#                 val_loss, val_acc = self.validate(X_val, y_val)
#                 print(f'Validation Loss: {val_loss:.4f}   |   Validation Accuracy: {val_acc:.2f}%')
#                 print('-' * 50)  # Creates a line of 50 dashes
                
#                 # Save best model
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     # Optionally save model weights here
#                     # torch.save(self.state_dict(), 'best_model.pt')

class ChromatinStateProbe(nn.Module):
    def __init__(self, input_dim, output_dim=18):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # self.softmax = nn.Softmax(dim=1)
        self.class_to_index = None  # Placeholder for the class-to-index mapping

    def forward(self, x, normalize=True):
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        x = self.linear(x)
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
            criterion = nn.CrossEntropyLoss()
            val_loss = criterion(output, y)
            
            # Calculate overall accuracy
            _, predicted = torch.max(output, 1)
            total = y.size(0)
            correct = (predicted == y).sum().item()
            overall_accuracy = 100 * correct / total
            
            # Calculate per-class metrics
            unique_labels = torch.unique(y)
            per_class_metrics = {}
            
            for label in unique_labels:
                # Create mask for current class
                mask = (y == label)
                
                # True positives, false positives, false negatives
                true_pos = ((predicted == label) & (y == label)).sum().item()
                false_pos = ((predicted == label) & (y != label)).sum().item()
                false_neg = ((predicted != label) & (y == label)).sum().item()
                
                # Calculate metrics
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = true_pos / mask.sum().item() if mask.sum().item() > 0 else 0
                
                per_class_metrics[self.index_to_class[label.item()]] = {
                    'accuracy': 100 * accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': mask.sum().item()
                }
            
            # Print metrics
            print(f'\nOverall Validation Loss: {val_loss:.4f}, Accuracy: {overall_accuracy:.2f}%')
            print('\nPer-class metrics:')
            print('Label | Accuracy | Precision | Recall |   F1   | Support')
            print('-' * 60)
            for label in sorted(per_class_metrics.keys()):
                metrics = per_class_metrics[label]
                print(f'{label:10s} | {metrics["accuracy"]:7.2f}% | {metrics["precision"]:8.4f} | {metrics["recall"]:6.4f} | {metrics["f1"]:6.4f} | {metrics["support"]:7d}')
            print('-' * 60)
            
        self.train()
        return val_loss.item(), overall_accuracy, per_class_metrics

    def train_loop(self, X_train, y_train, X_val, y_val, num_epochs=10, learning_rate=0.01, batch_size=200):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

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

            if epoch % 5 == 0:
                # Validate every epoch
                val_loss, val_acc, per_class_metrics = self.validate(X_val, y_val)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optionally save model weights here
                # torch.save(self.state_dict(), 'best_model.pt')

def chromatin_state_dataset_eic_train_test_val_split(solar_data_path="/project/compbio-lab/encode_data/"):
    bios_names = [t for t in os.listdir(solar_data_path) if t.startswith("T_")]
    # print(bios_names)

    cs_names = [t for t in os.listdir(os.path.join(solar_data_path, "chromatin_state_annotations"))]

    # Remove 'T_' prefix from biosample names for comparison
    bios_names_cleaned = [name.replace("T_", "") for name in bios_names]
    
    from difflib import SequenceMatcher

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

def train_chromatin_state_probe(
    model_path, hyper_parameters_path, 
    num_train_regions=10000, num_val_regions=3000, num_test_regions=30, 
    train_chrs=["chr19"], val_chrs=["chr21"], test_chrs=["chr21"],
    dataset_path="/project/compbio-lab/encode_data/", resolution=200, eic=True):

    candi = CANDIPredictor(model_path, hyper_parameters_path, data_path=dataset_path, DNA=True, eic=eic)

    probe = ChromatinStateProbe(candi.model.d_model, 18)

    splits = chromatin_state_dataset_eic_train_test_val_split(dataset_path)
    splits["train"] = splits["train"][:5]
    # splits["test"] = splits["test"][:1]
    splits["val"] = splits["val"][:3]
    
    def prepare_data(split, chrs, num_regions):
        chromatin_state_data = {}
        # Process each chromosome
        for chr in chrs:  
            candi.chr = chr

            cs_data = {}
            # Load chromatin state data for each cell type in training split
            for pair in splits[split]:
                bios_name = pair['biosample']
                cs_name = pair['chromatin_state']
                cs_dir = os.path.join(dataset_path, "chromatin_state_annotations", cs_name)
                parsed_dirs = [d for d in os.listdir(cs_dir) if d.startswith(f'parsed{resolution}_')]

                for idx, parsed_cs in enumerate(parsed_dirs):
                    cs_data[f"{cs_name}|{idx}"] = load_region_chromatin_states(os.path.join(cs_dir, parsed_cs), chr) 

            chromatin_state_data[chr] = {}  # chr : cell_type : [chromosome, start_pos, end_pos, chromatin_state_array]
            for ct in cs_data.keys():
                if ct.split("|")[0] not in chromatin_state_data[chr]:
                    chromatin_state_data[chr][ct.split("|")[0]] = []

                annot = cs_data[ct]
                # Make length divisible by (candi.model.l1 // resolution)
                context_len = (candi.model.l1 * 25) // resolution
                target_len = ((len(annot) // context_len) * context_len)
                annot = annot[:target_len]


                chromatin_state_data[chr][ct.split("|")[0]].append(annot)
        
            # Load chromatin state data for each cell type in training split
            for pair in splits[split]:
                bios_name = pair['biosample']
                cs_name = pair['chromatin_state']

                X, seq, mX = candi.load_encoder_input_bios(bios_name, x_dsf=1)
                Z = candi.get_latent_representations_cropped(X, mX, seq=seq)
                del X, seq, mX

                Z = Z.cpu()
                for annot in chromatin_state_data[chr][cs_name]:
                    print(annot)
                    exit()
                    # print(type(annot))
                    # print(f"annot shape: {annot.shape}, Z shape: {Z.shape}")
                    print(f"Z memory usage: {Z.element_size() * Z.nelement() / (1024*1024):.2f} MB")
                    
                    # Get available memory
                    available_mem = psutil.virtual_memory().available / (1024 * 1024) # Convert to MB
                    print(f"Available memory: {available_mem:.2f} MB")
                    print()
                    
                    # Only append if we have enough memory (e.g. 2x the tensor size)
                    z_size = Z.element_size() * Z.nelement() / (1024*1024)
                    if available_mem > 2 * z_size:
                        chromatin_state_data[chr][cs_name].append((annot, Z))
                    else:
                        print("Warning: Not enough memory to store tensor")

                gc.collect()

        return chromatin_state_data

    # structure ->  chr : cell_type : ([chromosome, start_pos, end_pos, chromatin_state_array], z_tensor)
    
    train_chromatin_state_data = prepare_data("train", train_chrs, num_train_regions)
    Z_train = [] 
    Y_train = []
    for chr in train_chromatin_state_data.keys():
        for ct in train_chromatin_state_data[chr].keys():
            for region, z in train_chromatin_state_data[chr][ct]:
                annots = region
                for bin in range(len(region)):
                    label = annots[bin]
                    latent_vector = z[bin]

                    if label is not None:
                        Z_train.append(latent_vector)
                        Y_train.append(label)
    
    # Convert lists to tensors first since Z contains torch tensors
    Z_train = np.stack(Z_train)
    Y_train = np.array(Y_train)

    val_chromatin_state_data = prepare_data("val", val_chrs, num_val_regions)
    Z_val = [] 
    Y_val = []
    for chr in val_chromatin_state_data.keys():
        for ct in val_chromatin_state_data[chr].keys():
            for region, z in val_chromatin_state_data[chr][ct]:
                annots = region
                for bin in range(len(region)):
                    label = annots[bin]
                    latent_vector = z[bin]

                    if label is not None:
                        Z_val.append(latent_vector)
                        Y_val.append(label)
    
    # Convert lists to tensors first since Z contains torch tensors
    Z_val = np.stack(Z_val)
    Y_val = np.array(Y_val)

    # test_chromatin_state_data = prepare_data("test", train_chrs, num_test_regions)
    # Z_test = [] 
    # Y_test = []
    # for chr in test_chromatin_state_data.keys():
    #     for ct in test_chromatin_state_data[chr].keys():
    #         for region, z in test_chromatin_state_data[chr][ct]:
    #             annots = region
    #             for bin in range(len(region)):
    #                 label = annots[bin]
    #                 latent_vector = z[bin]

    #                 if label is not None:
    #                     Z_test.append(latent_vector)
    #                     Y_test.append(label)
    
    # # Convert lists to tensors first since Z contains torch tensors
    # Z_test = np.stack(Z_test)
    # Y_test = np.array(Y_test)


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
        print(f"{label:5s} | {count:5d} | {percentage:6.2f}%")  # Changed :5d to :5s for label

    exit()
    # Use stratified training data for model training
    probe.train_loop(Z_train, Y_train, Z_val, Y_val, 
        num_epochs=500, learning_rate=0.005, batch_size=100)


if __name__ == "__main__":
    model_path = "models/CANDIeic_DNA_random_mask_Nov25_model_checkpoint_epoch5.pth"
    hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov25_20241126160857_params45093285.pkl"
    eic = True

    if sys.argv[1] == "cs_probe":
        train_chromatin_state_probe(model_path, hyper_parameters_path, dataset_path="/project/compbio-lab/encode_data/")

    elif sys.argv[1] == "latent_repr":
        repr1 = "ENCBS674MPN"
        repr2 = "ENCBS639AAA"

        latent_reproducibility(model_path, hyper_parameters_path, repr1, repr2, dataset_path="/project/compbio-lab/encode_data/")
    

