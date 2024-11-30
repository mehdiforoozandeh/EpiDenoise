import random
import torch
import pickle
import os, time, gc
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

    def load_encoder_input_bios(self, bios_name, x_dsf, chr=None, y_dsf=1):
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
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.class_to_index = None  # Placeholder for the class-to-index mapping

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)

    def encode_class_indices(self, class_names):
        """
        Convert a list of class names to class indices.
        """
        if self.class_to_index is None:
            unique_classes = sorted(set(class_names))
            self.class_to_index = {name: idx for idx, name in enumerate(unique_classes)}
        return [self.class_to_index[name] for name in class_names]

    def decode_class_indices(self, class_indices):
        """
        Convert class indices back to class names.
        """
        if self.class_to_index is None:
            raise ValueError("class_to_index mapping is not defined.")
        index_to_class = {idx: name for name, idx in self.class_to_index.items()}
        return [index_to_class[idx] for idx in class_indices]

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
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == y).sum().item()
            accuracy = 100 * correct / y.size(0)
            
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        self.train()
        return val_loss.item(), accuracy

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
                val_loss, val_acc = self.validate(X_val, y_val)

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
    random.seed(42)  # For reproducibility
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
    num_train_regions=2000, num_val_regions=1000, num_test_regions=30, 
    train_chrs=["chr19", "chr20", "chr21"], val_chrs=["chrX"], test_chrs=["chr21"],
    dataset_path="/project/compbio-lab/encode_data/", resolution=200, eic=True):

    candi = CANDIPredictor(model_path, hyper_parameters_path, data_path=dataset_path, DNA=True, eic=eic)

    probe = ChromatinStateProbe(candi.model.d_model, 18)

    splits = chromatin_state_dataset_eic_train_test_val_split(dataset_path)
    splits["train"] = splits["train"]
    # splits["test"] = splits["test"][:1]
    splits["val"] = splits["val"]
    
    def prepare_data(split, chrs, num_regions):
        chromatin_state_data = {}
        # Process each chromosome
        for chr in chrs:
            cs_data = {}

            # Load chromatin state data for each cell type in training split
            for pair in splits[split]:
                bios_name = pair['biosample']
                cs_name = pair['chromatin_state']
                cs_dir = os.path.join(dataset_path, "chromatin_state_annotations", cs_name)
                parsed_dirs = [d for d in os.listdir(cs_dir) if d.startswith(f'parsed{resolution}_')]

                for idx, parsed_cs in enumerate(parsed_dirs):
                    # print(parsed_cs)
                    chr_cs = load_region_chromatin_states(os.path.join(cs_dir, parsed_cs), chr)
                    cs_data[f"{cs_name}|{idx}"] = chr_cs
            
            # Convert to numpy array for easier processing
            cs_matrix = []
            cell_types = list(cs_data.keys())
            for ct in cell_types:
                cs_matrix.append(cs_data[ct])

            cs_matrix=np.array(cs_matrix)

            # Find valid columns (no None values)
            valid_cols = ~np.any(cs_matrix == None, axis=0)
            
            # Find valid indices (where no None values exist)
            valid_indices = np.where(valid_cols)[0]

            # Calculate number of regions needed per chromosome
            regions_per_chr = num_regions // len(chrs)
            min_distance = candi.model.l1 * 25   # minimum distance between regions

            # Randomly select indices from valid indices
            if len(valid_indices) < regions_per_chr:
                print(f"Warning: Only {len(valid_indices)} valid regions available for {chr}, less than requested {regions_per_chr}")
                top_indices = valid_indices
            else:
                # Randomly select regions_per_chr indices
                top_indices = np.random.choice(valid_indices, size=regions_per_chr, replace=False)

            # Sort indices for consistency
            top_indices = np.sort(top_indices)

            # Store selected regions and their coordinates
            selected_regions = []
            for idx in top_indices:

                offset = np.random.randint(0, min_distance)
                start = max(0, ((idx * resolution) - offset))
                end = start + (candi.model.l1 * 25)

                if end >= candi.chr_sizes[chr]:
                    continue
                
                region_info = {
                    'chr': chr,
                    'start': start,
                    'end': end
                }
                selected_regions.append(region_info)

            print(f"Number of selected regions: {len(selected_regions)} from chr {chr}")

            chromatin_state_data[chr] = {}  # chr : cell_type : [chromosome, start_pos, end_pos, chromatin_state_array]
            for region in selected_regions:
                
                # print(f"CS data length: {len(cs_data[ct])}, Region start: {region['start']}, Region end: {region['end']}, "
                #     f"Region size: {region['end'] - region['start']}, Bins: {(region['end'] - region['start']) // resolution}")

                for ct in cell_types:
                    if ct.split("|")[0] not in chromatin_state_data[chr]:
                        chromatin_state_data[chr][ct.split("|")[0]] = []

                    chromatin_state_data[chr][ct.split("|")[0]].append([
                        region['chr'], region['start'], region['end'], 
                        cs_data[ct][
                            ((region['start'])//resolution):
                            ((region['end'])//resolution)]
                        ])
        
        for chr in chrs:
            candi.chr = chr
            # Load chromatin state data for each cell type in training split
            for pair in splits[split]:
                bios_name = pair['biosample']
                cs_name = pair['chromatin_state']
                X, seq, mX = candi.load_encoder_input_bios(bios_name, x_dsf=1)

                X = X.reshape(-1, X.shape[-1])
                seq = seq.reshape(-1, seq.shape[-1])
                mX = mX[0]

                for idx, region in enumerate(chromatin_state_data[chr][cs_name]):
                    start = region[1] // 25
                    end = region[2] // 25

                    # Move input tensors to the same device as the model
                    x_input = X[start:end].unsqueeze(0).float().to(candi.device)
                    seq_input = seq[region[1]:region[2]].float().unsqueeze(0).to(candi.device)
                    mx_input = mX.unsqueeze(0).float().to(candi.device)

                    with torch.no_grad():
                        try:
                            z = candi.model.encode(x_input, seq_input, mx_input)
                        except:
                            print(start, end, end - start)
                            print(region[1], region[2], region[2] - region[1])
                            print(x_input.shapes, seq_input.shapes, mx_input.shapes)

                    z = z.cpu()

                    chromatin_state_data[chr][cs_name][idx] = (region, z)
                
                del X, seq, mX
                gc.collect()

        return chromatin_state_data

    # structure ->  chr : cell_type : ([chromosome, start_pos, end_pos, chromatin_state_array], z_tensor)
    
    train_chromatin_state_data = prepare_data("train", train_chrs, num_train_regions)
    Z_train = [] 
    Y_train = []
    for chr in train_chromatin_state_data.keys():
        for ct in train_chromatin_state_data[chr].keys():
            for region, z in train_chromatin_state_data[chr][ct]:
                z = z.squeeze(0)
                annots = region[3]
                for bin in range(len(region)):
                    label = annots[bin]
                    latent_vector = z[bin]

                    if label is not None:
                        Z_train.append(latent_vector)
                        Y_train.append(label)
    
    # Convert lists to tensors first since Z contains torch tensors
    Z_train = np.stack(Z_train)
    Y_train = np.array(Y_train)

    val_chromatin_state_data = prepare_data("val", train_chrs, num_val_regions)
    Z_val = [] 
    Y_val = []
    for chr in val_chromatin_state_data.keys():
        for ct in val_chromatin_state_data[chr].keys():
            for region, z in val_chromatin_state_data[chr][ct]:
                z = z.squeeze(0)
                annots = region[3]
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
    #             z = z.squeeze(0)
    #             annots = region[3]
    #             for bin in range(len(region)):
    #                 label = annots[bin]
    #                 latent_vector = z[bin]

    #                 if label is not None:
    #                     Z_test.append(latent_vector)
    #                     Y_test.append(label)
    
    # # Convert lists to tensors first since Z contains torch tensors
    # Z_test = np.stack(Z_test)
    # Y_test = np.array(Y_test)


    probe.train_loop(Z_train, Y_train, Z_val, Y_val, num_epochs=2000, learning_rate=0.005, batch_size=200)

    """
    chromatin_state_data = {}
    for chr in chrs:
        cs data = {}
        for members of the train split:
            chr_cs = get_chromatin_state_dataset(for the whole length of that chr)
            cs data[celltype name] = chr_cs

        all chr_cs in the cs data should have the same length. 
        find regions with most variability across different celltypes i.e. highest entropy across celltypes
        select top num_regions // len(chrs) of them 
        find their corresponding coordinates : chr, start = chr_cs_index * resolution end = (chr_cs_index * resolution )+ resolution

        chromatin_state_data[name of the celltype and coordinates as a list] = corresponding chromatin state 
    """ 


"""
for all cellypes with chromatin state annotations:
    match the input data to CANDI with its corresponding chromatin state annotation

for all chromosomes other than chr21:
    select num_regions random regions

for each epoch:
    for batch in range(0, num_regions, batch_size):
        create a batch of inputs to CANDI and corresponding chromatin state annotations (batch_size regions)
        for each region:
            retrieve chromatin state annotation for that region
            retrive the input data to CANDI for that region
    
        for that batch, get the latent representations from CANDI
        use latent representations as input to probe
        train probe using cross entropy loss on chromatin state annotations
"""

def test():
    # model_path = "models/CANDIeic_DNA_random_mask_test_20241125144433_params45093285.pt"
    # hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_test_20241125144433_params45093285.pkl"
    # eic = True
    # bios_name = "ENCBS674MPN"


    model_path = "models/CANDIfull_DNA_random_mask_test_model_checkpoint_epoch0.pth"
    hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_test_20241125150741_params45093285.pkl"
    eic = False
    bios_name = "upper_lobe_of_left_lung_nonrep"

    dataset_path = "/project/compbio-lab/encode_data/"
    output_dir = "output"
    DNA = True
    
    dsf = 1

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25, eic=eic)
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading biosample data for DNA analysis...")
    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf, fill_in_y_prompt=False)
    else:
        print("Loading biosample data for non-DNA analysis...")
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf, fill_in_y_prompt=False)
        seq = None

    # print("Evaluating leave-one-out for initial analysis...")
    # start_time = time.time()
    # metrics = CANDIP.evaluate_leave_one_out(X, mX, mY, avX, Y, P, seq=seq, crop_edges=True)
    # end_time = time.time()
    # print(f"Evaluation with crop_edges=True took {end_time - start_time:.2f} seconds.")

    print("Generating predictions with both pred and pred_cropped methods...")
    start_time_pred = time.time()
    n_pred, p_pred, mu_pred, var_pred, Z_pred = CANDIP.pred(X, mX, mY, avX, seq=seq, imp_target=[])
    end_time_pred = time.time()
    print(f"Prediction with pred method took {end_time_pred - start_time_pred:.2f} seconds.")

    start_time_pred_cropped = time.time()
    n_pred_cropped, p_pred_cropped, mu_pred_cropped, var_pred_cropped, Z_pred_cropped = CANDIP.pred_cropped(X, mX, mY, avX, imp_target=[], seq=seq, crop_percent=0.1)
    end_time_pred_cropped = time.time()
    print(f"Prediction with pred_cropped method took {end_time_pred_cropped - start_time_pred_cropped:.2f} seconds.")
    
    # Compare latent representations
    print("\nComparing latent representations...")
    
    # Normalize vectors for cosine distance
    Z_pred_norm = Z_pred / torch.norm(Z_pred, dim=1, keepdim=True)
    Z_pred_cropped_norm = Z_pred_cropped / torch.norm(Z_pred_cropped, dim=1, keepdim=True)
    
    # Calculate cosine distances (1 - cosine similarity)
    cosine_dist = 1 - torch.sum(Z_pred_norm * Z_pred_cropped_norm, dim=1)
    
    # Calculate euclidean distances
    euclidean_dist = torch.norm(Z_pred - Z_pred_cropped, dim=1)
    
    # Define thresholds
    cosine_thresholds = [0.001, 0.01, 0.1]
    euclidean_thresholds = [0.1, 1.0, 10.0]
    
    print("\nCosine Distance Analysis:")
    for threshold in cosine_thresholds:
        fraction = (cosine_dist > threshold).float().mean().item()
        print(f"Fraction of positions with cosine distance > {threshold:.3f}: {fraction:.4f}")
    
    print("\nEuclidean Distance Analysis:")
    for threshold in euclidean_thresholds:
        fraction = (euclidean_dist > threshold).float().mean().item()
        print(f"Fraction of positions with euclidean distance > {threshold:.1f}: {fraction:.4f}")
    
    # Print summary statistics
    print("\nDistance Statistics:")
    print(f"Cosine Distance - Mean: {cosine_dist.mean():.6f}, Std: {cosine_dist.std():.6f}")
    print(f"Euclidean Distance - Mean: {euclidean_dist.mean():.6f}, Std: {euclidean_dist.std():.6f}")



if __name__ == "__main__":
    model_path = "models/CANDIeic_DNA_random_mask_Nov25_model_checkpoint_epoch5.pth"
    hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov25_20241126160857_params45093285.pkl"
    eic = True

    train_chromatin_state_probe(model_path, hyper_parameters_path, dataset_path="/project/compbio-lab/encode_data/")

    

