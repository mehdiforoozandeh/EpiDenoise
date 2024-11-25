import torch
import pickle
import os, time
from CANDI import *
from scipy import stats
import numpy as np


# sequence clustering
class CANDIPredictor:
    def __init__(self, model, hyper_parameters_path, number_of_states, 
        split="test", DNA=False, eic=True, chr="chr21", resolution=25, context_length=1600,
        savedir="models/output/", data_path="/project/compbio-lab/encode_data/"):

        self.model = model
        self.number_of_states = number_of_states
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
                if chr_name == self.chr:
                    self.chr_sizes[chr_name] = int(chr_size)
                    break

        self.context_length = self.hyper_parameters["context_length"]
        self.batch_size = self.hyper_parameters["batch_size"]
        self.token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
            }

    def load_bios(self, bios_name, x_dsf, y_dsf=1, fill_in_y_prompt=False):
        # Load biosample data
        
        print(f"getting bios vals for {bios_name}")

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


def test():
    # model_path = "models/CANDIeic_DNA_random_mask_test_20241125144433_params45093285.pt"
    # hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_test_20241125144433_params45093285.pkl"
    # eic = True
    # bios_name = "ENCBS674MPN"


    model_path = "models/CANDIfull_DNA_random_mask_test_model_checkpoint_epoch1.pth"
    hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_test_20241125144500_params45093285.pkl"
    eic = False
    bios_name = "upper_lobe_of_left_lung_nonrep"

    dataset_path = "/project/compbio-lab/encode_data/"
    output_dir = "output"
    number_of_states = 10
    DNA = True
    
    dsf = 1

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25, eic=eic)
    
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
    test()
    
