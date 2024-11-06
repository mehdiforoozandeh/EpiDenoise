import torch
import pickle
import os
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
            self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5, eic=eic)

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

    def pred_cropped(self, X, mX, mY, avail, imp_target=[], seq=None, crop_percent=0.05):
        # Calculate dimensions
        crop_size = int(self.context_length * crop_percent)
        stride = self.context_length - (crop_size * 2)
        num_windows = X.shape[0]
        total_length = num_windows * self.context_length
        
        # Flatten only X and seq
        X_flat = X.view(-1, X.shape[-1])  # [total_length, feature_dim]
        if self.DNA:
            seq_flat = seq.view(-1, seq.shape[-1])  # [total_length*resolution, 4]
        
        # Initialize output tensors
        n = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        p = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        mu = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        var = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        
        # Coverage tracking tensor
        coverage_mask = torch.zeros(total_length, dtype=torch.bool, device="cpu")
        
        # Process sliding windows
        for i in range(0, total_length, stride):

            if i + self.context_length >= total_length:
                i = total_length - self.context_length

            # Extract window
            window_end = i + self.context_length
            x_window = X_flat[i:window_end].unsqueeze(0)  # [1, context_length, feature_dim]
            
            # Verify that all rows in metadata tensors are identical
            if not (mX == mX[0]).all():
                raise ValueError("Not all rows in mX are identical")
            if not (mY == mY[0]).all():
                raise ValueError("Not all rows in mY are identical")
            if not (avail == avail[0]).all():
                raise ValueError("Not all rows in avail are identical")

            # Use original metadata tensors directly
            mx_window = mX[0].unsqueeze(0)  # Already in shape [1, context_length, feature_dim]
            my_window = mY[0].unsqueeze(0)  # Already in shape [1, context_length, feature_dim]
            avail_window = avail[0].unsqueeze(0)  # Already in shape [1, feature_dim]
            
            if self.DNA:
                seq_start = i * self.resolution
                seq_end = window_end * self.resolution
                seq_window = seq_flat[seq_start:seq_end].unsqueeze(0)
            
            with torch.no_grad():
                if self.DNA:
                    outputs = self.model(
                        x_window.float().to(self.device),
                        seq_window.to(self.device),
                        mx_window.to(self.device),
                        my_window.to(self.device),
                        avail_window.to(self.device),
                        return_z=True
                    )
                else:
                    outputs = self.model(
                        x_window.float().to(self.device),
                        mx_window.to(self.device),
                        my_window.to(self.device),
                        avail_window.to(self.device),
                        return_z=True
                    )
                
                outputs_n, outputs_p, outputs_mu, outputs_var, _ = outputs
            
            # Determine which part of predictions to keep
            if i == 0:  # First window
                start_idx = 0
                end_idx = self.context_length - crop_size

            elif i + self.context_length >= total_length:  # Last window
                start_idx = crop_size
                end_idx = self.context_length

            else:  # Middle windows
                start_idx = crop_size
                end_idx = self.context_length - crop_size
                
            # Update predictions
            target_start = i + start_idx
            target_end = i + end_idx
            
            if torch.any(n[target_start:target_end, :] != 0):
                print(f"{target_start} Fraction of positions being overwritten in n: {torch.sum(n[target_start:target_end, :] != 0).item() / n[target_start:target_end, :].numel()}")
            n[target_start:target_end, :] = outputs_n[0, start_idx:end_idx, :].cpu()
            
            if torch.any(p[target_start:target_end, :] != 0):
                print(f"{target_start} Fraction of positions being overwritten in p: {torch.sum(p[target_start:target_end, :] != 0).item() / p[target_start:target_end, :].numel()}")
            p[target_start:target_end, :] = outputs_p[0, start_idx:end_idx, :].cpu()
            
            if torch.any(mu[target_start:target_end, :] != 0):
                print(f"{target_start} Fraction of positions being overwritten in mu: {torch.sum(mu[target_start:target_end, :] != 0).item() / mu[target_start:target_end, :].numel()}")
            mu[target_start:target_end, :] = outputs_mu[0, start_idx:end_idx, :].cpu()
            
            if torch.any(var[target_start:target_end, :] != 0):
                print(f"{target_start} Fraction of positions being overwritten in var: {torch.sum(var[target_start:target_end, :] != 0).item() / var[target_start:target_end, :].numel()}")
            var[target_start:target_end, :] = outputs_var[0, start_idx:end_idx, :].cpu()
        
            # Update coverage mask
            coverage_mask[target_start:target_end] = True
            
            del outputs
            torch.cuda.empty_cache()
        
        # Verify complete coverage
        if not coverage_mask.all():
            print(f"Missing predictions for positions: {torch.where(~coverage_mask)[0]}")
            raise ValueError("Missing predictions")
        
        # Reshape Z to match the number of windows
        num_output_windows = (total_length - crop_size) // stride
        Z = torch.empty((num_output_windows, self.model.latent_dim), device="cpu")
        
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

    def compare_prediction_methods(self, X, mX, mY, avX, Y, seq=None):
        available_indices = torch.where(avX[0, :] == 1)[0]
        
        # Initialize tensors for both methods
        n_imp_regular = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        p_imp_regular = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        n_imp_cropped = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        p_imp_cropped = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)

        # Perform leave-one-out validation for both methods
        for leave_one_out in available_indices:
            # Regular predictions
            n_reg, p_reg, _, _, _ = self.pred(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            n_imp_regular[:, leave_one_out] = n_reg[:, leave_one_out]
            p_imp_regular[:, leave_one_out] = p_reg[:, leave_one_out]
            
            # Cropped predictions
            n_crop, p_crop, _, _, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            n_imp_cropped[:, leave_one_out] = n_crop[:, leave_one_out]
            p_imp_cropped[:, leave_one_out] = p_crop[:, leave_one_out]
            
            print(f"Completed imputations for feature #{leave_one_out+1}")

        # Get full predictions without masking
        n_ups_regular, p_ups_regular, _, _, _ = self.pred(X, mX, mY, avX, imp_target=[], seq=seq)
        n_ups_cropped, p_ups_cropped, _, _, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[], seq=seq)
        
        # Reshape tensors
        def reshape_predictions(n_imp, p_imp, n_ups, p_ups, Y):
            # p_imp = p_imp.view(-1, p_imp.shape[-1])
            # n_imp = n_imp.view(-1, n_imp.shape[-1])
            # p_ups = p_ups.view(-1, p_ups.shape[-1])
            # n_ups = n_ups.view(-1, n_ups.shape[-1])
            Y = Y.view(-1, Y.shape[-1])
            
            return NegativeBinomial(p_imp, n_imp), NegativeBinomial(p_ups, n_ups), Y

        # Create distributions
        imp_dist_regular, ups_dist_regular, Y_flat = reshape_predictions(
            n_imp_regular, p_imp_regular, n_ups_regular, p_ups_regular, Y)
        imp_dist_cropped, ups_dist_cropped, _ = reshape_predictions(
            n_imp_cropped, p_imp_cropped, n_ups_cropped, p_ups_cropped, Y)

        # Compare predictions with true values
        def evaluate_predictions(imp_dist, ups_dist, Y, method_name):
            imp_mean = imp_dist.mean()
            ups_mean = ups_dist.mean()
            
            metrics = {}
            for idx in available_indices:
                # Calculate metrics for each feature
                true_vals = Y[:, idx]
                imp_vals = imp_mean[:, idx]
                ups_vals = ups_mean[:, idx]
                
                # Calculate correlations
                imp_pearson = stats.pearsonr(true_vals, imp_vals)[0]
                ups_pearson = stats.pearsonr(true_vals, ups_vals)[0]
                
                # Calculate MSE
                imp_mse = torch.mean((true_vals - imp_vals) ** 2).item()
                ups_mse = torch.mean((true_vals - ups_vals) ** 2).item()
                
                metrics[idx] = {
                    'imp_pearson': imp_pearson,
                    'ups_pearson': ups_pearson,
                    'imp_mse': imp_mse,
                    'ups_mse': ups_mse
                }
                
            print(f"\nResults for {method_name}:")
            print("Feature | Imp Pearson | Ups Pearson | Imp MSE | Ups MSE")
            print("-" * 60)
            for idx in available_indices:
                m = metrics[idx]
                print(f"{idx:7d} | {m['imp_pearson']:10.4f} | {m['ups_pearson']:10.4f} | "
                      f"{m['imp_mse']:7.4f} | {m['ups_mse']:7.4f}")
            
            return metrics

        # Evaluate both methods
        metrics_regular = evaluate_predictions(imp_dist_regular, ups_dist_regular, Y_flat, "Regular Prediction")
        metrics_cropped = evaluate_predictions(imp_dist_cropped, ups_dist_cropped, Y_flat, "Cropped Prediction")
        
        # Compare methods directly
        print("\nMethod Comparison (Cropped vs Regular):")
        print("Feature | Imp Pearson Diff | Ups Pearson Diff | Imp MSE Diff | Ups MSE Diff")
        print("-" * 70)
        for idx in available_indices:
            imp_pearson_diff = metrics_cropped[idx]['imp_pearson'] - metrics_regular[idx]['imp_pearson']
            ups_pearson_diff = metrics_cropped[idx]['ups_pearson'] - metrics_regular[idx]['ups_pearson']
            imp_mse_diff = metrics_cropped[idx]['imp_mse'] - metrics_regular[idx]['imp_mse']
            ups_mse_diff = metrics_cropped[idx]['ups_mse'] - metrics_regular[idx]['ups_mse']
            
            print(f"{idx:7d} | {imp_pearson_diff:15.4f} | {ups_pearson_diff:15.4f} | "
                  f"{imp_mse_diff:12.4f} | {ups_mse_diff:12.4f}")

        return metrics_regular, metrics_cropped


if __name__ == "__main__":
    model_path = "models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth"
    hyper_parameters_path = "models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl"
    dataset_path = "/project/compbio-lab/encode_data/"
    output_dir = "output"
    number_of_states = 10
    DNA = True
    bios_name = "ENCBS674MPN"
    dsf = 1


    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)

    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
    else:
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        seq = None
        
    print(X.shape, Y.shape, P.shape)
    print(mX.shape, mY.shape, avX.shape, avY.shape)

    n, p, mu, var, Z = CANDIP.pred_cropped(X, mX, mY, avX, seq=seq, crop_percent=0.05)

    # Get predictions from both methods
    n_regular, p_regular, mu_regular, var_regular, Z_regular = CANDIP.pred(X, mX, mY, avX, seq=seq)
    n_cropped, p_cropped, mu_cropped, var_cropped, Z_cropped = CANDIP.pred_cropped(X, mX, mY, avX, seq=seq, crop_percent=0.05)
    
    metrics_regular, metrics_cropped = CANDIP.compare_prediction_methods(X, mX, mY, avX, Y, seq=seq)
    print(metrics_regular)
    print(metrics_cropped)
    exit()
    # Compare predictions
    def compare_predictions(n1, p1, n2, p2, name):
        # Create NegativeBinomial distributions

        if name == "NegativeBinomial mean":
            nb1 = NegativeBinomial(p1, n1)
            nb2 = NegativeBinomial(p2, n2)
            
            # Get means
            pred1 = nb1.mean()
            pred2 = nb2.mean()
        elif name == "Gaussian mean":
            pred1 = torch.sinh(n1)
            pred2 = torch.sinh(n2)
        
        # Calculate differences
        differences = pred1 - pred2
        
        # Calculate statistics per feature
        mean_diff = differences.mean(dim=0)
        var_diff = differences.var(dim=0)
        
        # Calculate relative metrics
        relative_diff = mean_diff / pred1.mean(dim=0)  # Relative difference compared to original values
        
        # Calculate Cohen's d effect size
        pooled_std = torch.sqrt((pred1.var(dim=0) + pred2.var(dim=0)) / 2)
        cohens_d = mean_diff / pooled_std
        
        # Calculate normalized RMSE
        rmse = torch.sqrt(((pred1 - pred2) ** 2).mean(dim=0))
        nrmse = rmse / (pred1.max(dim=0)[0] - pred1.min(dim=0)[0])  # Normalized by range
        
        # Calculate correlations and R²
        pearson_corrs = []
        spearman_corrs = []
        r2_scores = []
        for i in range(pred1.shape[1]):
            # Convert to numpy for scipy stats
            p1 = pred1[:, i].numpy()
            p2 = pred2[:, i].numpy()
            
            # Calculate Pearson correlation
            pearson_corr = stats.pearsonr(p1, p2)[0]
            pearson_corrs.append(pearson_corr)
            
            # Calculate Spearman correlation
            spearman_corr = stats.spearmanr(p1, p2)[0]
            spearman_corrs.append(spearman_corr)
            
            # Calculate R² score
            ss_res = np.sum((p1 - p2) ** 2)
            ss_tot = np.sum((p1 - np.mean(p1)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)
        
        print(f"\n{name} differences per feature:")
        print("Feature | Mean Diff | Var Diff | Rel Diff % | Cohen's d | NRMSE | Pearson | Spearman | R²")
        print("-" * 105)
        for i in range(len(mean_diff)):
            print(f"{i:7d} | {mean_diff[i]:9.2e} | {var_diff[i]:9.2e} | "
                  f"{relative_diff[i]*100:9.2f} | {cohens_d[i]:9.2f} | {nrmse[i]:9.2f} | "
                  f"{pearson_corrs[i]:7.4f} | {spearman_corrs[i]:8.4f} | {r2_scores[i]:6.4f}")

    compare_predictions(n_regular, p_regular, n_cropped, p_cropped, "NegativeBinomial mean")
    compare_predictions(mu_regular, var_regular, mu_cropped, var_cropped, "Gaussian mean")

    

    

    
    
