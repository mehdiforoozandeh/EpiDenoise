from _utils import *    
from data import * 

import torch, math, random, time, json, os, pickle, sys, gc
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import datetime
from scipy.stats import nbinom
import imageio.v2 as imageio
from io import BytesIO
from torchinfo import summary

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class MONITOR_VALIDATION(object):
    def __init__(
        self, data_path, context_length, batch_size, must_have_chr_access=False,
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
            bios_min_exp_avail_threshold=3, eic=eic)

        # self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}
        
        self.expnames = list(self.dataset.aliases["experiment_aliases"].keys())
        self.mark_dict = {i: self.expnames[i] for i in range(len(self.expnames))}

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

        self.gaus_nll = torch.nn.GaussianNLLLoss(reduction="mean", full=True)
        self.nbin_nll = negative_binomial_loss

        self.chr_sizes = {}
        self.metrics = METRICS()
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

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
        
        # temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        # X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        # del temp_x, temp_mx
        
        # temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        # Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
        # if fill_in_y_prompt:
        #     mY = self.dataset.fill_in_y_prompt(mY)
        # del temp_y, temp_my

        # temp_p = self.dataset.load_bios_BW(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        # P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
        # assert (avlP == avY).all(), "avlP and avY do not match"
        # del temp_p

        # num_rows = (X.shape[0] // self.context_length) * self.context_length
        # X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

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
            adjusted_length = max(1, int(segment_length // self.context_length)) * self.context_length 
            adjusted_end = start + adjusted_length

            temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", start* self.resolution, adjusted_end* self.resolution ], x_dsf)
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", start* self.resolution, adjusted_end* self.resolution ], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_p = self.dataset.load_bios_BW(bios_name, ["chr21", start* self.resolution, adjusted_end*self.resolution ], y_dsf)
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            assert (avlP == avY).all(), "avlP and avY do not match"
            del temp_p

            subsets_X.append(X) #[start:adjusted_end, :])
            subsets_Y.append(Y) #[start:adjusted_end, :])
            subsets_P.append(P) #[start:adjusted_end, :])

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
        # temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        # X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        # del temp_x, temp_mx
        
        # # Load and process Y (target)
        # temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        # Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
        # if fill_in_y_prompt:
        #     mY = self.dataset.fill_in_y_prompt(mY)
        # del temp_y, temp_my

        # # Load and process P (probability)
        # temp_py = self.dataset.load_bios_BW(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        # temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        # temp_p = {**temp_py, **temp_px}

        # P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
        # del temp_py, temp_px, temp_p

        # num_rows = (X.shape[0] // self.context_length) * self.context_length
        # X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

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
            adjusted_length = max(1, int(segment_length // self.context_length)) * self.context_length 
            adjusted_end = start + adjusted_length

            temp_x, temp_mx = self.dataset.load_bios(
                bios_name.replace("V_", "T_"), ["chr21", start*self.resolution, adjusted_end*self.resolution], x_dsf)

            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(
                bios_name, ["chr21", start*self.resolution, adjusted_end*self.resolution], y_dsf)

            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            # Load and process P (probability)
            temp_py = self.dataset.load_bios_BW(
                bios_name, ["chr21", start*self.resolution, adjusted_end*self.resolution], y_dsf)
            temp_px = self.dataset.load_bios_BW(
                bios_name.replace("V_", "T_"), ["chr21", start*self.resolution, adjusted_end*self.resolution], x_dsf)
            temp_p = {**temp_py, **temp_px}

            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            del temp_py, temp_px, temp_p

            subsets_X.append(X) #[start:adjusted_end, :])
            subsets_Y.append(Y) #[start:adjusted_end, :])
            subsets_P.append(P) #[start:adjusted_end, :])

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

                        pred_n = imp_count_dist.n[:, j].numpy()
                        pred_p = imp_count_dist.p[:, j].numpy()
                        pred_mu = imp_pval_dist.mu[:, j].numpy()
                        pred_var = imp_pval_dist.var[:, j].numpy()
                        
                    elif comparison == "upsampled":
                        pred_count = ups_mean[:, j].numpy()
                        pred_pval = ups_gaussian_mean[:, j].numpy()
                        
                        pred_n = ups_count_dist.n[:, j].numpy()
                        pred_p = ups_count_dist.p[:, j].numpy()
                        pred_mu = ups_pval_dist.mu[:, j].numpy()
                        pred_var = ups_pval_dist.var[:, j].numpy()

                    target_count = Y[:, j].numpy()
                    target_pval = P[:, j].numpy()

                    metrics = {
                        'bios':bios_name,
                        'feature':  self.expnames[j],
                        'comparison': comparison,
                        'available assays': len(availability),

                        'MSE_count': self.metrics.mse(target_count, pred_count),
                        'Pearson_count': self.metrics.pearson(target_count, pred_count),
                        'Spearman_count': self.metrics.spearman(target_count, pred_count),
                        'r2_count': self.metrics.r2(target_count, pred_count),
                        'loss_count': self.nbin_nll(
                            torch.Tensor(target_count), 
                            torch.Tensor(pred_n), 
                            torch.Tensor(pred_p)
                            ).mean().item(),

                        'MSE_pval': self.metrics.mse(target_pval, pred_pval),
                        'Pearson_pval': self.metrics.pearson(target_pval, pred_pval),
                        'Spearman_pval': self.metrics.spearman(target_pval, pred_pval),
                        'r2_pval': self.metrics.r2(target_pval, pred_pval),
                        'loss_pval': self.gaus_nll(
                            torch.Tensor(pred_mu), 
                            torch.Tensor(target_pval), 
                            torch.Tensor(pred_var)
                            ).item()
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

            pred_n = ups_count_dist.n[:, j].numpy()
            pred_p = ups_count_dist.p[:, j].numpy()
            pred_mu = ups_pval_dist.mu[:, j].numpy()
            pred_var = ups_pval_dist.var[:, j].numpy()

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
                'feature': self.expnames[j],
                'comparison': comparison,
                'available assays': len(available_X_indices),

                'MSE_count': self.metrics.mse(target_count, pred_count),
                'Pearson_count': self.metrics.pearson(target_count, pred_count),
                'Spearman_count': self.metrics.spearman(target_count, pred_count),
                'r2_count': self.metrics.r2(target_count, pred_count),
                'loss_count': self.nbin_nll(
                    torch.Tensor(target_count), 
                    torch.Tensor(pred_n), 
                    torch.Tensor(pred_p)
                    ).mean().item(),

                'MSE_pval': self.metrics.mse(target_pval, pred_pval),
                'Pearson_pval': self.metrics.pearson(target_pval, pred_pval),
                'Spearman_pval': self.metrics.spearman(target_pval, pred_pval),
                'r2_pval': self.metrics.r2(target_pval, pred_pval),
                'loss_pval': self.gaus_nll(
                    torch.Tensor(pred_mu), 
                    torch.Tensor(target_pval), 
                    torch.Tensor(pred_var)
                    ).item()
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
                #     pass
            else:
                try:
                    imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, availability = self.get_bios_frame(
                        bios_name, x_dsf=x_dsf, y_dsf=y_dsf)

                    full_res += self.get_metric(imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, availability)
                except:
                    pass

        # del self.model
        # del model

        if len(full_res) == 0:
            print(f"No validation results found.")
            return "", {}
        
        df = pd.DataFrame(full_res)

        # Separate the data based on comparison type
        imputed_df = df[df['comparison'] == 'imputed']
        upsampled_df = df[df['comparison'] == 'upsampled']

        # Function to calculate mean, min, and max for a given metric
        def calculate_stats(df, metric):
            return df[metric].median(), df[metric].min(), df[metric].max()

        # Imputed statistics for count metrics
        imp_mse_count_stats = calculate_stats(imputed_df, 'MSE_count')
        imp_pearson_count_stats = calculate_stats(imputed_df, 'Pearson_count')
        imp_spearman_count_stats = calculate_stats(imputed_df, 'Spearman_count')
        imp_r2_count_stats = calculate_stats(imputed_df, 'r2_count')
        imp_loss_count_stats = calculate_stats(imputed_df, 'loss_count')

        # Imputed statistics for p-value metrics
        imp_mse_pval_stats = calculate_stats(imputed_df, 'MSE_pval')
        imp_pearson_pval_stats = calculate_stats(imputed_df, 'Pearson_pval')
        imp_spearman_pval_stats = calculate_stats(imputed_df, 'Spearman_pval')
        imp_r2_pval_stats = calculate_stats(imputed_df, 'r2_pval')
        imp_loss_pval_stats = calculate_stats(imputed_df, 'loss_pval')

        # Upsampled statistics for count metrics
        ups_mse_count_stats = calculate_stats(upsampled_df, 'MSE_count')
        ups_pearson_count_stats = calculate_stats(upsampled_df, 'Pearson_count')
        ups_spearman_count_stats = calculate_stats(upsampled_df, 'Spearman_count')
        ups_r2_count_stats = calculate_stats(upsampled_df, 'r2_count')
        ups_loss_count_stats = calculate_stats(upsampled_df, 'loss_count')

        # Upsampled statistics for p-value metrics
        ups_mse_pval_stats = calculate_stats(upsampled_df, 'MSE_pval')
        ups_pearson_pval_stats = calculate_stats(upsampled_df, 'Pearson_pval')
        ups_spearman_pval_stats = calculate_stats(upsampled_df, 'Spearman_pval')
        ups_r2_pval_stats = calculate_stats(upsampled_df, 'r2_pval')
        ups_loss_pval_stats = calculate_stats(upsampled_df, 'loss_pval')

        elapsed_time = datetime.datetime.now() - t0
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the updated print statement
        print_statement = f"""
        Took {int(minutes)}:{int(seconds)}
        For Imputed Counts:
        - MSE_count: median={imp_mse_count_stats[0]:.2f}, min={imp_mse_count_stats[1]:.2f}, max={imp_mse_count_stats[2]:.2f}
        - PCC_count: median={imp_pearson_count_stats[0]:.2f}, min={imp_pearson_count_stats[1]:.2f}, max={imp_pearson_count_stats[2]:.2f}
        - SRCC_count: median={imp_spearman_count_stats[0]:.2f}, min={imp_spearman_count_stats[1]:.2f}, max={imp_spearman_count_stats[2]:.2f}
        - R2_count: median={imp_r2_count_stats[0]:.2f}, min={imp_r2_count_stats[1]:.2f}, max={imp_r2_count_stats[2]:.2f}
        - loss_count: median={imp_loss_count_stats[0]:.2f}, min={imp_loss_count_stats[1]:.2f}, max={imp_loss_count_stats[2]:.2f}

        For Imputed P-values:
        - MSE_pval: median={imp_mse_pval_stats[0]:.2f}, min={imp_mse_pval_stats[1]:.2f}, max={imp_mse_pval_stats[2]:.2f}
        - PCC_pval: median={imp_pearson_pval_stats[0]:.2f}, min={imp_pearson_pval_stats[1]:.2f}, max={imp_pearson_pval_stats[2]:.2f}
        - SRCC_pval: median={imp_spearman_pval_stats[0]:.2f}, min={imp_spearman_pval_stats[1]:.2f}, max={imp_spearman_pval_stats[2]:.2f}
        - R2_pval: median={imp_r2_pval_stats[0]:.2f}, min={imp_r2_pval_stats[1]:.2f}, max={imp_r2_pval_stats[2]:.2f}
        - loss_pval: median={imp_loss_pval_stats[0]:.2f}, min={imp_loss_pval_stats[1]:.2f}, max={imp_loss_pval_stats[2]:.2f}

        For Upsampled Counts:
        - MSE_count: median={ups_mse_count_stats[0]:.2f}, min={ups_mse_count_stats[1]:.2f}, max={ups_mse_count_stats[2]:.2f}
        - PCC_count: median={ups_pearson_count_stats[0]:.2f}, min={ups_pearson_count_stats[1]:.2f}, max={ups_pearson_count_stats[2]:.2f}
        - SRCC_count: median={ups_spearman_count_stats[0]:.2f}, min={ups_spearman_count_stats[1]:.2f}, max={ups_spearman_count_stats[2]:.2f}
        - R2_count: median={ups_r2_count_stats[0]:.2f}, min={ups_r2_count_stats[1]:.2f}, max={ups_r2_count_stats[2]:.2f}
        - loss_count: median={ups_loss_count_stats[0]:.2f}, min={ups_loss_count_stats[1]:.2f}, max={ups_loss_count_stats[2]:.2f}

        For Upsampled P-values:
        - MSE_pval: median={ups_mse_pval_stats[0]:.2f}, min={ups_mse_pval_stats[1]:.2f}, max={ups_mse_pval_stats[2]:.2f}
        - PCC_pval: median={ups_pearson_pval_stats[0]:.2f}, min={ups_pearson_pval_stats[1]:.2f}, max={ups_pearson_pval_stats[2]:.2f}
        - SRCC_pval: median={ups_spearman_pval_stats[0]:.2f}, min={ups_spearman_pval_stats[1]:.2f}, max={ups_spearman_pval_stats[2]:.2f}
        - R2_pval: median={ups_r2_pval_stats[0]:.2f}, min={ups_r2_pval_stats[1]:.2f}, max={ups_r2_pval_stats[2]:.2f}
        - loss_pval: median={ups_loss_pval_stats[0]:.2f}, min={ups_loss_pval_stats[1]:.2f}, max={ups_loss_pval_stats[2]:.2f}
        """

        metrics_dict = {
            "imputed_counts": {
                "MSE_count": {"median": imp_mse_count_stats[0], "min": imp_mse_count_stats[1], "max": imp_mse_count_stats[2]},
                "PCC_count": {"median": imp_pearson_count_stats[0], "min": imp_pearson_count_stats[1], "max": imp_pearson_count_stats[2]},
                "SRCC_count": {"median": imp_spearman_count_stats[0], "min": imp_spearman_count_stats[1], "max": imp_spearman_count_stats[2]},
                "R2_count": {"median": imp_r2_count_stats[0], "min": imp_r2_count_stats[1], "max": imp_r2_count_stats[2]},
                "loss_count": {"median": imp_loss_count_stats[0], "min": imp_loss_count_stats[1], "max": imp_loss_count_stats[2]},
            },
            "imputed_pvals": {
                "MSE_pval": {"median": imp_mse_pval_stats[0], "min": imp_mse_pval_stats[1], "max": imp_mse_pval_stats[2]},
                "PCC_pval": {"median": imp_pearson_pval_stats[0], "min": imp_pearson_pval_stats[1], "max": imp_pearson_pval_stats[2]},
                "SRCC_pval": {"median": imp_spearman_pval_stats[0], "min": imp_spearman_pval_stats[1], "max": imp_spearman_pval_stats[2]},
                "R2_pval": {"median": imp_r2_pval_stats[0], "min": imp_r2_pval_stats[1], "max": imp_r2_pval_stats[2]},
                "loss_pval": {"median": imp_loss_pval_stats[0], "min": imp_loss_pval_stats[1], "max": imp_loss_pval_stats[2]},
            },
            "upsampled_counts": {
                "MSE_count": {"median": ups_mse_count_stats[0], "min": ups_mse_count_stats[1], "max": ups_mse_count_stats[2]},
                "PCC_count": {"median": ups_pearson_count_stats[0], "min": ups_pearson_count_stats[1], "max": ups_pearson_count_stats[2]},
                "SRCC_count": {"median": ups_spearman_count_stats[0], "min": ups_spearman_count_stats[1], "max": ups_spearman_count_stats[2]},
                "R2_count": {"median": ups_r2_count_stats[0], "min": ups_r2_count_stats[1], "max": ups_r2_count_stats[2]},
                "loss_count": {"median": ups_loss_count_stats[0], "min": ups_loss_count_stats[1], "max": ups_loss_count_stats[2]},
            },
            "upsampled_pvals": {
                "MSE_pval": {"median": ups_mse_pval_stats[0], "min": ups_mse_pval_stats[1], "max": ups_mse_pval_stats[2]},
                "PCC_pval": {"median": ups_pearson_pval_stats[0], "min": ups_pearson_pval_stats[1], "max": ups_pearson_pval_stats[2]},
                "SRCC_pval": {"median": ups_spearman_pval_stats[0], "min": ups_spearman_pval_stats[1], "max": ups_spearman_pval_stats[2]},
                "R2_pval": {"median": ups_r2_pval_stats[0], "min": ups_r2_pval_stats[1], "max": ups_r2_pval_stats[2]},
                "loss_pval": {"median": ups_loss_pval_stats[0], "min": ups_loss_pval_stats[1], "max": ups_loss_pval_stats[2]}
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

##=========================================== Loss Functions =============================================##

class CANDI_LOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(CANDI_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction, full=True)
        self.nbin_nll = negative_binomial_loss

    def forward(self, p_pred, n_pred, mu_pred, var_pred, true_count, true_pval, obs_map, masked_map):
        ups_true_count, ups_true_pval = true_count[obs_map], true_pval[obs_map]
        ups_n_pred, ups_p_pred = n_pred[obs_map], p_pred[obs_map]
        ups_mu_pred, ups_var_pred = mu_pred[obs_map], var_pred[obs_map]

        imp_true_count, imp_true_pval = true_count[masked_map], true_pval[masked_map]
        imp_n_pred, imp_p_pred = n_pred[masked_map], p_pred[masked_map]
        imp_mu_pred, imp_var_pred = mu_pred[masked_map], var_pred[masked_map]

        observed_count_loss = self.nbin_nll(ups_true_count, ups_n_pred, ups_p_pred) 
        imputed_count_loss = self.nbin_nll(imp_true_count, imp_n_pred, imp_p_pred)

        if self.reduction == "mean":
            observed_count_loss = observed_count_loss.mean()
            imputed_count_loss = imputed_count_loss.mean()
        elif self.reduction == "sum":
            observed_count_loss = observed_count_loss.sum()
            imputed_count_loss = imputed_count_loss.sum()

        observed_pval_loss = self.gaus_nll(ups_mu_pred, ups_true_pval, ups_var_pred)
        imputed_pval_loss = self.gaus_nll(imp_mu_pred, imp_true_pval, imp_var_pred)

        observed_pval_loss = observed_pval_loss.float()
        imputed_pval_loss = imputed_pval_loss.float()
        
        return observed_count_loss, imputed_count_loss, observed_pval_loss, imputed_pval_loss

##=========================================== CANDI Architecture =============================================##

class CANDI_Decoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size=2, expansion_factor=3, num_sequencing_platforms=10, num_runtypes=4):
        super(CANDI_Decoder, self).__init__()

        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model =  self.latent_dim = self.f2

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.ymd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, num_sequencing_platforms, num_runtypes, non_linearity=False)
        self.ymd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            )

        self.deconv = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])
    
    def forward(self, src, y_metadata):
        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusion(src)
        
        src = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv:
            src = dconv(src)

        src = src.permute(0, 2, 1) # to N, L, F1

        return src    

class CANDI_DNA_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3, num_sequencing_platforms=10, num_runtypes=4):
        super(CANDI_DNA_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2

        DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers+2)]

        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True, SE=False,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size) for i in range(n_cnn_layers + 2)])

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1, SE=False,
                pool_size=pool_size) for i in range(n_cnn_layers)])
        
        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, num_sequencing_platforms, num_runtypes, non_linearity=False)

        self.fusion = nn.Sequential(
            # nn.Linear((2*self.f2), self.f2), 
            nn.Linear((2*self.f2)+metadata_embedding_dim, self.f2), 
            # nn.Linear((self.f2)+metadata_embedding_dim, self.f2), 
            nn.LayerNorm(self.f2), 

            )

        self.transformer_encoder = nn.ModuleList([
            DualAttentionEncoderBlock(self.f2, nhead, self.l2, dropout=dropout, 
                max_distance=self.l2, pos_encoding_type="relative", max_len=self.l2
                ) for _ in range(n_sab_layers)])

    def forward(self, src, seq, x_metadata):
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to N, 4, 25*L
        seq = seq.float()

        ### DNA CONV ENCODER ###
        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)
        seq = seq.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        src = src.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL METADATA EMBEDDING ###
        xmd_embedding = self.xmd_emb(x_metadata).unsqueeze(1).expand(-1, self.l2, -1)

        ### FUSION ###
        src = torch.cat([src, xmd_embedding, seq], dim=-1)

        src = self.fusion(src)

        ### TRANSFORMER ENCODER ###
        for enc in self.transformer_encoder:
            src = enc(src)

        return src

class CANDI(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True, num_sequencing_platforms=10, num_runtypes=4):
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2

        self.encoder = CANDI_DNA_Encoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor, num_sequencing_platforms, num_runtypes)
        
        if self.separate_decoders:
            self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes)
            self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes)
        else:
            self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes)

        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def encode(self, src, seq, x_metadata):
        """Encode input data into latent representation."""
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        
        z = self.encoder(src, seq, x_metadata)
        return z
    
    def decode(self, z, y_metadata):
        """Decode latent representation into predictions."""
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        
        if self.separate_decoders:
            count_decoded = self.count_decoder(z, y_metadata)
            pval_decoded = self.pval_decoder(z, y_metadata)

            p, n = self.neg_binom_layer(count_decoded)
            mu, var = self.gaussian_layer(pval_decoded)
        else:
            decoded = self.decoder(z, y_metadata)
            p, n = self.neg_binom_layer(decoded)
            mu, var = self.gaussian_layer(decoded)
            
        return p, n, mu, var

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False):
        z = self.encode(src, seq, x_metadata)
        p, n, mu, var = self.decode(z, y_metadata)
        
        if return_z:
            return p, n, mu, var, z
        else:
            return p, n, mu, var

class CANDI_UNET(CANDI):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers,
                 nhead, n_sab_layers, pool_size=2, dropout=0.1, context_length=1600,
                 pos_enc="relative", expansion_factor=3, separate_decoders=True, num_sequencing_platforms=10, num_runtypes=4):
        super(CANDI_UNET, self).__init__(signal_dim, metadata_embedding_dim,
                                          conv_kernel_size, n_cnn_layers,
                                          nhead, n_sab_layers,
                                          pool_size, dropout,
                                          context_length, pos_enc,
                                          expansion_factor,
                                          separate_decoders, num_sequencing_platforms, num_runtypes)

    def _compute_skips(self, src):
        # mask as in encode
        src = torch.where(src == -2,
                          torch.tensor(-1, device=src.device), src)
        x = src.permute(0, 2, 1)  # (N, F1, L)
        skips = []
        for conv in self.encoder.convEnc:
            x = conv(x)
            skips.append(x)
        return skips

    def _unet_decode(self, z, y_metadata, skips, decoder):
        # mask metadata
        y_metadata = torch.where(y_metadata == -2,
                                 torch.tensor(-1, device=y_metadata.device),
                                 y_metadata)
        # embed and fuse metadata
        ymd_emb = decoder.ymd_emb(y_metadata)
        x = torch.cat([z, ymd_emb.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        x = decoder.ymd_fusion(x)
        x = x.permute(0, 2, 1)  # (N, C, L)

        # apply deconvs with UNet additions
        for i, dconv in enumerate(decoder.deconv):
            skip = skips[-(i + 1)]  # matching resolution
            x = x + skip
            x = dconv(x)

        x = x.permute(0, 2, 1)  # (N, L, F1)
        return x

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False):
        # compute skip features from signal branch
        skips = self._compute_skips(src)
        # standard encode (fuses seq + signal + metadata)
        z = self.encode(src, seq, x_metadata)

        # UNet-style decode for counts
        if self.separate_decoders:
            count_decoded = self._unet_decode(z, y_metadata, skips, self.count_decoder)
        else:
            count_decoded = self._unet_decode(z, y_metadata, skips, self.decoder)
        # Negative binomial parameters
        p, n = self.neg_binom_layer(count_decoded)

        # UNet-style decode for p-values
        if self.separate_decoders:
            pval_decoded = self._unet_decode(z, y_metadata, skips, self.pval_decoder)  
        else:
            pval_decoded = self._unet_decode(z, y_metadata, skips, self.decoder)  
        # Gaussian parameters
        mu, var = self.gaussian_layer(pval_decoded)

        if return_z:
            return p, n, mu, var, z
            
        return p, n, mu, var

#========================================================================================================#
#===========================================Building Blocks==============================================#
#========================================================================================================#

# ---------------------------
# Absolute Positional Encoding
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Creates positional encodings of shape (1, max_len, d_model).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term[:pe.size(2)//2])  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            x with added positional encoding for positions [0, L)
        """
        L = x.size(1)
        return x + self.pe[:, :L]

# ---------------------------
# Relative Positional Bias Module
# ---------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance):
        """
        Args:
            num_heads (int): number of attention heads.
            max_distance (int): maximum sequence length to support.
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, L):
        """
        Args:
            L (int): current sequence length.
        Returns:
            Tensor of shape (num_heads, L, L) to add as bias.
        """
        device = self.relative_bias.device
        pos = torch.arange(L, device=device)
        rel_pos = pos[None, :] - pos[:, None]  # shape (L, L)
        rel_pos = rel_pos + self.max_distance - 1  # shift to [0, 2*max_distance-2]
        bias = self.relative_bias[rel_pos]  # (L, L, num_heads)
        bias = bias.permute(2, 0, 1)  # (num_heads, L, L)
        return bias

# ---------------------------
# Dual Attention Encoder Block (Post-Norm)
# ---------------------------
class DualAttentionEncoderBlock(nn.Module):
    """
    Dual Attention Encoder Block with post-norm style.
    It has two parallel branches:
      - MHA1 (sequence branch): optionally uses relative or absolute positional encodings.
      - MHA2 (channel branch): operates along the channel dimension (no positional encoding).
    The outputs of the two branches are concatenated and fused via a FFN.
    Residual connections and layer norms are applied following the post-norm convention.
    """
    def __init__(self, d_model, num_heads, seq_length, dropout=0.1, 
                max_distance=128, pos_encoding_type="relative", max_len=5000):
        """
        Args:
            d_model (int): model (feature) dimension.
            num_heads (int): number of attention heads.
            seq_length (int): expected sequence length (used for channel branch).
            dropout (float): dropout rate.
            max_distance (int): max distance for relative bias.
            pos_encoding_type (str): "relative" or "absolute" for MHA1.
            max_len (int): max sequence length for absolute positional encoding.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.num_heads = num_heads
        self.pos_encoding_type = pos_encoding_type

        # Automatically determine the number of heads for each branch.
        self.num_heads_seq = get_divisible_heads(d_model, num_heads)
        self.num_heads_chan = get_divisible_heads(seq_length, num_heads)
        
        # Sequence branch (MHA1)
        if pos_encoding_type == "relative":
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.relative_bias = RelativePositionBias(num_heads, max_distance)
        elif pos_encoding_type == "absolute":
            # Use PyTorch's built-in MHA; we'll add absolute pos encodings.
            self.mha_seq = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.num_heads_seq, 
                                                  dropout=dropout, batch_first=True)
            self.abs_pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        else:
            raise ValueError("pos_encoding_type must be 'relative' or 'absolute'")
            
        # Channel branch (MHA2)
        # We transpose so that channels (d_model) become sequence tokens.
        # We set embed_dim for channel attention to seq_length.
        self.mha_channel = nn.MultiheadAttention(embed_dim=seq_length, num_heads=self.num_heads_chan,
                                                  dropout=dropout, batch_first=True)
        
        # Fusion: concatenate outputs from both branches (dimension becomes 2*d_model)
        # and then use an FFN to map it back to d_model.
        self.ffn = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Norms (applied after each sublayer, i.e., post-norm)
        self.norm_seq = nn.LayerNorm(d_model)
        self.norm_chan = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def relative_multihead_attention(self, x):
        """
        Custom multi-head self-attention with relative positional bias.
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        head_dim = self.d_model // self.num_heads
        q = self.q_proj(x)  # (B, L, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, num_heads, L, L)
        bias = self.relative_bias(L)  # (num_heads, L, L)
        scores = scores + bias.unsqueeze(0)  # (B, num_heads, L, L)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        out = torch.matmul(attn_weights, v)  # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)
        return out

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        
        # ----- Sequence Branch (MHA1) using post-norm -----
        if self.pos_encoding_type == "relative":
            # Compute sequence attention without pre-norm.
            seq_attn = self.relative_multihead_attention(x)  # (B, L, d_model)
        else:
            # Absolute positional encodings: add pos encoding and use default MHA.
            x_abs = self.abs_pos_enc(x)
            seq_attn, _ = self.mha_seq(x_abs, x_abs, x_abs)  # (B, L, d_model)
        # Add residual and then norm (post-norm)
        x_seq = self.norm_seq(x + seq_attn)  # (B, L, d_model)
        
        # ----- Channel Branch (MHA2) using post-norm -----
        # Transpose: (B, L, d_model) -> (B, d_model, L)
        x_trans = x.transpose(1, 2)
        # Apply channel attention (without pre-norm).
        chan_attn, _ = self.mha_channel(x_trans, x_trans, x_trans)  # (B, d_model, L)
        # Transpose back: (B, L, d_model)
        chan_attn = chan_attn.transpose(1, 2)
        # Add residual and norm
        x_chan = self.norm_chan(x + chan_attn)
        
        # ----- Fusion via FFN -----
        # Concatenate along feature dimension: (B, L, 2*d_model)
        fusion_input = torch.cat([x_seq, x_chan], dim=-1)
        ffn_out = self.ffn(fusion_input)  # (B, L, d_model)
        # Residual connection and final norm (post-norm)
        # out = self.norm_ffn(x + ffn_out)
        out = self.norm_ffn(x_seq + x_chan + ffn_out)
        return out

class EmbedMetadata(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_sequencing_platforms=10, num_runtypes=4, non_linearity=True):
        """
        Args:
            input_dim (int): Number of metadata features.
            embedding_dim (int): Final embedding dimension.
            num_sequencing_platforms (int): Number of sequencing platforms in the data.
            num_runtypes (int): Number of run types in the data.
            non_linearity (bool): Whether to apply ReLU at the end.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim 
        self.non_linearity = non_linearity
        # We divide the embedding_dim into 4 parts for continuous types.
        # (You can adjust the splitting scheme as needed.)
        self.continuous_size = embedding_dim // 4

        # For each feature (total input_dim features), create a separate linear transform.
        self.depth_transforms = nn.ModuleList(
            [nn.Linear(1, self.continuous_size) for _ in range(input_dim)]
        )
        # For sequencing platform, create separate embedding layers per feature.
        # Use dynamic size based on actual data
        self.sequencing_platform_embeddings = nn.ModuleList(
            [nn.Embedding(num_sequencing_platforms, self.continuous_size) for _ in range(input_dim)]
        )
        self.read_length_transforms = nn.ModuleList(
            [nn.Linear(1, self.continuous_size) for _ in range(input_dim)]
        )
        # For runtype, create separate embedding layers per feature.
        # Use dynamic size based on actual data
        self.runtype_embeddings = nn.ModuleList(
            [nn.Embedding(num_runtypes, self.continuous_size) for _ in range(input_dim)]
        )

        # Final projection: the concatenated vector for each feature will be of size 4*continuous_size.
        # For all features, that becomes input_dim * 4 * continuous_size.
        self.final_embedding = nn.Linear(input_dim * 4 * self.continuous_size, embedding_dim)
        self.final_emb_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, metadata):
        """
        Args:
            metadata: Tensor of shape (B, 4, input_dim)
                      where dimension 1 indexes the four metadata types in the order:
                      [depth, sequencing_platform, read_length, runtype]
        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        B = metadata.size(0)
        # Lists to collect per-feature embeddings.
        per_feature_embeds = []
        for i in range(self.input_dim):
            # Extract each metadata type for feature i.
            depth = metadata[:, 0, i].unsqueeze(-1).float() 
            sequencing_platform = metadata[:, 1, i].long() 
            read_length = metadata[:, 2, i].unsqueeze(-1).float() 
            runtype = metadata[:, 3, i].long() 
            
            # For runtype, map -1 -> 2 (missing) and -2 -> 3 (cloze_masked)
            runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype)
            runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype)
            
            # For sequencing platform, map -1 -> 2 (missing) and -2 -> 3 (cloze_masked)
            sequencing_platform = torch.where(sequencing_platform == -1, torch.tensor(2, device=sequencing_platform.device), sequencing_platform)
            sequencing_platform = torch.where(sequencing_platform == -2, torch.tensor(3, device=sequencing_platform.device), sequencing_platform)
            
            # Apply the separate transforms/embeddings for feature i.
            depth_embed = self.depth_transforms[i](depth)              # (B, continuous_size)
            sequencing_platform_embed = self.sequencing_platform_embeddings[i](sequencing_platform)  # (B, continuous_size)
            read_length_embed = self.read_length_transforms[i](read_length)  # (B, continuous_size)
            runtype_embed = self.runtype_embeddings[i](runtype)           # (B, continuous_size)
            
            # Concatenate the four embeddings along the last dimension.
            feature_embed = torch.cat([depth_embed, sequencing_platform_embed, read_length_embed, runtype_embed], dim=-1)  # (B, 4*continuous_size)
            per_feature_embeds.append(feature_embed)
        
        # Now stack along a new dimension for features -> shape (B, input_dim, 4*continuous_size)
        embeddings = torch.stack(per_feature_embeds, dim=1)
        # Flatten feature dimension: (B, input_dim * 4*continuous_size)
        embeddings = embeddings.view(B, -1)
        # Project to final embedding dimension.
        embeddings = self.final_embedding(embeddings)
        embeddings = self.final_emb_layer_norm(embeddings)
        
        if self.non_linearity:
            embeddings = F.relu(embeddings)
        
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D, norm="layer", groups=1, apply_act=False):
        super(ConvBlock, self).__init__()
        self.normtype = norm
        self.apply_act = apply_act
        
        if self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_C)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_C)
        
        self.conv = nn.Conv1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S, padding="same", groups=groups)
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.normtype == "layer":
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        elif self.normtype == "batch":
            x = self.norm(x)
        
        if self.apply_act:
            x = F.gelu(x)
        
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D, norm="layer", groups=1, apply_act=False):
        super(DeconvBlock, self).__init__()
        self.normtype = norm
        self.apply_act = apply_act
        
        if self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_C)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_C)
        
        padding = (W - 1) // 2
        output_padding = S - 1
        
        self.deconv = nn.ConvTranspose1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S,
            padding=padding, output_padding=output_padding, groups=groups)
    
    def forward(self, x):
        x = self.deconv(x)
        
        if self.normtype == "layer":
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        elif self.normtype == "batch":
            x = self.norm(x)
        
        if self.apply_act:
            x = F.gelu(x)
        
        return x

class DeconvTower(nn.Module):
    def __init__(self, in_C, out_C, W, S=1, D=1, residuals=True, groups=1, pool_size=2):
        super(DeconvTower, self).__init__()
        
        self.deconv1 = DeconvBlock(in_C, out_C, W, S, D, norm="layer", groups=groups, apply_act=False)
        self.resid = residuals
        
        if self.resid:
            self.rdeconv = nn.ConvTranspose1d(in_C, out_C, kernel_size=1, stride=S, output_padding=S - 1, groups=groups)
    
    def forward(self, x):
        y = self.deconv1(x)  # Output before activation
        
        if self.resid:
            y = y + self.rdeconv(x)
        
        y = F.gelu(y)  # Activation after residual
        return y

class ConvTower(nn.Module):
    def __init__(self, in_C, out_C, W, S=1, D=1, pool_type="max", residuals=True, groups=1, pool_size=2, SE=False):
        super(ConvTower, self).__init__()
        
        if pool_type == "max" or pool_type == "attn" or pool_type == "avg":
            self.do_pool = True
        else:
            self.do_pool = False
        
        if pool_type == "attn":
            self.pool = SoftmaxPooling1D(pool_size)
        elif pool_type == "max":
            self.pool = nn.MaxPool1d(pool_size)
        elif pool_type == "avg":
            self.pool = nn.AvgPool1d(pool_size)
        
        self.conv1 = ConvBlock(in_C, out_C, W, S, D, groups=groups, apply_act=False)
        self.resid = residuals
        
        if self.resid:
            self.rconv = nn.Conv1d(in_C, out_C, kernel_size=1, groups=groups)
        
        self.SE = SE
        if self.SE:
            self.se_block = SE_Block_1D(out_C)
    
    def forward(self, x):
        y = self.conv1(x)  # Output before activation
        
        if self.resid:
            y = y + self.rconv(x)
        
        y = F.gelu(y)  # Activation after residual
        
        if self.SE:
            y = self.se_block(y)
        
        if self.do_pool:
            y = self.pool(y)
        
        return y

class SE_Block_1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D convolutional layers.
    This module recalibrates channel-wise feature responses by modeling interdependencies between channels.
    """
    def __init__(self, c, r=8):
        super(SE_Block_1D, self).__init__()
        # Global average pooling for 1D
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        # Excitation network to produce channel-wise weights
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, recal=True):
        bs, c, l = x.shape  # Batch size, number of channels, length
        # Squeeze: Global average pooling to get the channel-wise statistics
        y = self.squeeze(x).view(bs, c)  # Shape becomes (bs, c)
        # Excitation: Fully connected layers to compute weights for each channel
        y = self.excitation(y).view(bs, c, 1)  # Shape becomes (bs, c, 1)
        # Recalibrate: Multiply the original input by the computed weights
        if recal:
            return x * y.expand_as(x)  # Shape matches (bs, c, l)
        else:
            return y.expand_as(x)  # Shape matches (bs, c, l)

class Sqeeze_Extend(nn.Module):
    def __init__(self, k=1):
        super(Sqeeze_Extend, self).__init__()
        self.k = k
        self.squeeze = nn.AdaptiveAvgPool1d(k)

    def forward(self, x):
        bs, c, l = x.shape  
        y = self.squeeze(x).view(bs, c, self.k)
        return y.expand_as(x)

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position

        # Get the current device from embeddings_table
        device = self.embeddings_table.device

        # Move final_mat to the same device as embeddings_table
        final_mat = final_mat.to(device)

        embeddings = self.embeddings_table[final_mat]

        return embeddings

class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))#.to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        self.scale = self.scale.to(attn1.device)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

class RelativeEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, feed_forward_hidden, dropout):
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.relative_multihead_attn = RelativeMultiHeadAttentionLayer(d_model, heads, dropout)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(d_model, feed_forward_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_hidden, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # Self-attention
        _src = self.relative_multihead_attn(src, src, src, src_mask)
        
        # Residual connection and layer norm
        src = self.layer_norm_1(src + self.dropout(_src))

        # Position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # Another residual connection and layer norm
        src = self.layer_norm_2(src + self.dropout(_src))

        return src

class RelativeDecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.layer_norm_cross_attn = nn.LayerNorm(hid_dim)
        self.layer_norm_ff = nn.LayerNorm(hid_dim)

        self.encoder_attention = RelativeMultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, src_mask=None):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # Encoder-decoder attention
        query = trg
        key = enc_src
        value = enc_src

        # Using the decoder input as the query, and the encoder output as key and value
        _trg = self.encoder_attention(query, key, value, src_mask)

        # Residual connection and layer norm
        trg = self.layer_norm_cross_attn(trg + self.dropout(_trg))

        # Positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # Residual connection and layer norm
        trg = self.layer_norm_ff(trg + self.dropout(_trg))

        return trg

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers):
        super(FeedForwardNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input Layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden Layers
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output Layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation Function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass through each layer
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        
        x = self.output_layer(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # Use the full div_term for both even and odd indices, handling odd d_model
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[:pe.size(2)//2])  # Ensure matching size

        self.register_buffer('pe', pe.permute(1, 0, 2))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

#========================================================================================================#
#========================================= Negative Binomial ============================================#
#========================================================================================================#

class NegativeBinomialLayer(nn.Module):
    def __init__(self, input_dim, output_dim, FF=False):
        super(NegativeBinomialLayer, self).__init__()
        self.FF = FF
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        self.fc_p = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Sigmoid()
        )

        self.fc_n = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        # using sigmoid to ensure it's between 0 and 1
        p = self.fc_p(x)

        # using softplus to ensure it's positive
        n = self.fc_n(x)

        return p, n

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

class GaussianLayer(nn.Module):
    def __init__(self, input_dim, output_dim, FF=False):
        super(GaussianLayer, self).__init__()

        self.FF = FF
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        # Define the layers for calculating mu (mean) parameter
        self.fc_mu = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Softplus()
        )

        # Define the layers for calculating var parameter
        self.fc_var = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Softplus() # Ensure var is positive
        )

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        mu = self.fc_mu(x)
        var = self.fc_var(x)

        return mu, var
#========================================================================================================#
#=============================================== Main ===================================================#
#========================================================================================================#

if __name__ == "__main__":
    hyper_parameters1678 = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.05,
        "nhead": 4,
        "d_model": 192,
        "nlayers": 3,
        "epochs": 4,
        "mask_percentage": 0.2,
        "chunk": True,
        "context_length": 200,
        "batch_size": 200,
        "learning_rate": 0.0001
    }  

    if sys.argv[1] == "epd16":
        train_epidenoise16(
            hyper_parameters1678, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd17":
        train_epidenoise17(
            hyper_parameters1678, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd18":
        train_epidenoise18(
            hyper_parameters1678, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd20":
        hyper_parameters20 = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.05,
            "nhead": 4,
            "d_model": 128,
            "nlayers": 2,
            "epochs": 10,
            "mask_percentage": 0.3,
            "kernel_size": [1, 3, 3],
            "conv_out_channels": [64, 64, 128],
            "dilation":1,
            "context_length": 800,
            "batch_size": 100,
            "learning_rate": 0.0001,
        }
        train_epidenoise20(
            hyper_parameters20, 
            checkpoint_path=None, 
            start_ds=0)
    
    elif sys.argv[1] == "epd21":
        hyper_parameters21 = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.1,
            "nhead": 4,
            "d_model": 256,
            "nlayers": 2,
            "epochs": 2,
            "kernel_size": [1, 9, 7, 5],
            "conv_out_channels": [64, 128, 192, 256],
            "dilation":1,
            "context_length": 800,
            "learning_rate": 1e-3,
        }
        train_epidenoise21(
            hyper_parameters21, 
            checkpoint_path=None, 
            start_ds=0)
    
    elif sys.argv[1] == "epd22":
        hyper_parameters22 = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.01,
            "context_length": 200,
            
            "kernel_size": [1, 3, 3, 3],
            "conv_out_channels": [128, 144, 192, 256],
            "dilation":1,

            "nhead": 2,
            "n_enc_layers": 1,
            "n_dec_layers": 1,
            
            "mask_percentage":0.15,
            "batch_size":400,
            "epochs": 10,
            "outer_loop_epochs":2,
            "learning_rate": 1e-4
        }
        train_epidenoise22(
            hyper_parameters22, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd30a":
        
        hyper_parameters30a = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 45,
            "dropout": 0.01,
            "nhead": 5,
            "d_model": 450,
            "nlayers": 2,
            "epochs": 1,
            "inner_epochs": 10,
            "mask_percentage": 0.15,
            "context_length": 200,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 200,
            "lr_halflife":1,
            "min_avail":10
        }
        if len(sys.argv) >= 3:
            if sys.argv[2] == "synth":
                synth_hyper_parameters30a = {
                    "data_path": "/project/compbio-lab/encode_data/",
                    "input_dim": 47,
                    "metadata_embedding_dim": 40,
                    "dropout": 0.01,
                    "nhead": 8,
                    "d_model": 416,
                    "nlayers": 6,
                    "epochs": 2000,
                    "inner_epochs": 100,
                    "mask_percentage": 0.1,
                    "context_length": 400,
                    "batch_size": 36,
                    "learning_rate": 1e-4,
                    "num_loci": 1600,
                    "lr_halflife":1,
                    "min_avail":5
                }
            
                train_epd30_synthdata(
                    synth_hyper_parameters30a, arch="a")

        else:
            train_epidenoise30(
                hyper_parameters30a, 
                checkpoint_path=None, 
                arch="a")
    
    elif sys.argv[1]  == "epd30b":
        hyper_parameters30b = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 40,
            "dropout": 0.01,

            "n_cnn_layers": 4,
            "conv_kernel_size" : 5,
            "n_decoder_layers" : 1,

            "nhead": 5,
            "d_model": 768,
            "nlayers": 6,
            "epochs": 10,
            "inner_epochs": 5,
            "mask_percentage": 0.15,
            "context_length": 810,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 1600,
            "lr_halflife":1,
            "min_avail":5
        }
        if len(sys.argv) >= 3:
            if sys.argv[2] == "synth":
                synth_hyper_parameters30b = {
                    "data_path": "/project/compbio-lab/encode_data/",
                    "input_dim": 47,
                    "metadata_embedding_dim": 40,
                    "dropout": 0.1,

                    "n_cnn_layers": 3,
                    "conv_kernel_size" : 7,
                    "n_decoder_layers" : 1,

                    "nhead": 8,
                    "d_model": 768,
                    "nlayers": 2,
                    "epochs": 4000,
                    "inner_epochs": 50,
                    "mask_percentage": 0.1,
                    "context_length": 810,
                    "batch_size": 50,
                    "learning_rate": 5e-4,
                    "num_loci": 800,
                    "lr_halflife":2,
                    "min_avail":8
                }
                train_epd30_synthdata(
                    synth_hyper_parameters30b, arch="b")

        else:
            if sys.argv[1] == "epd30b":
                train_epidenoise30(
                    hyper_parameters30b, 
                    checkpoint_path=None, 
                    arch="b")

    elif sys.argv[1] == "epd30c":
        hyper_parameters30c = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 45,
            "dropout": 0.05,

            "n_cnn_layers": 3,
            "conv_kernel_size" : 7,
            "pool_size" : 3,

            "nhead": 6,
            "d_model": (90)*(2**3),
            "nlayers": 2,
            "epochs": 1,
            "inner_epochs": 1,
            "mask_percentage": 0.1,
            "context_length": 810,
            "batch_size": 20,
            "learning_rate": 1e-4,
            "num_loci": 200,
            "lr_halflife":2,
            "min_avail":10
        }
        if len(sys.argv) >= 3:
            if sys.argv[2] == "synth":
                synth_hyper_parameters30cd = {
                    "data_path": "/project/compbio-lab/encode_data/",
                    "input_dim": 47,
                    "metadata_embedding_dim": 49,
                    "dropout": 0.05,

                    "n_cnn_layers": 3,
                    "conv_kernel_size" : 7,
                    "pool_size" : 3,

                    "nhead": 6,
                    "d_model": (47+49)*(2**3),
                    "nlayers": 3,
                    "epochs": 2000,
                    "inner_epochs": 50,
                    "mask_percentage": 0.1,
                    "context_length": 810,
                    "batch_size": 20,
                    "learning_rate": 1e-4,
                    "num_loci": 800,
                    "lr_halflife":2,
                    "min_avail":8
                }
                train_epd30_synthdata(
                    synth_hyper_parameters30cd, arch="c")

        else:
            train_epidenoise30(
                hyper_parameters30c, 
                checkpoint_path=None, 
                arch="c")
    
    elif sys.argv[1] == "epd30d":
        hyper_parameters30d = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 45,
            "dropout": 0.1,

            "n_cnn_layers": 5,
            "conv_kernel_size" : 5,
            "pool_size": 2,

            "nhead": 16,
            "d_model": 768,
            "nlayers": 8,
            "epochs": 10,
            "inner_epochs": 5,
            "mask_percentage": 0.2,
            "context_length": 1600,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 3200,
            "lr_halflife":1,
            "min_avail":5
        }
        train_epidenoise30(
            hyper_parameters30d, 
            checkpoint_path=None, 
            arch="d")
    
    elif sys.argv[1] == "epd30d_eic":
        hyper_parameters30d_eic = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 35,
            "metadata_embedding_dim": 35,
            "dropout": 0.1,

            "n_cnn_layers": 5,
            "conv_kernel_size" : 5,
            "pool_size": 2,

            "nhead": 16,
            "d_model": 768,
            "nlayers": 8,
            "epochs": 10,
            "inner_epochs": 1,
            "mask_percentage": 0.25,
            "context_length": 3200,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 3200,
            "lr_halflife":1,
            "min_avail":1
        }
        train_epd30_eic(
            hyper_parameters30d_eic, 
            checkpoint_path=None, 
            arch="d")