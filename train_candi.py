from model import *
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tracemalloc, sys, argparse

# tracemalloc.start()

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

class CANDI_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
        super(CANDI_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model =  self.latent_dim = self.f2

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])

        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        if self.pos_enc == "relative":
            self.transformer_encoder = nn.ModuleList([
                RelativeEncoderLayer(d_model=self.d_model, heads=nhead, feed_forward_hidden=expansion_factor*self.d_model, dropout=dropout) for _ in range(n_sab_layers)])
            
        else:
            self.posEnc = PositionalEncoding(self.d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=nhead, dim_feedforward=expansion_factor*self.d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

    def forward(self, src, x_metadata):
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)

        src = src.permute(0, 2, 1)  # to N, L', F2
        xmd_embedding = self.xmd_emb(x_metadata)
        src = torch.cat([src, xmd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.xmd_fusion(src)

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            src = self.posEnc(src)
            src = self.transformer_encoder(src)
        else:
            for enc in self.transformer_encoder:
                src = enc(src)
        
        return src

class CANDI_Decoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size=2, expansion_factor=3):
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

        self.ymd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=False)
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

class CANDI(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True):
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2
        print("d_model: ", self.d_model)

        self.encoder = CANDI_Encoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor)
        
        if self.separate_decoders:
            self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
            self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
        else:
            self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)

        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def encode(self, src, x_metadata):
        """Encode input data into latent representation."""
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        
        z = self.encoder(src, x_metadata)
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

    def forward(self, src, x_metadata, y_metadata, availability=None, return_z=False):
        z = self.encode(src, x_metadata)
        p, n, mu, var = self.decode(z, y_metadata)
        
        if return_z:
            return p, n, mu, var, z
        else:
            return p, n, mu, var

class CANDI_DNA_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
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
        
        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=False)

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
        # src = torch.cat([src, seq], dim=-1)
        # src = torch.cat([seq, xmd_embedding], dim=-1)
        src = self.fusion(src)

        ### TRANSFORMER ENCODER ###
        for enc in self.transformer_encoder:
            src = enc(src)

        return src

class CANDI_DNA(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True):
        super(CANDI_DNA, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2

        self.encoder = CANDI_DNA_Encoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor)
        
        if self.separate_decoders:
            self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
            self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)
        else:
            self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor)

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

class CANDI_UNET(CANDI_DNA):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers,
                 nhead, n_sab_layers, pool_size=2, dropout=0.1, context_length=1600,
                 pos_enc="relative", expansion_factor=3, separate_decoders=True):
        super(CANDI_UNET, self).__init__(signal_dim, metadata_embedding_dim,
                                          conv_kernel_size, n_cnn_layers,
                                          nhead, n_sab_layers,
                                          pool_size, dropout,
                                          context_length, pos_enc,
                                          expansion_factor,
                                          separate_decoders)

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

class PRETRAIN(object):
    def __init__(
        self, model, dataset, criterion, optimizer, 
        scheduler, device=None, HPO=False, cosine_sched=False):
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Training on device: {self.device}.")

        self.model = model.to(self.device)
        self.dataset = dataset
        self.HPO = HPO
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cosine_sched = cosine_sched

    def pretrain_CANDI(
        self, num_epochs, context_length, batch_size, inner_epochs, 
        arch="", mask_percentage=0.15, hook=False, DNA=False, 
        early_stop=True, early_stop_metric="imp_pval_r2", 
        early_stop_delta=0.01, patience=2, prog_monitor_patience=150, 
        prog_monitor_delta=1e-5, supertrack=False):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"CANDI{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/CANDI{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        images = []
        gif_filename = f"models/CANDI{arch}_TrainProg.gif"

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        num_assays = self.dataset.signal_dim
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.model)
        
        val_eval = MONITOR_VALIDATION(
            self.dataset.base_path, context_length, 4*batch_size, 
            must_have_chr_access=self.dataset.must_have_chr_access,
            token_dict=token_dict, eic=bool("eic" in arch), 
            DNA=DNA, device=self.device)
         
        if "test" not in arch:
            # try:
            validation_set_eval, val_metrics = val_eval.get_validation(self.model)
            torch.cuda.empty_cache()
            log_strs.append(validation_set_eval)
            print(validation_set_eval)
            log_resource_usage()
            # except:
            #     pass

        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)
        best_metric = None

        progress_monitor = {
            "ups_count_r2":[], "imp_count_r2":[],
            "ups_pval_r2":[], "imp_pval_r2":[],
            "ups_count_spearman":[], "imp_count_spearman":[],
            "ups_pval_spearman":[], "imp_pval_spearman":[],
            "ups_count_pearson":[], "imp_count_pearson":[],
            "ups_pval_pearson":[], "imp_pval_pearson":[]}
        
        prog_mon_ema = {}
        prog_mon_best_so_far = {}
        no_prog_mon_improvement = 0
        lr_sch_steps_taken = 0

        for epoch in range(num_epochs):
            if early_stop:
                epoch_rec = {
                    "ups_count_r2":[], "imp_count_r2":[],
                    "ups_pval_r2":[], "imp_pval_r2":[],
                    "ups_count_spearman":[], "imp_count_spearman":[],
                    "ups_pval_spearman":[], "imp_pval_spearman":[],
                    "ups_count_pearson":[], "imp_count_pearson":[],
                    "ups_pval_pearson":[], "imp_pval_pearson":[],

                    "val_count_median_ups_r2":[], "val_count_median_imp_r2":[], 
                    "val_count_median_ups_pcc":[], "val_count_median_imp_pcc":[], 
                    "val_count_median_ups_srcc":[], "val_count_median_imp_srcc":[], 
                    "val_count_median_ups_loss":[], "val_count_median_imp_loss":[], 
                    
                    "val_pval_median_ups_pcc":[],  "val_pval_median_imp_pcc":[], 
                    "val_pval_median_ups_r2":[],   "val_pval_median_imp_r2":[], 
                    "val_pval_median_ups_srcc":[], "val_pval_median_imp_srcc":[],
                    "val_pval_median_ups_loss":[], "val_pval_median_imp_loss":[]

                    }

            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                if DNA:
                    _X_batch, _mX_batch, _avX_batch, _dnaseq_batch= self.dataset.get_batch(side="x", dna_seq=True)
                else:
                    _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(side="x")

                _Y_batch, _mY_batch, _avY_batch, _pval_batch = self.dataset.get_batch(side="y", pval=True, y_prompt=supertrack)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_count_loss":[], "ups_count_loss":[],
                    "imp_pval_loss":[], "ups_pval_loss":[],
                    "ups_count_r2":[], "imp_count_r2":[],
                    "ups_pval_r2":[], "imp_pval_r2":[],
                    "ups_count_pp":[], "imp_count_pp":[],
                    "ups_pval_pp":[], "imp_pval_pp":[],
                    "ups_count_conf":[], "imp_count_conf":[],
                    "ups_pval_conf":[], "imp_pval_conf":[],
                    "ups_count_mse":[], "imp_count_mse":[],
                    "ups_pval_mse":[], "imp_pval_mse":[],
                    "ups_count_spearman":[], "imp_count_spearman":[],
                    "ups_pval_spearman":[], "imp_pval_spearman":[],
                    "ups_count_pearson":[], "imp_count_pearson":[],
                    "ups_pval_pearson":[], "imp_pval_pearson":[], 
                    "grad_norm":[]
                    }

                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                for _ in range(inner_epochs):
                    if DNA:
                        X_batch, mX_batch, avX_batch, dnaseq_batch= _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone(), _dnaseq_batch.clone()
                    else:
                        X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch, pval_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone(), _pval_batch.clone()

                    if "_prog_unmask" in arch or "_prog_mask" in arch:
                        X_batch, mX_batch, avX_batch = self.masker.progressive(X_batch, mX_batch, avX_batch, num_mask)

                    if "random_mask" in arch:
                        num_mask = random.randint(1, self.dataset.signal_dim - 1)
                        X_batch, mX_batch, avX_batch = self.masker.progressive(X_batch, mX_batch, avX_batch, num_mask)

                    else:
                        X_batch, mX_batch, avX_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)

                    masked_map = (X_batch == token_dict["cloze_mask"])
                    observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                    missing_map = (X_batch == token_dict["missing_mask"])
                    masked_map = masked_map.to(self.device) # imputation targets
                    observed_map = observed_map.to(self.device) # upsampling targets
                    
                    X_batch = X_batch.float().to(self.device).requires_grad_(True)
                    mX_batch = mX_batch.to(self.device)
                    avX_batch = avX_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    pval_batch = pval_batch.to(self.device)                        

                    if DNA:
                        dnaseq_batch = dnaseq_batch.to(self.device)
                        output_p, output_n, output_mu, output_var = self.model(X_batch, dnaseq_batch, mX_batch, mY_batch, avX_batch)
                    else:
                        output_p, output_n, output_mu, output_var = self.model(X_batch, mX_batch, mY_batch, avX_batch)

                    obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss = self.criterion(
                        output_p, output_n, output_mu, output_var, Y_batch, pval_batch, observed_map, masked_map) 
                    
                    # if "_prog_unmask" in arch or "_prog_mask" in arch or "random_mask" in arch:
                    if "random_mask" in arch:
                        msk_p = float(num_mask/num_assays)
                        
                        if "imponly" in arch:
                            loss = (msk_p*(imp_count_loss + imp_pval_loss))
                        
                        elif "pvalonly" in arch:
                            loss = (msk_p*imp_pval_loss) + ((1-msk_p)*obs_pval_loss)

                        else:
                            imp_pval_loss *= 4
                            obs_pval_loss *= 3
                            imp_count_loss *= 2
                            obs_count_loss *= 1
                            loss = (msk_p*(imp_count_loss + imp_pval_loss)) + ((1-msk_p)*(obs_pval_loss + obs_count_loss))

                    else:
                        loss = obs_count_loss + obs_pval_loss + imp_pval_loss + imp_count_loss

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss = loss.float()
                    loss.backward()  
                    
                    total_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)

                    #################################################################################

                    # IMP Count Predictions
                    neg_bin_imp = NegativeBinomial(output_p[masked_map].cpu().detach(), output_n[masked_map].cpu().detach())
                    imp_count_pred = neg_bin_imp.expect().numpy()
                    imp_count_std = neg_bin_imp.std().numpy()

                    imp_count_true = Y_batch[masked_map].cpu().detach().numpy()
                    imp_count_abs_error = torch.abs(torch.Tensor(imp_count_true) - torch.Tensor(imp_count_pred)).numpy()

                    imp_count_r2 = r2_score(imp_count_true, imp_count_pred)
                    imp_count_errstd = spearmanr(imp_count_std, imp_count_abs_error)
                    imp_count_pp = compute_perplexity(neg_bin_imp.pmf(imp_count_true))
                    imp_count_mse = ((imp_count_true - imp_count_pred)**2).mean()

                    imp_count_spearman = spearmanr(imp_count_true, imp_count_pred).correlation
                    imp_count_pearson = pearsonr(imp_count_true, imp_count_pred)[0]

                    # IMP P-value Predictions
                    imp_pval_pred = output_mu[masked_map].cpu().detach().numpy()
                    imp_pval_std = output_var[masked_map].cpu().detach().numpy() ** 0.5

                    imp_pval_true = pval_batch[masked_map].cpu().detach().numpy()
                    imp_pval_abs_error = torch.abs(torch.Tensor(imp_pval_true) - torch.Tensor(imp_pval_pred)).numpy()

                    imp_pval_r2 = r2_score(imp_pval_true, imp_pval_pred)
                    imp_pval_errstd = spearmanr(imp_pval_std, imp_pval_abs_error)
                    gaussian_imp = Gaussian(output_mu[masked_map].cpu().detach(), output_var[masked_map].cpu().detach())
                    imp_pval_pp = compute_perplexity(gaussian_imp.pdf(imp_pval_true))
                    imp_pval_mse = ((imp_pval_true - imp_pval_pred)**2).mean()

                    imp_pval_spearman = spearmanr(imp_pval_true, imp_pval_pred).correlation
                    imp_pval_pearson = pearsonr(imp_pval_true, imp_pval_pred)[0]

                    # UPS Count Predictions
                    neg_bin_ups = NegativeBinomial(output_p[observed_map].cpu().detach(), output_n[observed_map].cpu().detach())
                    ups_count_pred = neg_bin_ups.expect().numpy()
                    ups_count_std = neg_bin_ups.std().numpy()

                    ups_count_true = Y_batch[observed_map].cpu().detach().numpy()
                    ups_count_abs_error = torch.abs(torch.Tensor(ups_count_true) - torch.Tensor(ups_count_pred)).numpy()

                    ups_count_r2 = r2_score(ups_count_true, ups_count_pred)
                    ups_count_errstd = spearmanr(ups_count_std, ups_count_abs_error)
                    ups_count_pp = compute_perplexity(neg_bin_ups.pmf(ups_count_true))
                    ups_count_mse = ((ups_count_true - ups_count_pred)**2).mean()

                    ups_count_spearman = spearmanr(ups_count_true, ups_count_pred).correlation
                    ups_count_pearson = pearsonr(ups_count_true, ups_count_pred)[0]

                    # UPS P-value Predictions
                    ups_pval_pred = output_mu[observed_map].cpu().detach().numpy()
                    ups_pval_std = output_var[observed_map].cpu().detach().numpy() ** 0.5

                    ups_pval_true = pval_batch[observed_map].cpu().detach().numpy()
                    ups_pval_abs_error = torch.abs(torch.Tensor(ups_pval_true) - torch.Tensor(ups_pval_pred)).numpy()

                    ups_pval_r2 = r2_score(ups_pval_true, ups_pval_pred)
                    ups_pval_errstd = spearmanr(ups_pval_std, ups_pval_abs_error)
                    gaussian_ups = Gaussian(output_mu[observed_map].cpu().detach(), output_var[observed_map].cpu().detach())
                    ups_pval_pp = compute_perplexity(gaussian_ups.pdf(ups_pval_true))
                    ups_pval_mse = ((ups_pval_true - ups_pval_pred)**2).mean()

                    ups_pval_spearman = spearmanr(ups_pval_true, ups_pval_pred).correlation
                    ups_pval_pearson = pearsonr(ups_pval_true, ups_pval_pred)[0]

                    #################################################################################
                    batch_rec["grad_norm"].append(total_norm)

                    batch_rec["imp_count_loss"].append(imp_count_loss.item())
                    batch_rec["ups_count_loss"].append(obs_count_loss.item())
                    batch_rec["imp_pval_loss"].append(imp_pval_loss.item())
                    batch_rec["ups_pval_loss"].append(obs_pval_loss.item())

                    batch_rec["ups_count_r2"].append(ups_count_r2)
                    batch_rec["imp_count_r2"].append(imp_count_r2)

                    batch_rec["ups_pval_r2"].append(ups_pval_r2)
                    batch_rec["imp_pval_r2"].append(imp_pval_r2)

                    batch_rec["ups_count_pp"].append(ups_count_pp)
                    batch_rec["imp_count_pp"].append(imp_count_pp)

                    batch_rec["ups_pval_pp"].append(ups_pval_pp)
                    batch_rec["imp_pval_pp"].append(imp_pval_pp)

                    batch_rec["ups_count_conf"].append(ups_count_errstd)
                    batch_rec["imp_count_conf"].append(ups_count_errstd)

                    batch_rec["ups_pval_conf"].append(ups_pval_errstd)
                    batch_rec["imp_pval_conf"].append(imp_pval_errstd)

                    batch_rec["ups_count_mse"].append(ups_count_mse)
                    batch_rec["imp_count_mse"].append(imp_count_mse)

                    batch_rec["ups_pval_mse"].append(ups_pval_mse)
                    batch_rec["imp_pval_mse"].append(imp_pval_mse)

                    batch_rec["imp_count_spearman"].append(imp_count_spearman)
                    batch_rec["ups_count_spearman"].append(ups_count_spearman)
                    batch_rec["imp_pval_spearman"].append(imp_pval_spearman)
                    batch_rec["ups_pval_spearman"].append(ups_pval_spearman)

                    batch_rec["imp_count_pearson"].append(imp_count_pearson)
                    batch_rec["ups_count_pearson"].append(ups_count_pearson)
                    batch_rec["imp_pval_pearson"].append(imp_pval_pearson)
                    batch_rec["ups_pval_pearson"].append(ups_pval_pearson)

                    for k in [
                        "imp_pval_r2", "imp_pval_pearson", 
                        "imp_pval_spearman", "imp_count_r2", 
                        "imp_count_pearson", "imp_count_spearman",
                        "imp_count_loss", "ups_count_loss",
                        "imp_pval_loss", "ups_pval_loss"]:

                        mean_value = np.mean(batch_rec[k]) if not np.isnan(np.mean(batch_rec[k])) else 0
                        if "loss" in k:
                            mean_value = -1 * mean_value # since we are monitoring increasing trends

                        if k not in progress_monitor.keys():
                            progress_monitor[k] = []

                        progress_monitor[k].append(mean_value)

                        if k not in prog_mon_ema.keys():
                            prog_mon_ema[k] = mean_value
                        else:
                            alpha = 0.01 # APR4 change
                            prog_mon_ema[k] = alpha*mean_value + (1-alpha)*prog_mon_ema[k]

                        if k not in prog_mon_best_so_far.keys():
                            prog_mon_best_so_far[k] = mean_value
                        
                    if not self.cosine_sched:
                        # check if improvement in EMA
                        statement_prog_imp_pval_r2 = bool(prog_mon_ema["imp_pval_r2"] > prog_mon_best_so_far["imp_pval_r2"] + prog_monitor_delta)
                        statement_prog_imp_pval_pearson = bool(prog_mon_ema["imp_pval_pearson"] > prog_mon_best_so_far["imp_pval_pearson"] + prog_monitor_delta)
                        statement_prog_imp_pval_spearman = bool(prog_mon_ema["imp_pval_spearman"] > prog_mon_best_so_far["imp_pval_spearman"] + prog_monitor_delta)
                        statement_prog_imp_count_r2 = bool(prog_mon_ema["imp_count_r2"] > prog_mon_best_so_far["imp_count_r2"] + prog_monitor_delta)
                        statement_prog_imp_count_pearson = bool(prog_mon_ema["imp_count_pearson"] > prog_mon_best_so_far["imp_count_pearson"] + prog_monitor_delta)
                        statement_prog_imp_count_spearman = bool(prog_mon_ema["imp_count_spearman"] > prog_mon_best_so_far["imp_count_spearman"] + prog_monitor_delta)

                        for k in ["imp_pval_r2", "imp_pval_pearson", "imp_pval_spearman", "imp_count_r2", "imp_count_pearson", "imp_count_spearman"]:
                            if epoch > 0:
                                prog_mon_best_so_far[k] = max(prog_mon_best_so_far[k], prog_mon_ema[k])
                            else:
                                prog_mon_best_so_far[k] = 0.0

                        if not any([
                            statement_prog_imp_pval_r2, statement_prog_imp_pval_pearson, statement_prog_imp_pval_spearman,
                            statement_prog_imp_count_r2, statement_prog_imp_count_pearson, statement_prog_imp_count_spearman]):
                            no_prog_mon_improvement += 1
                        else:
                            no_prog_mon_improvement = 0
                        
                        if no_prog_mon_improvement >= prog_monitor_patience:
                            print(f"No improvement in EMA for {no_prog_mon_improvement} steps. Adjusting learning rate...")
                            current_lr = self.optimizer.param_groups[0]['lr']
                            self.scheduler.step()
                            lr_sch_steps_taken += 1
                            prog_monitor_patience *= 1.05
                            new_lr = self.optimizer.param_groups[0]['lr']
                            print(f"Learning rate adjusted from {current_lr:.2e} to {new_lr:.2e}")
                            no_prog_mon_improvement = 0

                    del output_p, output_n, output_mu, output_var, loss, obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss
                    del X_batch, mX_batch, mY_batch, avX_batch, Y_batch, pval_batch, observed_map, masked_map
                    if DNA:
                        del dnaseq_batch
                    gc.collect()
                
                if hook:
                    # Initialize variables to store maximum gradient norms and corresponding layer names
                    max_weight_grad_norm = 0
                    max_weight_grad_layer = None
                    max_bias_grad_norm = 0
                    max_bias_grad_layer = None

                    # Check and update maximum gradient norms
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                            if module.weight.grad_norm > max_weight_grad_norm:
                                max_weight_grad_norm = module.weight.grad_norm
                                max_weight_grad_layer = name

                        if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                            if module.bias.grad_norm > max_bias_grad_norm:
                                max_bias_grad_norm = module.bias.grad_norm
                                max_bias_grad_layer = name

                    if max_weight_grad_layer:
                        print(f"Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}")

                self.optimizer.step()
                if self.cosine_sched:
                    self.scheduler.step()

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                del _X_batch, _mX_batch, _avX_batch, _Y_batch, _mY_batch, _avY_batch, _pval_batch
                if DNA:
                    del _dnaseq_batch
                gc.collect()

                CurrentLR = self.optimizer.param_groups[0]['lr']
                if self.cosine_sched:
                    lr_printstatement = f"CurrentLR: {CurrentLR:.0e}" 
                    
                else:
                    lr_printstatement = f"LR_sch_steps_taken {lr_sch_steps_taken} | LR_patience {no_prog_mon_improvement} | CurrentLR: {CurrentLR:.0e}"

                logstr = [
                    f"Ep. {epoch}",
                    f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                    f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer / len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                    f"Bios Prog. {self.dataset.bios_pointer / self.dataset.num_bios:.2%}", "\n",
                    f"Imp_nbNLL {np.mean(batch_rec['imp_count_loss']):.2f}",
                    f"Ups_nbNLL {np.mean(batch_rec['ups_count_loss']):.2f}",
                    f"Imp_gNLL {np.mean(batch_rec['imp_pval_loss']):.2f}",
                    f"Ups_gNLL {np.mean(batch_rec['ups_pval_loss']):.2f}", "\n",
                    f"Imp_Count_R2 {np.mean(batch_rec['imp_count_r2']):.2f}",
                    f"Ups_Count_R2 {np.mean(batch_rec['ups_count_r2']):.2f}",
                    f"Imp_Pval_R2 {np.mean(batch_rec['imp_pval_r2']):.2f}",
                    f"Ups_Pval_R2 {np.mean(batch_rec['ups_pval_r2']):.2f}", "\n",
                    f"Imp_Count_PP {np.mean(batch_rec['imp_count_pp']):.2f}",
                    f"Ups_Count_PP {np.mean(batch_rec['ups_count_pp']):.2f}",
                    f"Imp_Pval_PP {np.mean(batch_rec['imp_pval_pp']):.2f}",
                    f"Ups_Pval_PP {np.mean(batch_rec['ups_pval_pp']):.2f}", "\n",
                    f"Imp_Count_Conf {np.mean(batch_rec['imp_count_conf']):.2f}",
                    f"Ups_Count_Conf {np.mean(batch_rec['ups_count_conf']):.2f}",
                    f"Imp_Pval_Conf {np.mean(batch_rec['imp_pval_conf']):.2f}",
                    f"Ups_Pval_Conf {np.mean(batch_rec['ups_pval_conf']):.2f}", "\n",
                    f"Imp_Count_MSE {np.mean(batch_rec['imp_count_mse']):.2f}",
                    f"Ups_Count_MSE {np.mean(batch_rec['ups_count_mse']):.2f}",
                    f"Imp_Pval_MSE {np.mean(batch_rec['imp_pval_mse']):.2f}",
                    f"Ups_Pval_MSE {np.mean(batch_rec['ups_pval_mse']):.2f}", "\n",
                    f"Imp_Count_SRCC {np.mean(batch_rec['imp_count_spearman']):.2f}",
                    f"Ups_Count_SRCC {np.mean(batch_rec['ups_count_spearman']):.2f}",
                    f"Imp_Pval_SRCC {np.mean(batch_rec['imp_pval_spearman']):.2f}",
                    f"Ups_Pval_SRCC {np.mean(batch_rec['ups_pval_spearman']):.2f}", "\n",
                    f"Imp_Count_PCC {np.mean(batch_rec['imp_count_pearson']):.2f}",
                    f"Ups_Count_PCC {np.mean(batch_rec['ups_count_pearson']):.2f}",
                    f"Imp_Pval_PCC {np.mean(batch_rec['imp_pval_pearson']):.2f}",
                    f"Ups_Pval_PCC {np.mean(batch_rec['ups_pval_pearson']):.2f}", "\n",
                    
                    
                    f"EMA_imp_pval_r2 {prog_mon_ema['imp_pval_r2']:.2f}",
                    f"EMA_imp_pval_PCC {prog_mon_ema['imp_pval_pearson']:.2f}",
                    f"EMA_imp_pval_SRCC {prog_mon_ema['imp_pval_spearman']:.2f}", 
                    f"EMA_imp_pval_loss {-1*prog_mon_ema['imp_pval_loss']:.2f}", "\n", # -1 since we multiplied it to -1 earlier :))

                    f"EMA_imp_count_r2 {prog_mon_ema['imp_count_r2']:.2f}",
                    f"EMA_imp_count_PCC {prog_mon_ema['imp_count_pearson']:.2f}",
                    f"EMA_imp_count_SRCC {prog_mon_ema['imp_count_spearman']:.2f}", 
                    f"EMA_imp_count_loss {-1*prog_mon_ema['imp_count_loss']:.2f}", "\n", # -1 since we multiplied it to -1 earlier :))
                    
                    f"took {int(minutes)}:{int(seconds):02d}", 
                    f"Gradient_Norm {np.mean(batch_rec['grad_norm']):.2f}",
                    f"num_mask {num_mask}", lr_printstatement, "\n"
                ]

                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                if lr_sch_steps_taken >= 100 and early_stop:
                    print("Early stopping due to super small learning rate...")
                    return self.model, best_metric
                
                #################################################################################
                #################################################################################
                if early_stop:
                    epoch_rec["imp_count_r2"].append(np.mean(batch_rec['imp_count_r2']))
                    epoch_rec["imp_pval_r2"].append(np.mean(batch_rec['imp_pval_r2']))
                    epoch_rec["imp_count_spearman"].append(np.mean(batch_rec['imp_count_spearman']))
                    epoch_rec["imp_pval_spearman"].append(np.mean(batch_rec['imp_pval_spearman']))
                    epoch_rec["imp_count_pearson"].append(np.mean(batch_rec['imp_count_pearson']))
                    epoch_rec["imp_pval_pearson"].append(np.mean(batch_rec['imp_pval_pearson']))
                #################################################################################
                #################################################################################

                chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                dsf_pointer0 = self.dataset.dsf_pointer
                bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                dsf_pointer1 = self.dataset.dsf_pointer
                chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                bios_pointer1 = self.dataset.bios_pointer

                if (chr0 != chr1) or (dsf_pointer0 != dsf_pointer1) or (bios_pointer0 != bios_pointer1):
                    logfile = open(f"models/CANDI{arch}_log.txt", "w")
                    logfile.write("\n".join(log_strs))
                    logfile.close()

                    if (chr0 != chr1) and "test" not in arch:
                        try:
                            validation_set_eval, val_metrics = val_eval.get_validation(self.model)
                            torch.cuda.empty_cache()
                            log_strs.append(validation_set_eval)
                            print(validation_set_eval)
                            log_resource_usage()

                            if early_stop:
                                epoch_rec["val_count_median_imp_r2"].append(val_metrics["imputed_counts"]["R2_count"]["median"])
                                epoch_rec["val_count_median_imp_pcc"].append(val_metrics["imputed_counts"]["PCC_count"]["median"])
                                epoch_rec["val_count_median_imp_srcc"].append(val_metrics["imputed_counts"]["SRCC_count"]["median"])
                                
                                epoch_rec["val_pval_median_imp_r2"].append(val_metrics["imputed_pvals"]["R2_pval"]["median"])
                                epoch_rec["val_pval_median_imp_pcc"].append(val_metrics["imputed_pvals"]["PCC_pval"]["median"])
                                epoch_rec["val_pval_median_imp_srcc"].append(val_metrics["imputed_pvals"]["SRCC_pval"]["median"])
                        except:
                            pass
                    
                    if self.HPO==False:
                        try:
                            # try:
                            #     if os.path.exists(f'models/CANDI{arch}_model_checkpoint_epoch{epoch}_{chr0}.pth'):
                            #         os.system(f"rm -rf models/CANDI{arch}_model_checkpoint_epoch{epoch}_{chr0}.pth")
                            #     torch.save(self.model.state_dict(), f'models/CANDI{arch}_model_checkpoint_epoch{epoch}_{chr1}.pth')
                            # except:
                            #     pass

                            # Generate and process the plot
                            fig_title = " | ".join([
                                f"Ep. {epoch}", f"DSF{self.dataset.dsf_list[dsf_pointer0]}->{1}",
                                f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]}"])
                            
                            if "eic" in arch:
                                plot_buf = val_eval.generate_training_gif_frame_eic(self.model, fig_title)
                            else:
                                plot_buf = val_eval.generate_training_gif_frame(self.model, fig_title)

                            images.append(imageio.imread(plot_buf))
                            plot_buf.close()
                            imageio.mimsave(gif_filename, images, duration=0.5 * len(images))
                        except Exception as e:
                            pass

                if next_epoch and "test" not in arch:
                    try:
                        validation_set_eval, val_metrics = val_eval.get_validation(self.model)
                        torch.cuda.empty_cache()
                        log_strs.append(validation_set_eval)
                        print(validation_set_eval)
                        log_resource_usage()

                        if early_stop:
                            epoch_rec["val_count_mean_imp_r2"].append(val_metrics["imputed_counts"]["R2_count"]["mean"])
                            epoch_rec["val_count_mean_imp_pcc"].append(val_metrics["imputed_counts"]["PCC_count"]["mean"])
                            epoch_rec["val_count_mean_imp_srcc"].append(val_metrics["imputed_counts"]["SRCC_count"]["mean"])
                            
                            epoch_rec["val_pval_mean_imp_r2"].append(val_metrics["imputed_pvals"]["R2_pval"]["mean"])
                            epoch_rec["val_pval_mean_imp_pcc"].append(val_metrics["imputed_pvals"]["PCC_pval"]["mean"])
                            epoch_rec["val_pval_mean_imp_srcc"].append(val_metrics["imputed_pvals"]["SRCC_pval"]["mean"])
                    except:
                        pass 

            # if early_stop:
            #     # Initialize the best metrics if it's the first epoch
            #     if best_metric is None:
            #         best_metric = {key: None for key in epoch_rec.keys()}
            #         patience_counter = {key: 0 for key in epoch_rec.keys()}

            #     # Loop over all metrics
            #     for metric_name in epoch_rec.keys():
            #         current_metric = np.mean(epoch_rec[metric_name])  # Calculate the current epoch's mean for this metric

            #         if best_metric[metric_name] is None or current_metric > best_metric[metric_name] + early_stop_delta:
            #             best_metric[metric_name] = current_metric  # Update the best metric for this key
            #             patience_counter[metric_name] = 0  # Reset the patience counter
            #         else:
            #             patience_counter[metric_name] += 1  # Increment the patience counter if no improvement

            #     # Check if all patience counters have exceeded the limit (e.g., 3 epochs of no improvement)
            #     if all(patience_counter[metric] >= patience for metric in patience_counter.keys()):
            #         print(f"Early stopping at epoch {epoch}. No significant improvement across metrics.")
            #         logfile = open(f"models/CANDI{arch}_log.txt", "w")
            #         logfile.write("\n".join(log_strs))
            #         logfile.write(f"\n\nFinal best metric records:\n")
            #         for metric_name, value in best_metric.items():
            #             logfile.write(f"{metric_name}: {value}\n")
            #         logfile.close()
            #         return self.model, best_metric
            #     else:
            #         print(f"best metric records so far: \n{best_metric}")
            #         logfile = open(f"models/CANDI{arch}_log.txt", "w") 
            #         logfile.write("\n".join(log_strs))
            #         logfile.write(f"\n\nBest metric records so far:\n")
            #         for metric_name, value in best_metric.items():
            #             logfile.write(f"{metric_name}: {value}\n")
            #         logfile.close()
                
            if self.HPO==False and epoch != (num_epochs-1):
                try:
                    os.system(f"rm -rf models/CANDI{arch}_model_checkpoint_epoch{epoch-1}.pth")
                    torch.save(self.model.state_dict(), f'models/CANDI{arch}_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
        
        if early_stop:
            return self.model, best_metric
        else:
            return self.model

def Train_CANDI(hyper_parameters, eic=False, checkpoint_path=None, DNA=False, suffix="", prog_mask=False, device=None, HPO=False):
    if eic:
        arch="eic"
    else:
        arch="full"
    
    if DNA:
        arch = f"{arch}_DNA"

    if prog_mask:
        arch = f"{arch}_prog_mask"
    else:
        arch = f"{arch}_random_mask"

    arch = f"{arch}_{suffix}"
    # Defining the hyperparameters
    resolution = 25
    n_sab_layers = hyper_parameters["n_sab_layers"]
    
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    n_cnn_layers = hyper_parameters["n_cnn_layers"]
    conv_kernel_size = hyper_parameters["conv_kernel_size"]
    pool_size = hyper_parameters["pool_size"]
    expansion_factor = hyper_parameters["expansion_factor"]
    pos_enc = hyper_parameters["pos_enc"]
    separate_decoders = hyper_parameters["separate_decoders"]
    merge_ct = hyper_parameters["merge_ct"]
    loci_gen = hyper_parameters["loci_gen"]
    supertrack = hyper_parameters["supertrack"]

    dataset = ExtendedEncodeDataHandler(hyper_parameters["data_path"])
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, loci_gen=loci_gen, 
        bios_min_exp_avail_threshold=min_avail, check_completeness=True, 
        eic=eic, merge_ct=merge_ct,
        DSF_list=[1,2], 
        must_have_chr_access=hyper_parameters["must_have_chr_access"])

    signal_dim = dataset.signal_dim
    metadata_embedding_dim = dataset.signal_dim * 4

    if DNA:
        if hyper_parameters["unet"]:
            model = CANDI_UNET(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, hyper_parameters["nhead"],
                n_sab_layers, pool_size=pool_size, dropout=hyper_parameters["dropout"], context_length=context_length, 
                pos_enc=pos_enc, expansion_factor=expansion_factor, separate_decoders=separate_decoders)
        else:
            model = CANDI_DNA(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, hyper_parameters["nhead"],
                n_sab_layers, pool_size=pool_size, dropout=hyper_parameters["dropout"], context_length=context_length, 
                pos_enc=pos_enc, expansion_factor=expansion_factor, separate_decoders=separate_decoders)

    else:
        model = CANDI(
            signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, hyper_parameters["nhead"],
            n_sab_layers, pool_size=pool_size, dropout=hyper_parameters["dropout"], context_length=context_length,
            pos_enc=pos_enc, expansion_factor=expansion_factor, separate_decoders=separate_decoders)

    if hyper_parameters["optim"].lower()=="adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif hyper_parameters["optim"].lower()=="adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif hyper_parameters["optim"].lower()=="adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if hyper_parameters["LRschedule"] is None:
        cos_sch=False
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hyper_parameters["lr_halflife"], gamma=1)
    elif hyper_parameters["LRschedule"].lower()=="cosine":
        cos_sch=True
        num_total_epochs = epochs * inner_epochs * len(dataset.m_regions) * 2
        warmup_epochs = inner_epochs * len(dataset.m_regions) * 2
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs), 
                CosineAnnealingLR(optimizer, T_max=(num_total_epochs - warmup_epochs), eta_min=0.0)],
            milestones=[warmup_epochs])
    else:
        cos_sch=False
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hyper_parameters["lr_halflife"], gamma=0.95)

    print(f"Using optimizer: {optimizer.__class__.__name__}")

    if checkpoint_path is not None:
        print("loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"CANDI_{arch} # model_parameters: {count_parameters(model)}")
    summary(model)
    
    model_name = f"CANDI{arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = CANDI_LOSS()

    start_time = time.time()

    trainer = PRETRAIN(
        model, dataset, criterion, optimizer, scheduler, 
        device=device, HPO=HPO, cosine_sched=cos_sch)

    model, best_metric = trainer.pretrain_CANDI(
        num_epochs=epochs, mask_percentage=mask_percentage, context_length=context_length, 
        batch_size=batch_size, inner_epochs=inner_epochs, arch=arch, DNA=DNA, supertrack=supertrack)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    if not HPO:
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

        # Write a description text file
        description = {
            "hyper_parameters": hyper_parameters,
            "model_architecture": str(model),
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "number_of_model_parameters": count_parameters(model),
            "training_duration": int(end_time - start_time)
        }
        with open(os.path.join(model_dir, model_name.replace(".pt", ".txt")), 'w') as f:
            f.write(json.dumps(description, indent=4))

    return model, best_metric

class CANDI_LOADER(object):
    def __init__(self, model_path, hyper_parameters, DNA=False):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DNA = DNA

    def load_CANDI(self):
        signal_dim = self.hyper_parameters["signal_dim"]
        dropout = self.hyper_parameters["dropout"]
        nhead = self.hyper_parameters["nhead"]
        n_sab_layers = self.hyper_parameters["n_sab_layers"]
        metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
        context_length = self.hyper_parameters["context_length"]

        n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
        conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
        pool_size = self.hyper_parameters["pool_size"]
        separate_decoders = self.hyper_parameters["separate_decoders"]
        
        if self.DNA:
            if self.hyper_parameters["unet"]:
                model = CANDI_UNET(
                    signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                    n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
                    separate_decoders=separate_decoders)
            else:
                model = CANDI_DNA(
                    signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                    n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
                    separate_decoders=separate_decoders)
        else:
            model = CANDI(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length,
                separate_decoders=separate_decoders)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 

        model = model.to(self.device)
        return model
    
def main():
    parser = argparse.ArgumentParser(description="Train the model with specified hyperparameters")

    # Hyperparameters
    parser.add_argument('--data_path', type=str, default="/project/compbio-lab/encode_data/", help='Path to the data')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--n_cnn_layers', type=int, default=3, help='Number of CNN layers')
    parser.add_argument('--conv_kernel_size', type=int, default=3, help='Convolution kernel size')
    parser.add_argument('--pool_size', type=int, default=2, help='Pooling size')
    parser.add_argument('--expansion_factor', type=int, default=3, help='Expansion factor for the model')

    parser.add_argument('--nhead', type=int, default=9, help='Number of attention heads')
    parser.add_argument('--n_sab_layers', type=int, default=4, help='Number of SAB layers')
    parser.add_argument('--pos_enc', type=str, default="relative", help='Transformer Positional Encodings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--inner_epochs', type=int, default=1, help='Number of inner epochs')
    parser.add_argument('--mask_percentage', type=float, default=0.2, help='Masking percentage (if used)')
    parser.add_argument('--context_length', type=int, default=1200, help='Context length')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_loci', type=int, default=5000, help='Number of loci')
    parser.add_argument('--lr_halflife', type=int, default=1, help='Learning rate halflife')
    parser.add_argument('--min_avail', type=int, default=3, help='Minimum available')
    parser.add_argument('--hpo', action='store_true', help='Flag to enable hyperparameter optimization')
    parser.add_argument('--shared_decoders', action='store_true', help='Flag to enable shared decoders for pval and count')
    parser.add_argument('--suffix', type=str, default='', help='Optional suffix for model name')
    parser.add_argument('--merge_ct', action='store_true', help='Flag to enable merging celltypes')
    parser.add_argument('--loci_gen', type=str, default="ccre", help='Loci generation method')

    parser.add_argument('--optim', type=str, default="sgd", help='optimizer')
    parser.add_argument('--unet', action='store_true', help='whether to use unet skip connections')
    parser.add_argument('--supertrack', action='store_true', help='whether to use train using supertrack setting (fill in y_prompt)')
    parser.add_argument('--LRschedule', type=str, default=None, help='optimizer lr scheduler')
    
    # Flags for DNA and EIC
    parser.add_argument('--eic', action='store_true', help='Flag to enable EIC')
    parser.add_argument('--dna', action='store_true', help='Flag to enable DNA')
    parser.add_argument('--prog_mask', action='store_true', help='Flag to enable progressive masking')

    # Add checkpoint argument
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint model for continued training')

    # Parse the arguments
    args = parser.parse_args()
    separate_decoders = not args.shared_decoders
    merge_ct = True
    must_have_chr_access = True

    # Convert parsed arguments into a dictionary for hyperparameters
    hyper_parameters = {
        "data_path": args.data_path,
        "dropout": args.dropout,
        "n_cnn_layers": args.n_cnn_layers,
        "conv_kernel_size": args.conv_kernel_size,
        "pool_size": args.pool_size,
        "expansion_factor": args.expansion_factor,
        "nhead": args.nhead,
        "n_sab_layers": args.n_sab_layers,
        "pos_enc": args.pos_enc,
        "epochs": args.epochs,
        "inner_epochs": args.inner_epochs,
        "mask_percentage": args.mask_percentage,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_loci": args.num_loci,
        "lr_halflife": args.lr_halflife,
        "min_avail": args.min_avail,
        "hpo": args.hpo,
        "separate_decoders": separate_decoders,
        "merge_ct": merge_ct,
        "loci_gen": args.loci_gen,
        "must_have_chr_access": must_have_chr_access,

        "optim": args.optim,
        "unet": args.unet,
        "LRschedule": args.LRschedule,
        "supertrack": args.supertrack
    }

    # Call your training function with parsed arguments, including checkpoint
    Train_CANDI(hyper_parameters, eic=args.eic, checkpoint_path=args.checkpoint, 
                DNA=True, suffix=args.suffix, prog_mask=args.prog_mask, HPO=args.hpo)

if __name__ == "__main__":
    main()

#  watch -n 20 "squeue -u mfa76 && echo  && tail -n 15 models/*sab*txt && echo  && tail -n 15 models/*def*txt && echo  && tail -n 15 models/*XL*txt"

# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_CosSched --LRschedule cosine 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_adamw --optim adamw 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_adam --optim adam 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun9_onedec --shared_decoders

# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet --unet
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet_CosSched --unet --LRschedule cosine 
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet_adamax --unet --optim adamax
# python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_abljun12_unet_onedec --unet --shared_decoders

# python CANDI.py --dna --eic --hpo --suffix def_jun13_unet_onedec_admx --unet --shared_decoders --optim adamax
# python CANDI.py --dna --eic --hpo --context_length 2400 --expansion_factor 2 --n_cnn_layers 5 --suffix XLcntx_jun13_unet_onedec --unet --shared_decoders --optim adamax

# srun python CANDI.py --dna --eic --hpo --epochs 6 --suffix def_unet_imponly --unet
# srun python CANDI.py --dna --eic --hpo --epochs 6 --suffix unet_xlc_imponly --unet --expansion_factor 2 --n_cnn_layers 5 --n_sab_layers 1  --nhead 8 --context_length 4800
# srun python CANDI.py --dna --eic --hpo --epochs 6 --suffix unet_xlc_imponly_adamax --unet --expansion_factor 2 --n_cnn_layers 5 --n_sab_layers 1  --nhead 8 --context_length 4800 --optim adamax
# srun python CANDI.py --dna --eic --hpo --epochs 6 --suffix unet_xlc --unet --expansion_factor 2 --n_cnn_layers 5 --n_sab_layers 1  --nhead 8 --context_length 4800 

# watch -n 20 "squeue -u mfa76 && tail -n 11 models/*jun21_xlc_imponly_adamax*txt && echo && tail -n 11 models/*unet_xlc_imponly_log*txt && echo && tail -n 11 models/*unet_xlc_imponly_adamax_log*txt && echo && tail -n 11 models/*unet_xlc_log*txt"



# srun python CANDI.py --dna --eic --hpo --epochs 10 --suffix def_jun24_unet_admx_onedec --unet --optim adamax --shared_decoders
# srun python CANDI.py --dna --eic --hpo --epochs 10 --suffix def_jun24_unet_Cos_admx_onedec --unet --optim adamax --LRschedule cosine --shared_decoders

# srun python CANDI.py --dna --hpo --epochs 10 --suffix def_jun24_unet_admx_onedec --unet --optim adamax --shared_decoders
# srun python CANDI.py --dna --hpo --epochs 10 --suffix def_jun24_unet_Cos_admx_onedec --unet --optim adamax --LRschedule cosine --shared_decoders




# python CANDI.py --dna --eic --epochs 15 --suffix admx_cos_shdc --optim adamax --LRschedule cosine --shared_decoders
# python CANDI.py --dna --epochs 15 --suffix admx_cos_shdc --optim adamax --LRschedule cosine --shared_decoders

# python CANDI.py --dna --eic --epochs 15 --suffix unt_cos_shdc --unet --LRschedule cosine --shared_decoders
# python CANDI.py --dna --epochs 15 --suffix unt_cos_shdc --unet --LRschedule cosine --shared_decoders

# python CANDI.py --dna --eic --epochs 15 --suffix unt_admx_shdc --unet --optim adamax --shared_decoders
# python CANDI.py --dna --epochs 15 --suffix unt_admx_shdc --unet --optim adamax --shared_decoders

# python CANDI.py --dna --eic --epochs 15 --suffix unt_admx_cos --unet --optim adamax --LRschedule cosine
# python CANDI.py --dna --epochs 15 --suffix unt_admx_cos --unet --optim adamax --LRschedule cosine


