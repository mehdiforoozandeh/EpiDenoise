from model import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class CANDI(nn.Module):
    def __init__(
        self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=2000, pos_enc="absolute", expansion_factor=4):
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (2**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2

        conv_channels = [(self.f1)*(2**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [2 * x for x in conv_channels[::-1]]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.signal_layer_norm = nn.LayerNorm(self.f1)

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else 2 * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])

        self.SE_enc = SE_Block_1D(self.f2)

        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        self.ymd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.ymd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(
                d_model=d_model, heads=nhead, feed_forward_hidden=expansion_factor*d_model, dropout=dropout)
            self.transformer_encoder = nn.ModuleList([self.encoder_layer for _ in range(n_sab_layers)])
            
        else:
            self.posEnc = PositionalEncoding(d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=expansion_factor*d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

        self.deconv = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / 2),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])
        
        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        # availability = torch.where(availability == -2, torch.tensor(-1, device=availability.device), availability)

        src = self.signal_layer_norm(src)
        ### CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        # e_src.shape = N, F2, L'
        src = self.SE_enc(src)

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

        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusion(src)
        
        src = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv:
            src = dconv(src)

        src = src.permute(0, 2, 1) # to N, L, F1
        
        p, n = self.neg_binom_layer(src)
        mu, var = self.gaussian_layer(src)

        return p, n, mu, var

class CANDI_DNA():
    pass


class CANDI_NLL_LOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(CANDI_NLL_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction)
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
        
        return observed_count_loss, imputed_count_loss, observed_pval_loss, imputed_pval_loss


class PRETRAIN(object):
    def __init__(self, model, dataset, criterion, optimizer, scheduler):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}.")

        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def pretrain_CANDI(self, num_epochs, context_length, batch_size, inner_epochs, arch="", mask_percentage=0.15, hook=False):
        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"CANDI{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/CANDI{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        # images = []
        # gif_filename = f"models/EPD30{arch}_TrainProg.gif"

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.model)
            
        # val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, arch=arch, token_dict=token_dict)
        
        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)
        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(side="x")
                _Y_batch, _mY_batch, _avY_batch, _pval_batch = self.dataset.get_batch(side="y", pval=True)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_count_loss":[], "ups_count_loss":[],
                    "imp_pval_loss":[], "ups_pval_loss":[],
                    "ups_count_r2":[], "imp_count_r2":[],
                    "ups_count_pmf":[], "imp_count_pmf":[],
                    "ups_pval_pmf":[], "imp_pval_pmf":[],
                    "ups_count_conf":[], "imp_count_conf":[],
                    "ups_pval_conf":[], "imp_pval_conf":[]
                    }

                for _ in range(inner_epochs):
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch, pval_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone(), _pval_batch.clone()

                    # X_batch, mX_batch, avX_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)
                    X_batch, mX_batch, avX_batch = self.masker.mask_chunk_features_30(X_batch, mX_batch, avX_batch)

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

                    output_p, output_n, output_mu, output_var = self.model(X_batch, mX_batch, mY_batch, avX_batch)

                    obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss = self.criterion(
                        output_p, output_n, output_mu, output_var, Y_batch, pval_batch, observed_map, masked_map) 

                    # if torch.isnan(pred_loss).any():
                    #     if len(batch_rec["imp_loss"]) > 0:
                    #         pred_loss = torch.tensor(np.mean(batch_rec["imp_loss"]))
                    #     else:
                    #         pred_loss = torch.tensor(1e5)

                    # if torch.isnan(obs_loss).any():
                    #     if len(batch_rec["ups_loss"]) > 0:
                    #         obs_loss = torch.tensor(np.mean(batch_rec["ups_loss"]))
                    #     else:
                    #         obs_loss = torch.tensor(1e5)

                    loss = obs_count_loss + imp_count_loss + obs_pval_loss + imp_pval_loss

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss, obs_loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss.backward()  

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

                #################################################################################
                #################################################################################
                #################################################################################

                    output_std = output_var**(1/2)
                    imp_count_pred = NegativeBinomial(
                        output_p[masked_map].cpu().detach(), 
                        output_n[masked_map].cpu().detach()
                        ).expect().cpu().detach().numpy()
                    imp_count_std = NegativeBinomial(
                        output_p[masked_map].cpu().detach(), 
                        output_n[masked_map].cpu().detach()
                        ).std().cpu().detach().numpy()

                    imp_count_true = Y_batch[masked_map].cpu().detach().numpy()
                    imp_count_abs_error = torch.abs(torch.Tensor(imp_count_true) - torch.Tensor(imp_count_pred)).cpu().detach().numpy()
                    
                    imp_count_r2 = r2_score(imp_count_true, imp_count_pred)
                    imp_count_errstd = spearmanr(imp_count_std, imp_count_abs_error)
                    imp_count_pmf = NegativeBinomial(
                        output_p[masked_map].cpu().detach(), 
                        output_n[masked_map].cpu().detach()).pmf(imp_count_true).mean()
                    #################################################################################
                    imp_pval_pred = output_mu[masked_map].cpu().detach().numpy()
                    imp_pval_std = output_std[masked_map].cpu().detach().numpy()

                    imp_pval_true = pval_batch[masked_map].cpu().detach().numpy()
                    imp_pval_abs_error = torch.abs(torch.Tensor(imp_pval_true) - torch.Tensor(imp_pval_pred)).cpu().detach().numpy()

                    imp_pval_r2 = r2_score(imp_pval_true, imp_pval_pred)
                    imp_pval_errstd = spearmanr(imp_pval_std, imp_pval_abs_error)
                    imp_pval_pmf = Gaussian(
                        output_mu[masked_map].cpu().detach(), 
                        output_var[masked_map].cpu().detach()).pmf(imp_pval_true).mean()
                    #################################################################################
                    ups_count_pred = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()
                        ).expect().cpu().detach().numpy()
                    ups_count_std = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()
                        ).std().cpu().detach().numpy()

                    ups_count_true = Y_batch[observed_map].cpu().detach().numpy()
                    ups_count_abs_error = torch.abs(torch.Tensor(ups_count_true) - torch.Tensor(ups_count_pred)).cpu().detach().numpy()

                    ups_count_r2 = r2_score(ups_count_true, ups_count_pred)
                    ups_count_errstd = spearmanr(ups_count_std, ups_count_abs_error)
                    ups_count_pmf = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()).pmf(ups_count_true).mean()
                    #################################################################################
                    ups_pval_pred = output_mu[observed_map].cpu().detach().numpy()
                    ups_pval_std = output_std[observed_map].cpu().detach().numpy()

                    ups_pval_true = pval_batch[observed_map].cpu().detach().numpy()
                    ups_pval_abs_error = torch.abs(torch.Tensor(ups_pval_true) - torch.Tensor(ups_pval_pred)).cpu().detach().numpy()

                    ups_pval_r2 = r2_score(ups_pval_true, ups_pval_pred)
                    ups_pval_errstd = spearmanr(ups_pval_std, ups_pval_abs_error)
                    ups_pval_pmf = Gaussian(
                        output_mu[observed_map].cpu().detach(), 
                        output_var[observed_map].cpu().detach()).pmf(ups_pval_true).mean()

                    
                #################################################################################
                #################################################################################
                #################################################################################
                    # # IMP Count Predictions
                    # neg_bin_imp = NegativeBinomial(output_p[masked_map].cpu().detach(), output_n[masked_map].cpu().detach())
                    # imp_count_pred = neg_bin_imp.expect().numpy()
                    # imp_count_std = neg_bin_imp.std().numpy()

                    # imp_count_true = Y_batch[masked_map].cpu().detach().numpy()
                    # imp_count_abs_error = torch.abs(torch.Tensor(imp_count_true) - torch.Tensor(imp_count_pred)).numpy()

                    # imp_count_r2 = r2_score(imp_count_true, imp_count_pred)
                    # imp_count_errstd = spearmanr(imp_count_std, imp_count_abs_error)
                    # imp_count_pmf = neg_bin_imp.pmf(imp_count_true).mean()

                    # # IMP P-value Predictions
                    # imp_pval_pred = output_mu[masked_map].cpu().detach().numpy()
                    # imp_pval_std = output_var[masked_map].cpu().detach().numpy() ** 0.5

                    # imp_pval_true = pval_batch[masked_map].cpu().detach().numpy()
                    # imp_pval_abs_error = torch.abs(torch.Tensor(imp_pval_true) - torch.Tensor(imp_pval_pred)).numpy()

                    # imp_pval_r2 = r2_score(imp_pval_true, imp_pval_pred)
                    # imp_pval_errstd = spearmanr(imp_pval_std, imp_pval_abs_error)
                    # gaussian_imp = Gaussian(output_mu[masked_map].cpu().detach(), output_var[masked_map].cpu().detach())
                    # imp_pval_pmf = gaussian_imp.pmf(imp_pval_true).mean()

                    # # UPS Count Predictions
                    # neg_bin_ups = NegativeBinomial(output_p[observed_map].cpu().detach(), output_n[observed_map].cpu().detach())
                    # ups_count_pred = neg_bin_ups.expect().numpy()
                    # ups_count_std = neg_bin_ups.std().numpy()

                    # ups_count_true = Y_batch[observed_map].cpu().detach().numpy()
                    # ups_count_abs_error = torch.abs(torch.Tensor(ups_count_true) - torch.Tensor(ups_count_pred)).numpy()

                    # ups_count_r2 = r2_score(ups_count_true, ups_count_pred)
                    # ups_count_errstd = spearmanr(ups_count_std, ups_count_abs_error)
                    # ups_count_pmf = neg_bin_ups.pmf(ups_count_true).mean()

                    # # UPS P-value Predictions
                    # ups_pval_pred = output_mu[observed_map].cpu().detach().numpy()
                    # ups_pval_std = output_var[observed_map].cpu().detach().numpy() ** 0.5

                    # ups_pval_true = pval_batch[observed_map].cpu().detach().numpy()
                    # ups_pval_abs_error = torch.abs(torch.Tensor(ups_pval_true) - torch.Tensor(ups_pval_pred)).numpy()

                    # ups_pval_r2 = r2_score(ups_pval_true, ups_pval_pred)
                    # ups_pval_errstd = spearmanr(ups_pval_std, ups_pval_abs_error)
                    # gaussian_ups = Gaussian(output_mu[observed_map].cpu().detach(), output_var[observed_map].cpu().detach())
                    # ups_pval_pmf = gaussian_ups.pmf(ups_pval_true).mean()
                    #################################################################################
                    #################################################################################
                    #################################################################################
                    #################################################################################                    
                    batch_rec["imp_count_loss"].append(imp_count_loss.item())
                    batch_rec["ups_count_loss"].append(obs_count_loss.item())
                    batch_rec["imp_pval_loss"].append(imp_pval_loss.item())
                    batch_rec["ups_pval_loss"].append(obs_pval_loss.item())

                    batch_rec["ups_count_r2"].append(ups_count_r2)
                    batch_rec["imp_count_r2"].append(imp_count_r2)

                    batch_rec["ups_count_pmf"].append(ups_count_pmf)
                    batch_rec["imp_count_pmf"].append(imp_count_pmf)

                    batch_rec["ups_pval_pmf"].append(ups_pval_pmf)
                    batch_rec["imp_pval_pmf"].append(imp_pval_pmf)

                    batch_rec["ups_count_conf"].append(ups_count_conf)
                    batch_rec["imp_count_conf"].append(imp_count_conf)

                    batch_rec["ups_pval_conf"].append(ups_pval_conf)
                    batch_rec["imp_pval_conf"].append(imp_pval_conf)


                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                # logstr = [
                #     f"Ep. {epoch}",
                #     f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                #     f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer/len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                #     f"Bios Prog. {self.dataset.bios_pointer/self.dataset.num_bios:.2%}",
                #     f"Imp_Loss {np.mean(batch_rec['imp_loss']):.2f}",
                #     f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                #     f"Imp_R2 {np.mean(batch_rec['imp_r2']):.2f}",
                #     f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                #     # f"Imp_pmf {np.mean(batch_rec['imp_pmf']):.2f}",
                #     # f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                #     f"Imp_MSE {np.mean(batch_rec['imp_mse']):.2f}",
                #     f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                #     f"Imp_Conf {np.mean(batch_rec['imp_conf']):.2f}",
                #     f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                #     f"took {int(minutes)}:{int(seconds)}"]
                
                logstr = [
                    f"Ep. {epoch}",
                    f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                    f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer / len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                    f"Bios Prog. {self.dataset.bios_pointer / self.dataset.num_bios:.2%}",
                    f"Imp_Count_Loss {np.mean(batch_rec['imp_count_loss']):.2f}",
                    f"Ups_Count_Loss {np.mean(batch_rec['ups_count_loss']):.2f}",
                    f"Imp_Pval_Loss {np.mean(batch_rec['imp_pval_loss']):.2f}",
                    f"Ups_Pval_Loss {np.mean(batch_rec['ups_pval_loss']):.2f}",
                    f"Imp_Count_R2 {np.mean(batch_rec['imp_count_r2']):.2f}",
                    f"Ups_Count_R2 {np.mean(batch_rec['ups_count_r2']):.2f}",
                    f"Imp_Count_PMF {np.mean(batch_rec['imp_count_pmf']):.2f}",
                    f"Ups_Count_PMF {np.mean(batch_rec['ups_count_pmf']):.2f}",
                    f"Imp_Pval_PMF {np.mean(batch_rec['imp_pval_pmf']):.2f}",
                    f"Ups_Pval_PMF {np.mean(batch_rec['ups_pval_pmf']):.2f}",
                    f"Imp_Count_Conf {np.mean(batch_rec['imp_count_conf']):.2f}",
                    f"Ups_Count_Conf {np.mean(batch_rec['ups_count_conf']):.2f}",
                    f"Imp_Pval_Conf {np.mean(batch_rec['imp_pval_conf']):.2f}",
                    f"Ups_Pval_Conf {np.mean(batch_rec['ups_pval_conf']):.2f}",
                    f"took {int(minutes)}:{int(seconds):02d}", "\n"
                ]

                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                logfile = open(f"models/CANDI{arch}_log.txt", "w")
                logfile.write("\n".join(log_strs))
                logfile.close()

                
                #################################################################################
                #################################################################################
                #################################################################################


                # chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                # dsf_pointer0 = self.dataset.dsf_pointer
                # bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                # dsf_pointer1 = self.dataset.dsf_pointer
                # chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                # bios_pointer1 = self.dataset.bios_pointer

                # if dsf_pointer0 != dsf_pointer1 or chr0 != chr1 or bios_pointer0 != bios_pointer1:
                #     # Generate and process the plot
                #     fig_title = " | ".join([
                #         f"Ep. {epoch}", f"DSF{self.dataset.dsf_list[dsf_pointer0]}->{1}",
                #         f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]}"])
                    
                #     plot_buf = val_eval.generate_training_gif_frame(self.model, fig_title)
                #     images.append(imageio.imread(plot_buf))
                #     plot_buf.close()
                #     imageio.mimsave(gif_filename, images, duration=0.5 * len(images))

                # if chr0 != chr1:
                #     validation_set_eval = val_eval.get_validation(self.model)
                #     torch.cuda.empty_cache()
                #     log_strs.append(validation_set_eval)
                #     print(validation_set_eval)
                #     log_resource_usage()
            
            self.scheduler.step()
            print("learning rate scheduler step...")
            if epoch%1==0:
                try:
                    torch.save(self.model.state_dict(), f'models/CANDI{arch}_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
                
        return self.model


def Train_CANDI_EIC(hyper_parameters, eic=False, arch="", checkpoint_path=None):
    # Defining the hyperparameters
    resolution = 25
    data_path = hyper_parameters["data_path"]

    signal_dim = hyper_parameters["signal_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    n_sab_layers = hyper_parameters["n_sab_layers"]
    metadata_embedding_dim = hyper_parameters["metadata_embedding_dim"]
    
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    lr_halflife = hyper_parameters["lr_halflife"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    n_cnn_layers = hyper_parameters["n_cnn_layers"]
    conv_kernel_size = hyper_parameters["conv_kernel_size"]
    pool_size = hyper_parameters["pool_size"]

    model = CANDI(
        signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length)

    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_halflife, gamma=0.5)

    if checkpoint_path is not None:
        print("loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"CANDI{arch} # model_parameters: {count_parameters(model)}")
    summary(model)

    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, loci_gen="random",#["chr19", "chr20"], 
        bios_min_exp_avail_threshold=min_avail, check_completeness=True)
    
    model_name = f"CANDI{arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters{arch}_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    
    criterion = CANDI_NLL_LOSS()

    start_time = time.time()

    trainer = PRETRAIN(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_CANDI(
        num_epochs=epochs, mask_percentage=mask_percentage, context_length=context_length, 
        batch_size=batch_size, inner_epochs=inner_epochs, arch=arch)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
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

    return model

    # if eic:
    #     pass
    # else:
    #     pass


if __name__ == "__main__":
    hyper_parameters = {
        "data_path": "/project/compbio-lab/encode_data/",
        "signal_dim": 45,
        "metadata_embedding_dim": 45,
        "dropout": 0.1,

        "n_cnn_layers": 5,
        "conv_kernel_size" : 5,
        "pool_size": 2,

        "nhead": 16,
        "n_sab_layers": 8,
        "epochs": 5,
        "inner_epochs": 5,
        "mask_percentage": 0.15,
        "context_length": 1600,
        "batch_size": 50,
        "learning_rate": 1e-4,
        "num_loci": 3200,
        "lr_halflife":1,
        "min_avail":5}
    Train_CANDI_EIC(hyper_parameters)