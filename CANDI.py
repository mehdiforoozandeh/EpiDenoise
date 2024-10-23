from model import *
import tracemalloc, sys, argparse

# tracemalloc.start()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CANDI(nn.Module):
    def __init__(
        self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=2000, pos_enc="relative", 
        expansion_factor=3):
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2
        print("d_model: ", d_model)

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        # self.signal_layer_norm = nn.LayerNorm(self.f1)

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])

        # self.SE_enc = SE_Block_1D(self.f2)

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
            self.transformer_encoder = nn.ModuleList([
                RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=expansion_factor*d_model, dropout=dropout) for _ in range(n_sab_layers)])
            
        else:
            self.posEnc = PositionalEncoding(d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=expansion_factor*d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

        self.deconv_count = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])

        self.deconv_pval = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])
        
        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def forward(self, src, x_metadata, y_metadata, availability, return_z=False):
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        # availability = torch.where(availability == -2, torch.tensor(-1, device=availability.device), availability)

        # src = self.signal_layer_norm(src)
        ### CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        # e_src.shape = N, F2, L'
        # src = self.SE_enc(src)

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

        ### Count Decoder ###
        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusion(src)
        z = src
        
        src_count = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv_count:
            src_count = dconv(src_count)

        src_count = src_count.permute(0, 2, 1) # to N, L, F1
        p, n = self.neg_binom_layer(src_count)

        ### Pval Decoder ###
        src_pval = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv_pval:
            src_pval = dconv(src_pval)

        src_pval = src_pval.permute(0, 2, 1) # to N, L, F1
        mu, var = self.gaussian_layer(src_pval)

        if return_z:
            return p, n, mu, var, z
        else:
            return p, n, mu, var

class CANDI_DNA(nn.Module):
    def __init__(
        self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=2000, pos_enc="relative", expansion_factor=2):
        super(CANDI_DNA, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2
        print("d_model: ", d_model)

        ################################################################################
        # self.convEncDNA_stem = nn.ModuleList(
        #     [ConvTower(
        #         4, self.f2//2, W=conv_kernel_size, S=1, D=1,
        #         pool_type="max", residuals=True, groups=1, pool_size=5),
        #     ConvTower(
        #         self.f2//2, self.f2//2, W=conv_kernel_size, S=1, D=1,
        #         pool_type="max", residuals=True, groups=1, pool_size=5)
        #     ])

        # DNA_conv_channels = exponential_linspace_int(self.f2//2, self.f2, n_cnn_layers+1)
        # DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        # self.convEncDNA = nn.ModuleList(
        #     [ConvTower(
        #         DNA_conv_channels[i], DNA_conv_channels[i + 1],
        #         DNA_kernel_size[i], S=1, D=1,
        #         pool_type="max", residuals=True,
        #         groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])

        ################################################################################

        # self.convEncDNA_stem = nn.ModuleList(
        #     [ConvTower(
        #         4, self.f2//4, W=conv_kernel_size, S=1, D=1,
        #         pool_type="max", residuals=True, groups=1, pool_size=pool_size),
        #     ConvTower(
        #         self.f2//4, self.f2//2, W=conv_kernel_size, S=1, D=1,
        #         pool_type="max", residuals=True, groups=1, pool_size=pool_size)
        #     ])


        # DNA_conv_channels = exponential_linspace_int(self.f2//2, self.f2, n_cnn_layers+1)
        # DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        # self.convEncDNA = nn.ModuleList(
        #     [ConvTower(
        #         DNA_conv_channels[i], DNA_conv_channels[i + 1],
        #         DNA_kernel_size[i], S=1, D=1,
        #         pool_type="max", residuals=True,
        #         groups=1, pool_size=5 if i >= n_cnn_layers - 2 else pool_size) for i in range(n_cnn_layers)])

        ################################################################################

        DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers+2)]

        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size) for i in range(n_cnn_layers + 2)])
        
        ################################################################################

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]

        # self.signal_layer_norm = nn.LayerNorm(self.f1)

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])

        # self.SE_enc = SE_Block_1D(self.f2)

        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        ################################################################################
        # self.SE_DNA_enc = SE_Block_1D(self.f2)
        self.DNA_Epig_fusion = nn.Sequential(
            nn.Linear(2*self.f2, self.f2), 
            nn.LayerNorm(self.f2), 
            nn.ReLU())
        ################################################################################

        self.ymd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.ymd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        if self.pos_enc == "relative":
            self.transformer_encoder = nn.ModuleList([
                RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=expansion_factor*d_model, dropout=dropout) for _ in range(n_sab_layers)])
            
        else:
            self.posEnc = PositionalEncoding(d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=expansion_factor*d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

        self.deconv_count = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size_list[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])

        self.deconv_pval = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size_list[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])
        
        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1, FF=False)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1, FF=False)
    
    def forward(self, src, seq, x_metadata, y_metadata, availability, return_z=False):
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        # availability = torch.where(availability == -2, torch.tensor(-1, device=availability.device), availability)

        ################################################################################
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to N, 4, 25*L
        seq = seq.float()
        # for seq_conv in self.convEncDNA_stem:
        #     seq = seq_conv(seq)

        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)

        # seq = self.SE_DNA_enc(seq)
        seq = seq.permute(0, 2, 1)  # to N, L', F2

        ################################################################################
        # src = self.signal_layer_norm(src)
        ### CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        # e_src.shape = N, F2, L'
        # src = self.SE_enc(src)

        src = src.permute(0, 2, 1)  # to N, L', F2
        xmd_embedding = self.xmd_emb(x_metadata)
        src = torch.cat([src, xmd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.xmd_fusion(src)
        
        ################################################################################
        src = torch.cat([src, seq], dim=-1)
        src = self.DNA_Epig_fusion(src)
        ################################################################################

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            src = self.posEnc(src)
            src = self.transformer_encoder(src)
        else:
            for enc in self.transformer_encoder:
                src = enc(src)

        ### Count Decoder ###
        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusion(src)
        z = src

        src_count = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv_count:
            src_count = dconv(src_count)

        src_count = src_count.permute(0, 2, 1) # to N, L, F1
        p, n = self.neg_binom_layer(src_count)

        ### Pval Decoder ###
        src_pval = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv_pval:
            src_pval = dconv(src_pval)

        src_pval = src_pval.permute(0, 2, 1) # to N, L, F1
        mu, var = self.gaussian_layer(src_pval)

        if return_z:
            return p, n, mu, var, z

        else:
            return p, n, mu, var

class CANDI_LOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(CANDI_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction, full=True)#, eps= 1e-3)
        self.mse = nn.MSELoss(reduction=reduction)
        self.nbin_nll = negative_binomial_loss
    
    def mse_loss(self, Y_true, Y_pred):
        return torch.square(torch.subtract(Y_true,Y_pred)).mean() 

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
        # observed_pval_loss = self.mse_loss(ups_mu_pred, ups_true_pval)
        # imputed_pval_loss = self.mse_loss(imp_mu_pred, imp_true_pval)

        observed_pval_loss = observed_pval_loss.float()
        imputed_pval_loss = imputed_pval_loss.float()
        
        return observed_count_loss, imputed_count_loss, observed_pval_loss, imputed_pval_loss

class PRETRAIN(object):
    def __init__(self, model, dataset, criterion, optimizer, scheduler, device=None, HPO=False):
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

    def pretrain_CANDI(
        self, num_epochs, context_length, batch_size, inner_epochs, 
        arch="", mask_percentage=0.15, hook=True, DNA=False, 
        early_stop=True, early_stop_metric="imp_pval_r2", early_stop_delta=0.03, patience=1):

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

        if "_prog_unmask" in arch or "_prog_mask" in arch:
            N = len(self.dataset.m_regions) * len(self.dataset.dsf_list) * self.dataset.num_batches
            mask_step = N // (num_assays - 1)

        if hook:
            register_hooks(self.model)
        
        if "eic" in arch:
            val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, token_dict=token_dict, eic=True, DNA=DNA, device=self.device)
        else:
            val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, token_dict=token_dict, eic=False, DNA=DNA, device=self.device)

        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)

        best_metric = None

        for epoch in range(num_epochs):
            if early_stop:
                epoch_rec = {
                    "ups_count_r2":[], "imp_count_r2":[],
                    "ups_pval_r2":[], "imp_pval_r2":[],
                    "ups_count_spearman":[], "imp_count_spearman":[],
                    "ups_pval_spearman":[], "imp_pval_spearman":[],
                    "ups_count_pearson":[], "imp_count_pearson":[],
                    "ups_pval_pearson":[], "imp_pval_pearson":[],

                    "val_count_mean_ups_r2":[], "val_count_mean_imp_r2":[], 
                    "val_pval_mean_ups_r2":[], "val_pval_mean_imp_r2":[], 
                    "val_count_mean_ups_pcc":[], "val_count_mean_imp_pcc":[], 
                    "val_pval_mean_ups_pcc":[], "val_pval_mean_imp_pcc":[], 
                    "val_count_mean_ups_srcc":[], "val_count_mean_imp_srcc":[], 
                    "val_pval_mean_ups_srcc":[], "val_pval_mean_imp_srcc":[]
                    }

            self.dataset.new_epoch()
            next_epoch = False

            if "_prog_unmask" in arch or "_prog_mask" in arch:
                M_i = 1
                if "_prog_unmask" in arch:
                    num_mask = num_assays - 1

                elif "_prog_mask" in arch:
                    num_mask = 1

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                if DNA:
                    _X_batch, _mX_batch, _avX_batch, _dnaseq_batch= self.dataset.get_batch(side="x", dna_seq=True)
                else:
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

                if "_prog_unmask" in arch or "_prog_mask" in arch:
                    if M_i % mask_step == 0:
                        if "_prog_unmask" in arch:
                            if num_mask > 1:
                                num_mask -= 1

                        elif "_prog_mask" in arch:
                            if num_mask < num_assays:
                                num_mask += 1

                    M_i += 1

                for _ in range(inner_epochs):
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

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
                        # X_batch, mX_batch, avX_batch = self.masker.mask_chunk_features_30(X_batch, mX_batch, avX_batch)

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
                    
                    if "_prog_unmask" in arch or "_prog_mask" in arch or "random_mask" in arch:
                        msk_p = float(num_mask/num_assays)
                        
                        if "imponly" in arch:
                            loss = (msk_p*(imp_count_loss + imp_pval_loss))
                        else:
                            loss = (msk_p*(imp_count_loss + imp_pval_loss)) + ((1-msk_p)*(obs_pval_loss + obs_count_loss))

                    else:
                        # loss = (mask_percentage*(obs_count_loss + obs_pval_loss)) + ((1-mask_percentage)*(imp_pval_loss + imp_count_loss))
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
                    
                    self.optimizer.step()
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

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                del _X_batch, _mX_batch, _avX_batch, _Y_batch, _mY_batch, _avY_batch, _pval_batch
                if DNA:
                    del _dnaseq_batch
                gc.collect()

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
                    f"Gradient_Norm {np.mean(batch_rec['grad_norm']):.2f}",
                    f"took {int(minutes)}:{int(seconds):02d}"
                ]
                if "_prog_unmask" in arch or "_prog_mask" in arch or "_random_mask" in arch:
                    logstr.append(f"num_mask {num_mask}")
                    logstr.append("\n")
                else:
                    logstr.append("\n")

                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                logfile = open(f"models/CANDI{arch}_log.txt", "w")
                logfile.write("\n".join(log_strs))
                logfile.close()
                
                #################################################################################
                #################################################################################
                if early_stop:
                    epoch_rec["imp_count_r2"].append(np.mean(batch_rec['imp_count_r2']))
                    # epoch_rec["ups_count_r2"].append(np.mean(batch_rec['ups_count_r2']))
                    epoch_rec["imp_pval_r2"].append(np.mean(batch_rec['imp_pval_r2']))
                    # epoch_rec["ups_pval_r2"].append(np.mean(batch_rec['ups_pval_r2']))
                    epoch_rec["imp_count_spearman"].append(np.mean(batch_rec['imp_count_spearman']))
                    # epoch_rec["ups_count_spearman"].append(np.mean(batch_rec['ups_count_spearman']))
                    epoch_rec["imp_pval_spearman"].append(np.mean(batch_rec['imp_pval_spearman']))
                    # epoch_rec["ups_pval_spearman"].append(np.mean(batch_rec['ups_pval_spearman']))
                    epoch_rec["imp_count_pearson"].append(np.mean(batch_rec['imp_count_pearson']))
                    # epoch_rec["ups_count_pearson"].append(np.mean(batch_rec['ups_count_pearson']))
                    epoch_rec["imp_pval_pearson"].append(np.mean(batch_rec['imp_pval_pearson']))
                    # epoch_rec["ups_pval_pearson"].append(np.mean(batch_rec['ups_pval_pearson']))
                #################################################################################
                #################################################################################

                chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                dsf_pointer0 = self.dataset.dsf_pointer
                bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                dsf_pointer1 = self.dataset.dsf_pointer
                chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                bios_pointer1 = self.dataset.bios_pointer

                # if dsf_pointer0 != dsf_pointer1 or chr0 != chr1 or bios_pointer0 != bios_pointer1:
                if self.HPO==False and chr0 != chr1:
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

                # if next_epoch:
                #     validation_set_eval, val_metrics = val_eval.get_validation(self.model)
                #     torch.cuda.empty_cache()
                #     log_strs.append(validation_set_eval)
                #     print(validation_set_eval)
                #     log_resource_usage()

                #     if early_stop:
                #         # epoch_rec["val_count_mean_ups_r2"].append(val_metrics["upsampled_counts"]["R2_count"]["mean"])
                #         epoch_rec["val_count_mean_imp_r2"].append(val_metrics["imputed_counts"]["R2_count"]["mean"])
                #         # epoch_rec["val_count_mean_ups_pcc"].append(val_metrics["upsampled_counts"]["PCC_count"]["mean"])
                #         epoch_rec["val_count_mean_imp_pcc"].append(val_metrics["imputed_counts"]["PCC_count"]["mean"])
                #         # epoch_rec["val_count_mean_ups_srcc"].append(val_metrics["upsampled_counts"]["SRCC_count"]["mean"])
                #         epoch_rec["val_count_mean_imp_srcc"].append(val_metrics["imputed_counts"]["SRCC_count"]["mean"])
                        
                #         # epoch_rec["val_pval_mean_ups_r2"].append(val_metrics["upsampled_pvals"]["R2_pval"]["mean"])
                #         epoch_rec["val_pval_mean_imp_r2"].append(val_metrics["imputed_pvals"]["R2_pval"]["mean"])
                #         # epoch_rec["val_pval_mean_ups_pcc"].append(val_metrics["upsampled_pvals"]["PCC_pval"]["mean"])
                #         epoch_rec["val_pval_mean_imp_pcc"].append(val_metrics["imputed_pvals"]["PCC_pval"]["mean"])
                #         # epoch_rec["val_pval_mean_ups_srcc"].append(val_metrics["upsampled_pvals"]["SRCC_pval"]["mean"])
                #         epoch_rec["val_pval_mean_imp_srcc"].append(val_metrics["imputed_pvals"]["SRCC_pval"]["mean"])

            self.scheduler.step()
            print("learning rate scheduler step...")

            if early_stop:
                # Initialize the best metrics if it's the first epoch
                if best_metric is None:
                    best_metric = {key: None for key in epoch_rec.keys()}
                    patience_counter = {key: 0 for key in epoch_rec.keys()}

                # Loop over all metrics
                for metric_name in epoch_rec.keys():
                    current_metric = np.mean(epoch_rec[metric_name])  # Calculate the current epoch's mean for this metric

                    if best_metric[metric_name] is None or current_metric > best_metric[metric_name] + early_stop_delta:
                        best_metric[metric_name] = current_metric  # Update the best metric for this key
                        patience_counter[metric_name] = 0  # Reset the patience counter
                    else:
                        patience_counter[metric_name] += 1  # Increment the patience counter if no improvement

                # Check if all patience counters have exceeded the limit (e.g., 3 epochs of no improvement)
                if all(patience_counter[metric] >= patience for metric in patience_counter.keys()):
                    print(f"Early stopping at epoch {epoch}. No significant improvement across metrics.")
                    logfile = open(f"models/CANDI{arch}_log.txt", "w")
                    logfile.write(f"best metric records: \n{best_metric}")
                    logfile.close()
                    return  self.model
                else:
                    print(f"best metric records so far: \n{best_metric}")
                    logfile = open(f"models/CANDI{arch}_log.txt", "w")
                    logfile.write(f"best metric records so far: \n{best_metric}")
                    logfile.close()
                
            if self.HPO==False and epoch%5==0 and epoch != (num_epochs-1):
                try:
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
    data_path = hyper_parameters["data_path"]
    
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    n_sab_layers = hyper_parameters["n_sab_layers"]
    
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
    expansion_factor = hyper_parameters["expansion_factor"]
    pos_enc = hyper_parameters["pos_enc"]

    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, loci_gen="debug",#"ccre", #["chr19", "chr20"], 
        bios_min_exp_avail_threshold=min_avail, check_completeness=True, eic=eic)

    signal_dim = dataset.signal_dim
    metadata_embedding_dim = dataset.signal_dim

    if DNA:
        model = CANDI_DNA(
            signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
            pos_enc=pos_enc, expansion_factor=expansion_factor)
    else:
        model = CANDI(
            signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length,
            pos_enc=pos_enc, expansion_factor=expansion_factor)

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_halflife, gamma=0.9)

    if checkpoint_path is not None:
        print("loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"CANDI_{arch} # model_parameters: {count_parameters(model)}")
    summary(model)
    
    model_name = f"CANDI{arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters_{arch}_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = CANDI_LOSS()

    start_time = time.time()

    trainer = PRETRAIN(model, dataset, criterion, optimizer, scheduler, device=device, HPO=HPO)
    model, best_metric = trainer.pretrain_CANDI(
        num_epochs=epochs, mask_percentage=mask_percentage, context_length=context_length, 
        batch_size=batch_size, inner_epochs=inner_epochs, arch=arch, DNA=DNA)

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

        if self.DNA:
            model = CANDI_DNA(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length)
        else:
            model = CANDI(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 
        model = model.to(self.device)
        return model
    
    
def main():
    parser = argparse.ArgumentParser(description="Train the model with specified hyperparameters")

    # Hyperparameters
    parser.add_argument('--data_path', type=str, default="/project/compbio-lab/encode_data/", help='Path to the data')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--n_cnn_layers', type=int, default=3, help='Number of CNN layers')
    parser.add_argument('--conv_kernel_size', type=int, default=5, help='Convolution kernel size')
    parser.add_argument('--pool_size', type=int, default=2, help='Pooling size')
    parser.add_argument('--expansion_factor', type=int, default=2, help='Expansion factor for the model')

    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_sab_layers', type=int, default=4, help='Number of SAB layers')
    parser.add_argument('--pos_enc', type=str, default="relative", help='Transformer Positional Encodings')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--inner_epochs', type=int, default=1, help='Number of inner epochs')
    parser.add_argument('--mask_percentage', type=float, default=0.2, help='Masking percentage (if used)')
    parser.add_argument('--context_length', type=int, default=800, help='Context length')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_loci', type=int, default=3750, help='Number of loci')
    parser.add_argument('--lr_halflife', type=int, default=1, help='Learning rate halflife')
    parser.add_argument('--min_avail', type=int, default=5, help='Minimum available')
    

    # Flags for DNA and EIC
    parser.add_argument('--eic', action='store_true', help='Flag to enable EIC')
    parser.add_argument('--dna', action='store_true', help='Flag to enable DNA')
    parser.add_argument('--prog_mask', action='store_true', help='Flag to enable progressive masking')

    # Parse the arguments
    args = parser.parse_args()

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
        "min_avail": args.min_avail
    }

    # Call your training function with parsed arguments
    Train_CANDI(hyper_parameters, eic=args.eic, DNA=args.dna, suffix="oct17-expan2", prog_mask=args.prog_mask)

if __name__ == "__main__":
    main()