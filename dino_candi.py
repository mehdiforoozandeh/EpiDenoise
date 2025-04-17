from CANDI import *  # Import all components from your CANDI model (including your encoder, decoder, etc.)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gc
import os
import math
from tqdm import tqdm


###############################################
# CANDI_DNA_Encoder (with DNA) with Projection Head
###############################################

class DINO_CANDI_DNA_Encoder(nn.Module):
    def __init__(
        self,
        signal_dim,
        metadata_embedding_dim,
        conv_kernel_size,
        n_cnn_layers,
        nhead,
        n_sab_layers,
        pool_size=2,
        dropout=0.1,
        context_length=1600,
        pos_enc="relative",
        expansion_factor=3,
        pooling_type="attention"):
        
        super().__init__()
        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        self.f1 = signal_dim
        self.f2 = self.f1 * (expansion_factor**n_cnn_layers)
        d_model = self.latent_dim = self.f2

        # DNA convolution stack
        DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        self.convEncDNA = nn.ModuleList([
            ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i+1], conv_kernel_size,
                S=1, D=1, pool_type="max", residuals=True, SE=False,
                groups=1,
                pool_size=5 if i >= n_cnn_layers else pool_size
            ) for i in range(n_cnn_layers+2)
        ])

        # Signal convolution stack
        conv_channels = [self.f1 * (expansion_factor**i) for i in range(n_cnn_layers)]
        self.convEnc = nn.ModuleList([
            ConvTower(
                conv_channels[i],
                conv_channels[i+1] if i+1 < n_cnn_layers else expansion_factor*conv_channels[i],
                conv_kernel_size,
                S=1, D=1, pool_type="avg", residuals=True,
                groups=self.f1, SE=False, pool_size=pool_size
            ) for i in range(n_cnn_layers)
        ])

        # Metadata embedding + fusion
        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=False)
        self.fusion = nn.Sequential(
            nn.Linear((2*self.f2)+metadata_embedding_dim, self.f2),
            nn.LayerNorm(self.f2)
        )

        # Transformer encoder blocks
        self.transformer_encoder = nn.ModuleList([
            DualAttentionEncoderBlock(self.f2, nhead, self.l2, dropout=dropout,
                                      max_distance=self.l2, pos_encoding_type="relative",
                                      max_len=self.l2)
            for _ in range(n_sab_layers)
        ])

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # Pooling options
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attn_pool = nn.Linear(self.latent_dim, 1)
        elif pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

    def forward(self, src, seq, x_metadata, return_projected=True):
        # DNA conv
        if seq.dim() != src.dim():
            seq = seq.unsqueeze(0).expand(src.size(0), -1, -1)
        seq = seq.permute(0, 2, 1).float()
        for conv in self.convEncDNA:
            seq = conv(seq)
        seq = seq.permute(0, 2, 1)

        # Signal conv
        src = src.permute(0, 2, 1)
        for conv in self.convEnc:
            src = conv(src)
        src = src.permute(0, 2, 1)

        # Metadata embed + fusion
        xmd = self.xmd_emb(x_metadata).unsqueeze(1).expand(-1, self.l2, -1)
        x = torch.cat([src, xmd, seq], dim=-1)
        x = self.fusion(x)

        # CLS token prep if needed
        if self.pooling_type == "cls":
            bsz = x.size(0)
            cls_tokens = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Transformer blocks
        for blk in self.transformer_encoder:
            x = blk(x)

        # Pooling
        if self.pooling_type == "mean":
            rep = x.mean(dim=1)
        elif self.pooling_type == "attention":
            scores = self.attn_pool(x).squeeze(-1)       # [bsz, seq_len]
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            rep = (weights * x).sum(dim=1)
        elif self.pooling_type == "cls":
            rep = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        if return_projected:
            return self.projection_head(rep)
        else:
            if self.pooling_type == "cls":
                return x[:, 1:, :]
            else:
                return x

###############################################
# CANDI_Decoder & loss
###############################################

class DINO_CANDI_Decoder(nn.Module):
    def __init__(
        self, signal_dim, metadata_embedding_dim, conv_kernel_size,
         n_cnn_layers, context_length, pool_size=2, expansion_factor=3):
        super(DINO_CANDI_Decoder, self).__init__()

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
            # nn.ReLU()
            )

        self.deconv_c = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])

        self.deconv_p = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=1, pool_size=pool_size) for i in range(n_cnn_layers)])

        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1)
    
    def forward(self, src, y_metadata):
        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusion(src)
        src = src.permute(0, 2, 1) # to N, F2, L'

        upsampled_p = src
        upsampled_c = src

        for dconv in self.deconv_c:
            upsampled_c = dconv(upsampled_c)

        for dconv in self.deconv_p:
            upsampled_p = dconv(upsampled_p)

        upsampled_p = upsampled_p.permute(0, 2, 1) # to N, L, F1
        upsampled_c = upsampled_c.permute(0, 2, 1) # to N, L, F1

        p, n = self.neg_binom_layer(upsampled_c)
        mu, var = self.gaussian_layer(upsampled_p)

        return p, n, mu, var

class CANDI_Decoder_LOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(CANDI_Decoder_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction, full=True)
        self.nbin_nll = negative_binomial_loss

    def forward(self, p_pred, n_pred, mu_pred, var_pred, true_count, true_pval, obs_map):
        ups_true_count, ups_true_pval = true_count[obs_map], true_pval[obs_map]
        ups_n_pred, ups_p_pred = n_pred[obs_map], p_pred[obs_map]
        ups_mu_pred, ups_var_pred = mu_pred[obs_map], var_pred[obs_map]

        observed_count_loss = self.nbin_nll(ups_true_count, ups_n_pred, ups_p_pred) 

        if self.reduction == "mean":
            observed_count_loss = observed_count_loss.mean()
        elif self.reduction == "sum":
            observed_count_loss = observed_count_loss.sum()

        observed_pval_loss = self.gaus_nll(ups_mu_pred, ups_true_pval, ups_var_pred)
        observed_pval_loss = observed_pval_loss.float()
        
        return observed_count_loss, observed_pval_loss

###############################################
# DINO_CANDI Class: DINO-style training for CANDI's encoder
###############################################

class DINO_CANDI:
    def __init__(
        self, student_encoder, teacher_encoder, dataset, 
        optimizer, ema_decay, center_update, t_student,
        t_teacher, device_student, device_teacher,

        decoder, decoder_optimizer, decoder_dataset, decoder_criterion):
        """
        Initialize the DINO_CANDI training framework.

        Args:
            student_encoder (nn.Module): The student encoder (instance of your CANDI encoder).
            teacher_encoder (nn.Module): The teacher encoder (same architecture, updated by EMA).
            optimizer (torch.optim.Optimizer): Optimizer for the student encoder.
            num_epochs (int): Number of epochs to train.
            ema_decay (float): EMA decay coefficient for teacher updates.
            center_update (float): Coefficient for updating the center vector.
            t_student (float): Temperature parameter for student outputs.
            t_teacher (float): Temperature parameter for teacher outputs.
            device_student (torch.device): Device for student (e.g., cuda:0).
            device_teacher (torch.device): Device for teacher (e.g., cuda:1).
        """
        self.device = device_student
        self.student = student_encoder.to(device_student)
        self.teacher = teacher_encoder.to(device_teacher)
        self.dataset = dataset
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.center_update = center_update
        self.t_student = t_student
        self.t_teacher = t_teacher
        self.device_student = device_student
        self.device_teacher = device_teacher

        self.token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        self.masker = DataMasker(self.token_dict["cloze_mask"], 0.15)

        self.decoder = decoder.to(self.device)
        self.decoder_optimizer = decoder_optimizer

        self.decoder_dataset = decoder_dataset
        self.decoder_dataset.new_epoch()

        self.decoder_criterion = decoder_criterion

        # Initialize a center vector based on the output projection dimension of your encoder.
        self.center = torch.zeros(self.student.projection_head[-1].out_features, device=device_student)
        
    def update_teacher(self):
        """
        Update the teacher parameters using an EMA on the student parameters.
        """
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data.to(self.device_teacher)

    def generate_local_views(self, X_batch, mX_batch, avX_batch): # input is global view
        num_mask = random.randint(1, self.dataset.signal_dim - 1)
        return self.masker.progressive(X_batch, mX_batch, avX_batch, num_mask)

    def dino_loss(self, student_out, teacher_out):
        """
        Compute the DINO loss between the student output and teacher output.
        
        Both outputs are assumed to have shape [batch_size, projection_dim]. 
        Teacher outputs should be detached and centered using self.center.
        
        Args:
            student_out (Tensor): Student output from a view.
            teacher_out (Tensor): Teacher output from the global view.
        
        Returns:
            loss_val (Tensor): Scalar loss value.
        """
        # Stop gradient on teacher.
        teacher_out = teacher_out.detach()
        
        # Apply softmax with temperature scaling.
        student_probs = F.softmax(student_out / self.t_student, dim=1)
        teacher_probs = F.softmax((teacher_out - self.center) / self.t_teacher, dim=1)
        
        # Compute cross-entropy loss (averaged over the batch).
        loss_val = - (teacher_probs * torch.log(student_probs + 1e-7)).sum(dim=1).mean()
        return loss_val

    def train_dino(self, num_epochs, context_length, batch_size, inner_epochs, 
        arch="", hook=False, DNA=True, 
        early_stop=True, accumulation_steps=1, num_local_views=1):

        self.log_strs = []
        self.log_strs.append(str(self.device))
        self.log_strs.append(f"DINO CANDI{arch} # model_parameters: {count_parameters(self.student)}")
        logfile = open(f"models/DINO_CANDI{arch}_log.txt", "w")
        logfile.write("\n".join(self.log_strs))
        logfile.close()
        
        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                self.teacher.eval()
                self.student.train()

                if DNA:
                    _X_batch, _mX_batch, _avX_batch, _dnaseq_batch= self.dataset.get_batch(side="x", dna_seq=True)
                else:
                    _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(side="x")

                _Y_batch, _mY_batch, _avY_batch, _pval_batch = self.dataset.get_batch(side="y", pval=True)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                    
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                for _ in range(inner_epochs):
                    if DNA:
                        X_batch, mX_batch, avX_batch, dnaseq_batch= _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone(), _dnaseq_batch.clone()
                    else:
                        X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                        
                    Y_batch, mY_batch, avY_batch, pval_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone(), _pval_batch.clone()

                    masked_map = (X_batch == self.token_dict["cloze_mask"])
                    observed_map = (X_batch != self.token_dict["missing_mask"]) & (X_batch != self.token_dict["cloze_mask"])
                    missing_map = (X_batch == self.token_dict["missing_mask"])

                    # masked_map = masked_map.to(self.device) # imputation targets
                    # observed_map = observed_map.to(self.device) # upsampling targets
                    # pval_batch = pval_batch.to(self.device)

                    
                    Y_batch = Y_batch.float().to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    
                    X_batch = X_batch.float().to(self.device)
                    mX_batch = mX_batch.to(self.device)
                    
                    avX_batch = avX_batch.to(self.device)
                    
                    local_views = []
                    for gen_local in range(num_local_views):
                        local_views.append(
                            (self.generate_local_views(X_batch, mX_batch, avX_batch))
                        )
                    n_views = len(local_views)

                    student_outputs = []
                    if DNA:
                        dnaseq_batch = dnaseq_batch.to(self.device)
                        with torch.no_grad():
                            teacher_output = self.teacher(Y_batch, dnaseq_batch, mY_batch)

                        student_outputs.append(self.student(Y_batch, dnaseq_batch, mY_batch))
                        student_outputs.append(self.student(X_batch, dnaseq_batch, mX_batch))
                        for loc in local_views:
                            student_outputs.append(self.student(loc[0], dnaseq_batch, loc[1]))

                    else:
                        with torch.no_grad():
                            teacher_output = self.teacher(Y_batch, mY_batch)

                        student_outputs.append(self.student(Y_batch, mY_batch))
                        student_outputs.append(self.student(X_batch, mX_batch))
                        for loc in local_views:
                            student_outputs.append(self.student(loc[0], loc[1]))

                    # -----------------------------
                    # Compute loss, student entropy, teacher entropy, and KL divergence.
                    max_entropy = math.log(teacher_output.size(1))
                    
                    # Compute normalized teacher entropy.
                    teacher_probs = F.softmax(teacher_output, dim=1)
                    teacher_entropy = - (teacher_probs * torch.log(teacher_probs + 1e-7)).sum(dim=1).mean().item()
                    normalized_teacher_entropy = teacher_entropy / max_entropy
                    
                    loss_sum = 0.0
                    student_entropies = []
                    for view_out in student_outputs:
                        loss_sum += self.dino_loss(view_out, teacher_output)
                        s_probs = F.softmax(view_out, dim=1)
                        s_entropy = - (s_probs * torch.log(s_probs + 1e-7)).sum(dim=1).mean().item()
                        student_entropies.append(s_entropy)
                    loss = loss_sum / len(student_outputs)
                    avg_student_entropy = sum(student_entropies) / len(student_entropies)
                    normalized_student_entropy = avg_student_entropy / max_entropy

                    # Compute KL divergence.
                    kl_div_sum = 0.0
                    for view_out in student_outputs:
                        student_log_probs = F.log_softmax(view_out, dim=1)
                        teacher_probs_scaled = F.softmax(teacher_output, dim=1)
                        kl_div = F.kl_div(student_log_probs, teacher_probs_scaled, reduction='batchmean')
                        kl_div_sum += kl_div.item()
                    batch_avg_kl_div = kl_div_sum / len(student_outputs)
                    # epoch_kl_div += batch_avg_kl_div
                    # -----------------------------

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, teacher_output, loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss = loss.float()
                    loss.backward()
                    del X_batch, mX_batch, mY_batch, avX_batch, Y_batch, pval_batch, observed_map, masked_map

                self.optimizer.step() # update student
                self.update_teacher()

                del _X_batch, _mX_batch, _avX_batch, _Y_batch, _mY_batch, _avY_batch, _pval_batch
                if DNA:
                    del _dnaseq_batch
                gc.collect()

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                logstr = [
                    "ENCODER",
                    f"Ep. {epoch}",
                    f"Loss: {loss.item():.4f}",
                    f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                    f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer / len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                    f"Bios Prog. {self.dataset.bios_pointer / self.dataset.num_bios:.2%}",
                    f"T_Ent: {normalized_teacher_entropy:.4f}",
                    f"S_Ent: {normalized_student_entropy:.4f}",
                    f"KL_div: {batch_avg_kl_div:.4f}",
                    f"took {int(minutes)}:{int(seconds):02d}",
                ]
                logstr = " | ".join(logstr)
                self.log_strs.append(logstr)
                print(logstr)

                chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                dsf_pointer0 = self.dataset.dsf_pointer
                bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                dsf_pointer1 = self.dataset.dsf_pointer
                chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                bios_pointer1 = self.dataset.bios_pointer

                # if chr0 != chr1 or dsf_pointer0 != dsf_pointer1 or bios_pointer0 != bios_pointer1:
                if chr0 != chr1:
                    logfile = open(f"models/DINO_CANDI{arch}_log.txt", "w")
                    logfile.write("\n".join(self.log_strs))
                    logfile.close()

                    self.train_decoder(context_length, batch_size, arch=arch)
        
    def train_decoder(self, context_length, batch_size, early_stop=True, DNA=True, arch=""):
        next_epoch = False
        self.student.eval()
        self.decoder.train()
        
        while (next_epoch==False):
            t0 = datetime.now()
            batch_rec = {
                "ups_count_loss":[], "ups_pval_loss":[],
                "ups_count_r2":[], "ups_pval_r2":[],
                "ups_count_pp":[], "ups_pval_pp":[],
                "ups_count_conf":[], "ups_pval_conf":[], 
                "ups_count_mse":[],  "ups_pval_mse":[], 
                "ups_count_spearman":[], "ups_pval_spearman":[],
                "ups_count_pearson":[], "ups_pval_pearson":[], 
                }

            if DNA:
                _X_batch, _mX_batch, _avX_batch, _dnaseq_batch= self.decoder_dataset.get_batch(side="x", dna_seq=True)
            else:
                _X_batch, _mX_batch, _avX_batch = self.decoder_dataset.get_batch(side="x")

            _Y_batch, _mY_batch, _avY_batch, _pval_batch = self.decoder_dataset.get_batch(side="y", pval=True)

            if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                self.decoder_dataset.update_batch_pointers()
                print("mismatch in shapes! skipped batch...")
                continue
                
            self.decoder_optimizer.zero_grad()
            torch.cuda.empty_cache()

            for _ in range(1):
                if DNA:
                    X_batch, mX_batch, avX_batch, dnaseq_batch= _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone(), _dnaseq_batch.clone()
                else:
                    X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    
                Y_batch, mY_batch, avY_batch, pval_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone(), _pval_batch.clone()

                masked_map = (X_batch == self.token_dict["cloze_mask"])
                observed_map = (X_batch != self.token_dict["missing_mask"]) & (X_batch != self.token_dict["cloze_mask"])
                missing_map = (X_batch == self.token_dict["missing_mask"])

                masked_map = masked_map.to(self.device) # imputation targets
                observed_map = observed_map.to(self.device) # upsampling targets
                pval_batch = pval_batch.to(self.device)

                Y_batch = Y_batch.float().to(self.device)
                mY_batch = mY_batch.to(self.device)

                X_batch = X_batch.float().to(self.device)
                mX_batch = mX_batch.to(self.device)
                avX_batch = avX_batch.to(self.device)

                ###################################
                dnaseq_batch = dnaseq_batch.to(self.device)
                with torch.no_grad():
                    latent = self.student(X_batch, dnaseq_batch, mX_batch, return_projected=False)
                output_p, output_n, output_mu, output_var = self.decoder(latent, mY_batch)

                count_loss, pval_loss = self.decoder_criterion(
                    output_p, output_n, output_mu, output_var, Y_batch, pval_batch, observed_map)
                ###################################
                loss = count_loss + pval_loss
                if torch.isnan(loss).sum() > 0:
                    skipmessage = "Encountered nan loss! Skipping batch..."
                    log_strs.append(skipmessage)
                    del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss
                    print(skipmessage)
                    torch.cuda.empty_cache() 
                    continue
                
                loss = loss.float()
                loss.backward()

                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), clip_value=5)

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

                del X_batch, mX_batch, mY_batch, avX_batch, Y_batch, pval_batch, observed_map, masked_map

                batch_rec["ups_count_loss"].append(count_loss.item())
                batch_rec["ups_pval_loss"].append(pval_loss.item())

                batch_rec["ups_count_r2"].append(ups_count_r2)
                batch_rec["ups_pval_r2"].append(ups_pval_r2)

                batch_rec["ups_count_pp"].append(ups_count_pp)
                batch_rec["ups_pval_pp"].append(ups_pval_pp)

                batch_rec["ups_count_conf"].append(ups_count_errstd)
                batch_rec["ups_pval_conf"].append(ups_pval_errstd)

                batch_rec["ups_count_mse"].append(ups_count_mse)
                batch_rec["ups_pval_mse"].append(ups_pval_mse)

                batch_rec["ups_count_spearman"].append(ups_count_spearman)
                batch_rec["ups_pval_spearman"].append(ups_pval_spearman)

                batch_rec["ups_count_pearson"].append(ups_count_pearson)
                batch_rec["ups_pval_pearson"].append(ups_pval_pearson)

            self.decoder_optimizer.step()

            del _X_batch, _mX_batch, _avX_batch, _Y_batch, _mY_batch, _avY_batch, _pval_batch
            if DNA:
                del _dnaseq_batch
            gc.collect()

            elapsed_time = datetime.now() - t0
            hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logstr = [
                "\tDECODER",
                # f"Ep. {epoch}",
                f"Loss: {loss.item():.4f}",
                f"DSF{self.decoder_dataset.dsf_list[self.decoder_dataset.dsf_pointer]}->{1}",
                f"{list(self.decoder_dataset.loci.keys())[self.decoder_dataset.chr_pointer]} Prog. {self.decoder_dataset.chr_loci_pointer / len(self.decoder_dataset.loci[list(self.decoder_dataset.loci.keys())[self.decoder_dataset.chr_pointer]]):.2%}",
                f"Bios Prog. {self.decoder_dataset.bios_pointer / self.decoder_dataset.num_bios:.2%}",  "\n\t",
                
                f"nbNLL {np.mean(batch_rec['ups_count_loss']):.2f}",
                f"gNLL {np.mean(batch_rec['ups_pval_loss']):.2f}", 
                f"Ct_R2 {np.mean(batch_rec['ups_count_r2']):.2f}", 
                f"P_R2 {np.mean(batch_rec['ups_pval_r2']):.2f}",  "\n\t",

                f"Ct_SRCC {np.mean(batch_rec['ups_count_spearman']):.2f}",
                f"P_SRCC {np.mean(batch_rec['ups_pval_spearman']):.2f}",
                f"Ct_PCC {np.mean(batch_rec['ups_count_pearson']):.2f}",
                f"P_PCC {np.mean(batch_rec['ups_pval_pearson']):.2f}",  "\n\t",

                f"Ct_PPL {np.mean(batch_rec['ups_count_pp']):.2f}",
                f"P_PPL {np.mean(batch_rec['ups_pval_pp']):.2f}",
                f"Ct_Conf {np.mean(batch_rec['ups_count_conf']):.2f}",
                f"P_Conf {np.mean(batch_rec['ups_pval_conf']):.2f}",  "\n\t",
                f"took {int(minutes)}:{int(seconds):02d}",
            ]
            logstr = " | ".join(logstr)
            self.log_strs.append(logstr)
            print(logstr)

            chr0 = list(self.decoder_dataset.loci.keys())[self.decoder_dataset.chr_pointer]
            dsf_pointer0 = self.decoder_dataset.dsf_pointer
            bios_pointer0 = self.decoder_dataset.bios_pointer

            next_epoch = self.decoder_dataset.update_batch_pointers()
            
            if next_epoch:
                self.decoder_dataset.new_epoch()
                
            dsf_pointer1 = self.decoder_dataset.dsf_pointer
            chr1 = list(self.decoder_dataset.loci.keys())[self.decoder_dataset.chr_pointer]
            bios_pointer1 = self.decoder_dataset.bios_pointer

            # if chr0 != chr1 or dsf_pointer0 != dsf_pointer1 or bios_pointer0 != bios_pointer1:
            if chr0 != chr1:
                logfile = open(f"models/DINO_CANDI{arch}_log.txt", "w")
                logfile.write("\n".join(self.log_strs))
                logfile.close()
                return

###############################################
# Main function: Parse arguments and run training.
###############################################

def main():
    context_length = 1200
    num_epochs = 100            # Adjust as needed.
    learning_rate = 1e-3        # Learning rate for the student encoder.
    ema_decay = 0.996            # EMA decay coefficient for teacher updates.
    center_update = 0.9        # Center update coefficient.
    t_student = 0.4             # Temperature for student outputs.
    t_teacher = 0.04            # Temperature for teacher outputs.
    batch_size = 50             # Batch size to be used by your dataset (if applicable).
    inner_epochs = 1            # Number of inner iterations per batch.
    num_local_views = 1    
    loci_gen = "debug"   

    # -------------------------------
    student_encoder = DINO_CANDI_DNA_Encoder(
        signal_dim=35, metadata_embedding_dim=4*35, conv_kernel_size=3, n_cnn_layers=3, nhead=9,
        n_sab_layers=4, pool_size=2, dropout=0.1, context_length=context_length, pos_enc="relative", expansion_factor=3)
                 
    teacher_encoder =  DINO_CANDI_DNA_Encoder(
        signal_dim=35, metadata_embedding_dim=4*35, conv_kernel_size=3, n_cnn_layers=3, nhead=9,
        n_sab_layers=4, pool_size=2, dropout=0.1, context_length=context_length, pos_enc="relative", expansion_factor=3)

    teacher_encoder.load_state_dict(student_encoder.state_dict())

    # -------------------------------
    data_path = "/project/compbio-lab/encode_data/"
    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=10,                  # number of loci
        context_length=context_length*25,     # context length (adjust based on your application)
        bios_batchsize=10,       # batch size for bios samples
        loci_batchsize=1,        # batch size for loci
        loci_gen=loci_gen,         # loci generation method
        bios_min_exp_avail_threshold=7,  # minimum available bios
        check_completeness=True,
        eic=True,
        merge_ct=True
    )

      # Number of local views to generate per batch.
    
    # -------------------------------
    device_student = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_teacher = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else device_student)
    # -------------------------------
    optimizer = optim.SGD(student_encoder.parameters(), lr=learning_rate)

    # -------------------------------
    candi_decoder = DINO_CANDI_Decoder(
        signal_dim=35, metadata_embedding_dim=4*35, conv_kernel_size=3, n_cnn_layers=3, 
        context_length=context_length, pool_size=2, expansion_factor=3)

    decoder_optimizer = optim.Adam(candi_decoder.parameters(), lr=learning_rate)
    decoder_criterion = CANDI_Decoder_LOSS(reduction='mean')
    decoder_dataset = ExtendedEncodeDataHandler(data_path)
    decoder_dataset.initialize_EED(
        m= 10,                  # number of loci
        context_length=context_length*25,
        bios_batchsize=50,       # batch size for bios samples
        loci_batchsize=1,        # batch size for loci
        loci_gen=loci_gen,         # loci generation method
        bios_min_exp_avail_threshold=7,  # minimum available bios
        check_completeness=True,
        eic=True,
        merge_ct=True,
        DSF_list=[1, 2]
    )

    # -------------------------------
    dino_trainer = DINO_CANDI(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        dataset=dataset,
        optimizer=optimizer,
        ema_decay=ema_decay,
        center_update=center_update,
        t_student=t_student,
        t_teacher=t_teacher,
        device_student=device_student,
        device_teacher=device_teacher,
        decoder=candi_decoder, 
        decoder_optimizer=decoder_optimizer,
        decoder_dataset=decoder_dataset,
        decoder_criterion=decoder_criterion
    )

    # -------------------------------
    # Start training.
    dino_trainer.train_dino(
        num_epochs=num_epochs,
        context_length=context_length,
        batch_size=batch_size,
        inner_epochs=inner_epochs,
        arch="",
        hook=False,
        DNA=True,  # Set to True if you use DNA-specific inputs.
        early_stop=True,
        accumulation_steps=5,
        num_local_views=num_local_views
    )

if __name__ == "__main__":
    main()
