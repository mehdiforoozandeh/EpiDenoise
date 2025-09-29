import os, sys, time, json, math, argparse, random
from torch import nn
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Import project modules (do not import from train_candi.py)
from data import ExtendedEncodeDataHandler
# from model import CANDI, CANDI_DNA, CANDI_UNET, MONITOR_VALIDATION
from model import MONITOR_VALIDATION
from _utils import DataMasker, negative_binomial_loss, Gaussian, NegativeBinomial, count_parameters


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


def init_distributed(backend="nccl"):
    """Initialize DDP process group from torchrun environment variables."""
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_model(h, signal_dim, metadata_embedding_dim):
    if h["dna"]:
        if h["unet"]:
            model = CANDI_UNET(
                signal_dim, metadata_embedding_dim, h["conv_kernel_size"], h["n_cnn_layers"],
                h["nhead"], h["n_sab_layers"], pool_size=h["pool_size"], dropout=h["dropout"],
                context_length=h["context_length"], pos_enc=h["pos_enc"],
                expansion_factor=h["expansion_factor"], separate_decoders=h["separate_decoders"])
        else:
            model = CANDI_DNA(
                signal_dim, metadata_embedding_dim, h["conv_kernel_size"], h["n_cnn_layers"],
                h["nhead"], h["n_sab_layers"], pool_size=h["pool_size"], dropout=h["dropout"],
                context_length=h["context_length"], pos_enc=h["pos_enc"],
                expansion_factor=h["expansion_factor"], separate_decoders=h["separate_decoders"])
    else:
        model = CANDI(
            signal_dim, metadata_embedding_dim, h["conv_kernel_size"], h["n_cnn_layers"],
            h["nhead"], h["n_sab_layers"], pool_size=h["pool_size"], dropout=h["dropout"],
            context_length=h["context_length"], pos_enc=h["pos_enc"],
            expansion_factor=h["expansion_factor"], separate_decoders=h["separate_decoders"])
    return model


def build_optimizer_and_scheduler(model, h):
    if h["optim"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=h["learning_rate"])
    elif h["optim"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=h["learning_rate"], weight_decay=0.01)
    elif h["optim"].lower() == "adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=h["learning_rate"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=h["learning_rate"], momentum=0.9)

    if h["LRschedule"] is None:
        cos_sch = False
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=h["lr_halflife"], gamma=1)
    elif h["LRschedule"].lower() == "cosine":
        cos_sch = True
        # approximate total steps to keep behavior similar to single-GPU
        num_total_epochs = h["epochs"] * h["inner_epochs"] * 2
        warmup_epochs = h["inner_epochs"] * 2
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=(num_total_epochs - warmup_epochs), eta_min=0.0),
            ],
            milestones=[warmup_epochs],
        )
    else:
        cos_sch = False
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=h["lr_halflife"], gamma=0.95)

    return optimizer, scheduler, cos_sch


class LossComputer:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.gaus_nll = torch.nn.GaussianNLLLoss(reduction=self.reduction, full=True)
        self.nbin_nll = negative_binomial_loss

    def __call__(self, p_pred, n_pred, mu_pred, var_pred, true_count, true_pval, obs_map, masked_map):
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


def apply_feature_mask(masker, arch_flags, X_batch, mX_batch, avX_batch, num_assays):
    if "random_mask" in arch_flags:
        num_mask = random.randint(1, num_assays - 1)
        X_batch, mX_batch, avX_batch = masker.progressive(X_batch, mX_batch, avX_batch, num_mask)
        return X_batch, mX_batch, avX_batch, num_mask
    else:
        X_batch, mX_batch, avX_batch = masker.mask_feature30(X_batch, mX_batch, avX_batch)
        return X_batch, mX_batch, avX_batch, None


class TrainerDDP:
    def __init__(self, h, device, rank, world_size):
        self.h = h
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # dataset init (sharded by bios_batchsize across processes via pointers stride)
        self.dataset = ExtendedEncodeDataHandler(h["data_path"])
        self.dataset.initialize_EED(
            m=h["num_loci"], context_length=h["context_length"] * 25,
            bios_batchsize=h["batch_size"], loci_batchsize=1, loci_gen=h["loci_gen"],
            bios_min_exp_avail_threshold=h["min_avail"], check_completeness=True,
            eic=h["eic"], merge_ct=h["merge_ct"], DSF_list=[1, 2], must_have_chr_access=h["must_have_chr_access"] if "must_have_chr_access" in h else True)

        signal_dim = self.dataset.signal_dim
        metadata_embedding_dim = self.dataset.signal_dim * 4

        self.model = build_model(h, signal_dim, metadata_embedding_dim).to(self.device)
        self.criterion = LossComputer()
        self.optimizer, self.scheduler, self.cosine_sched = build_optimizer_and_scheduler(self.model, h)

        # wrap with DDP
        self.model = DDP(self.model, device_ids=[self.device.index], output_device=self.device.index, find_unused_parameters=False)

        # masking helper
        token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
        self.masker = DataMasker(token_dict["cloze_mask"], h["mask_percentage"])
        self.token_dict = token_dict

    def _shard_epoch_start(self):
        # advance dataset internal pointers so each rank starts at a different bios batch
        self.dataset.new_epoch()
        for _ in range(self.rank):
            self.dataset.update_batch_pointers()

    def train(self):
        h = self.h
        best_val = None
        arch_flags = f"{'eic' if h['eic'] else 'full'}_{'DNA' if h['dna'] else ''}_random_mask_{h['suffix']}"

        # validation on rank 0
        if self.rank == 0 and not h.get("test_mode", False):
            val_eval = MONITOR_VALIDATION(
                self.dataset.base_path, h["context_length"], 4 * h["batch_size"],
                must_have_chr_access=self.dataset.must_have_chr_access, token_dict=self.token_dict,
                eic=h["eic"], DNA=h["dna"], device=self.device)
            try:
                _ = val_eval.get_validation(self.model.module)
                torch.cuda.empty_cache()
            except Exception:
                pass

        num_assays = self.dataset.signal_dim
        for epoch in range(h["epochs"]):
            self._shard_epoch_start()
            next_epoch = False
            while next_epoch == False:
                if h["dna"]:
                    _X, _mX, _avX, _seq = self.dataset.get_batch(side="x", dna_seq=True)
                else:
                    _X, _mX, _avX = self.dataset.get_batch(side="x")
                _Y, _mY, _avY, _pval = self.dataset.get_batch(side="y", pval=True, y_prompt=h["supertrack"])

                if _X.shape != _Y.shape or _mX.shape != _mY.shape or _avX.shape != _avY.shape:
                    next_epoch = self.dataset.update_batch_pointers()
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                for _ in range(h["inner_epochs"]):
                    if h["dna"]:
                        X, mX, avX, seq = _X.clone(), _mX.clone(), _avX.clone(), _seq.clone()
                    else:
                        X, mX, avX = _X.clone(), _mX.clone(), _avX.clone()
                    Y, mY, avY, pval = _Y.clone(), _mY.clone(), _avY.clone(), _pval.clone()

                    X, mX, avX, num_mask = apply_feature_mask(self.masker, arch_flags, X, mX, avX, num_assays)

                    masked_map = (X == self.token_dict["cloze_mask"]).to(self.device)
                    observed_map = ((X != self.token_dict["missing_mask"]) & (X != self.token_dict["cloze_mask"])).to(self.device)

                    X = X.float().to(self.device, non_blocking=True).requires_grad_(True)
                    mX = mX.to(self.device, non_blocking=True)
                    avX = avX.to(self.device, non_blocking=True)
                    mY = mY.to(self.device, non_blocking=True)
                    Y = Y.to(self.device, non_blocking=True)
                    pval = pval.to(self.device, non_blocking=True)

                    if h["dna"]:
                        seq = seq.to(self.device, non_blocking=True)
                        p_pred, n_pred, mu_pred, var_pred = self.model(X, seq, mX, mY, avX)
                    else:
                        p_pred, n_pred, mu_pred, var_pred = self.model(X, mX, mY, avX)

                    obs_c, imp_c, obs_p, imp_p = self.criterion(p_pred, n_pred, mu_pred, var_pred, Y, pval, observed_map, masked_map)

                    if num_mask is not None:
                        msk_p = float(num_mask / num_assays)
                        imp_p *= 4
                        obs_p *= 3
                        imp_c *= 2
                        obs_c *= 1
                        loss = (msk_p * (imp_c + imp_p)) + ((1 - msk_p) * (obs_p + obs_c))
                    else:
                        loss = obs_c + obs_p + imp_p + imp_c

                    if torch.isnan(loss).any():
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5)

                self.optimizer.step()
                if self.cosine_sched:
                    self.scheduler.step()

                next_epoch = self.dataset.update_batch_pointers()

            # rank-0 checkpointing per epoch
            if self.rank == 0 and not h.get("hpo", False):
                model_name = f"CANDI_DDP_{time.strftime('%Y%m%d%H%M%S')}_params{count_parameters(self.model.module)}.pt"
                os.makedirs(h["model_root"], exist_ok=True)
                torch.save(self.model.module.state_dict(), os.path.join(h["model_root"], model_name))


def parse_args():
    p = argparse.ArgumentParser(description="DDP training for CANDI models (multi-GPU)")
    p.add_argument('--data_path', type=str, default="/project/compbio-lab/encode_data/")
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--n_cnn_layers', type=int, default=3)
    p.add_argument('--conv_kernel_size', type=int, default=3)
    p.add_argument('--pool_size', type=int, default=2)
    p.add_argument('--expansion_factor', type=int, default=3)
    p.add_argument('--nhead', type=int, default=9)
    p.add_argument('--n_sab_layers', type=int, default=4)
    p.add_argument('--pos_enc', type=str, default="relative")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--inner_epochs', type=int, default=1)
    p.add_argument('--mask_percentage', type=float, default=0.2)
    p.add_argument('--context_length', type=int, default=1200)
    p.add_argument('--batch_size', type=int, default=25)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--num_loci', type=int, default=5000)
    p.add_argument('--lr_halflife', type=int, default=1)
    p.add_argument('--min_avail', type=int, default=3)
    p.add_argument('--optim', type=str, default="sgd")
    p.add_argument('--unet', action='store_true')
    p.add_argument('--supertrack', action='store_true')
    p.add_argument('--LRschedule', type=str, default=None)
    p.add_argument('--eic', action='store_true')
    p.add_argument('--dna', action='store_true')
    p.add_argument('--suffix', type=str, default='')
    p.add_argument('--merge_ct', action='store_true')
    p.add_argument('--loci_gen', type=str, default="ccre")
    p.add_argument('--model_root', type=str, default="models/")
    return p.parse_args()


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    h = {
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
        "separate_decoders": not False,
        "merge_ct": args.merge_ct,
        "loci_gen": args.loci_gen,
        "must_have_chr_access": True,
        "optim": args.optim,
        "unet": args.unet,
        "LRschedule": args.LRschedule,
        "supertrack": args.supertrack,
        "eic": args.eic,
        "dna": args.dna,
        "suffix": args.suffix,
        "model_root": args.model_root,
    }

    if rank == 0:
        os.makedirs(h["model_root"], exist_ok=True)
        with open(os.path.join(h["model_root"], f'hyper_parameters_ddp_{time.strftime("%Y%m%d%H%M%S")}.json'), 'w') as f:
            json.dump(h, f, indent=2)

    trainer = TrainerDDP(h, device, rank, world_size)
    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()


