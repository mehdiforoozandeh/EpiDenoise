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
# CANDI_Encoder (without DNA) with Projection Head
###############################################

class DINO_CANDI_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                 n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
        super(DINO_CANDI_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2

        conv_channels = [(self.f1) * (expansion_factor**l) for l in range(n_cnn_layers)]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)]
        )

        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU()
        )

        if self.pos_enc == "relative":
            self.transformer_encoder = nn.ModuleList([
                RelativeEncoderLayer(d_model=self.d_model, heads=nhead,
                                     feed_forward_hidden=expansion_factor * self.d_model,
                                     dropout=dropout) for _ in range(n_sab_layers)
            ])
        else:
            self.posEnc = PositionalEncoding(self.d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=nhead,
                dim_feedforward=expansion_factor * self.d_model, dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

        # --- Add projection head ---
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
    
    def forward(self, src, x_metadata):
        # Input src shape: [batch, L, f1]
        src = src.permute(0, 2, 1)  # to [batch, f1, L]
        for conv in self.convEnc:
            src = conv(src)
        
        src = src.permute(0, 2, 1)  # to [batch, L', f2]
        xmd_embedding = self.xmd_emb(x_metadata)
        # Concatenate metadata embedding expanded to the spatial dimension.
        src = torch.cat([src, xmd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.xmd_fusion(src)

        # Transformer encoder stage:
        if self.pos_enc != "relative":
            src = self.posEnc(src)
            src = self.transformer_encoder(src)
        else:
            for enc in self.transformer_encoder:
                src = enc(src)
        
        # Pool across sequence dimension (mean pooling)
        rep = src.mean(dim=1)  # shape [batch, d_model]
        proj = self.projection_head(rep)
        return proj

###############################################
# CANDI_DNA_Encoder (with DNA) with Projection Head
###############################################

class DINO_CANDI_DNA_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                 n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
        super(DINO_CANDI_DNA_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2

        # DNA convolution stack with exponentially spaced channels.
        DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers+2)]
        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True, SE=False,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size) for i in range(n_cnn_layers + 2)]
        )

        conv_channels = [(self.f1) * (expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]
        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1, SE=False,
                pool_size=pool_size) for i in range(n_cnn_layers)]
        )
        
        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=False)
        self.fusion = nn.Sequential(
            nn.Linear((2 * self.f2) + metadata_embedding_dim, self.f2), 
            nn.LayerNorm(self.f2)
            # nn.ReLU() is commented out based on your design.
        )

        self.transformer_encoder = nn.ModuleList([
            DualAttentionEncoderBlock(self.f2, nhead, self.l2, dropout=dropout, 
                max_distance=self.l2, pos_encoding_type="relative", max_len=self.l2)
            for _ in range(n_sab_layers)
        ])

        # --- Add projection head ---
        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, src, seq, x_metadata):
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to [batch, 4, 25*L]
        seq = seq.float()

        # DNA Conv encoder.
        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)
        seq = seq.permute(0, 2, 1)  # to [batch, L', f2]

        # Signal Conv encoder.
        src = src.permute(0, 2, 1)  # to [batch, f1, L]
        for conv in self.convEnc:
            src = conv(src)
        src = src.permute(0, 2, 1)  # to [batch, L', f2]

        # Signal metadata embedding.
        xmd_embedding = self.xmd_emb(x_metadata).unsqueeze(1).expand(-1, self.l2, -1)

        # Fusion.
        src = torch.cat([src, xmd_embedding, seq], dim=-1)
        src = self.fusion(src)

        # Transformer encoder.
        for enc in self.transformer_encoder:
            src = enc(src)
        
        # Pool along the sequence dimension.
        rep = src.mean(dim=1)  # shape [batch, latent_dim]
        proj = self.projection_head(rep)
        return proj

###############################################
# DINO_CANDI Class: DINO-style training for CANDI's encoder
###############################################

class DINO_CANDI:
    def __init__(
        self, student_encoder, teacher_encoder, decoder,
        dataset, optimizer, ema_decay, center_update, t_student,
        t_teacher, device_student, device_teacher):
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

        # Initialize a center vector based on the output projection dimension of your encoder.
        self.center = torch.zeros(self.student.projection_head[-1].out_features, device=device_student)
        self.decoder = decoder
    
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
        arch="", mask_percentage=0.15, hook=False, DNA=False, 
        early_stop=True, accumulation_steps=1, num_local_views=1):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"DINO CANDI{arch} # model_parameters: {count_parameters(self.student)}")
        logfile = open(f"models/DINO_CANDI{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        num_assays = self.dataset.signal_dim
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.student)
        
        if "eic" in arch:
            val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, token_dict=token_dict, eic=True, DNA=DNA, device=self.device)
        else:
            val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, token_dict=token_dict, eic=False, DNA=DNA, device=self.device)

        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)

        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

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
                    
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                for _ in range(inner_epochs):
                    if DNA:
                        X_batch, mX_batch, avX_batch, dnaseq_batch= _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone(), _dnaseq_batch.clone()
                    else:
                        X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                        
                    Y_batch, mY_batch, avY_batch, pval_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone(), _pval_batch.clone()

                    masked_map = (X_batch == token_dict["cloze_mask"])
                    observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                    missing_map = (X_batch == token_dict["missing_mask"])

                    # masked_map = masked_map.to(self.device) # imputation targets
                    # observed_map = observed_map.to(self.device) # upsampling targets
                    # pval_batch = pval_batch.to(self.device)
                    # mY_batch = mY_batch.to(self.device)
                    # Y_batch = Y_batch.to(self.device)
                    
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
                            teacher_output = self.teacher(X_batch, dnaseq_batch, mX_batch)

                        student_outputs.append(self.student(X_batch, dnaseq_batch, mX_batch))
                        for loc in local_views:
                            student_outputs.append(self.student(loc[0], dnaseq_batch, loc[1]))

                    else:
                        with torch.no_grad():
                            teacher_output = self.teacher(X_batch, mX_batch)

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
                    
                self.optimizer.step()
                self.update_teacher()

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
                    f"Bios Prog. {self.dataset.bios_pointer / self.dataset.num_bios:.2%}",
                    f"Loss: {loss.item():.4f}",
                    f"T_Ent: {normalized_teacher_entropy:.4f}",
                    f"S_Ent: {normalized_student_entropy:.4f}",
                    f"KL_div: {batch_avg_kl_div:.4f}",
                    f"took {int(minutes)}:{int(seconds):02d}",
                ]
                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                # (Rest of logging, model saving, and validation code as needed.)
                next_epoch = self.dataset.update_batch_pointers()
        
    def evaluate_encoder(self):
        """
        Evaluate the current quality of the encoder representations.
        This evaluation should train (or fine-tune) a candidate decoder to reconstruct the global input
        and report reconstruction metrics (e.g., MSE, R2 score).

        TODO: Please implement your candidate decoder training and evaluation here.
        You can use a simple regression model or any other decoder architecture that fits your task.
        """
        # Example placeholder:
        print("Running encoder evaluation with candidate decoder...")
        # TODO: Replace the following with your decoder evaluation code.
        if self.candidate_decoder is None:
            print("Candidate decoder is not implemented. Please implement candidate decoder evaluation.")
        else:
            # Example: train candidate decoder for a few iterations and compute reconstruction loss.
            pass

###############################################
# Main function: Parse arguments and run training.
###############################################

def main():
    context_length = 1600
    # -------------------------------
    # Instantiate the Student and Teacher Encoders.
    # Replace 'create_candi_encoder()' with your actual function that returns a CANDI encoder instance.
    student_encoder = DINO_CANDI_DNA_Encoder(
        signal_dim=35, metadata_embedding_dim=4*35, conv_kernel_size=3, n_cnn_layers=3, nhead=9,
        n_sab_layers=4, pool_size=2, dropout=0.1, context_length=context_length, pos_enc="relative", expansion_factor=3)
                 
    teacher_encoder =  DINO_CANDI_DNA_Encoder(
        signal_dim=35, metadata_embedding_dim=4*35, conv_kernel_size=3, n_cnn_layers=3, nhead=9,
        n_sab_layers=4, pool_size=2, dropout=0.1, context_length=context_length, pos_enc="relative", expansion_factor=3)

    teacher_encoder.load_state_dict(student_encoder.state_dict())

    # -------------------------------
    # Instantiate the candidate decoder (for evaluation) if available.
    # If you haven't implemented a candidate decoder yet, you can simply set this to None.
    # Optionally, you can create one using a function like create_candi_decoder().
    candidate_decoder = None  # TODO: Replace with your candidate decoder instantiation if available.

    # -------------------------------
    # Instantiate your dataset.
    # Your dataset is expected to implement methods such as new_epoch(), get_batch(), update_batch_pointers(), etc.
    # For instance, if you are using ExtendedEncodeDataHandler, instantiate and initialize it here.
    data_path = "/project/compbio-lab/encode_data/"
    # The following parameters are placeholders. Adjust them as needed.
    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=3000,                  # number of loci
        context_length=context_length*25,     # context length (adjust based on your application)
        bios_batchsize=10,       # batch size for bios samples
        loci_batchsize=1,        # batch size for loci
        loci_gen="random",         # loci generation method
        bios_min_exp_avail_threshold=7,  # minimum available bios
        check_completeness=True,
        eic=True,
        merge_ct=True
    )

    # -------------------------------
    # Set hyperparameters for DINO_CANDI.
    num_epochs = 100            # Adjust as needed.
    learning_rate = 1e-3        # Learning rate for the student encoder.
    ema_decay = 0.996            # EMA decay coefficient for teacher updates.
    center_update = 0.9        # Center update coefficient.
    t_student = 0.1             # Temperature for student outputs.
    t_teacher = 0.04            # Temperature for teacher outputs.
    batch_size = 50             # Batch size to be used by your dataset (if applicable).
    inner_epochs = 1            # Number of inner iterations per batch.
    mask_percentage = 0.15      # Fraction of assays to mask.
    num_local_views = 1         # Number of local views to generate per batch.
    
    # -------------------------------
    # Device Setup.
    device_student = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_teacher = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else device_student)

    # -------------------------------
    # Define the optimizer for the student encoder.
    optimizer = optim.SGD(student_encoder.parameters(), lr=learning_rate)

    # -------------------------------
    # Instantiate the DINO_CANDI trainer.
    # Note: The trainer expects (student_encoder, teacher_encoder, candidate_decoder, dataset, optimizer, 
    # ema_decay, center_update, t_student, t_teacher, device_student, device_teacher).
    dino_trainer = DINO_CANDI(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        decoder=candidate_decoder,
        dataset=dataset,
        optimizer=optimizer,
        ema_decay=ema_decay,
        center_update=center_update,
        t_student=t_student,
        t_teacher=t_teacher,
        device_student=device_student,
        device_teacher=device_teacher
    )

    # -------------------------------
    # Start training.
    dino_trainer.train_dino(
        num_epochs=num_epochs,
        context_length=context_length,
        batch_size=batch_size,
        inner_epochs=inner_epochs,
        arch="",
        mask_percentage=mask_percentage,
        hook=False,
        DNA=True,  # Set to True if you use DNA-specific inputs.
        early_stop=True,
        accumulation_steps=4,
        num_local_views=num_local_views
    )


if __name__ == "__main__":
    main()
