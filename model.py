import torch, math, random, time, json, os, pickle
from torch import nn
import torch.optim as optim
from data import ENCODE_IMPUTATION_DATASET
import torch.nn.functional as F
import pandas as pd
import numpy as np
from _utils import *
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# # Set the seed value
# seed_value = 42
# # Set the seed for Python's built-in random module
# random.seed(seed_value)
# # Set the seed for NumPy
# np.random.seed(seed_value)
# # Set the seed for PyTorch
# torch.manual_seed(seed_value)
# torch.cuda.manual_seed(seed_value)
# torch.cuda.manual_seed_all(seed_value)

#========================================================================================================#
#===========================================building blocks==============================================#
#========================================================================================================#

class PositionalEncoding10(nn.Module):

     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
         super().__init__()
         self.dropout = nn.Dropout(p=dropout)

         position = torch.arange(max_len).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
         pe = torch.zeros(max_len, 1, d_model)
         pe[:, 0, 0::2] = torch.sin(position * div_term)
         pe[:, 0, 1::2] = torch.cos(position * div_term)
         self.register_buffer('pe', pe)

     def forward(self, x):
         """
         Arguments:
             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
         """
         x = x + self.pe[:x.size(0)]
         return self.dropout(x)

class MaskedLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaskedLinear, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # Xavier initialization
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, mask):
        masked_weight = self.weights * mask
        output = torch.matmul(x, masked_weight) + self.bias
        return output

class AbsPositionalEmbedding15(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pe = pe.to(device)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe

class ComboEmbedding15(nn.Module):
    """
    Combo Embedding which is consisted with under features
        1. AbsPositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding segment info, (seg_A:1, seg_B:2)
        sum of all these features are output of ComboEmbedding
    """

    def __init__(self, d_model, seq_len=64, dropout=0.1):
        """
        :param d_model: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.segment = torch.nn.Embedding(3, d_model, padding_idx=0)
        self.position = AbsPositionalEmbedding15(d_model=d_model, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence, segment_label):
        x = sequence + self.position(sequence).unsqueeze(1) + self.segment(segment_label).unsqueeze(1) 
        return self.dropout(x)

class ComboLoss15(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss15, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.alpha = alpha

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac):

        mse_loss = self.mse_loss(pred_signals, true_signals)

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_loss) or torch.isnan(bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return self.alpha * mse_loss + (1 - self.alpha) * bce_loss

class ComboLoss16(nn.Module):
    def __init__(self):
        super(ComboLoss16, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, cloze_mask, obs_mask):

        mse_obs_loss = self.mse_loss(pred_signals[obs_mask], true_signals[obs_mask])
        mse_pred_loss = self.mse_loss(pred_signals[cloze_mask], true_signals[cloze_mask])
        bce_mask_loss = self.bce_loss(pred_mask, ~obs_mask)

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        SAP_bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss + mse_pred_loss + SAP_bce_loss + bce_mask_loss

class ComboLoss17(nn.Module):
    def __init__(self):
        super(ComboLoss17, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, obs_mask):

        mse_obs_loss = self.mse_loss(pred_signals[obs_mask], true_signals[obs_mask])
        mse_pred_loss = self.mse_loss(pred_signals[pred_mask], true_signals[pred_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss + mse_pred_loss + bce_loss

#========================================================================================================#
#=======================================EpiDenoise Versions==============================================#
#========================================================================================================#

class EpiDenoise10(nn.Module): 
    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise10, self).__init__()
        
        self.masked_linear = MaskedLinear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding10(d_model=d_model, max_len=context_length) 

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, src, pmask, fmask):
        src = self.masked_linear(src, fmask)
        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_key_padding_mask=pmask) 
        src = self.decoder(src)
        src = torch.permute(src, (1, 0, 2))
        return src

class EpiDenoise15(nn.Module):
    """
    updates since EpiDenoise1.0:
        1. add CLS and SEP tokens
        2. segment adjacency prediction (SAP):
            - add segment encodings
            - custom loss function (masking + segment adjacency)
        3. dynamic masking chunks (gradually increasing)
    """

    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise15, self).__init__()
        
        self.masked_linear = MaskedLinear(input_dim, d_model)
        self.embeddings = ComboEmbedding15(d_model=d_model, seq_len=context_length+3, dropout=dropout) # segment + positional

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, output_dim)

        self.SAP = nn.Linear(d_model, 2) # segment adjacency prediction head
        self.softmax = torch.nn.Softmax(dim=-1)

class EpiDenoise16(nn.Module):
    """
    VIME
    gets masked_x as input
    loss: masked value prediction + observation reconstruction + mask reconstruction + segment adjacency prediction
    returns:
        - reconstructed input
        - reconstructed mask
        - SAP
    """

    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise16, self).__init__()

        self.embedding_linear = nn.Linear(input_dim, d_model)
        self.embeddings = ComboEmbedding15(d_model=d_model, seq_len=context_length+3, dropout=dropout) # segment + positional

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.signal_decoder = nn.Linear(d_model, output_dim)
        self.mask_decoder = nn.Linear(d_model, output_dim)

        self.SAP = nn.Linear(d_model, 2) # segment adjacency prediction head
        self.softmax = torch.nn.Softmax(dim=-1)
    

    def forward(self, src, segment_label):
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.embeddings(src, segment_label)
        src = self.transformer_encoder(src) 

        cls_token = src[0, :, :].unsqueeze(0)
        SAP = self.softmax(self.SAP(cls_token))
        
        msk = self.mask_decoder(src)
        src = self.signal_decoder(src)

        src = torch.permute(src, (1, 0, 2))  # to N, L, F
        msk = torch.permute(msk, (1, 0, 2))  # to N, L, F

        return src, msk, SAP   

class EpiDenoise17(nn.Module):
    """
    SAITS:
    concatenates x and mask from assays dimension:
        (N, L, A) -> (N, L, 2A)

    LOSS: observed reconstruction + masked value prediction + segment adjacency prediction
    returns:
        - reconstructed input
        - SAP

    """

    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise17, self).__init__()

        self.embedding_linear = nn.Linear(2*input_dim, d_model)
        self.embeddings = ComboEmbedding15(d_model=d_model, seq_len=context_length+3, dropout=dropout) # segment + positional

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.decoder = nn.Linear(d_model, output_dim)
        self.SAP = nn.Linear(d_model, 2) # segment adjacency prediction head
        self.softmax = torch.nn.Softmax(dim=-1)
    

    def forward(self, src, mask, segment_label):
        src = torch.cat([src, mask.float()], dim=2)
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.embeddings(src, segment_label)
        src = self.transformer_encoder(src) 

        cls_token = src[0, :, :].unsqueeze(0)
        SAP = self.softmax(self.SAP(cls_token))

        src = self.decoder(src)
        src = torch.permute(src, (1, 0, 2))  # to N, L, F

        return src, SAP  

#========================================================================================================#
#=========================================Pretraining====================================================#
#========================================================================================================#

class PRE_TRAINER(object):  
    def __init__(self, model, dataset, criterion, optimizer, scheduler):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def pretrain_epidenoise_10(self, 
        d_model, outer_loop_epochs=1, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    x = torch.arcsinh_(x)
                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]
                        fmask = torch.ones(num_features, d_model)

                        for i in pattern:
                            fmask[i,:] = 0

                        fmask = fmask.to(self.device)

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            torch.cuda.empty_cache()
                            x_batch = pattern_batch[i:i+batch_size]
                            missing_mask_batch = missing_mask_patten_batch[i:i+batch_size]

                            # Masking a subset of the input data
                            masked_x_batch, cloze_mask = mask_data(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                            
                            pmask = cloze_mask[:,:,0].squeeze()
                            pmask = pmask.to(self.device)

                            cloze_mask = cloze_mask & ~missing_mask_batch
                            x_batch = x_batch.to(self.device)

                            masked_x_batch = masked_x_batch.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs = self.model(masked_x_batch, pmask, fmask)
                            loss = self.criterion(outputs[cloze_mask], x_batch[cloze_mask])

                            mean_pred, std_pred = outputs[cloze_mask].mean().item(), outputs[cloze_mask].std().item()
                            mean_target, std_target = x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].std().item()

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del pmask
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del pmask
                            del masked_x_batch
                            del outputs
                            epoch_loss.append(loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                        
                        if p%8 == 0:
                            logfile = open("models/log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Loss: {loss.item():.4f}", 
                                f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                                f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    # update parameters over all batches and all patterns of missing data
                    self.optimizer.step()
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss)}", 
                        f"Epoch Loss std: {np.std(epoch_loss)}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    torch.save(self.model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_15(self, 
        d_model, outer_loop_epochs=6, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):

                            torch.cuda.empty_cache()
                            seg_length = context_length // 2
                            is_adjacent = random.choice([True, False])

                            seg_1 = pattern_batch[i:i+batch_size, :seg_length, :]
                            seg1m = missing_mask_patten_batch[i:i+batch_size, :seg_length, :]
                            
                            if is_adjacent:
                                seg_2 = pattern_batch[i:i+batch_size, seg_length:, :]
                                seg2m = missing_mask_patten_batch[i:i+batch_size, seg_length:, :]
                                
                            else:
                                seg_1.shape[0]
                                # Randomly select a start index
                                start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                # If the start index overlaps with the range i:i+batch_size, choose again
                                while i <= start < i + seg_1.shape[0]:
                                    start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                seg_2 = pattern_batch[start:start+seg_1.shape[0], :seg_length, :]
                                seg2m = missing_mask_patten_batch[start:start+seg_1.shape[0], :seg_length, :]

                            CLS = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -3)
                            SEP = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -4)
                            
                            x_batch = torch.cat((CLS, seg_1, SEP, seg_2, SEP), 1)
                            missing_mask_batch = torch.cat((seg1m[:,0,:].unsqueeze(1), seg1m, seg1m[:,0,:].unsqueeze(1), seg2m, seg2m[:,0,:].unsqueeze(1)), 1)

                            special_token_indices = [0, seg_length, (2*seg_length)+1]

                            # 0 are for special tokens, 1 for segment1 and 2 for segment2
                            segment_label = [0] + [1 for i in range(seg_1.shape[1])] + [0] + [2 for i in range(seg_2.shape[1])] + [0]
                            segment_label = torch.from_numpy(np.array(segment_label))
                            segment_label = segment_label.to(self.device)

                            # create segment adjacency prediction labels based on is_adjacent
                            target_SAP = torch.full((1, x_batch.shape[0], 2), float(0))
                            target_SAP[:,:,int(is_adjacent)] = 1 
                            target_SAP = target_SAP.to(self.device)

                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data15(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                            
                            pmask = cloze_mask[:,:,0].squeeze()
                            
                            cloze_mask = cloze_mask & ~missing_mask_batch

                            """
                            if num_available features > 1, 
                                in each batch, randomly mask one of the available features
                                update the fmask
                                get the model to predict the whole track based on input
                            """

                            fmask = torch.ones(num_features, d_model)

                            for i in pattern:
                                fmask[i,:] = 0

                            if len(available_assays_ind) > 1:
                                assaymask_ind = random.choice(available_assays_ind)
                                masked_x_batch[:,:,available_assays_ind] = -1
                                fmask[assaymask_ind, :] = 0
                                cloze_mask[:, :, available_assays_ind] = True
                                cloze_mask[:, special_token_indices, :] = False

                            # move to GPU
                            fmask = fmask.to(self.device)
                            pmask = pmask.to(self.device)
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, SAP = self.model(masked_x_batch, pmask, fmask, segment_label)

                            loss = self.criterion(outputs[cloze_mask], x_batch[cloze_mask], SAP, target_SAP)

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del pmask
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue
                            
                            mean_pred, std_pred = outputs[cloze_mask].mean().item(), outputs[cloze_mask].std().item()
                            mean_target, std_target = x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].std().item()

                            del x_batch
                            del pmask
                            del masked_x_batch
                            del outputs
                            epoch_loss.append(loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                        
                        if p%8 == 0:
                            logfile = open("models/log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Loss: {loss.item():.4f}", 
                                f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                                f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    # update parameters over all batches and all patterns of missing data
                    self.optimizer.step()
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    torch.save(self.model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_16(self, 
        d_model, outer_loop_epochs=3, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            self.optimizer.zero_grad()

                            torch.cuda.empty_cache()
                            seg_length = context_length // 2
                            is_adjacent = random.choice([True, False])

                            seg_1 = pattern_batch[i:i+batch_size, :seg_length, :]
                            seg1m = missing_mask_patten_batch[i:i+batch_size, :seg_length, :]
                            
                            if is_adjacent:
                                seg_2 = pattern_batch[i:i+batch_size, seg_length:, :]
                                seg2m = missing_mask_patten_batch[i:i+batch_size, seg_length:, :]
                                
                            else:
                                seg_1.shape[0]
                                # Randomly select a start index
                                start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                # If the start index overlaps with the range i:i+batch_size, choose again
                                while i <= start < i + seg_1.shape[0]:
                                    start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                seg_2 = pattern_batch[start:start+seg_1.shape[0], :seg_length, :]
                                seg2m = missing_mask_patten_batch[start:start+seg_1.shape[0], :seg_length, :]

                            CLS = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -2)
                            SEP = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -3)
                            
                            x_batch = torch.cat((CLS, seg_1, SEP, seg_2, SEP), 1)
                            missing_mask_batch = torch.cat((seg1m[:,0,:].unsqueeze(1), seg1m, seg1m[:,0,:].unsqueeze(1), seg2m, seg2m[:,0,:].unsqueeze(1)), 1)

                            special_token_indices = [0, seg_length, (2*seg_length)+1]

                            # 0 are for special tokens, 1 for segment1 and 2 for segment2
                            segment_label = [0] + [1 for i in range(seg_1.shape[1])] + [0] + [2 for i in range(seg_2.shape[1])] + [0]
                            segment_label = torch.from_numpy(np.array(segment_label))
                            segment_label = segment_label.to(self.device)

                            # create segment adjacency prediction labels based on is_adjacent
                            target_SAP = torch.full((1, x_batch.shape[0], 2), float(0))
                            target_SAP[:,:,int(is_adjacent)] = 1 
                            target_SAP = target_SAP.to(self.device)

                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                            union_mask = cloze_mask | missing_mask_batch

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask, SAP = self.model(masked_x_batch, segment_label)

                            #pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, obs_mask
                            loss = self.criterion(outputs, x_batch, SAP, target_SAP, pred_mask, cloze_mask, ~union_mask)

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs
                            epoch_loss.append(loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                            self.scheduler.step()
                        
                        if p%8 == 0:
                            logfile = open("models/log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Loss: {loss.item():.4f}", 
                                # f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                                # f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    # update parameters over all batches and all patterns of missing data
                    # self.optimizer.step()
                    # self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    torch.save(self.model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model
    
    def pretrain_epidenoise_17(self, 
        d_model, outer_loop_epochs=3, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            self.optimizer.zero_grad()

                            torch.cuda.empty_cache()
                            seg_length = context_length // 2
                            is_adjacent = random.choice([True, False])

                            seg_1 = pattern_batch[i:i+batch_size, :seg_length, :]
                            seg1m = missing_mask_patten_batch[i:i+batch_size, :seg_length, :]
                            
                            if is_adjacent:
                                seg_2 = pattern_batch[i:i+batch_size, seg_length:, :]
                                seg2m = missing_mask_patten_batch[i:i+batch_size, seg_length:, :]
                                
                            else:
                                seg_1.shape[0]
                                # Randomly select a start index
                                start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                # If the start index overlaps with the range i:i+batch_size, choose again
                                while i <= start < i + seg_1.shape[0]:
                                    start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                seg_2 = pattern_batch[start:start+seg_1.shape[0], :seg_length, :]
                                seg2m = missing_mask_patten_batch[start:start+seg_1.shape[0], :seg_length, :]

                            CLS = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -2)
                            SEP = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -3)
                            
                            x_batch = torch.cat((CLS, seg_1, SEP, seg_2, SEP), 1)
                            missing_mask_batch = torch.cat((seg1m[:,0,:].unsqueeze(1), seg1m, seg1m[:,0,:].unsqueeze(1), seg2m, seg2m[:,0,:].unsqueeze(1)), 1)

                            special_token_indices = [0, seg_length, (2*seg_length)+1]

                            # 0 are for special tokens, 1 for segment1 and 2 for segment2
                            segment_label = [0] + [1 for i in range(seg_1.shape[1])] + [0] + [2 for i in range(seg_2.shape[1])] + [0]
                            segment_label = torch.from_numpy(np.array(segment_label))
                            segment_label = segment_label.to(self.device)

                            # create segment adjacency prediction labels based on is_adjacent
                            target_SAP = torch.full((1, x_batch.shape[0], 2), float(0))
                            target_SAP[:,:,int(is_adjacent)] = 1 
                            target_SAP = target_SAP.to(self.device)

                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                            union_mask = cloze_mask | missing_mask_batch

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, SAP = self.model(masked_x_batch, ~union_mask, segment_label)

                            loss = self.criterion(outputs, x_batch, SAP, target_SAP, cloze_mask, ~union_mask)

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs
                            epoch_loss.append(loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                            self.scheduler.step()
                        
                        if p%8 == 0:
                            logfile = open("models/log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Loss: {loss.item():.4f}", 
                                # f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                                # f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    # update parameters over all batches and all patterns of missing data
                    # self.optimizer.step()
                    # self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    torch.save(self.model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

class MODEL_LOADER(object):
    def __init__(self, model_path, hyper_parameters):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_epidenoise10(self):
        input_dim = output_dim = self.hyper_parameters["input_dim"]
        dropout = self.hyper_parameters["dropout"]
        nhead = self.hyper_parameters["nhead"]
        d_model = self.hyper_parameters["d_model"]
        nlayers = self.hyper_parameters["nlayers"]
        context_length = self.hyper_parameters["context_length"]
        
        # Assuming model is an instance of the correct class
        model = EpiDenoise10(
            input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
            output_dim=output_dim, dropout=dropout, context_length=context_length)

        model.load_state_dict(torch.load(self.model_path))
        model = model.to(self.device)
        return model

def train_epidenoise10(hyper_parameters, checkpoint_path=None, start_ds=0):

    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    chunk = hyper_parameters["chunk"]
    context_length = hyper_parameters["context_length"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    n_chunks = (mask_percentage * context_length) // 4 # change it to 6 later

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise10(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.97)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise10_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters10_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_10(d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, chunk=chunk, n_chunks=n_chunks,
        context_length=context_length, batch_size=batch_size, start_ds=start_ds)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    # os.system(f"mv models/hyper_parameters.pkl models/hyper_parameters_{model_name.replace( '.pt', '.pkl' )}")

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

def train_epidenoise15(hyper_parameters, checkpoint_path=None, start_ds=0):

    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    chunk = hyper_parameters["chunk"]
    context_length = hyper_parameters["context_length"]
    alpha = hyper_parameters["alpha"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise15(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise15_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters15_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = ComboLoss15(alpha=alpha)

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_15(d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, chunk=chunk, n_chunks=n_chunks,
        context_length=context_length, batch_size=batch_size, start_ds=start_ds)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    # os.system(f"mv models/hyper_parameters.pkl models/hyper_parameters_{model_name.replace( '.pt', '.pkl' )}")

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

def train_epidenoise16(hyper_parameters, checkpoint_path=None, start_ds=0):

    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    chunk = hyper_parameters["chunk"]
    context_length = hyper_parameters["context_length"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise16(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.75)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise16_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters16_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss16()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_16(d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, chunk=chunk, n_chunks=n_chunks,
        context_length=context_length, batch_size=batch_size, start_ds=start_ds)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    # os.system(f"mv models/hyper_parameters.pkl models/hyper_parameters_{model_name.replace( '.pt', '.pkl' )}")

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

def train_epidenoise17(hyper_parameters, checkpoint_path=None, start_ds=0):

    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    chunk = hyper_parameters["chunk"]
    context_length = hyper_parameters["context_length"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise17(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.75)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise17_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters17_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = ComboLoss17()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_17(d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, chunk=chunk, n_chunks=n_chunks,
        context_length=context_length, batch_size=batch_size, start_ds=start_ds)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    # os.system(f"mv models/hyper_parameters.pkl models/hyper_parameters_{model_name.replace( '.pt', '.pkl' )}")

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

#========================================================================================================#
#================================================main====================================================#
#========================================================================================================#

if __name__ == "__main__":
    # EPIDENOISE_1.7
    hyper_parameters = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.1,
        "nhead": 8,
        "d_model": 64,
        "nlayers": 2,
        "epochs": 15,
        "mask_percentage": 0.15,
        "chunk": True,
        "context_length": 400,
        "batch_size": 50,
        "learning_rate": 0.005,
    }

    train_epidenoise16(
        hyper_parameters, 
        checkpoint_path=None, 
        start_ds=0)

    exit()
    # EPIDENOISE_1.5
    hyper_parameters = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.1,
        "nhead": 8,
        "d_model": 64,
        "nlayers": 2,
        "epochs": 15,
        "mask_percentage": 0.15,
        "chunk": True,
        "context_length": 400,
        "batch_size": 50,
        "learning_rate": 0.05,
        "alpha":0.75
    }

    train_epidenoise15(
        hyper_parameters, 
        checkpoint_path=None, 
        start_ds=0)

    exit()
    # EPIDENOISE_1.0-LARGE
    hyper_parameters_large = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.1,
            "nhead": 8,
            "d_model": 128,
            "nlayers": 4,
            "epochs": 20,
            "mask_percentage": 0.15,
            "chunk": True,
            "context_length": 400,
            "batch_size": 80,
            "learning_rate": 0.01
        }

    # EPIDENOISE_1.0-SMALL
    hyper_parameters_small = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.1,
        "nhead": 4,
        "d_model": 64,
        "nlayers": 2,
        "epochs": 10,
        "mask_percentage": 0.15,
        "chunk": True,
        "context_length": 400,
        "batch_size": 50,
        "learning_rate": 0.005
    }

    train_epidenoise10(
        hyper_parameters_small, 
        checkpoint_path=None, 
        start_ds=0)