import torch, math, random, time, json, os, pickle, sys
from scipy.stats import spearmanr
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

#========================================================================================================#
#===========================================building blocks==============================================#
#========================================================================================================#

class ConvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D):
        super(ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_C)
        self.conv = nn.Conv1d(in_C, out_C, kernel_size=W, dilation=D, stride=S, padding="same")
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = F.gelu(x)
        x = self.conv(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D):
        super(DeconvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_C)
        padding = W // 2
        output_padding = 1 
        self.deconv = nn.ConvTranspose1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S, 
            padding=padding, output_padding=output_padding)
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = F.gelu(x)
        x = self.deconv(x)
        return x

class RConvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D):
        super(RConvBlock, self).__init__()
        self.conv_block = ConvBlock(in_C, out_C, W, S, D)
        
    def forward(self, x):
        return x + self.conv_block(x)

class ConvTower(nn.Module):
    def __init__(self, in_C, out_C, W, S, D):
        super(ConvTower, self).__init__()
        self.conv  =   ConvBlock(in_C, out_C, W, S, D)
        self.rconv = RConvBlock(out_C, out_C, W, S, D)
        self.pool  = nn.MaxPool1d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.rconv(x)
        return self.pool(x)


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
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

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

class RelativeEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads, feed_forward_hidden, dropout=0.1):
        super(RelativeEncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = RelativeMultiHeadAttentionLayer(
            hid_dim=d_model, n_heads=heads, dropout=dropout)
        self.feed_forward = FeedForwardNN(
            input_size=d_model, hidden_size=feed_forward_hidden, output_size=d_model, n_hidden_layers=1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings,src_key_padding_mask=None, src_mask=None, mask=None, is_causal=None):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded

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
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, cloze_mask, union_mask):
        mse_obs_loss =  self.l1_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.l1_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any() or torch.isnan(pred_mask).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_mask_loss = self.bce_loss(pred_mask, union_mask.float())
        # SAP_bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_mask_loss):# or torch.isnan(SAP_bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss + mse_pred_loss + bce_mask_loss #+ SAP_bce_loss

class ComboLoss17(nn.Module):
    def __init__(self):
        super(ComboLoss17, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, obs_mask):

        mse_obs_loss = self.l1_loss(pred_signals[obs_mask], true_signals[obs_mask])
        mse_pred_loss = self.l1_loss(pred_signals[pred_mask], true_signals[pred_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss + mse_pred_loss #+ bce_loss

class ComboLoss18(nn.Module):
    def __init__(self):
        super(ComboLoss18, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_mask, cloze_mask, union_mask):
        mse_obs_loss =  self.l1_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.l1_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_signals).any() or torch.isnan(pred_mask).any():
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)

        bce_mask_loss = self.bce_loss(pred_mask, union_mask.float())

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_mask_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss, mse_pred_loss, bce_mask_loss

class ComboLoss20(nn.Module):
    def __init__(self):
        super(ComboLoss20, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_mask, cloze_mask, union_mask):
        mse_obs_loss =  self.l1_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.l1_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_signals).any() or torch.isnan(pred_mask).any():
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)

        bce_mask_loss = self.bce_loss(pred_mask, union_mask.float())

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_mask_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss, mse_pred_loss, bce_mask_loss

class MatrixFactorizationEmbedding(nn.Module):
    """
    learns two factorizations from input matrix M of size l*d
    U of size l*k
    V of size d*k
    UV^T is should reconstruct the input matrix M
    """

    def __init__(self, l, d, k):
        super().__init__()
        self.dense_U = nn.Linear(l, k)
        self.dense_V = nn.Linear(d, k)
        self.relu = nn.ReLU()
        # self.dense_U = FeedForwardNN(l, 4*k, k, 2)
        # self.dense_V = FeedForwardNN(d, 4*k, k, 2)
       
    def forward(self, M, linear=False):
        # shape of M is (N, L, D)
        U = self.dense_U(torch.permute(M, (0, 2, 1))) 
        V = self.dense_V(M)

        if not linear:
            U = self.relu(U)
            V = self.relu(V)
        
        V = torch.permute(V, (0, 2, 1))
        M = torch.matmul(U, V)

        return torch.permute(M, (0, 2, 1))

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
        
        msk = torch.sigmoid(self.mask_decoder(src))
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

class EpiDenoise18(nn.Module):
    """
    no SAP
    added relative position encodings
    """
    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise18, self).__init__()

        self.mf_embedding = MatrixFactorizationEmbedding(l=context_length, d=input_dim, k=d_model)
        self.embedding_linear = nn.Linear(input_dim, d_model)

        self.encoder_layer = RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.signal_decoder =  nn.Linear(d_model, output_dim)
        # self.signal_decoder = FeedForwardNN(d_model, 4*d_model, output_dim, 2)
        self.mask_decoder = nn.Linear(d_model, output_dim)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, src):
        src = self.mf_embedding(src, linear=True)
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        
        src = self.transformer_encoder(src) 
        
        msk = torch.sigmoid(self.mask_decoder(src))
        src = self.signal_decoder(src)

        src = torch.permute(src, (1, 0, 2))  # to N, L, F
        msk = torch.permute(msk, (1, 0, 2))  # to N, L, F

        return src, msk


# self.convblock1 =   ConvBlock(2*input_dim, d_model, kernel_size, 1, 1)
# self.rconvblock1 = RConvBlock(d_model, d_model, kernel_size, 1, 1)
# self.pool1 = nn.MaxPool1d(2, stride=2)


class EpiDenoise20(nn.Module):
    def __init__(self, 
                 input_dim, kernel_size, n_cnn_layer, dilation,
                 nhead, d_model, n_encoder_layers, output_dim, dropout=0.1):
        super(EpiDenoise20, self).__init__()

        stride = 1
        # Convolutional layers
        self.conv1 = ConvTower(input_dim, int((0.75 * (d_model // (2**n_cnn_layer)))), kernel_size, stride, dilation)
        self.convm = ConvTower(input_dim, int((0.25 * (d_model // (2**n_cnn_layer)))), 1, stride, dilation)

        self.convtower = nn.Sequential(*[
            ConvTower(
                d_model // (2**(n_cnn_layer-i)), 
                d_model // (2**(n_cnn_layer-i-1)),
                kernel_size//2, stride, dilation
            ) for i in range(n_cnn_layer)
        ])

        # Transformer Encoder Layer (replace with your actual RelativeEncoderLayer)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_encoder_layers)

        # Deconvolution layers
        self.deconvtower = nn.Sequential(*[
            DeconvBlock(
                d_model // (2**(i)), d_model // (2**(i+1)), 
                kernel_size//2, 2, dilation) for i in range(n_cnn_layer - 1)
        ])
        self.deconv1 = DeconvBlock(d_model // (2**(n_cnn_layer-1)), d_model // (2**(n_cnn_layer)), kernel_size//2, 2, dilation)
        self.deconv2 = DeconvBlock(d_model // (2**(n_cnn_layer)), input_dim, kernel_size, 2, dilation)

        self.signal_decoder = nn.Linear(input_dim, output_dim)
        self.mask_decoder = nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, m):
        x = x.permute(0, 2, 1) # to N, F, L
        m = m.permute(0, 2, 1) # to N, F, L
        m = self.convm(m.float())

        x = self.conv1(x)
        x = torch.cat([x, m], dim=1)
        x = self.convtower(x)

        x = x.permute(2, 0, 1)  # to L, N, F
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0) # to N, F, L'

        x = self.deconvtower(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = x.permute(2, 0, 1)  # to L, N, F

        mask = self.softmax(self.mask_decoder(x))
        x = self.signal_decoder(x)

        x = torch.permute(x, (1, 0, 2))  # to N, L, F
        mask = torch.permute(mask, (1, 0, 2))  # to N, L, F

        return x, mask

#========================================================================================================#
#=========================================Pretraining====================================================#
#========================================================================================================#

class PRE_TRAINER(object):  
    def __init__(
        self, model, dataset, criterion, optimizer, scheduler):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.test_x_dir = "/project/compbio-lab/EIC/training_data/C23_chr21_25.pt"
        self.test_y_dir = "/project/compbio-lab/EIC/validation_data/C23_chr21_25.pt"
    
    def test_model(self, context_length, version, is_arcsin, batch_size):
        """
        load X and Y
        pred = model(X)
        compare X and Y
        """
        missing_x_i = []
        missing_y_i = []

        X = torch.load(self.test_x_dir)
        # fill-in missing_ind
        for i in range(X.shape[1]):
            if (X[:, i] == -1).all():
                missing_x_i.append(i)

        Y = torch.load(self.test_y_dir)
        # fill-in missing_ind
        for i in range(Y.shape[1]):
            if (Y[:, i] == -1).all():
                missing_y_i.append(i)

        num_rows = (X.shape[0] // context_length) * context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]

        if version == "10":
            fmask = torch.ones(d_model, self.hyper_parameters["d_model"])
            for i in missing_x_i: # input fmask
                fmask[i,:] = 0
            fmask = fmask.to(self.device)

        elif version == "16" or version == "17":
            CLS_x = torch.full((X.shape[0], 1, X.shape[2]), -2)
            SEP_x = torch.full((X.shape[0], 1, X.shape[2]), -3)
            CLS_y = torch.full((Y.shape[0], 1, Y.shape[2]), -2)
            SEP_y = torch.full((Y.shape[0], 1, Y.shape[2]), -3)

            X = torch.cat([CLS_x, X[:, :context_length//2, :], SEP_x, X[:, context_length//2:, :], SEP_x], dim=1)
            Y = torch.cat([CLS_y, Y[:, :context_length//2, :], SEP_y, Y[:, context_length//2:, :], SEP_y], dim=1)

            segment_label = [0] + [1 for i in range(context_length//2)] + [0] + [2 for i in range(context_length//2)] + [0]
            segment_label = torch.from_numpy(np.array(segment_label))
            segment_label = segment_label.to(self.device)

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                if version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif version == "17":
                    mask = torch.zeros_like(x_batch, dtype=torch.bool)
                    for i in missing_x_i: 
                        mask[:,:,i] = True

                    outputs, SAP = self.model(x_batch, ~mask, segment_label)
                
                elif version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif version == "20":
                    mask = torch.zeros_like(x_batch, dtype=torch.bool)
                    for i in missing_x_i: 
                        mask[:,:,i] = True

                    outputs, pred_mask = self.model(x_batch, ~mask)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        mses = []
        spearmans = []
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()   
                mse_GW = np.mean((np.array(target) - np.array(pred))**2)
                spearman_GW = spearmanr(pred, target)[0]

                mses.append(mse_GW)
                spearmans.append(spearman_GW)
        
        return sum(mses)/len(mses), sum(spearmans)/len(spearmans) 

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
        logfile = open("models/EPD16_log.txt", "w")
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
                            loss = self.criterion(outputs, x_batch, SAP, target_SAP, pred_mask, cloze_mask, union_mask)

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
                            # self.scheduler.step()
                        
                        if p%8 == 0:
                            logfile = open("models/EPD16_log.txt", "w")

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
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD16_log.txt", "w")
                    
                    test_mse, test_corr = self.test_model(
                        context_length, version="16", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Test_MSE: {test_mse:.4f}",
                        f"Test Corr: {test_corr:.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD16_model_checkpoint_ds_{ds}.pth')
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
        logfile = open("models/EPD17_log.txt", "w")
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
                        
                        if p%8 == 0:
                            logfile = open("models/EPD17_log.txt", "w")

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
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD17_log.txt", "w")

                    test_mse, test_corr = self.test_model(
                        context_length, version="17", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Test_MSE: {test_mse:.4f}",
                        f"Test Corr: {test_corr:.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD17_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_18(self, 
        d_model, outer_loop_epochs=2, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD18_log.txt", "w")
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
                    epoch_obs_loss = []
                    epoch_msk_loss = []
                    epoch_clz_loss = []
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

                            x_batch = pattern_batch[i:i+batch_size]
                            missing_mask_batch = missing_mask_patten_batch[i:i+batch_size]
                            
                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(
                                x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)

                            union_mask = cloze_mask | missing_mask_batch

                            masked_x_batch = add_noise(masked_x_batch, 0.4)
                            masked_x_batch[union_mask] = -1

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask = self.model(masked_x_batch)
                            mse_obs_loss, mse_pred_loss, bce_mask_loss = self.criterion(
                                outputs, x_batch, pred_mask, cloze_mask, union_mask)
                            loss = mse_obs_loss + mse_pred_loss + bce_mask_loss

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                print(len(available_assays_ind), mse_obs_loss + mse_pred_loss + bce_mask_loss)
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
                            epoch_obs_loss.append(mse_obs_loss.item())
                            epoch_clz_loss.append(mse_pred_loss.item())
                            epoch_msk_loss.append(bce_mask_loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                        
                        if p == 1 or p%8 == 0:
                            logfile = open("models/EPD18_log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Obs Loss: {mse_obs_loss.item():.4f}",
                                f"Clz Loss: {mse_pred_loss.item():.4f}",
                                f"Msk Loss: {bce_mask_loss.item():.4f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD18_log.txt", "w")
                    
                    test_mse, test_corr = self.test_model(
                        context_length, version="18", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Obs Loss: {np.mean(epoch_obs_loss):.4f}", 
                        f"Epoch Clz Loss: {np.mean(epoch_clz_loss):.4f}", 
                        f"Epoch Msk Loss: {np.mean(epoch_msk_loss):.4f}", 
                        # f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Test_MSE: {test_mse:.4f}",
                        f"Test Corr: {test_corr:.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD18_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_20(self, 
        d_model, outer_loop_epochs=2, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD20_log.txt", "w")
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
                    epoch_obs_loss = []
                    epoch_msk_loss = []
                    epoch_clz_loss = []
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

                            x_batch = pattern_batch[i:i+batch_size]
                            missing_mask_batch = missing_mask_patten_batch[i:i+batch_size]
                            
                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(
                                x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)

                            union_mask = cloze_mask | missing_mask_batch

                            masked_x_batch = add_noise(masked_x_batch, 0.25)
                            masked_x_batch[union_mask] = -1

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask = self.model(masked_x_batch, union_mask)
                            mse_obs_loss, mse_pred_loss, bce_mask_loss = self.criterion(
                                outputs, x_batch, pred_mask, cloze_mask, union_mask)
                            loss = mse_obs_loss + mse_pred_loss + bce_mask_loss

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                print(len(available_assays_ind), mse_obs_loss + mse_pred_loss + bce_mask_loss)
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
                            epoch_obs_loss.append(mse_obs_loss.item())
                            epoch_clz_loss.append(mse_pred_loss.item())
                            epoch_msk_loss.append(bce_mask_loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                        
                        if p == 1 or p%8 == 0:
                            logfile = open("models/EPD20_log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', 
                                f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Obs Loss: {mse_obs_loss.item():.4f}",
                                f"Clz Loss: {mse_pred_loss.item():.4f}",
                                f"Msk Loss: {bce_mask_loss.item():.4f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD20_log.txt", "w")
                    
                    test_mse, test_corr = self.test_model(
                        context_length, version="20", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Obs Loss: {np.mean(epoch_obs_loss):.4f}", 
                        f"Epoch Clz Loss: {np.mean(epoch_clz_loss):.4f}", 
                        f"Epoch Msk Loss: {np.mean(epoch_msk_loss):.4f}", 
                        # f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Test_MSE: {test_mse:.4f}",
                        f"Test Corr: {test_corr:.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD20_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

class MODEL_LOADER(object):
    def __init__(self, model_path, hyper_parameters):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_epidenoise(self, version= "16"):
        input_dim = output_dim = self.hyper_parameters["input_dim"]
        dropout = self.hyper_parameters["dropout"]
        nhead = self.hyper_parameters["nhead"]
        d_model = self.hyper_parameters["d_model"]
        nlayers = self.hyper_parameters["nlayers"]
        context_length = self.hyper_parameters["context_length"]
        
        # Assuming model is an instance of the correct class
        if version == "10":
            model = EpiDenoise10(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)

        elif version == "15":
            model = EpiDenoise15(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)
        
        elif version == "16":
            model = EpiDenoise16(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)

        elif version == "17":
            model = EpiDenoise17(
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=330, gamma=0.5)

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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=330, gamma=0.5)

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

def train_epidenoise18(hyper_parameters, checkpoint_path=None, start_ds=0):
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

    model = EpiDenoise18(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=330, gamma=0.5)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise18_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters18_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss18()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_18(d_model=d_model, num_epochs=epochs, 
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

def train_epidenoise20(hyper_parameters, checkpoint_path=None, start_ds=0):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    n_cnn_layer = hyper_parameters["n_cnn_layer"]
    dilation = hyper_parameters["dilation"]

    kernel_size = hyper_parameters["kernel_size"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome


    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise20(
        input_dim=input_dim, 
        kernel_size=kernel_size, n_cnn_layer=n_cnn_layer, dilation=dilation,
        nhead=nhead, d_model=d_model, n_encoder_layers=nlayers, 
        output_dim=output_dim, dropout=dropout)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise20_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters20_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss20()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_20(d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, context_length=context_length, 
        batch_size=batch_size, start_ds=start_ds)

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
    hyper_parameters1678 = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.05,
        "nhead": 4,
        "d_model": 64,
        "nlayers": 4,
        "epochs": 10,
        "mask_percentage": 0.2,
        "chunk": True,
        "context_length": 400,
        "batch_size": 100,
        "learning_rate": 0.0001,
    }

    hyper_parameters20 = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.05,
        "nhead": 8,
        "d_model": 512,
        "nlayers": 6,
        "epochs": 10,
        "mask_percentage": 0.2,
        "kernel_size": 7,
        "n_cnn_layer": 4,
        "dilation":1,
        "context_length": 1600,
        "batch_size": 100,
        "learning_rate": 0.1,
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
        train_epidenoise20(
            hyper_parameters20, 
            checkpoint_path=None, 
            start_ds=0)