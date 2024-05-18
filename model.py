import torch, math, random, time, json, os, pickle, sys
from scipy.stats import spearmanr
from torch import nn
import torch.optim as optim
from data import ENCODE_IMPUTATION_DATASET, ExtendedEncodeDataHandler
import torch.nn.functional as F
import pandas as pd
import numpy as np
from _utils import *
from sklearn.metrics import r2_score
from datetime import datetime
from scipy.stats import nbinom

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#========================================================================================================#
#===========================================Building Blocks==============================================#
#========================================================================================================#

class MetadataEmbeddingModule(nn.Module):
    def __init__(self, input_dim, embedding_dim, non_linearity=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim 
        self.non_linearity = non_linearity
        self.continuous_size = embedding_dim // 3

        self.avail_embedding = nn.Embedding(3, self.continuous_size)  # 3 classes: available, missing, cloze_masked

        # X metadata embedding parameters
        self.xruntype_embedding = nn.Embedding(4, self.continuous_size)  # 4 classes: single_end, pair_end, missing, cloze_masked
        self.xdepth_transform = nn.Linear(1, self.continuous_size) 
        self.xcoverage_transform = nn.Linear(1, self.continuous_size)
        self.xread_length_transform = nn.Linear(1, self.continuous_size)

        # Y metadata embedding parameters
        self.yruntype_embedding = nn.Embedding(3, self.continuous_size)  # 4 classes: single_end, pair_end, missing
        self.ydepth_transform = nn.Linear(1, self.continuous_size) 
        self.ycoverage_transform = nn.Linear(1, self.continuous_size)
        self.yread_length_transform = nn.Linear(1, self.continuous_size)

        # Final layer to combine all embeddings
        self.final_embedding = nn.Linear(self.input_dim * self.continuous_size * 9, embedding_dim)  # Adjusted for all inputs
        # self.final_embedding = FeedForwardNN(self.input_dim * self.continuous_size * 9, embedding_dim, embedding_dim, 3)
        # self.final_emb_layer_norm = nn.LayerNorm(embedding_dim)

    def embed_metadata(self, metadata, side="x"):
        depth = metadata[:, 0, :].unsqueeze(-1).float() 
        coverage = metadata[:, 1, :].unsqueeze(-1).float() 
        read_length = metadata[:, 2, :].unsqueeze(-1).float() 
        runtype = metadata[:, 3, :].long() 
        
        runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype) # missing
        runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype) # cloze_masked

        if side == "x":
            depth_embed = self.xdepth_transform(depth)
            coverage_embed = self.xcoverage_transform(coverage)
            read_length_embed = self.xread_length_transform(read_length)
            runtype_embed = self.xruntype_embedding(runtype)

        elif side == "y":
            depth_embed = self.ydepth_transform(depth)
            coverage_embed = self.ycoverage_transform(coverage)
            read_length_embed = self.yread_length_transform(read_length)
            runtype_embed = self.yruntype_embedding(runtype)

        if self.non_linearity:
            depth_embed = F.relu(depth_embed)
            coverage_embed = F.relu(coverage_embed)
            read_length_embed = F.relu(read_length_embed)

        # Concatenate all embeddings
        embeddings = torch.cat([depth_embed, coverage_embed, read_length_embed, runtype_embed], dim=-1)
        return embeddings

    def embed_avail(self, availability):
        availability = availability.long()
        availability = torch.where(
            availability == -2, torch.tensor(2, device=availability.device), availability)

        availability_embed = self.avail_embedding(availability)
        return availability_embed

    def forward(self, x_metadata, y_metadata, availability):
        Xmd_embed = self.embed_metadata(x_metadata, side="x")
        Ymd_embed = self.embed_metadata(y_metadata, side="y")
        av_embed = self.embed_avail(availability)

        # Concatenate all embeddings along the last dimension
        full_embed = torch.cat([Xmd_embed, Ymd_embed, av_embed], dim=-1)

        full_embed = full_embed.view(full_embed.shape[0], -1)
        full_embed = self.final_embedding(full_embed)

        # full_embed = self.final_emb_layer_norm(full_embed)

        return full_embed

class DualConvEmbedding(nn.Module):
    def __init__(self, in_C, out_C, do_batchnorm=True):
        super(DualConvEmbedding, self).__init__()
        self.do_batchnorm = do_batchnorm
        if self.do_batchnorm:
            self.batch_norm = nn.BatchNorm1d(out_C)
        
        self.convF = nn.Conv1d(in_C, out_C, kernel_size=1, dilation=1, stride=1, padding="same")
        self.convM = nn.Conv1d(in_C, out_C, kernel_size=1, dilation=1, stride=1, padding="same")
        
        self.act = nn.ReLU()
        
    def forward(self, F, M): # F is feature matrix # M is the binary mask matrix
        F = self.convF(F)
        if self.do_batchnorm:
            F = self.batch_norm(F)
        F = self.act(F)

        M = self.convM(M)
        if self.do_batchnorm:
            M = self.batch_norm(M)
        M = self.act(M)
        
        return F * M

class SoftmaxPooling1D(nn.Module):
    def __init__(self, pool_size, per_channel=False, w_init_scale=1.0):
        super(SoftmaxPooling1D, self).__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.w_init_scale = w_init_scale
        self.weights_initialized = False  # Track whether weights have been initialized

    def forward(self, inputs):
        device = inputs.device
        inputs = inputs.permute(0, 2, 1)
        batch_size, length, num_features = inputs.size()
        
        if not self.weights_initialized:
            output_size = num_features if self.per_channel else 1
            identity = torch.eye(num_features, device=device)
            init_val = identity * self.w_init_scale
            weights_val = init_val.repeat(1, output_size) if self.per_channel else init_val.mean(dim=0, keepdim=True).repeat(output_size, 1)
            self.weights = nn.Parameter(weights_val)
            self.register_parameter('softmax_weights', self.weights)
            self.weights_initialized = True
        else:
            self.weights = self.weights.to(device)
        
        # Ensure the length is divisible by the pool size for simplicity
        if length % self.pool_size != 0:
            padding = self.pool_size - length % self.pool_size
            inputs = F.pad(inputs, (0, 0, 0, padding))
            _, length, _ = inputs.size()
        
        # Reshape inputs for pooling
        inputs = inputs.unfold(
            1, self.pool_size, self.pool_size).contiguous().view(batch_size, -1, self.pool_size, num_features)

        # Calculate logits using einsum for simplicity here
        logits = torch.einsum('bnpc,pc->bnp', inputs, self.weights)

        # Apply softmax to the logits along the pooling window dimension
        softmax_weights = F.softmax(logits, dim=2)
        
        # Multiply inputs by softmax weights and sum over the pooling window
        pooled = torch.einsum('bnpc,bnp->bnc', inputs, softmax_weights)
        
        return pooled.permute(0,2,1)

class AttentionPooling1D(nn.Module):
    def __init__(self, in_channels, pooling_size=2):
        super(AttentionPooling1D, self).__init__()
        self.pooling_size = pooling_size
        self.attention_weights = nn.Parameter(torch.randn(in_channels, pooling_size))

    def forward(self, x):
        # x shape: (batch_size, channels, length)

        # Split the input tensor into pools
        unfolded = x.unfold(dimension=2, size=self.pooling_size, step=self.pooling_size)
        # unfolded shape: (batch_size, channels, num_pools, pooling_size)

        # Compute attention scores and apply them
        scores = F.softmax(self.attention_weights, dim=1)
        pooled = (unfolded * scores.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        # pooled shape: (batch_size, channels, num_pools)

        pooled = pooled.permute(0, 2, 1)
        
        return pooled

class ConvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D, norm="layer"):
        super(ConvBlock, self).__init__()
        self.norm = norm

        if self.norm == "batch":
            self.norm = nn.BatchNorm1d(out_C)
        elif self.norm == "layer":
            self.norm = nn.LayerNorm(out_C)

        self.conv = nn.Conv1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S, padding="same")
        
    def forward(self, x):
        x = self.conv(x)

        if self.norm in ["batch", "layer"]:
            x = self.norm(x)
            
        x = F.relu(x)
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
        x = self.deconv(x)
        x = F.gelu(x)
        return x

class ConvTower(nn.Module):
    def __init__(self, in_C, out_C, W, S=1, D=1, pool_type="max", residuals=True):
        super(ConvTower, self).__init__()
        
        if pool_type == "max" or pool_type == "attn" or pool_type == "avg":
            self.do_pool = True
        else:
            self.do_pool = False

        if pool_type == "attn":
            self.pool = SoftmaxPooling1D(2)
        elif pool_type == "max":
            self.pool  = nn.MaxPool1d(2)
        elif pool_type == "avg":
            self.pool  = nn.AvgPool1d(2)

        self.conv1 = ConvBlock(in_C, out_C, W, S, D)

        self.resid = residuals
        if self.resid:
            self.rconv = nn.Conv1d(in_C, out_C, kernel_size=1)
        
    def forward(self, x):
        y = self.conv1(x)

        if self.resid:
            y = y + self.rconv(x)

        if self.do_pool:
            y = self.pool(y)
        
        return y

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
        if torch.cuda.is_available():
            final_mat = torch.LongTensor(final_mat).cuda()
            embeddings = self.embeddings_table[final_mat].cuda()
        else:
            final_mat = torch.LongTensor(final_mat)
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
        return self.pe.unsqueeze(1)

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
#========================================= Negative Binomial ============================================#
#========================================================================================================#

class NegativeBinomialLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NegativeBinomialLayer, self).__init__()

        self.fc_p = nn.Linear(input_dim, output_dim)
        self.fc_n = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # using sigmoid to ensure it's between 0 and 1
        p = torch.sigmoid(self.fc_p(x))

        # using softplus to ensure it's positive
        n = F.softplus(self.fc_n(x))

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
    
    # Calculate the negative log likelihood using PyTorch functions
    nll = (
        torch.lgamma(n_pred)
        + torch.lgamma(y_true + 1)
        - torch.lgamma(n_pred + y_true)
        - n_pred * torch.log(p_pred)
        - y_true * torch.log(1 - p_pred)
    )
    
    return nll

class NegativeBinomial(object):
    def __init__(self, p, n):
        self.p = p.numpy()
        self.n = n.numpy()
        
    def expect(self, stat="mean"):
        if stat == "median":
            self.median_value = torch.tensor(nbinom.median(self.n, self.p), dtype=torch.float32)
            return self.median_value

        elif stat == "mean":
            self.mean_value = torch.tensor(nbinom.mean(self.n, self.p), dtype=torch.float32)
            return self.mean_value
    
    def mean(self):
        self.mean_value = torch.tensor(nbinom.mean(self.n, self.p), dtype=torch.float32)
        return self.mean_value

    def interval(self, confidence):
        lower, upper = nbinom.interval(confidence, self.n, self.p)
        return torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32)
    
    def std(self):
        std_value = nbinom.std(self.n, self.p)
        return torch.tensor(std_value, dtype=torch.float32)

    def var(self):
        var_value = nbinom.var(self.n, self.p)
        return torch.tensor(var_value, dtype=torch.float32)

#========================================================================================================#
#=========================================== Loss Functions =============================================#
#========================================================================================================#

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
        self.l1_loss = nn.MSELoss(reduction='mean')
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

class ComboLoss21(nn.Module):
    def __init__(self):
        super(ComboLoss21, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, union_mask, next_pos_mask):
        # mse_obs_loss =  self.mse_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_next_pos = self.mse_loss(pred_signals[next_pos_mask], true_signals[next_pos_mask])

        if torch.isnan(pred_signals).any() or torch.isnan(mse_next_pos):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        return mse_next_pos

class ComboLoss22(nn.Module):
    def __init__(self, alpha=0.75):
        super(ComboLoss22, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, cloze_mask, union_mask):#, aggrmean, aggrstd, aggr_mask):

        # true_seq_mean = true_signals.mean(dim=1)
        # true_seq_std = true_signals.std(dim=1)

        # mse_aggrmean_loss = self.mse_loss(aggrmean[aggr_mask], true_seq_mean[aggr_mask]) 
        # mse_aggrstd_loss  = self.mse_loss(aggrstd[aggr_mask], true_seq_std[aggr_mask]) 

        mse_obs_loss =  self.mse_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.mse_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        if torch.isnan(mse_pred_loss).any() or torch.isnan(mse_obs_loss).any():
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(mse_pred_loss.device), torch.tensor(float('nan')).to(mse_pred_loss.device)

        return self.alpha*mse_pred_loss, (1-self.alpha)*mse_obs_loss#, mse_aggrmean_loss, mse_aggrstd_loss

class ComboLoss_PoissonNLL(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss30a_PoissonNLL, self).__init__()
        self.alpha = alpha
        self.nll_loss = nn.PoissonNLLLoss(log_input=False, reduction='mean', full=True)

    def forward(self, pred_signals, true_signals, masked_map, obs_map):

        obs_loss =  self.nll_loss(pred_signals[obs_map], true_signals[obs_map])
        pred_loss = self.nll_loss(pred_signals[masked_map], true_signals[masked_map])
        
        return self.alpha*pred_loss, (1-self.alpha)*obs_loss

class ComboLoss_NBNLL(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss_NBNLL, self).__init__()
        self.alpha = alpha
        self.reduction = 'sum'

    def forward(self, p_pred, n_pred, true_signals, masked_map, obs_map):
        ups_y_true, ups_n_pred, ups_p_pred = true_signals[obs_map], n_pred[obs_map], p_pred[obs_map]
        imp_y_true, imp_n_pred, imp_p_pred = true_signals[masked_map], n_pred[masked_map], p_pred[masked_map]

        upsampling_loss = negative_binomial_loss(ups_y_true, ups_n_pred, ups_p_pred)
        imputation_loss = negative_binomial_loss(imp_y_true, imp_n_pred, imp_p_pred)
        
        if self.reduction == "mean":
            upsampling_loss = upsampling_loss.mean()
            imputation_loss = imputation_loss.mean()
        else:
            upsampling_loss = upsampling_loss.sum()
            imputation_loss = imputation_loss.sum()

        return self.alpha * imputation_loss, (1-self.alpha) * upsampling_loss

class ComboLoss_NBNLL_msk(nn.Module):
    def __init__(self):
        super(ComboLoss_NBNLL_msk, self).__init__()
        self.alpha = alpha
        self.reduction = 'mean'
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, p_pred, n_pred, pred_mask, true_signals, masked_map, obs_map):
        ups_y_true, ups_n_pred, ups_p_pred = true_signals[obs_map], n_pred[obs_map], p_pred[obs_map]
        imp_y_true, imp_n_pred, imp_p_pred = true_signals[masked_map], n_pred[masked_map], p_pred[masked_map]

        upsampling_loss = negative_binomial_loss(ups_y_true, ups_n_pred, ups_p_pred)
        imputation_loss = negative_binomial_loss(imp_y_true, imp_n_pred, imp_p_pred)

        if self.reduction == "mean":
            upsampling_loss = upsampling_loss.mean()
            imputation_loss = imputation_loss.mean()

        bce_mask_loss = self.bce_loss(pred_mask, obs_map.float())

        return imputation_loss, upsampling_loss, bce_mask_loss

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

        # self.mf_embedding = MatrixFactorizationEmbedding(l=context_length, d=input_dim, k=d_model)
        self.embedding_linear = nn.Linear(input_dim, d_model)

        self.encoder_layer = RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.signal_decoder =  nn.Linear(d_model, output_dim)
        self.signal_softplus = nn.Softplus()
        # self.signal_decoder = FeedForwardNN(d_model, 4*d_model, output_dim, 2)
        self.mask_decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src = self.mf_embedding(src, linear=True)
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        
        src = self.transformer_encoder(src) 
        
        msk = torch.sigmoid(self.mask_decoder(src))
        src = self.signal_softplus(self.signal_decoder(src))

        src = torch.permute(src, (1, 0, 2))  # to N, L, F
        msk = torch.permute(msk, (1, 0, 2))  # to N, L, F

        return src, msk

class EpiDenoise20(nn.Module):
    def __init__(self, 
                 input_dim, conv_out_channels, conv_kernel_sizes, dilation,
                 nhead, d_model, n_encoder_layers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise20, self).__init__()

        stride = 1
        n_cnn_layers = len(conv_out_channels)

        # Convolutional layers
        self.conv1 = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convm = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convtower = nn.Sequential(*[
            ConvTower(
                conv_out_channels[i], 
                conv_out_channels[i + 1],
                conv_kernel_sizes[i + 1], stride, dilation
            ) for i in range(n_cnn_layers - 1)
        ])

        # self.position = AbsPositionalEmbedding15(d_model=d_model, max_len=context_length)
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.encoder_layer, num_layers=n_encoder_layers)

        self.encoder_layer = RelativeEncoderLayer(
            d_model=d_model, heads=nhead, feed_forward_hidden=2*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_encoder_layers)

        # Deconvolution layers
        reversed_channels = list(reversed(conv_out_channels))
        reversed_kernels = list(reversed(conv_kernel_sizes))

        self.deconvtower = nn.Sequential(*[
            DeconvBlock(
                reversed_channels[i], reversed_channels[i + 1], 
                reversed_kernels[i + 1], 2, dilation) for i in range(n_cnn_layers - 2)
        ])
        self.deconvF = DeconvBlock(reversed_channels[-2], output_dim, reversed_kernels[-1], 2, dilation)

        self.signal_decoder = nn.Linear(output_dim, output_dim)
        self.mask_decoder = nn.Linear(output_dim, output_dim)

    def forward(self, x, m):
        x = x.permute(0, 2, 1) # to N, F, L
        m = m.permute(0, 2, 1) # to N, F, L

        m = self.convm(m.float())
        x = self.conv1(x)

        x = x + m

        x = self.convtower(x)

        x = x.permute(0, 2, 1)  # to N, L, F

        # x = x + self.position(x)
        x = self.transformer_encoder(x)

        x = x.permute(0, 2, 1) # to N, F, L'
        x = self.deconvtower(x)
        x = self.deconvF(x)
        x = x.permute(2, 0, 1)  # to L, N, F

        mask = torch.sigmoid(self.mask_decoder(x))
        x = self.signal_decoder(x)

        x = torch.permute(x, (1, 0, 2))  # to N, L, F
        mask = torch.permute(mask, (1, 0, 2))  # to N, L, F

        return x, mask

class EpiDenoise21(nn.Module):
    def __init__(self, 
                input_dim, conv_out_channels, conv_kernel_sizes, dilation, nhead, 
                d_model, n_encoder_layers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise21, self).__init__()

        stride = 1
        n_cnn_layers = len(conv_out_channels)

        # Convolutional layers
        self.conv1 = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convm = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convtower = nn.Sequential(*[
            ConvTower(
                conv_out_channels[i], 
                conv_out_channels[i + 1],
                conv_kernel_sizes[i + 1], stride, dilation
            ) for i in range(n_cnn_layers - 1)
        ])

        self.encoder_layer = RelativeEncoderLayer(
            d_model=d_model, heads=nhead, feed_forward_hidden=2*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_encoder_layers)

        # Convolutional layers
        self.d_conv1 = ConvTower(
            input_dim, d_model, 
            1, stride, dilation, pool_type="None", residuals=False)

        self.d_convm = ConvTower(
            input_dim, d_model, 
            1, stride, dilation, pool_type="None", residuals=False)

        self.relative_decoder = RelativeDecoderLayer(
            hid_dim=d_model, 
            n_heads=nhead, 
            pf_dim=2*d_model, 
            dropout=dropout
        )

        self.linear_output = nn.Linear(d_model, output_dim)

    def forward(self, src, src_missing_mask, trg, trg_missing_mask, trg_mask):
        src_missing_mask = src_missing_mask.permute(0, 2, 1) # to N, F, L
        src_missing_mask = ~src_missing_mask
        src_missing_mask = self.convm(src_missing_mask.float())

        src = src.permute(0, 2, 1) # to N, F, L
        src = self.conv1(src)

        src = src + src_missing_mask        
        src = self.convtower(src)

        src = src.permute(0, 2, 1)  # to N, L, F
        src = self.transformer_encoder(src)
        
        trg = trg.permute(0, 2, 1) # to N, F, L
        trg = self.d_conv1(trg) 

        trg_missing_mask = trg_missing_mask.permute(0, 2, 1) # to N, F, L
        trg_missing_mask = ~trg_missing_mask
        trg_missing_mask = self.d_convm(trg_missing_mask.float())

        trg = trg + trg_missing_mask  
        trg = trg.permute(0, 2, 1)  # to N, L, F

        # Apply the relative decoder
        trg = self.relative_decoder(trg, src, trg_mask)

        # Apply the final linear layers
        trg = self.linear_output(trg)

        return trg

class EpiDenoise22(nn.Module):
    def __init__(
        self, input_dim, conv_out_channels, conv_kernel_sizes, nhead, 
        d_model, n_encoder_layers, n_decoder_layers, output_dim, dilation=1, 
        dropout=0.1, context_length=2000, aggr=False):

        super(EpiDenoise22, self).__init__()

        stride = 1
        n_cnn_layers = len(conv_out_channels)
        self.aggr=aggr

        self.dual_conv_emb_src = DualConvEmbedding(in_C=input_dim, out_C=conv_out_channels[0])

        self.convtower = nn.ModuleList([ConvTower(
                conv_out_channels[i], conv_out_channels[i + 1],
                conv_kernel_sizes[i + 1], stride, dilation, 
                pool_type="attn", residuals=True
            ) for i in range(n_cnn_layers - 1)])

        if self.aggr:
            self.aggr_token = nn.Parameter(torch.randn(1, 1, d_model))
            # Additional linear layers for predicting mean and standard deviation
            self.mean_prediction = nn.Linear(d_model, output_dim)
            self.stddev_prediction = nn.Linear(d_model, output_dim)


        self.transformer_encoder = nn.ModuleList([RelativeEncoderLayer(
                d_model=d_model, heads=nhead, 
                feed_forward_hidden=2*d_model, 
                dropout=dropout) for _ in range(n_encoder_layers)])

        self.dual_conv_emb_trg = DualConvEmbedding(in_C=input_dim, out_C=d_model)

        self.transformer_decoder = nn.ModuleList([RelativeDecoderLayer(
            hid_dim=d_model, n_heads=nhead, 
            pf_dim=2*d_model, dropout=dropout) for _ in range(n_decoder_layers)])
    
        self.linear_output = nn.Linear(d_model, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, seq, mask):
        mask = mask.permute(0, 2, 1) # to N, F, L
        mask = mask.float()
        src = seq.permute(0, 2, 1) # to N, F, L
        trg = seq.permute(0, 2, 1) # to N, F, L

        src = self.dual_conv_emb_src(src, mask)
        trg = self.dual_conv_emb_trg(trg, mask)

        for conv in self.convtower:
            src = conv(src)

        src = src.permute(0, 2, 1)  # to N, L, F
        if self.aggr:
            # Concatenate special token to src
            batch_size, seq_len, _ = src.shape
            aggr_token = self.aggr_token.repeat(batch_size, 1, 1)
            src = torch.cat([aggr_token, src], dim=1)
            # print("agg+src", src.shape)

        for enc in self.transformer_encoder:
            src = enc(src)
            # print("src", src.shape)
        
        # print(src.shape)
        if self.aggr:
            aggr_token = src[:, 0, :]
            src = src[:, 1:, :]

            aggrmean = self.softplus(self.mean_prediction(aggr_token))
            aggrstddev = self.softplus(self.stddev_prediction(aggr_token))

        # print("trg",trg.shape)
        trg = trg.permute(0, 2, 1)  # to N, L, F
        for dec in self.transformer_decoder:
            trg = dec(trg, src, pad)
            # print("trg", trg.shape)
        
        trg = self.linear_output(trg)
        trg = self.softplus(trg)

        if self.aggr:
            return trg, aggrmean, aggrstddev
        else:
            return trg

# class EpiDenoise30a(nn.Module):
#     def __init__(
#         self, input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
#         dropout=0.1, context_length=2000, pos_enc="relative"):
#         super(EpiDenoise30a, self).__init__()
#         self.pos_enc = pos_enc
#         self.context_length = context_length
        
#         self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim)
#         self.embedding_linear = nn.Linear(input_dim + metadata_embedding_dim, d_model)

#         if self.pos_enc == "relative":
#             self.encoder_layer = RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=2*d_model, dropout=dropout)
#         else:
#             self.position = AbsPositionalEmbedding15(d_model=d_model, max_len=context_length)
#             self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        
#         self.transformer_encoder = nn.ModuleList(
#             [self.encoder_layer for _ in range(nlayers)])

#         self.neg_binom_layer = NegativeBinomialLayer(d_model, output_dim)
    
#     def forward(self, src, x_metadata, y_metadata, availability):
#         md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
#         md_embedding = md_embedding.unsqueeze(1).expand(-1, self.context_length, -1)

#         src = torch.cat([src, md_embedding], dim=-1)
#         src = F.relu(self.embedding_linear(src))

#         src = torch.permute(src, (1, 0, 2)) # to L, N, F

#         if self.pos_enc != "relative":
#             src = src + self.position(src)
        
#         for enc in self.transformer_encoder:
#             src = enc(src)

#         p, n = self.neg_binom_layer(src)

#         p = torch.permute(p, (1, 0, 2))  # to N, L, F
#         n = torch.permute(n, (1, 0, 2))  # to N, L, F

#         return p, n

class EpiDenoise30a(nn.Module):
    def __init__(
        self, input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30a, self).__init__()
        self.pos_enc = pos_enc
        self.context_length = context_length
        
        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim)
        self.embedding_linear = nn.Linear(input_dim + metadata_embedding_dim, d_model)

        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=2*d_model, dropout=dropout)
        else:
            self.position = AbsPositionalEmbedding15(d_model=d_model, max_len=context_length)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        
        self.transformer_encoder = nn.ModuleList(
            [self.encoder_layer for _ in range(nlayers)])

        self.neg_binom_layer = NegativeBinomialLayer(d_model, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
        md_embedding = md_embedding.unsqueeze(1).expand(-1, self.context_length, -1)

        src = torch.cat([src, md_embedding], dim=-1)
        src = F.relu(self.embedding_linear(src))

        src = torch.permute(src, (1, 0, 2)) # to L, N, F

        if self.pos_enc != "relative":
            src = src + self.position(src)
        
        for enc in self.transformer_encoder:
            src = enc(src)

        p, n = self.neg_binom_layer(src)

        p = torch.permute(p, (1, 0, 2))  # to N, L, F
        n = torch.permute(n, (1, 0, 2))  # to N, L, F

        return p, n

class EpiDenoise30b(nn.Module):
    def __init__(self, 
        input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30b, self).__init__()

        conv_out_channels = exponential_linspace_int(
            d_model//n_cnn_layers, d_model, n_cnn_layers, divisible_by=2)

        stride = 1
        dilation=1
        self.context_length = context_length
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim, non_linearity=True)
        self.lin = nn.Linear(input_dim + metadata_embedding_dim, d_model)

        self.conv0 = ConvTower(
                input_dim + metadata_embedding_dim, conv_out_channels[0],
                conv_kernel_size[0], stride, dilation, 
                pool_type="max", residuals=True)

        self.convtower = nn.ModuleList([ConvTower(
                conv_out_channels[i], conv_out_channels[i + 1],
                conv_kernel_size[i + 1], stride, dilation, 
                pool_type="max", residuals=True
            ) for i in range(n_cnn_layers - 1)])

        self.transformer_encoder = nn.ModuleList(
            [RelativeEncoderLayer(
                d_model=d_model, heads=nhead, 
                feed_forward_hidden=2*d_model, 
                dropout=dropout) for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList(
            [RelativeDecoderLayer(
                hid_dim=d_model, n_heads=nhead, 
                pf_dim=2*d_model, dropout=dropout) for _ in range(n_decoder_layers)])
        
        self.neg_binom_layer = NegativeBinomialLayer(d_model, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
        md_embedding = md_embedding.unsqueeze(1).expand(-1, self.context_length, -1)

        src = F.relu(torch.cat([src, md_embedding], dim=-1))

        e_src = src.permute(0, 2, 1) # to N, F, L
        e_src = self.conv0(e_src)

        for conv in self.convtower:
            e_src = conv(e_src)
        
        e_src = e_src.permute(0, 2, 1)  # to N, L, F
        for enc in self.transformer_encoder:
            e_src = enc(e_src)
        
        src = self.lin(src)
        for dec in self.transformer_decoder:
            src = dec(src, e_src)

        p, n = self.neg_binom_layer(src)
        return p, n

#========================================================================================================#
#=========================================Pretraining====================================================#
#========================================================================================================#

class PRE_TRAINER(object):  
    def __init__(self, model, dataset, criterion, optimizer, scheduler, eed=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        print(self.device)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if not eed:
            self.test_x_dir = "/project/compbio-lab/EIC/training_data/C23_chr21_25.pt"
            self.test_y_dir = "/project/compbio-lab/EIC/validation_data/C23_chr21_25.pt"
    
    def test_model(self, context_length, version, is_arcsin, batch_size):
        self.model.eval()
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
                mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                for ii in missing_x_i: 
                    mask[:,:,ii] = True
                mask = mask.to(self.device)

                if version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif version == "17":
                    outputs, SAP = self.model(x_batch, ~mask, segment_label)
                
                elif version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif version == "20":

                    outputs, pred_mask = self.model(x_batch, mask)
                
                elif version == "21":
                    outputs, pred_mask = self.model(x_batch, mask)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        mses = []
        spearmans = []
        peak_overlaps = []
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()   
                mse_GW = np.mean((np.array(target) - np.array(pred))**2)
                spearman_GW = spearmanr(pred, target)[0]
                ovr = peak_overlap(target, pred, p=0.05)

                mses.append(mse_GW)
                spearmans.append(spearman_GW)
                peak_overlaps.append(ovr)

        self.model.train()

        if version in ["10", "16", "17"]:
            return sum(mses)/len(mses), sum(spearmans)/len(spearmans)
        else:
            return mses, spearmans, peak_overlaps

    def test_autoregressive_model(self, context_length, is_arcsin, step_size, p=0.01):
        self.model.eval()

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

        L, num_features = X.shape

        subset_size = int(L * p)  # The total number of rows to include in the subset
        start_index = (L - subset_size) // 2  # Start index for the middle subset
        end_index = start_index + subset_size  # End index for the middle subset

        start_index -= start_index % step_size
        end_index -= end_index % step_size
        
        # Slice X and Y to get the middle subset
        X = X[start_index:end_index, :]
        Y = Y[start_index:end_index, :]
        
        if is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])
        
        mask = torch.zeros_like(X, dtype=torch.bool)
        for ii in missing_x_i: 
            mask[:,ii] = True

        # ============================================ #
        #               SENSE PREDICTION
        # ============================================ #
        src_context = []
        trg_context = []

        missing_mask = []

        for i in range(0, X.shape[0] - context_length, step_size):
            if i+context_length+step_size > X.shape[0]:
                break

            src_context.append(X[i : i+context_length, :].numpy())
            trg_context.append(X[i+step_size : i+context_length+step_size, :].numpy())

            missing_mask.append(mask[i : i+context_length, :].numpy())

        src_context = torch.from_numpy(np.array(src_context)).to(self.device)
        trg_context = torch.from_numpy(np.array(trg_context)).to(self.device)

        missing_mask = torch.from_numpy(np.array(missing_mask)).to(self.device)

        trg_msk = torch.zeros((trg_context.shape[0], trg_context.shape[1]), dtype=torch.bool, device=self.device)
        trg_msk[:, -step_size:] = True

        with torch.no_grad():
            fw_outputs = self.model(
                src_context, missing_mask, trg_context, missing_mask, trg_msk) 
            
            fw_outputs = fw_outputs[:,-step_size:, :]
        
        fw_outputs = fw_outputs.reshape(fw_outputs.shape[0]*fw_outputs.shape[1], fw_outputs.shape[2])
        
        # ============================================ #
        #           ANTI-SENSE PREDICTION
        # ============================================ #
        X = torch.flip(X, dims=(0,))
        src_context = []
        trg_context = []
        missing_mask = []

        start_i = X.shape[0] - (2 * context_length) - 1

        for i in range(start_i, X.shape[0] - context_length, step_size):
            if i+context_length+step_size > X.shape[0]:
                break

            src_context.append(X[i : i+context_length, :].numpy())
            trg_context.append(X[i+step_size : i+context_length+step_size, :].numpy())

            missing_mask.append(mask[i : i+context_length, :].numpy())

        src_context = torch.from_numpy(np.array(src_context)).to(self.device)
        trg_context = torch.from_numpy(np.array(trg_context)).to(self.device)

        missing_mask = torch.from_numpy(np.array(missing_mask)).to(self.device)

        trg_msk = torch.zeros((trg_context.shape[0], trg_context.shape[1]), dtype=torch.bool, device=self.device)
        trg_msk[:, -step_size:] = True

        with torch.no_grad():
            bw_outputs = self.model(
                src_context, missing_mask, trg_context, missing_mask, trg_msk) 
            
            bw_outputs = bw_outputs[:,-step_size:, :]
        
        bw_outputs = bw_outputs.reshape(bw_outputs.shape[0]*bw_outputs.shape[1], bw_outputs.shape[2])
        bw_outputs = torch.flip(bw_outputs, dims=(0,))

        P = torch.cat((bw_outputs, fw_outputs), dim=0).cpu()

        del src_context, trg_context, missing_mask, trg_msk, bw_outputs, fw_outputs

        mses = []
        spearmans = []
        peak_overlaps = []
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()   
                mse_GW = np.mean((np.array(target) - np.array(pred))**2)
                spearman_GW = spearmanr(pred, target)[0]
                ovr = peak_overlap(target, pred, p=0.05)

                mses.append(mse_GW)
                spearmans.append(spearman_GW)
                peak_overlaps.append(ovr)

        self.model.train()
        return mses, spearmans, peak_overlaps

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

                            masked_x_batch = add_noise(masked_x_batch, 0.2)
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
                    
                    test_mse, test_corr, test_ovr = self.test_model(
                        context_length, version="18", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    test_mse = np.mean(test_mse)
                    test_corr = np.mean(test_corr)

                    test_ovr_mean = np.mean(test_ovr)
                    test_ovr_min = np.min(test_ovr)
                    test_ovr_max = np.max(test_ovr)

                    logstr = [
                        "\n----------------------------------------------------\n"
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Obs Loss: {np.mean(epoch_obs_loss):.3f}", 
                        f"Clz Loss: {np.mean(epoch_clz_loss):.3f}", 
                        f"Msk Loss: {np.mean(epoch_msk_loss):.3f}", 
                        f"Val_MSE: {test_mse:.4f}",
                        f"Val_POmean: {test_ovr_mean:.3f}",
                        f"Val_POmin: {test_ovr_min:.3f}",
                        f"Val_POmax: {test_ovr_max:.3f}",
                        f"Epoch took: {t1 - t0}"
                        "\n----------------------------------------------------\n"
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
        d_model, outer_loop_epochs=3, arcsinh_transform=True,
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
                            
                            # if len(available_assays_ind) == 1:
                            masked_x_batch, cloze_mask = mask_data16(
                                x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                            # else:
                            #     masked_x_batch, cloze_mask = mask_data18(
                                    # x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                                
                            union_mask = cloze_mask | missing_mask_batch

                            masked_x_batch = add_noise(masked_x_batch, 0.2)
                            masked_x_batch[union_mask] = -1

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask = self.model(masked_x_batch, ~union_mask)

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
                    
                    test_mse, test_corr, test_ovr = self.test_model(
                        context_length, version="20", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    test_mse = np.mean(test_mse)
                    test_corr = np.mean(test_corr)

                    test_ovr_mean = np.mean(test_ovr)
                    # test_ovr_min = np.min(test_ovr)
                    # test_ovr_max = np.max(test_ovr)

                    logstr = [
                        "\n----------------------------------------------------\n"
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Obs Loss: {np.mean(epoch_obs_loss):.3f}", 
                        f"Clz Loss: {np.mean(epoch_clz_loss):.3f}", 
                        f"Msk Loss: {np.mean(epoch_msk_loss):.3f}", 
                        f"Val_MSE: {test_mse:.3f}",
                        f"Val_Corr: {test_corr:.3f}",
                        f"Val_POmean: {test_ovr_mean:.3f}",
                        # f"Val_POmin: {test_ovr_min:.3f}",
                        # f"Val_POmax: {test_ovr_max:.3f}",
                        f"Epoch took: {t1 - t0}"
                        "\n----------------------------------------------------\n"
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
    
    def pretrain_epidenoise_21(self, 
        d_model, outer_loop_epochs=1, arcsinh_transform=True, step_size=40,
        num_epochs=25, context_length=2000, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD21_log.txt", "w")
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
                N, L, num_features = x.shape

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    epoch_loss = []

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        p_loss = []
                        x_batch = x[indices]

                        if epoch%2==1:
                            x_batch = torch.flip(x_batch, dims=(1,))

                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]
                        if len(available_assays_ind) < 2:
                            continue

                        for i in range(0, L - context_length, step_size):
                            self.optimizer.zero_grad()

                            if i+context_length+step_size > L:
                                break

                            # Extract the context and the target for this step
                            context = x_batch[:, i:i+context_length, :].to(self.device)
                            missing_msk_src = missing_mask_patten_batch[:, i:i+context_length, :].to(self.device)

                            target_context = x_batch[:, i+step_size:i+context_length+step_size, :].to(self.device)

                            if torch.isnan(context).sum() > 0 or torch.isnan(target_context).sum() > 0:
                                skipmessage = "Encountered nan data! Skipping..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                continue

                            trg_msk = torch.zeros(
                                (target_context.shape[0], target_context.shape[1]), 
                                dtype=torch.bool, device=self.device)
                            
                            trg_msk[:, -step_size:] = True

                            next_pos_mask = torch.zeros_like(target_context, dtype=torch.bool, device=self.device)
                            next_pos_mask[:,-step_size:, :] = True
                            next_pos_mask = next_pos_mask & ~missing_msk_src

                            try:
                                outputs = self.model(
                                    context, missing_msk_src, target_context, missing_msk_src, trg_msk)

                                loss = self.criterion(outputs, target_context, missing_msk_src, next_pos_mask)

                                if torch.isnan(loss).sum() > 0:
                                    skipmessage = "Encountered nan loss! Skipping batch..."
                                    log_strs.append(skipmessage)
                                    print(skipmessage)
                                    del context, target_context, missing_msk_src, outputs, next_pos_mask, loss
                                    torch.cuda.empty_cache()
                                    continue
                                
                                p_loss.append(loss.item())
                                loss.backward()  
                                self.optimizer.step()

                                del context, target_context, missing_msk_src, outputs, next_pos_mask, loss

                            except:
                                skipmessage = f"Exception! Skipping... [e:{epoch}, p:{p}]"
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del context, target_context, missing_msk_src, outputs, next_pos_mask, loss
                                torch.cuda.empty_cache()
                                continue

                        # Clear GPU memory again
                        torch.cuda.empty_cache()
                        if not math.isnan(np.mean(p_loss)):
                            epoch_loss.append(np.mean(p_loss))

                        if p==1 or p%4==0:
                            logfile = open("models/EPD21_log.txt", "w")
                            logstr = [
                                "\n----------------------------------------------------\n"
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Pattern #: {p}/{len(missing_f_pattern)}', 
                                f'P_epoch: {np.mean(p_loss):.4f}'
                                "\n----------------------------------------------------"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)

                    test_mse, test_corr, test_ovr = self.test_autoregressive_model(
                        context_length, is_arcsin=arcsinh_transform, step_size=step_size)

                    test_mse = np.mean(test_mse)
                    test_corr = np.mean(test_corr)

                    test_ovr_mean = np.mean(test_ovr)
                    torch.cuda.empty_cache()

                    t1 = datetime.now()
                    logfile = open("models/EPD21_log.txt", "w")
                    logstr = [
                        "\n----------------------------------------------------\n"
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Pattern #: {p}/{len(missing_f_pattern)}', 
                        f'epoch_loss: {np.mean(epoch_loss):.4f}', 
                        f"Val_MSE: {test_mse:.3f}",
                        f"Val_POmean: {test_ovr_mean:.3f}",
                        f"Val_Corr: {test_corr:.3f}",
                        f"Epoch took: {t1 - t0}"
                        "\n----------------------------------------------------\n"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)
                        
                    self.scheduler.step()

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD21_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_22(self, 
        d_model, outer_loop_epochs=1, arcsinh_transform=True, focus_middle=False, num_random_segs=10,
        num_epochs=25, mask_percentage=0.15, context_length=2000, start_ds=0, batch_size=50):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD22_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }

        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        for ole in range(outer_loop_epochs):

            ds=0
            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                N, L, num_features = x.shape

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    t0 = datetime.now()
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    epoch_loss = []

                    pred_loss = []
                    obs_loss = []

                    mseobs_loss = []
                    msepred_loss = []

                    r2obs_loss = []
                    r2pred_loss = []

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1
                        
                        p_batch = x[indices]
                        missing_p_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]
                        
                        if focus_middle:
                            # Select m random points along the L dimension
                            random_points = torch.randint(low=0, high=L, size=(num_random_segs,))

                            # Initialize the output tensor with padding value (e.g., 0) and the padding tracker
                            xp_batch = torch.zeros((num_random_segs * p_batch.shape[0], context_length, num_features))

                            for i, point in enumerate(random_points):
                                start = point - context_length // 2
                                end = point + context_length // 2

                                adjusted_start = max(start, 0)
                                adjusted_end = min(end, L)
                                ival = p_batch[:, adjusted_start:adjusted_end, :]

                                if ival.shape[1] < context_length:
                                    pad_length = context_length - ival.shape[1]
                                    pad = torch.full((ival.shape[0], pad_length, ival.shape[2]), token_dict["pad"])
                                    if start < 0:
                                        ival = torch.cat([pad, ival], dim=1)
                                    elif end > L:
                                        ival = torch.cat([ival, pad], dim=1)

                                xp_batch[i*p_batch.shape[0]:(i+1)*p_batch.shape[0]] = ival   
                        
                        else:
                            context_length_factor = context_length / p_batch.shape[1]

                            xp_batch = reshape_tensor(p_batch, context_length_factor)
                            missing_p_batch = reshape_tensor(missing_p_batch, context_length_factor)
                            del p_batch

                        for x_p in range(0, xp_batch.shape[0], batch_size):
                            self.optimizer.zero_grad()
                            torch.cuda.empty_cache()
                            x_batch = xp_batch[x_p:x_p+batch_size, :, :]

                            """
                            src: cloze(x_batch)
                            trg: cloze(x_batch)
                            trg_pad: x_batch_pad

                            src_mask and trg_msk
                                union_mask: union(missing, cloze, x_batch_pad)
                            """

                            x_batch_pad = (x_batch == token_dict["pad"])

                            if focus_middle:
                                masked_x_batch, cloze_mask = self.masker.mid_slice_focused_full_feature_mask(x_batch, token_dict["missing_mask"], available_assays_ind)
                            else:                                
                                masked_x_batch, cloze_mask = self.masker.mask_chunk_features(x_batch, available_assays_ind)

                            # ensure that padded regions remain padded
                            x_batch[x_batch_pad] = token_dict["pad"]
                            
                            if torch.isnan(x_batch).sum() > 0:
                                skipmessage = "Encountered nan input! Skipping batch..."
                                log_strs.append(skipmessage)
                                # print(skipmessage)
                                del x_batch
                                torch.cuda.empty_cache()
                                continue

                            x_batch_missing = (x_batch == token_dict["missing_mask"])
                            # ensure that padded regions remain padded
                            
                            union_mask = x_batch_pad | cloze_mask | x_batch_missing

                            x_batch_pad = x_batch_pad[:, :, 0]

                            x_batch_pad = x_batch_pad.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)
                            x_batch = x_batch.to(self.device)

                            # outputs, aggrmean, aggrstd = self.model(masked_x_batch, union_mask, x_batch_pad) #(sequence, mask, pad)
                            outputs = self.model(masked_x_batch, union_mask, x_batch_pad) #(sequence, mask, pad)

                            aggr_mask = torch.zeros((x_batch.shape[0], x_batch.shape[2]), dtype=torch.bool)
                            for av_i in available_assays_ind:
                                aggr_mask[:,av_i] = True

                            # mse_pred_loss, mse_obs_loss, mse_aggrmean_loss, mse_aggrstd_loss = self.criterion(
                            #     outputs, x_batch, cloze_mask, union_mask, aggrmean, aggrstd, aggr_mask)
                            mse_pred_loss, mse_obs_loss = self.criterion(
                                outputs, x_batch, cloze_mask, union_mask)#, aggrmean, aggrstd, aggr_mask)


                            obs_mse = (((x_batch[~union_mask]) - (outputs[~union_mask]))**2).mean().item()
                            mseobs_loss.append(obs_mse)
                            prd_mse = (((x_batch[cloze_mask]) - (outputs[cloze_mask]))**2).mean().item()
                            msepred_loss.append(prd_mse)

                            r2_obs = r2_score(
                                (x_batch[~union_mask]).cpu().detach().numpy(), 
                                (outputs[~union_mask]).cpu().detach().numpy())
                            r2obs_loss.append(r2_obs)
                            r2_pred = r2_score(
                                (x_batch[cloze_mask]).cpu().detach().numpy(), 
                                (outputs[cloze_mask]).cpu().detach().numpy())
                            r2pred_loss.append(r2_pred)

                            loss = mse_pred_loss + mse_obs_loss #+ mse_aggrmean_loss + mse_aggrstd_loss
                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                print(len(available_assays_ind), loss)
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

                            pred_loss.append(mse_pred_loss.item())
                            obs_loss.append(mse_obs_loss.item())
                            
                            # aggrmean_loss.append(mse_aggrmean_loss.item())
                            # aggrstd_loss.append(mse_aggrstd_loss.item())

                            epoch_loss.append(loss.item())
                            loss.backward()  
                            self.optimizer.step()
                    
                    # self.scheduler.step(np.mean(r2pred_loss))
                    logfile = open("models/EPD22_log.txt", "w")

                    elapsed_time = datetime.now() - t0
                    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}',
                        f"L_tot: {np.mean(epoch_loss):.2f}",
                        f"L_clz: {np.mean(pred_loss):.2f}",
                        f"L_obs: {np.mean(obs_loss):.2f}",
                        f"mse_prd: {np.mean(msepred_loss):.2f}",
                        f"mse_obs: {np.mean(mseobs_loss):.2f}",
                        f"R2_prd: {np.mean(r2pred_loss):.2f}",
                        f"R2_obs: {np.mean(r2obs_loss):.2f}",
                        f"took: {int(minutes)}:{int(seconds)}"]

                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                try:
                    if ds%11 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD22_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_30(self, 
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=50, inner_epochs=5, arch="a"):
        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"EPD30{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/EPD30{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        dsf_list = [1, 2, 4]#, 8]

        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        """
        - each epoch consists of going through all dataset.m_regions and all biosamples in the dataset.navigation
        - total number of training samples in each epoch is  len(dataset.m_regions) * len(dataset.navigation)
            - each batch consists of batch_size number of biosamples for 1 region
        """
        # register_hooks(self.model)
            
        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)
        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

            while (next_epoch==False) and (self.dataset.current_loci_batch_pointer < self.dataset.num_regions or self.dataset.current_bios_batch_pointer < self.dataset.num_bios):
                t0 = datetime.now()
                # Randomly choose two downsampling factors and assign them to dsf_X and dsf_Y based on their values
                dsf_X, dsf_Y = sorted(random.choices(dsf_list, k=2), reverse=True) # dsf_X is of equal or higher dsf

                _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(dsf_X)
                _Y_batch, _mY_batch, _avY_batch = self.dataset.get_batch(dsf_Y)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_loss":[],
                    "ups_loss":[],
                    "ups_r2":[],
                    "imp_r2":[],
                    "imp_spearman":[],
                    "ups_spearman":[],
                    "ups_mse":[],
                    "imp_mse":[]
                }
                # for _ in range(inner_epochs):
                while True:
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone()

                    # X_batch, mX_batch, avail_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)
                    avail_batch = avX_batch

                    masked_map = (X_batch == token_dict["cloze_mask"])
                    observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                    missing_map = (X_batch == token_dict["missing_mask"])

                    X_batch = X_batch.float().to(self.device).requires_grad_(True)
                    mX_batch = mX_batch.to(self.device)
                    avail_batch = avail_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    masked_map = masked_map.to(self.device) # imputation targets
                    observed_map = observed_map.to(self.device) # upsampling targets

                    output_p, output_n = self.model(X_batch, mX_batch, mY_batch, avail_batch)

                    # Retain gradients for intermediate tensors
                    # output_p.retain_grad()
                    # output_n.retain_grad()

                    pred_loss, obs_loss = self.criterion(
                        output_p, output_n, Y_batch, masked_map, observed_map) # p_pred, n_pred, true_signals, masked_map, obs_map

                    if torch.isnan(pred_loss).any():
                        loss = obs_loss
                    else:
                        loss = pred_loss+obs_loss  

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss.backward()  

                    # Initialize variables to store maximum gradient norms and corresponding layer names
                    # max_weight_grad_norm = 0
                    # max_weight_grad_layer = None
                    # max_bias_grad_norm = 0
                    # max_bias_grad_layer = None

                    # # Check and update maximum gradient norms
                    # for name, module in self.model.named_modules():
                    #     if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                    #         if module.weight.grad_norm > max_weight_grad_norm:
                    #             max_weight_grad_norm = module.weight.grad_norm
                    #             max_weight_grad_layer = name

                    #     if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                    #         if module.bias.grad_norm > max_bias_grad_norm:
                    #             max_bias_grad_norm = module.bias.grad_norm
                    #             max_bias_grad_layer = name

                    # Print the layers with the maximum weight and bias gradients
                    # if max_weight_grad_layer:
                    #     print(f"Epoch {epoch}, Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}")
                    # if max_bias_grad_layer:
                    #     print(f"Epoch {epoch}, Max Bias Grad Layer: {max_bias_grad_layer}, Bias Grad Norm: {max_bias_grad_norm:.3f}")

                    print(obs_loss.item(), pred_loss.item())
                    self.optimizer.step()
                    continue
                    
                    ups_pred = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()
                        ).expect().cpu().detach().numpy()

                    imp_pred = NegativeBinomial(
                        output_p[masked_map].cpu().detach(), 
                        output_n[masked_map].cpu().detach()
                        ).expect().cpu().detach().numpy()

                    ups_true = Y_batch[observed_map].cpu().detach().numpy()
                    imp_true = Y_batch[masked_map].cpu().detach().numpy()
                    
                    ups_r2 = r2_score(ups_true, ups_pred)
                    imp_r2 = r2_score(imp_true, imp_pred)

                    ups_mse = ((ups_true - ups_pred)**2).mean()
                    imp_mse = ((imp_true - imp_pred)**2).mean()

                    ups_spearman = spearmanr(ups_pred, ups_true)[0]
                    imp_spearman = spearmanr(imp_pred, imp_true)[0]
                    
                    batch_rec["imp_loss"].append(pred_loss.item())
                    batch_rec["ups_loss"].append(obs_loss.item())
                    batch_rec["ups_r2"].append(ups_r2)
                    batch_rec["imp_r2"].append(imp_r2)
                    batch_rec["imp_spearman"].append(imp_spearman)
                    batch_rec["ups_spearman"].append(ups_spearman)
                    batch_rec["ups_mse"].append(ups_mse)
                    batch_rec["imp_mse"].append(imp_mse)
                
                lopr = int((self.dataset.current_loci_batch_pointer/self.dataset.num_regions) * 100)
                if lopr > 1 and lopr % 10 == 0:
                    # self.scheduler.step()
                    try:
                        torch.save(
                            self.model.state_dict(), 
                            f'models/EPD30{arch}_model_checkpoint_epoch{epoch}_LociProg{lopr}.pth')
                    except:
                        pass

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                logstr = [
                    f"Ep. {epoch}",
                    f"DSF{dsf_X}->{dsf_Y}",
                    f"Loci Prog. {self.dataset.current_loci_batch_pointer/self.dataset.num_regions:.2%}",
                    f"Bios Prog. {self.dataset.current_bios_batch_pointer/self.dataset.num_bios:.2%}",
                    f"Imp_Loss {np.mean(batch_rec['imp_loss']):.2f}",
                    f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                    f"Imp_R2 {np.mean(batch_rec['imp_r2']):.2f}",
                    f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                    f"Imp_Sp.r {np.mean(batch_rec['imp_spearman']):.2f}",
                    f"Ups_Sp.r {np.mean(batch_rec['ups_spearman']):.2f}",
                    f"Imp_MSE {np.mean(batch_rec['imp_mse']):.2f}",
                    f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                    f"took {int(minutes)}:{int(seconds)}"]

                logfile = open(f"models/EPD30{arch}_log.txt", "w")
                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                logfile.write("\n".join(log_strs))
                logfile.close()
                print(logstr)

                next_epoch = self.dataset.update_batch_pointers()
                
            if epoch%1==0:
                try:
                    torch.save(self.model.state_dict(), f'models/EPD30{arch}_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
                
        return self.model

#========================================================================================================#
#==========================================  Loader  ====================================================#
#========================================================================================================#

class MODEL_LOADER(object):
    def __init__(self, model_path, hyper_parameters):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_epidenoise(self, version= "16"):
        if version in ["10", "15", "16", "17", "18"]:
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
        
        elif version == "18":
            model = EpiDenoise18(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)

        elif version == "22":
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            n_enc_layers = self.hyper_parameters["n_enc_layers"]
            n_dec_layers = self.hyper_parameters["n_dec_layers"]
            context_length = self.hyper_parameters["context_length"]

            conv_out_channels = self.hyper_parameters["conv_out_channels"]
            d_model = conv_out_channels[-1]

            dilation = self.hyper_parameters["dilation"]
            kernel_size = self.hyper_parameters["kernel_size"]

            model = EpiDenoise22(
                input_dim, conv_out_channels, kernel_size, nhead, 
                d_model, n_enc_layers, n_dec_layers, output_dim, 
                dilation=dilation, dropout=dropout, context_length=context_length)
        
        elif version == "30a":
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
            context_length = self.hyper_parameters["context_length"]

            model = EpiDenoise30a(
                input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
                dropout=dropout, context_length=context_length, pos_enc="relative")
        
        elif version == "30b":          
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
            context_length = self.hyper_parameters["context_length"]

            n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
            conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
            n_decoder_layers = self.hyper_parameters["n_decoder_layers"]

            model = EpiDenoise30b(input_dim, metadata_embedding_dim, conv_kernel_size, 
                n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
                dropout=dropout, context_length=context_length, pos_enc="relative")

        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 
        model = model.to(self.device)
        return model

#========================================================================================================#
#=========================================  Trainer  ====================================================#
#========================================================================================================#

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=66, gamma=0.5)

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

    conv_out_channels = hyper_parameters["conv_out_channels"]
    dilation = hyper_parameters["dilation"]
    kernel_size = hyper_parameters["kernel_size"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise20(
        input_dim=input_dim, conv_out_channels=conv_out_channels, conv_kernel_sizes=kernel_size,
        dilation=dilation, nhead=nhead, d_model=d_model, n_encoder_layers=nlayers, 
        output_dim= output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.75)

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

def train_epidenoise21(hyper_parameters, checkpoint_path=None, start_ds=0):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    context_length = hyper_parameters["context_length"]

    conv_out_channels = hyper_parameters["conv_out_channels"]
    dilation = hyper_parameters["dilation"]
    kernel_size = hyper_parameters["kernel_size"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise21(
        input_dim, conv_out_channels, kernel_size, dilation, nhead, 
        d_model, nlayers, output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9326035)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise21_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters21_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss21()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_21(
        d_model=d_model, num_epochs=epochs,  
        context_length=context_length, start_ds=start_ds)

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

def train_epidenoise22(hyper_parameters, checkpoint_path=None, start_ds=0):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    n_enc_layers = hyper_parameters["n_enc_layers"]
    n_dec_layers = hyper_parameters["n_dec_layers"]
    epochs = hyper_parameters["epochs"]
    context_length = hyper_parameters["context_length"]

    conv_out_channels = hyper_parameters["conv_out_channels"]
    d_model = conv_out_channels[-1]

    dilation = hyper_parameters["dilation"]
    kernel_size = hyper_parameters["kernel_size"]

    learning_rate = hyper_parameters["learning_rate"]
    batch_size = hyper_parameters["batch_size"]
    mask_percentage = hyper_parameters["mask_percentage"]
    outer_loop_epochs = hyper_parameters["outer_loop_epochs"]
    # end of hyperparameters

    model = EpiDenoise22(
        input_dim, conv_out_channels, kernel_size, nhead, 
        d_model, n_enc_layers, n_dec_layers, output_dim, dilation=dilation, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=epochs*5, threshold=1e-3)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise22_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters22_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = ComboPoissonNLLloss()
    criterion = ComboLoss22(alpha=0.95)

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_22(
        d_model=d_model, num_epochs=epochs, mask_percentage=mask_percentage, outer_loop_epochs=outer_loop_epochs, 
        context_length=context_length, start_ds=start_ds, batch_size=batch_size)
        
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

def train_epidenoise30(hyper_parameters, checkpoint_path=None, arch="a"):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]

    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    metadata_embedding_dim = hyper_parameters["metadata_embedding_dim"]
    
    resolution = 25
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    lr_halflife = hyper_parameters["lr_halflife"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    # end of hyperparameters
    if arch == "a":
        model = EpiDenoise30a(input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
            dropout=dropout, context_length=context_length, pos_enc="relative")
            
    elif arch == "b":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        n_decoder_layers = hyper_parameters["n_decoder_layers"]

        model = EpiDenoise30b( input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
        dropout=dropout, context_length=context_length, pos_enc="relative")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_halflife, gamma=1)
    scheduler = None

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"EPD30{arch} # model_parameters: {count_parameters(model)}")

    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, ccre=True, 
        bios_min_exp_avail_threshold=min_avail, check_completeness=True)
    
    model_name = f"EpiDenoise30{arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters30{arch}_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss_NBNLL(alpha=1-mask_percentage)

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_30(
        num_epochs=epochs, mask_percentage=mask_percentage, 
        context_length=context_length, batch_size=batch_size, inner_epochs=inner_epochs, arch=arch)

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
#================================================Main====================================================#
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
            "input_dim": 47,
            "metadata_embedding_dim": 47,
            "dropout": 0.00,
            "nhead": 4,
            "d_model": 128,
            "nlayers": 2,
            "epochs": 2,
            "inner_epochs": 50,
            "mask_percentage": 0.5,
            "context_length": 100,
            "batch_size": 2,
            "learning_rate": 5e-4,
            "num_loci": 1200,
            "lr_halflife":1,
            "min_avail":15
        }
        train_epidenoise30(
            hyper_parameters30a, 
            checkpoint_path=None, arch="a")
    
    elif sys.argv[1] == "epd30b":
        hyper_parameters30b = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 47,
            "metadata_embedding_dim": 47,
            "dropout": 0.05,

            "n_cnn_layers": 5,
            "conv_kernel_size" : 7,
            "n_decoder_layers" : 1,

            "nhead": 4,
            "d_model": 384,
            "nlayers": 12,
            "epochs": 1,
            "inner_epochs": 50,
            "mask_percentage": 0.25,
            "context_length": 3200,
            "batch_size": 50,
            "learning_rate": 1e-3,
            "num_loci": 1200,
            "lr_halflife":2,
            "min_avail":4
        }
        train_epidenoise30(
            hyper_parameters30b, 
            checkpoint_path=None, arch="b")
