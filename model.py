import torch, math, random, time, json, os, pickle, sys, gc
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from datetime import datetime
from scipy.stats import nbinom
import imageio.v2 as imageio
from io import BytesIO
from torchinfo import summary

from _utils import *    
from data import * 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
    def __init__(self, input_dim, embedding_dim, non_linearity=True):
        """
        Args:
            input_dim (int): Number of metadata features.
            embedding_dim (int): Final embedding dimension.
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
        self.coverage_transforms = nn.ModuleList(
            [nn.Linear(1, self.continuous_size) for _ in range(input_dim)]
        )
        self.read_length_transforms = nn.ModuleList(
            [nn.Linear(1, self.continuous_size) for _ in range(input_dim)]
        )
        # For runtype, create separate embedding layers per feature.
        # Assuming 4 classes for runtype.
        self.runtype_embeddings = nn.ModuleList(
            [nn.Embedding(4, self.continuous_size) for _ in range(input_dim)]
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
                      [depth, coverage, read_length, runtype]
        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        B = metadata.size(0)
        # Lists to collect per-feature embeddings.
        per_feature_embeds = []
        for i in range(self.input_dim):
            # Extract each metadata type for feature i.
            depth = metadata[:, 0, i].unsqueeze(-1).float() 
            coverage = metadata[:, 1, i].unsqueeze(-1).float() 
            read_length = metadata[:, 2, i].unsqueeze(-1).float() 
            runtype = metadata[:, 3, i].long() 
            
            # For runtype, map -1 -> 2 (missing) and -2 -> 3 (cloze_masked)
            runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype)
            runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype)
            
            # Apply the separate transforms/embeddings for feature i.
            depth_embed = self.depth_transforms[i](depth)              # (B, continuous_size)
            coverage_embed = self.coverage_transforms[i](coverage)        # (B, continuous_size)
            read_length_embed = self.read_length_transforms[i](read_length)  # (B, continuous_size)
            runtype_embed = self.runtype_embeddings[i](runtype)           # (B, continuous_size)
            
            # Concatenate the four embeddings along the last dimension.
            feature_embed = torch.cat([depth_embed, coverage_embed, read_length_embed, runtype_embed], dim=-1)  # (B, 4*continuous_size)
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