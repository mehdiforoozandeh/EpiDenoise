import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np

# -----------------------
# Transformer Model Code
# -----------------------

class SimplifiedPerFeatureTransformer(nn.Module):
    def __init__(self, num_features, E, nhead, nhid, nlayers, dropout=0.1, parallel_attention=False, second_mlp=False):
        super(SimplifiedPerFeatureTransformer, self).__init__()
        self.num_features = num_features  # Number of features
        self.E = E  # Embedding dimension
        self.nhead = nhead  # Number of attention heads
        self.nhid = nhid  # Hidden size of the MLP
        self.nlayers = nlayers  # Number of transformer layers
        self.dropout = dropout
        self.parallel_attention = parallel_attention  # Whether to use parallel attentions
        self.second_mlp = second_mlp  # Whether to include a second MLP after feature attention
        
        # Embedding layer with separate parameters for each feature
        self.embedding = nn.ModuleList([nn.Linear(1, E) for _ in range(num_features)])
        
        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            PerFeatureEncoderLayer(E, nhead, nhid, dropout, parallel_attention, second_mlp) for _ in range(nlayers)
        ])
        
        # Output layer to map back to (B, L, num_features)
        self.output_layer = nn.Linear(E, 1)
        
    def forward(self, x, feat_mask=None):
        B, L, num_features = x.shape
        # Embed each feature separately: (B, L, num_features) -> (B, L, num_features, E)
        x_embedded = torch.stack([self.embedding[f](x[..., f].unsqueeze(-1)) for f in range(num_features)], dim=2)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x_embedded = layer(x_embedded, feat_mask)
        
        # Map back to original feature space: (B, L, num_features, E) -> (B, L, num_features, 1)
        x_output = self.output_layer(x_embedded)
        # Remove the extra dimension: (B, L, num_features, 1) -> (B, L, num_features)
        x_output = x_output.squeeze(-1)
        return x_output

class PerFeatureEncoderLayer(nn.Module):
    def __init__(self, E, nhead, nhid, dropout, parallel_attention, second_mlp):
        super(PerFeatureEncoderLayer, self).__init__()
        self.E = E
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.parallel_attention = parallel_attention
        self.second_mlp = second_mlp
        
        # Add dropout to MultiheadAttention for stability
        self.feature_attention = nn.MultiheadAttention(embed_dim=E, num_heads=nhead, dropout=dropout)
        self.position_attention = nn.MultiheadAttention(embed_dim=E, num_heads=nhead, dropout=dropout)
        
        if parallel_attention:
            # MLP for combined attentions (concatenated outputs)
            self.mlp = nn.Sequential(
                nn.Linear(2 * E, nhid),
                nn.GELU(),
                nn.Linear(nhid, E)
            )
        else:
            # MLP for sequential attentions
            self.mlp = nn.Sequential(
                nn.Linear(E, nhid),
                nn.GELU(),
                nn.Linear(nhid, E)
            )
        
        if second_mlp:
            # Second MLP after feature attention (only used in sequential mode)
            self.second_mlp_layer = nn.Sequential(
                nn.Linear(E, nhid),
                nn.GELU(),
                nn.Linear(nhid, E)
            )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(E)
        self.norm2 = nn.LayerNorm(E)
        self.norm3 = nn.LayerNorm(E)
        if second_mlp:
            self.norm4 = nn.LayerNorm(E)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, feat_mask=None):
        B, L, num_features, E = x.shape
        
        if self.parallel_attention:
            # --- Parallel attentions ---
            # Feature attention (masking applied)
            x_feat = x.view(B * L, num_features, E).permute(1, 0, 2)  # (num_features, B*L, E)
            if feat_mask is not None:
                feat_mask_expanded = feat_mask.view(B, 1, num_features).expand(B, L, num_features).reshape(B * L, num_features)
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat, key_padding_mask=feat_mask_expanded)
            else:
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat)
            feat_attn_output = feat_attn_output.permute(1, 0, 2).view(B, L, num_features, E)
            
            # Position attention
            x_pos = x.permute(0, 2, 1, 3).reshape(B * num_features, L, E).permute(1, 0, 2)  # (L, B*num_features, E)
            pos_attn_output, _ = self.position_attention(x_pos, x_pos, x_pos)
            pos_attn_output = pos_attn_output.permute(1, 0, 2).view(B, num_features, L, E).permute(0, 2, 1, 3)
            
            # Concatenate outputs along the embedding dimension: (B, L, num_features, 2*E)
            combined = torch.cat([feat_attn_output, pos_attn_output], dim=-1)
            # MLP to mix the combined features
            mlp_output = self.mlp(combined)
            x = self.norm1(x + self.dropout_layer(mlp_output))
        else:
            # --- Sequential attentions ---
            # Feature attention (with masking)
            x_feat = x.view(B * L, num_features, E).permute(1, 0, 2)
            if feat_mask is not None:
                feat_mask_expanded = feat_mask.view(B, 1, num_features).expand(B, L, num_features).reshape(B * L, num_features)
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat, key_padding_mask=feat_mask_expanded)
            else:
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat)
            feat_attn_output = feat_attn_output.permute(1, 0, 2).view(B, L, num_features, E)
            x = self.norm1(x + self.dropout_layer(feat_attn_output))
            
            if self.second_mlp:
                # Second MLP after feature attention
                second_mlp_output = self.second_mlp_layer(x)
                x = self.norm4(x + self.dropout_layer(second_mlp_output))
            
            # Position attention (across L dimension)
            x_pos = x.permute(0, 2, 1, 3).reshape(B * num_features, L, E).permute(1, 0, 2)
            pos_attn_output, _ = self.position_attention(x_pos, x_pos, x_pos)
            pos_attn_output = pos_attn_output.permute(1, 0, 2).view(B, num_features, L, E).permute(0, 2, 1, 3)
            x = self.norm2(x + self.dropout_layer(pos_attn_output))
            
            # Final MLP
            mlp_output = self.mlp(x)
            x = self.norm3(x + self.dropout_layer(mlp_output))
        
        return x

# --------------------------
# Synthetic Data Generation
# --------------------------
def generate_synthetic_data(B, L, num_features, device):
    """
    Generate synthetic data of shape (B, L, num_features) where each sample is based on an underlying
    sine function (with a random offset per sample) and each feature is a different transformation
    (using a different amplitude and phase shift) of that sequence.
    """
    t = torch.linspace(0, 2 * math.pi, L, device=device)
    offsets = torch.rand(B, device=device) * 2 * math.pi
    base = torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1))
    
    x = torch.zeros(B, L, num_features, device=device)
    for f in range(num_features):
        amplitude = 1.0 + 0.2 * f
        phase_shift = f * 0.5
        x[:, :, f] = amplitude * torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1) + phase_shift)
    return x

# --------------------------
# Training and Evaluation
# --------------------------
def train_model(model, optimizer, num_epochs, B, L, num_features, device, mask_prob=0.3):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x = generate_synthetic_data(B, L, num_features, device)
        feat_mask = (torch.rand(B, num_features, device=device) < mask_prob)
        
        # Create input where masked features are replaced with zero
        x_input = x.clone()
        mask_expanded = feat_mask.unsqueeze(1).expand(-1, L, -1)
        x_input[mask_expanded] = 0.0
        
        output = model(x_input, feat_mask)
        
        loss = F.mse_loss(output[mask_expanded], x[mask_expanded])
        loss.backward()
        
        # Check if loss is NaN
        if torch.isnan(loss).item():
            print(f"Epoch {epoch + 1}: Loss is NaN, skipping parameter update.")
            optimizer.zero_grad()  # clear gradients
        else:
            # Check if any gradient contains NaN
            grad_has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"Epoch {epoch + 1}: Gradient for {name} contains NaN, skipping update.")
                    grad_has_nan = True
                    break
            if grad_has_nan:
                optimizer.zero_grad()  # clear gradients
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        losses.append(loss.item())
        
        # Compute R² score on masked features safely
        pred_masked = output[mask_expanded].detach().cpu().numpy()
        target_masked = x[mask_expanded].detach().cpu().numpy()
        if np.isnan(pred_masked).any() or np.isnan(target_masked).any():
            r2 = float('nan')
        else:
            r2 = r2_score(target_masked, pred_masked)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, R² (masked): {r2:.4f}")
    return losses

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    B = 500                  # Batch size
    L = 300                  # Sequence length
    num_features = 20        # Number of features (assays)
    E = 20                  # Embedding dimension
    nhead = 4               # Number of attention heads
    nhid = 64               # Hidden size of the MLP
    nlayers = 2             # Number of transformer layers
    dropout = 0.1
    num_epochs = 2000
    mask_prob = 0.3         # Probability to mask each feature per sample
    
    # Set parallel_attention=False for sequential attention (more stable)
    model = SimplifiedPerFeatureTransformer(num_features, E, nhead, nhid, nlayers, dropout=dropout,
                                            parallel_attention=False, second_mlp=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    losses = train_model(model, optimizer, num_epochs, B, L, num_features, device, mask_prob=mask_prob)
    
    print("\nTraining complete.")
    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")

if __name__ == '__main__':
    main()
