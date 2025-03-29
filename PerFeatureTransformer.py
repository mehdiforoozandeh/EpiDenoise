import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np

# -----------------------
# Special Transformer Code (Per-Feature Attention)
# -----------------------

class SimplifiedPerFeatureTransformer(nn.Module):
    def __init__(self, num_features, E, nhead, nhid, nlayers, dropout=0.1, parallel_attention=False, second_mlp=False):
        super(SimplifiedPerFeatureTransformer, self).__init__()
        self.num_features = num_features  # Number of features
        self.E = E                      # Embedding dimension
        self.nhead = nhead              # Number of attention heads
        self.nhid = nhid                # Hidden size of the MLP
        self.nlayers = nlayers          # Number of transformer layers
        self.dropout = dropout
        self.parallel_attention = parallel_attention  # Whether to use parallel attentions
        self.second_mlp = second_mlp                  # Whether to include a second MLP after feature attention

        # Each feature is embedded separately (F independent linear layers)
        self.embedding = nn.ModuleList([nn.Linear(1, E) for _ in range(num_features)])

        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            PerFeatureEncoderLayer(E, nhead, nhid, dropout, parallel_attention, second_mlp)
            for _ in range(nlayers)
        ])

        # Output layer to map back to original feature dimension
        self.output_layer = nn.Linear(E, 1)

    def forward(self, x, feat_mask=None):
        # x: (B, L, num_features)
        B, L, num_features = x.shape
        # Embed each feature separately -> (B, L, num_features, E)
        x_embedded = torch.stack([self.embedding[f](x[..., f].unsqueeze(-1))
                                  for f in range(num_features)], dim=2)
        # Process with transformer layers
        for layer in self.transformer_layers:
            x_embedded = layer(x_embedded, feat_mask)
        # Map back to (B, L, num_features, 1) and squeeze to (B, L, num_features)
        x_output = self.output_layer(x_embedded).squeeze(-1)
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
            # Process the concatenated outputs of feature and position attentions
            self.mlp = nn.Sequential(
                nn.Linear(2 * E, nhid),
                nn.GELU(),
                nn.Linear(nhid, E)
            )
        else:
            # Sequential processing
            self.mlp = nn.Sequential(
                nn.Linear(E, nhid),
                nn.GELU(),
                nn.Linear(nhid, E)
            )

        if second_mlp:
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

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, feat_mask=None):
        # x: (B, L, num_features, E)
        B, L, num_features, E = x.shape
        if self.parallel_attention:
            # --- Parallel attentions ---
            x_feat = x.view(B * L, num_features, E).permute(1, 0, 2)  # (num_features, B*L, E)
            if feat_mask is not None:
                feat_mask_expanded = feat_mask.view(B, 1, num_features).expand(B, L, num_features).reshape(B * L, num_features)
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat, key_padding_mask=feat_mask_expanded)
            else:
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat)
            feat_attn_output = feat_attn_output.permute(1, 0, 2).view(B, L, num_features, E)

            x_pos = x.permute(0, 2, 1, 3).reshape(B * num_features, L, E).permute(1, 0, 2)  # (L, B*num_features, E)
            pos_attn_output, _ = self.position_attention(x_pos, x_pos, x_pos)
            pos_attn_output = pos_attn_output.permute(1, 0, 2).view(B, num_features, L, E).permute(0, 2, 1, 3)

            combined = torch.cat([feat_attn_output, pos_attn_output], dim=-1)  # (B, L, num_features, 2*E)
            mlp_output = self.mlp(combined)
            x = self.norm1(x + self.dropout_layer(mlp_output))
        else:
            # --- Sequential attentions ---
            x_feat = x.view(B * L, num_features, E).permute(1, 0, 2)
            if feat_mask is not None:
                feat_mask_expanded = feat_mask.view(B, 1, num_features).expand(B, L, num_features).reshape(B * L, num_features)
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat, key_padding_mask=feat_mask_expanded)
            else:
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat)
            feat_attn_output = feat_attn_output.permute(1, 0, 2).view(B, L, num_features, E)
            x = self.norm1(x + self.dropout_layer(feat_attn_output))

            if self.second_mlp:
                second_mlp_output = self.second_mlp_layer(x)
                x = self.norm4(x + self.dropout_layer(second_mlp_output))

            x_pos = x.permute(0, 2, 1, 3).reshape(B * num_features, L, E).permute(1, 0, 2)
            pos_attn_output, _ = self.position_attention(x_pos, x_pos, x_pos)
            pos_attn_output = pos_attn_output.permute(1, 0, 2).view(B, num_features, L, E).permute(0, 2, 1, 3)
            x = self.norm2(x + self.dropout_layer(pos_attn_output))

            mlp_output = self.mlp(x)
            x = self.norm3(x + self.dropout_layer(mlp_output))
        return x

# -----------------------
# Standard Transformer Code
# -----------------------

class StandardTransformer(nn.Module):
    def __init__(self, num_features, E, nlayers, dropout=0.1, nhead=4):
        """
        A standard transformer that processes the entire feature vector at each time step.
        Input shape: (B, L, num_features)
        """
        super(StandardTransformer, self).__init__()
        self.input_linear = nn.Linear(num_features, E)  # Embed F-dimensional input into E-dimensional space
        encoder_layer = nn.TransformerEncoderLayer(d_model=E, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.output_linear = nn.Linear(E, num_features)  # Map back to F

    def forward(self, x):
        # x: (B, L, num_features)
        x_emb = self.input_linear(x)        # (B, L, E)
        x_emb = x_emb.transpose(0, 1)         # (L, B, E) for Transformer
        encoded = self.transformer_encoder(x_emb)  # (L, B, E)
        encoded = encoded.transpose(0, 1)     # (B, L, E)
        out = self.output_linear(encoded)     # (B, L, num_features)
        return out

# -----------------------
# Synthetic Data Generation (Generalized)
# -----------------------

def generate_synthetic_dataset(N, L, num_features, device):
    """
    Generate a synthetic dataset of shape (N, L, num_features) with complex sequence functions.
    
    The features are divided into three groups:
      - Group 1 (~15% of features, at least 1): Related low-noise sine waves.
      - Group 2 (~35% of features, at least 1): A shared combination of sine and cosine functions
        with moderate noise and larger numerical ranges.
      - Group 3 (remaining features): Independent functions per feature with varied frequency, amplitude,
        offset, and noise levels.
    
    Group sizes are computed as:
         g1 = max(1, round(0.15 * num_features))
         g2 = max(1, round(0.35 * num_features))
         Adjust g2 if g1 + g2 > num_features, and let g3 = num_features - g1 - g2.
         
    Returns:
         x: Tensor of shape (N, L, num_features)
    """
    # Compute group sizes
    g1 = max(1, round(0.15 * num_features))
    g2 = max(1, round(0.35 * num_features))
    if g1 + g2 > num_features:
        g2 = num_features - g1
    g3 = num_features - g1 - g2

    t = torch.linspace(0, 2 * math.pi, L, device=device)
    x = torch.zeros(N, L, num_features, device=device)

    # Group 1: Related low-noise sine waves
    for i in range(N):
        offset = torch.rand(1).item() * 2 * math.pi
        base = torch.sin(t + offset)
        for f in range(g1):
            amp = 1.0 + 0.1 * (f - (g1 - 1) / 2)  # Center amplitudes around 1.0
            noise = 0.1 * torch.randn(L, device=device)
            x[i, :, f] = amp * base + noise

    # Group 2: Shared combination of sine and cosine with moderate noise
    for i in range(N):
        offset1 = torch.rand(1).item() * 2 * math.pi
        offset2 = torch.rand(1).item() * 2 * math.pi
        base2 = torch.sin(t + offset1) + 0.5 * torch.cos(2 * t + offset2)
        for idx, f in enumerate(range(g1, g1 + g2)):
            amp = 2.0 + 0.5 * idx  # Increasing amplitude
            noise = 0.5 * torch.randn(L, device=device)
            x[i, :, f] = amp * base2 + noise

    # Group 3: Independent functions with varied properties
    for i in range(N):
        for f in range(g1 + g2, num_features):
            freq = 1 + 2 * torch.rand(1).item()           # frequency between 1 and 3
            phase = torch.rand(1).item() * 2 * math.pi
            amp = 0.5 + 5.0 * torch.rand(1).item()          # amplitude between 0.5 and 5.5
            offset_feature = -5 + 10 * torch.rand(1).item()   # offset between -5 and 5
            noise_std = 0.2 + 1.0 * torch.rand(1).item()      # noise std between 0.2 and 1.2
            base_feature = amp * torch.sin(freq * t + phase) + offset_feature
            noise = noise_std * torch.randn(L, device=device)
            x[i, :, f] = base_feature + noise

    return x

def get_batches(data, batch_size):
    """
    Yield mini-batches (along the first dimension) from the dataset.
    """
    N = data.size(0)
    indices = torch.randperm(N)
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield data[batch_indices]

# -----------------------
# Whole-Feature Missing Mask Generator
# -----------------------

def generate_whole_feature_mask(B, num_features, missing_prob):
    """
    Generate a mask of shape (B, num_features) where each feature is entirely missing (i.e. across all time steps)
    with probability missing_prob.
    """
    return torch.rand(B, num_features) < missing_prob

# -----------------------
# Training and Evaluation for Both Models
# -----------------------

def train_and_evaluate_both_models(model_special, model_standard, optimizer_special, optimizer_standard,
                                   train_data, test_data, num_epochs, batch_size, mask_prob, device):
    # To record epoch-level test metrics (using random masking as before)
    test_losses_special, test_losses_standard = [], []
    test_r2s_special, test_r2s_standard = [], []

    for epoch in range(num_epochs):
        model_special.train()
        model_standard.train()
        # --- Training ---
        for x in get_batches(train_data, batch_size):
            B_curr = x.size(0)
            # Generate random feature mask (column-wise missing is not used during training)
            feat_mask = (torch.rand(B_curr, model_special.num_features, device=device) < mask_prob)
            x_input = x.clone()
            mask_expanded = feat_mask.unsqueeze(1).expand(-1, x.size(1), -1)
            x_input[mask_expanded] = 0.0

            # --- Special Transformer Update ---
            optimizer_special.zero_grad()
            output_special = model_special(x_input, feat_mask)
            loss_special = F.mse_loss(output_special[mask_expanded], x[mask_expanded])
            loss_special.backward()
            update_special = True
            if torch.isnan(loss_special).item():
                print(f"Epoch {epoch + 1}: Special Transformer loss is NaN, skipping update.")
                update_special = False
                optimizer_special.zero_grad()
            else:
                for name, param in model_special.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Epoch {epoch + 1}: Special Transformer gradient for {name} is NaN, skipping update.")
                        update_special = False
                        optimizer_special.zero_grad()
                        break
            if update_special:
                torch.nn.utils.clip_grad_norm_(model_special.parameters(), max_norm=1.0)
                optimizer_special.step()

            # --- Standard Transformer Update ---
            optimizer_standard.zero_grad()
            output_standard = model_standard(x_input)
            loss_standard = F.mse_loss(output_standard[mask_expanded], x[mask_expanded])
            loss_standard.backward()
            update_standard = True
            if torch.isnan(loss_standard).item():
                print(f"Epoch {epoch + 1}: Standard Transformer loss is NaN, skipping update.")
                update_standard = False
                optimizer_standard.zero_grad()
            else:
                for name, param in model_standard.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Epoch {epoch + 1}: Standard Transformer gradient for {name} is NaN, skipping update.")
                        update_standard = False
                        optimizer_standard.zero_grad()
                        break
            if update_standard:
                torch.nn.utils.clip_grad_norm_(model_standard.parameters(), max_norm=1.0)
                optimizer_standard.step()

        # --- Evaluation on Test Data with Random Masking (as before) ---
        model_special.eval()
        model_standard.eval()
        test_preds_special = []
        test_preds_standard = []
        test_targets = []
        test_loss_special_epoch = 0.0
        test_loss_standard_epoch = 0.0
        count = 0
        with torch.no_grad():
            for x in get_batches(test_data, batch_size):
                B_curr = x.size(0)
                feat_mask = (torch.rand(B_curr, model_special.num_features, device=device) < mask_prob)
                x_input = x.clone()
                mask_expanded = feat_mask.unsqueeze(1).expand(-1, x.size(1), -1)
                x_input[mask_expanded] = 0.0

                output_special = model_special(x_input, feat_mask)
                loss_special = F.mse_loss(output_special[mask_expanded], x[mask_expanded])
                test_loss_special_epoch += loss_special.item() * B_curr

                output_standard = model_standard(x_input)
                loss_standard = F.mse_loss(output_standard[mask_expanded], x[mask_expanded])
                test_loss_standard_epoch += loss_standard.item() * B_curr

                test_preds_special.append(output_special[mask_expanded].detach().cpu().numpy())
                test_preds_standard.append(output_standard[mask_expanded].detach().cpu().numpy())
                test_targets.append(x[mask_expanded].detach().cpu().numpy())
                count += B_curr

        avg_test_loss_special = test_loss_special_epoch / count
        avg_test_loss_standard = test_loss_standard_epoch / count
        test_preds_special = np.concatenate(test_preds_special, axis=0)
        test_preds_standard = np.concatenate(test_preds_standard, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        r2_special = r2_score(test_targets, test_preds_special)
        r2_standard = r2_score(test_targets, test_preds_standard)

        test_losses_special.append(avg_test_loss_special)
        test_losses_standard.append(avg_test_loss_standard)
        test_r2s_special.append(r2_special)
        test_r2s_standard.append(r2_standard)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} -- Random Mask Test Metrics:")
            print(f"  Special Transformer: Loss: {avg_test_loss_special:.4f}, R²: {r2_special:.4f}")
            print(f"  Standard Transformer: Loss: {avg_test_loss_standard:.4f}, R²: {r2_standard:.4f}")

    return test_losses_special, test_losses_standard, test_r2s_special, test_r2s_standard

def evaluate_whole_feature_missing(model_special, model_standard, test_data, batch_size, missing_prob, device):
    """
    Evaluate both models on the test set using a whole-feature mask.
    For each sample, generate a mask of shape (B, num_features) such that entire features are missing
    with probability missing_prob, then compute the MSE and R² on these features.
    """
    model_special.eval()
    model_standard.eval()
    all_preds_special, all_preds_standard, all_targets = [], [], []
    total_loss_special = 0.0
    total_loss_standard = 0.0
    count = 0
    with torch.no_grad():
        for x in get_batches(test_data, batch_size):
            B_curr = x.size(0)
            feat_mask = (torch.rand(B_curr, model_special.num_features, device=device) < missing_prob)
            # Expand mask to all time steps (B, L, num_features)
            mask_expanded = feat_mask.unsqueeze(1).expand(-1, x.size(1), -1)
            x_input = x.clone()
            x_input[mask_expanded] = 0.0

            output_special = model_special(x_input, feat_mask)
            output_standard = model_standard(x_input)

            loss_special = F.mse_loss(output_special[mask_expanded], x[mask_expanded])
            loss_standard = F.mse_loss(output_standard[mask_expanded], x[mask_expanded])
            total_loss_special += loss_special.item() * B_curr
            total_loss_standard += loss_standard.item() * B_curr

            all_preds_special.append(output_special[mask_expanded].detach().cpu().numpy())
            all_preds_standard.append(output_standard[mask_expanded].detach().cpu().numpy())
            all_targets.append(x[mask_expanded].detach().cpu().numpy())
            count += B_curr

    avg_loss_special = total_loss_special / count
    avg_loss_standard = total_loss_standard / count
    all_preds_special = np.concatenate(all_preds_special, axis=0)
    all_preds_standard = np.concatenate(all_preds_standard, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    r2_special = r2_score(all_targets, all_preds_special)
    r2_standard = r2_score(all_targets, all_preds_standard)

    print("\nWhole-Feature Missing Evaluation:")
    print(f"  Special Transformer: Loss: {avg_loss_special:.4f}, R²: {r2_special:.4f}")
    print(f"  Standard Transformer: Loss: {avg_loss_standard:.4f}, R²: {r2_standard:.4f}")

    return avg_loss_special, r2_special, avg_loss_standard, r2_standard

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Hyperparameters
    N = 500                # Total number of samples in the dataset
    train_ratio = 0.8       # 80% training, 20% test
    L = 100                 # Sequence length
    num_features = 20       # Number of features
    E = 20                  # Embedding dimension
    nhead = 4               # Number of attention heads (for both models)
    nhid = 64               # Hidden size for special transformer MLP
    nlayers = 2             # Number of transformer layers
    dropout = 0.1
    num_epochs = 2000
    batch_size = 50         # Mini-batch size
    mask_prob = 0.5         # Probability to randomly mask each feature (per sample) during training/evaluation
    whole_missing_prob = 0.5  # For whole-feature missing evaluation

    # Generate synthetic dataset
    dataset = generate_synthetic_dataset(N, L, num_features, device)
    indices = torch.randperm(N)
    train_size = int(train_ratio * N)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_data = dataset[train_indices]
    test_data = dataset[test_indices]

    # Instantiate both models
    model_special = SimplifiedPerFeatureTransformer(num_features, E, nhead, nhid, nlayers, dropout=dropout,
                                                     parallel_attention=False, second_mlp=True).to(device)
    model_standard = StandardTransformer(num_features, E, nlayers, dropout=dropout, nhead=nhead).to(device)

    optimizer_special = optim.Adam(model_special.parameters(), lr=1e-4)
    optimizer_standard = optim.Adam(model_standard.parameters(), lr=1e-4)

    # Train both models and evaluate on test split (with random masking) at each epoch
    test_losses_special, test_losses_standard, test_r2s_special, test_r2s_standard = train_and_evaluate_both_models(
        model_special, model_standard, optimizer_special, optimizer_standard,
        train_data, test_data, num_epochs, batch_size, mask_prob, device
    )

    print("\nTraining complete on the training split.")
    print("\nFinal Test Metrics (Random Masking):")
    print(f"Special Transformer - Test Loss: {test_losses_special[-1]:.4f}, Test R²: {test_r2s_special[-1]:.4f}")
    print(f"Standard Transformer - Test Loss: {test_losses_standard[-1]:.4f}, Test R²: {test_r2s_standard[-1]:.4f}")

    if test_losses_special[-1] < test_losses_standard[-1] and test_r2s_special[-1] >= test_r2s_standard[-1]:
        better = "Special Transformer"
    else:
        better = "Standard Transformer"
    print(f"\nOverall (Random Masking), {better} performed better on the test split.")

    # Now, perform a dedicated evaluation on test data with whole-feature missingness
    evaluate_whole_feature_missing(model_special, model_standard, test_data, batch_size, whole_missing_prob, device)

if __name__ == '__main__':
    main()
