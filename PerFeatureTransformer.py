import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np

# -----------------------
# OGPerFeatureTransformer (Original) – updated usage based on new code
# -----------------------
from tabpfn.model.transformer import PerFeatureTransformer
from tabpfn.model.encoders import SequentialEncoder, LinearInputEncoderStep

class OGPerFeatureTransformer(nn.Module):
    def __init__(self, num_features, E, nlayers, dropout=0.1, nhead=4):
        super().__init__()
        self.transformer = PerFeatureTransformer(
            encoder=SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=num_features,
                    emsize=E,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
            ),
            ninp=E,
            nhead=nhead,
            nhid=4 * E,  # Feed-forward dimension
            nlayers=nlayers,
            decoder_dict={"standard": (None, num_features)},
            # Removed 'layer_kwargs' here.
        )
    
    def forward(self, x):
        # x: (B, L, num_features)
        x_transposed = x.transpose(0, 1)  # (L, B, num_features)
        output = self.transformer(x_transposed, None, single_eval_pos=None)
        return output.transpose(0, 1)  # (B, L, num_features)

# -----------------------
# Simplified PerFeatureTransformer Code (Our custom implementation)
# -----------------------

class SimplifiedPerFeatureTransformer(nn.Module):
    def __init__(self, num_features, E, nhead, nhid, nlayers, dropout=0.1, parallel_attention=False, second_mlp=False):
        super(SimplifiedPerFeatureTransformer, self).__init__()
        self.num_features = num_features
        self.E = E
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.parallel_attention = parallel_attention
        self.second_mlp = second_mlp

        # Each feature is embedded separately
        self.embedding = nn.ModuleList([nn.Linear(1, E) for _ in range(num_features)])
        self.transformer_layers = nn.ModuleList([
            PerFeatureEncoderLayer(E, nhead, nhid, dropout, parallel_attention, second_mlp)
            for _ in range(nlayers)
        ])
        self.output_layer = nn.Linear(E, 1)

    def forward(self, x, feat_mask=None):
        # x: (B, L, num_features)
        B, L, num_features = x.shape
        x_embedded = torch.stack([self.embedding[f](x[..., f].unsqueeze(-1))
                                  for f in range(num_features)], dim=2)
        for layer in self.transformer_layers:
            x_embedded = layer(x_embedded, feat_mask)
        return self.output_layer(x_embedded).squeeze(-1)  # (B, L, num_features)

class PerFeatureEncoderLayer(nn.Module):
    def __init__(self, E, nhead, nhid, dropout, parallel_attention, second_mlp):
        super(PerFeatureEncoderLayer, self).__init__()
        self.E = E
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.parallel_attention = parallel_attention
        self.second_mlp = second_mlp

        self.feature_attention = nn.MultiheadAttention(embed_dim=E, num_heads=nhead, dropout=dropout)
        self.position_attention = nn.MultiheadAttention(embed_dim=E, num_heads=nhead, dropout=dropout)

        if parallel_attention:
            self.mlp = nn.Sequential(
                nn.Linear(2 * E, nhid),
                nn.GELU(),
                nn.Linear(nhid, E)
            )
        else:
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
            x_feat = x.view(B * L, num_features, E).permute(1, 0, 2)
            if feat_mask is not None:
                feat_mask_expanded = feat_mask.view(B, 1, num_features).expand(B, L, num_features).reshape(B * L, num_features)
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat, key_padding_mask=feat_mask_expanded)
            else:
                feat_attn_output, _ = self.feature_attention(x_feat, x_feat, x_feat)
            feat_attn_output = feat_attn_output.permute(1, 0, 2).view(B, L, num_features, E)

            x_pos = x.permute(0, 2, 1, 3).reshape(B * num_features, L, E).permute(1, 0, 2)
            pos_attn_output, _ = self.position_attention(x_pos, x_pos, x_pos)
            pos_attn_output = pos_attn_output.permute(1, 0, 2).view(B, num_features, L, E).permute(0, 2, 1, 3)

            combined = torch.cat([feat_attn_output, pos_attn_output], dim=-1)
            mlp_output = self.mlp(combined)
            x = self.norm1(x + self.dropout_layer(mlp_output))
        else:
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
        super(StandardTransformer, self).__init__()
        self.input_linear = nn.Linear(num_features, E)
        encoder_layer = nn.TransformerEncoderLayer(d_model=E, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.output_linear = nn.Linear(E, num_features)

    def forward(self, x):
        # x: (B, L, num_features)
        x_emb = self.input_linear(x)
        x_emb = x_emb.transpose(0, 1)
        encoded = self.transformer_encoder(x_emb)
        encoded = encoded.transpose(0, 1)
        return self.output_linear(encoded)

# -----------------------
# Linear Baseline Model
# -----------------------

class LinearBaseline(nn.Module):
    def __init__(self, num_features):
        super(LinearBaseline, self).__init__()
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, x):
        return self.linear(x)

# -----------------------
# MLP Baseline Model
# -----------------------

class MLPBaseline(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(MLPBaseline, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x):
        B, L, F = x.shape
        x_flat = x.view(B * L, F)
        out = self.mlp(x_flat)
        return out.view(B, L, F)

# -----------------------
# 1D CNN Baseline Model
# -----------------------

class CNNBaseline(nn.Module):
    def __init__(self, num_features, hidden_channels=32, kernel_size=3):
        super(CNNBaseline, self).__init__()
        hidden_channels = num_features * 2
        self.conv1 = nn.Conv1d(num_features, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(hidden_channels, num_features, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, L, num_features) -> (B, num_features, L)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x.transpose(1, 2)

# -----------------------
# Synthetic Data Generation (Complicated Transformations)
# -----------------------

def generate_synthetic_dataset(N, L, num_features, device):
    """
    Generate a synthetic dataset of shape (N, L, num_features) by first generating a base sequence S
    (of length L, positive) and then applying F different numerical transformations (with optional noise)
    to S. For each feature, we randomly choose one of the following transformations:
        - log2(S)
        - log10(S)
        - exp(S)
        - arcsinh(S)
        - sinh(S)
        - linear: w * S (with a random weight w)
        - taylor: 1 + S + S^2/2 + S^3/6 (Taylor expansion of exp(S))
        - sqrt(S)
        - square: S^2 (here modified to S^3.2 for more nonlinearity)
    Gaussian noise (with a randomly chosen standard deviation) is then added.
    """
    transformations = ["log2", "log10", "exp", "arcsinh", "sinh", "linear", "taylor", "sqrt", "square"]
    feat_transforms = []
    feat_params = []
    noise_levels = []
    for f in range(num_features):
        choice = random.choice(transformations)
        feat_transforms.append(choice)
        if choice == "linear":
            w = random.uniform(0.1, 10.0)
            feat_params.append({"w": w})
        else:
            feat_params.append({})
        noise = random.uniform(0.01, 0.2)
        noise_levels.append(noise)

    x = torch.zeros(N, L, num_features, device=device)
    t = torch.linspace(0, 2 * math.pi, L, device=device)
    for i in range(N):
        offset = random.uniform(0, 2 * math.pi)
        S = torch.sin(t + offset) + 1.1  # S in [0.1, 2.1]
        for f in range(num_features):
            choice = feat_transforms[f]
            params = feat_params[f]
            if choice == "log2":
                Y = torch.log2(S)
            elif choice == "log10":
                Y = torch.log10(S)
            elif choice == "exp":
                Y = torch.exp(S)
            elif choice == "arcsinh":
                Y = torch.asinh(S)
            elif choice == "sinh":
                Y = torch.sinh(S)
            elif choice == "linear":
                w = params.get("w", 1.0)
                Y = w * S
            elif choice == "taylor":
                Y = 1 + S + (S**2)/2 + (S**3)/6
            elif choice == "sqrt":
                Y = torch.sqrt(S)
            elif choice == "square":
                Y = S**3.2
            else:
                Y = S
            noise = noise_levels[f] * torch.randn(L, device=device)
            x[i, :, f] = Y + noise
    return x

def get_batches(data, batch_size):
    N = data.size(0)
    indices = torch.randperm(N)
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield data[batch_indices]

# -----------------------
# Whole-Feature Missing Mask Generator
# -----------------------

def generate_whole_feature_mask(B, num_features, missing_prob):
    return torch.rand(B, num_features) < missing_prob

# -----------------------
# Training and Evaluation for All Models
# -----------------------

def train_and_evaluate_models(models, optimizers, train_data, test_data, num_epochs, batch_size, mask_prob, device):
    test_losses = {name: [] for name in models}
    test_r2s = {name: [] for name in models}
    num_feats = list(models.values())[0].num_features if hasattr(list(models.values())[0], 'num_features') else train_data.shape[-1]

    for epoch in range(num_epochs):
        for x in get_batches(train_data, batch_size):
            B_curr = x.size(0)
            feat_mask = (torch.rand(B_curr, num_feats, device=device) < mask_prob)
            mask_expanded = feat_mask.unsqueeze(1).expand(-1, x.size(1), -1)
            x_input = x.clone()
            x_input[mask_expanded] = 0.0
            for name, model in models.items():
                optimizers[name].zero_grad()
                if name == "PerFeature":
                    output = model(x_input, feat_mask)
                else:
                    output = model(x_input)
                loss = F.mse_loss(output[mask_expanded], x[mask_expanded])
                loss.backward()
                valid_update = True
                if torch.isnan(loss).item():
                    valid_update = False
                    optimizers[name].zero_grad()
                else:
                    for param in model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            valid_update = False
                            optimizers[name].zero_grad()
                            break
                if valid_update:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizers[name].step()
        all_losses = {name: 0.0 for name in models}
        all_counts = 0
        preds = {name: [] for name in models}
        targets = []
        with torch.no_grad():
            for x in get_batches(test_data, batch_size):
                B_curr = x.size(0)
                feat_mask = (torch.rand(B_curr, num_feats, device=device) < mask_prob)
                mask_expanded = feat_mask.unsqueeze(1).expand(-1, x.size(1), -1)
                x_input = x.clone()
                x_input[mask_expanded] = 0.0
                for name, model in models.items():
                    if name == "PerFeature":
                        output = model(x_input, feat_mask)
                    else:
                        output = model(x_input)
                    loss = F.mse_loss(output[mask_expanded], x[mask_expanded])
                    all_losses[name] += loss.item() * B_curr
                    preds[name].append(output[mask_expanded].detach().cpu().numpy())
                targets.append(x[mask_expanded].detach().cpu().numpy())
                all_counts += B_curr
        for name in models:
            avg_loss = all_losses[name] / all_counts
            test_losses[name].append(avg_loss)
            preds_concat = np.concatenate(preds[name], axis=0)
            targets_concat = np.concatenate(targets, axis=0)
            r2 = r2_score(targets_concat, preds_concat)
            test_r2s[name].append(r2)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} -- Random Mask Test Metrics:")
            for name in models:
                print(f"  {name}: Loss: {test_losses[name][-1]:.4f}, R²: {test_r2s[name][-1]:.4f}")
    return test_losses, test_r2s

def evaluate_whole_feature_missing(models, test_data, batch_size, missing_prob, device):
    print("\nWhole-Feature Missing Evaluation:")
    num_feats = list(models.values())[0].num_features if hasattr(list(models.values())[0], 'num_features') else test_data.shape[-1]
    results = {}
    with torch.no_grad():
        total_loss = {name: 0.0 for name in models}
        count = 0
        preds = {name: [] for name in models}
        targets = []
        for x in get_batches(test_data, batch_size):
            B_curr = x.size(0)
            feat_mask = generate_whole_feature_mask(B_curr, num_feats, missing_prob).to(device)
            mask_expanded = feat_mask.unsqueeze(1).expand(-1, x.size(1), -1)
            x_input = x.clone()
            x_input[mask_expanded] = 0.0
            for name, model in models.items():
                if name == "PerFeature":
                    output = model(x_input, feat_mask)
                else:
                    output = model(x_input)
                loss = F.mse_loss(output[mask_expanded], x[mask_expanded])
                total_loss[name] += loss.item() * B_curr
                preds[name].append(output[mask_expanded].detach().cpu().numpy())
            targets.append(x[mask_expanded].detach().cpu().numpy())
            count += B_curr
        for name in models:
            avg_loss = total_loss[name] / count
            preds_concat = np.concatenate(preds[name], axis=0)
            targets_concat = np.concatenate(targets, axis=0)
            r2 = r2_score(targets_concat, preds_concat)
            results[name] = (avg_loss, r2)
            print(f"  {name}: Loss: {avg_loss:.4f}, R²: {r2:.4f}")
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # Hyperparameters
    N = 1000                # Total samples
    train_ratio = 0.8
    L = 100                 # Sequence length
    num_features = 50       # Number of features
    E = 30                  # Embedding dimension
    nhead = 5               # For Transformer models
    nhid = 64               # Hidden size for PerFeatureTransformer
    nlayers = 2             # Number of Transformer layers
    dropout = 0.1
    num_epochs = 1000
    batch_size = 10
    mask_prob = 0.5         # For random masking during training/evaluation
    whole_missing_prob = 0.5  # For whole-feature missing evaluation

    # Generate synthetic dataset using complicated transformations
    dataset = generate_synthetic_dataset(N, L, num_features, device)
    indices = torch.randperm(N)
    train_size = int(train_ratio * N)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_data = dataset[train_indices]
    test_data = dataset[test_indices]

    # Instantiate models (include OGPerFeatureTransformer)
    models = {
        "Linear": LinearBaseline(num_features).to(device),
        "MLP": MLPBaseline(num_features, hidden_dim=64).to(device),
        "CNN": CNNBaseline(num_features, hidden_channels=32, kernel_size=3).to(device),
        "Standard": StandardTransformer(num_features, E, nlayers, dropout=dropout, nhead=nhead).to(device),
        "PerFeature": SimplifiedPerFeatureTransformer(num_features, E, nhead, nhid, nlayers, dropout=dropout,
                                                      parallel_attention=False, second_mlp=True).to(device),
        "OGPerFeature": OGPerFeatureTransformer(num_features, E, nlayers, dropout=dropout, nhead=nhead).to(device),
    }

    optimizers = {name: optim.Adam(model.parameters(), lr=1e-4) for name, model in models.items()}

    test_losses, test_r2s = train_and_evaluate_models(models, optimizers, train_data, test_data,
                                                      num_epochs, batch_size, mask_prob, device)

    print("\nFinal Test Metrics (Random Masking):")
    for name in models:
        print(f"{name}: Loss: {test_losses[name][-1]:.4f}, R²: {test_r2s[name][-1]:.4f}")

    print("\nWhole-Feature Missing Evaluation:")
    whole_results = evaluate_whole_feature_missing(models, test_data, batch_size, whole_missing_prob, device)

    print("\nOverall Results:")
    for name in models:
        rand_loss, rand_r2 = test_losses[name][-1], test_r2s[name][-1]
        whole_loss, whole_r2 = whole_results[name]
        print(f"{name}: Random Mask -> Loss: {rand_loss:.4f}, R²: {rand_r2:.4f};  Whole-Feature -> Loss: {whole_loss:.4f}, R²: {whole_r2:.4f}")

if __name__ == '__main__':
    main()
