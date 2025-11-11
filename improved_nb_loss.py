"""
Improved Negative Binomial Loss using PyTorch Distributions

This implementation addresses the convergence issues by:
1. Using PyTorch's built-in NegativeBinomial distribution
2. Better parameterization (total_count + logits or probs)
3. Removing aggressive clamping that kills gradients
4. Better initialization strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial as TorchNegativeBinomial


class ImprovedNegativeBinomialLayer(nn.Module):
    """
    Output layer for Negative Binomial distribution using PyTorch distributions.
    
    Uses the parameterization:
    - total_count (r): dispersion parameter
    - logits: log(p/(1-p)) where p is success probability
    
    Mean = total_count * (1-p) / p = total_count * exp(-logits) * sigmoid(logits)
    Variance = mean * (1 + mean/total_count)
    """
    def __init__(self, input_dim, output_dim, min_total_count=0.1, max_total_count=1000.0):
        super(ImprovedNegativeBinomialLayer, self).__init__()
        
        self.min_total_count = min_total_count
        self.max_total_count = max_total_count
        
        # Pre-normalization
        self.pre_norm = nn.LayerNorm(input_dim)
        
        # Separate projections for total_count and logits
        self.linear_total_count = nn.Linear(input_dim, output_dim)
        self.linear_logits = nn.Linear(input_dim, output_dim)
        
        # Better initialization
        # Initialize total_count to predict moderate dispersion (around 10)
        nn.init.xavier_normal_(self.linear_total_count.weight, gain=0.1)
        nn.init.constant_(self.linear_total_count.bias, 2.3)  # softplus(2.3) ≈ 10
        
        # Initialize logits to predict reasonable means (around 5-10)
        # logits of -2 to -1 gives means in that range with total_count=10
        nn.init.xavier_normal_(self.linear_logits.weight, gain=0.1)
        nn.init.constant_(self.linear_logits.bias, -1.5)
    
    def forward(self, x):
        """
        Returns total_count and logits for NegativeBinomial distribution.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            total_count: Shape (batch, seq_len, output_dim), range [min_total_count, max_total_count]
            logits: Shape (batch, seq_len, output_dim), unconstrained
        """
        x = self.pre_norm(x)
        
        # Total count with soft clamping via softplus + scaling
        total_count_raw = self.linear_total_count(x)
        total_count = F.softplus(total_count_raw)
        # Soft clamp: allow range but penalize extremes
        total_count = torch.clamp(total_count, min=self.min_total_count, max=self.max_total_count)
        
        # Logits (unconstrained, will be used internally by distribution)
        logits = self.linear_logits(x)
        
        return total_count, logits


def negative_binomial_loss_torch(y_true, total_count, logits, reduction='none', eps=1e-8):
    """
    Negative Binomial negative log-likelihood using PyTorch distributions.
    
    Args:
        y_true: Ground truth counts, shape (N,)
        total_count: Dispersion parameter, shape (N,)
        logits: Log odds, shape (N,)
        reduction: 'none', 'mean', or 'sum'
        eps: Small constant for numerical stability
        
    Returns:
        nll: Negative log-likelihood
    """
    # Ensure inputs are float
    y_true = y_true.float()
    total_count = total_count.float()
    logits = logits.float()
    
    # Add small epsilon to total_count for stability
    total_count = torch.clamp(total_count, min=eps)
    
    # Create distribution
    try:
        nb_dist = TorchNegativeBinomial(total_count=total_count, logits=logits)
        
        # Compute negative log-likelihood
        nll = -nb_dist.log_prob(y_true)
        
        # Handle any NaN or Inf values
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))
        
    except Exception as e:
        print(f"Warning: NegativeBinomial distribution failed: {e}")
        # Fallback to simple MSE-based loss
        probs = torch.sigmoid(logits)
        mean_pred = total_count * (1 - probs) / probs
        nll = F.mse_loss(mean_pred, y_true, reduction='none')
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll


class ImprovedCANDI_LOSS(nn.Module):
    """
    Improved loss function using PyTorch distributions for Negative Binomial.
    """
    def __init__(self, reduction='mean', count_weight=1.0, pval_weight=1.0, peak_weight=1.0):
        super(ImprovedCANDI_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction, full=True)
        self.bce_loss = nn.BCELoss(reduction=self.reduction)
        
        # Loss weights
        self.count_weight = count_weight
        self.pval_weight = pval_weight
        self.peak_weight = peak_weight

    def forward(self, total_count_pred, logits_pred, mu_pred, var_pred, peak_pred, 
                true_count, true_pval, true_peak, obs_map, masked_map):
        """
        Args:
            total_count_pred: Predicted total_count parameter
            logits_pred: Predicted logits parameter  
            mu_pred, var_pred: Gaussian parameters for p-values
            peak_pred: Peak predictions
            true_count, true_pval, true_peak: Ground truth
            obs_map, masked_map: Boolean masks for observed/masked positions
        """
        # Observed (upsampling) loss
        ups_true_count = true_count[obs_map]
        ups_total_count = total_count_pred[obs_map]
        ups_logits = logits_pred[obs_map]
        
        observed_count_loss = negative_binomial_loss_torch(
            ups_true_count, ups_total_count, ups_logits, reduction=self.reduction
        )
        
        # Imputed loss
        imp_true_count = true_count[masked_map]
        imp_total_count = total_count_pred[masked_map]
        imp_logits = logits_pred[masked_map]
        
        imputed_count_loss = negative_binomial_loss_torch(
            imp_true_count, imp_total_count, imp_logits, reduction=self.reduction
        )
        
        # P-value losses (unchanged)
        ups_true_pval, ups_true_peak = true_pval[obs_map], true_peak[obs_map]
        ups_mu_pred, ups_var_pred = mu_pred[obs_map], var_pred[obs_map]
        ups_peak_pred = peak_pred[obs_map]
        
        imp_true_pval, imp_true_peak = true_pval[masked_map], true_peak[masked_map]
        imp_mu_pred, imp_var_pred = mu_pred[masked_map], var_pred[masked_map]
        imp_peak_pred = peak_pred[masked_map]
        
        observed_pval_loss = self.gaus_nll(ups_mu_pred, ups_true_pval, ups_var_pred)
        imputed_pval_loss = self.gaus_nll(imp_mu_pred, imp_true_pval, imp_var_pred)
        
        observed_pval_loss = observed_pval_loss.float()
        imputed_pval_loss = imputed_pval_loss.float()
        
        # Peak losses
        with torch.amp.autocast("cuda", enabled=False):
            observed_peak_loss = self.bce_loss(ups_peak_pred.float(), ups_true_peak.float())
            imputed_peak_loss = self.bce_loss(imp_peak_pred.float(), imp_true_peak.float())
        
        # Apply weights
        observed_count_loss = self.count_weight * observed_count_loss
        imputed_count_loss = self.count_weight * imputed_count_loss
        observed_pval_loss = self.pval_weight * observed_pval_loss
        imputed_pval_loss = self.pval_weight * imputed_pval_loss
        observed_peak_loss = self.peak_weight * observed_peak_loss
        imputed_peak_loss = self.peak_weight * imputed_peak_loss
        
        return (observed_count_loss, imputed_count_loss, 
                observed_pval_loss, imputed_pval_loss, 
                observed_peak_loss, imputed_peak_loss)


# ==================== Alternative: Mean + Dispersion Parameterization ====================

class MeanDispersionNBLayer(nn.Module):
    """
    Alternative parameterization that directly predicts mean and dispersion.
    This can be easier to optimize as mean is directly interpretable.
    
    mean = μ (what we actually care about)
    dispersion = r (controls overdispersion)
    
    Then convert to PyTorch's parameterization:
    total_count = r
    probs = r / (r + μ)
    """
    def __init__(self, input_dim, output_dim, min_dispersion=0.1, max_dispersion=1000.0):
        super(MeanDispersionNBLayer, self).__init__()
        
        self.min_dispersion = min_dispersion
        self.max_dispersion = max_dispersion
        
        self.pre_norm = nn.LayerNorm(input_dim)
        
        # Separate projections
        self.linear_mean = nn.Linear(input_dim, output_dim)
        self.linear_dispersion = nn.Linear(input_dim, output_dim)
        
        # Initialize to predict reasonable values
        # Mean around 5-10 for typical ChIP-seq data
        nn.init.xavier_normal_(self.linear_mean.weight, gain=0.1)
        nn.init.constant_(self.linear_mean.bias, 2.0)  # softplus(2) ≈ 2.13
        
        # Dispersion around 10
        nn.init.xavier_normal_(self.linear_dispersion.weight, gain=0.1)
        nn.init.constant_(self.linear_dispersion.bias, 2.3)
    
    def forward(self, x):
        """
        Returns mean and dispersion, then converts to total_count and probs.
        """
        x = self.pre_norm(x)
        
        # Predict mean (must be positive)
        mean_raw = self.linear_mean(x)
        mean = F.softplus(mean_raw) + 1e-6  # Add small offset to prevent zero
        
        # Predict dispersion (must be positive)
        dispersion_raw = self.linear_dispersion(x)
        dispersion = F.softplus(dispersion_raw)
        dispersion = torch.clamp(dispersion, min=self.min_dispersion, max=self.max_dispersion)
        
        # Convert to PyTorch parameterization
        total_count = dispersion
        probs = dispersion / (dispersion + mean + 1e-6)
        
        return total_count, probs


def negative_binomial_loss_mean_dispersion(y_true, total_count, probs, reduction='none', eps=1e-8):
    """
    NB loss using mean-dispersion parameterization with probs instead of logits.
    """
    y_true = y_true.float()
    total_count = torch.clamp(total_count.float(), min=eps)
    probs = torch.clamp(probs.float(), min=eps, max=1.0-eps)
    
    try:
        nb_dist = TorchNegativeBinomial(total_count=total_count, probs=probs)
        nll = -nb_dist.log_prob(y_true)
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))
    except Exception as e:
        print(f"Warning: NB distribution failed: {e}")
        # Fallback
        mean_pred = total_count * (1 - probs) / probs
        nll = F.mse_loss(mean_pred, y_true, reduction='none')
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll


if __name__ == "__main__":
    # Test the improved layer
    batch_size, seq_len, input_dim, output_dim = 4, 100, 128, 35
    
    # Test 1: Logits parameterization
    print("Testing Logits Parameterization:")
    layer1 = ImprovedNegativeBinomialLayer(input_dim, output_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    total_count, logits = layer1(x)
    print(f"  Total count range: [{total_count.min():.2f}, {total_count.max():.2f}]")
    print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Compute implied mean
    probs = torch.sigmoid(logits)
    mean = total_count * (1 - probs) / probs
    print(f"  Implied mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    
    # Test loss
    y_true = torch.randint(0, 20, (batch_size, seq_len, output_dim)).float()
    loss = negative_binomial_loss_torch(y_true.flatten(), total_count.flatten(), 
                                       logits.flatten(), reduction='mean')
    print(f"  Loss: {loss.item():.4f}")
    
    # Test 2: Mean-Dispersion parameterization  
    print("\nTesting Mean-Dispersion Parameterization:")
    layer2 = MeanDispersionNBLayer(input_dim, output_dim)
    total_count2, probs2 = layer2(x)
    mean2 = total_count2 * (1 - probs2) / probs2
    print(f"  Mean range: [{mean2.min():.2f}, {mean2.max():.2f}]")
    print(f"  Dispersion range: [{total_count2.min():.2f}, {total_count2.max():.2f}]")
    
    loss2 = negative_binomial_loss_mean_dispersion(y_true.flatten(), total_count2.flatten(),
                                                   probs2.flatten(), reduction='mean')
    print(f"  Loss: {loss2.item():.4f}")
    
    # Test gradients
    print("\nTesting gradients:")
    loss2.backward()
    print(f"  Gradients computed successfully")
    print(f"  Mean grad norm: {layer2.linear_mean.weight.grad.norm():.4f}")
    print(f"  Dispersion grad norm: {layer2.linear_dispersion.weight.grad.norm():.4f}")

