# Negative Binomial Loss Convergence Issues - Complete Analysis

## **Problem Summary**

Your negative binomial count loss is **not converging properly**, resulting in:
- **R² near 0** (or highly negative) for count imputation
- **Very poor Pearson/Spearman correlations** (near 0)
- **Loss values stable but not decreasing** (stuck around 1.0-1.8)
- **P-value branch converges fine** (R² around -0.4 to -0.9, correlations 0.001-0.01)

The count branch is essentially **not learning meaningful representations**.

---

## **Root Causes**

### **1. Overly Aggressive Robust Loss Clamping**

```python
# Current implementation in _utils.py
if robust:
    med = torch.median(nll).detach()
    lo, hi = med - delta, med + delta  # delta=2.0
    nll = torch.clamp(nll, min=lo, max=hi)
```

**Problem**: This clamps the loss to a **very narrow range** (±2.0 around the median), which:
- **Kills gradients** for samples with high loss
- **Prevents the model from learning** from difficult examples
- **Stops convergence** once loss reaches this range

**Impact**: Your loss plateaus early because gradients are clipped.

---

### **2. Poor Parameterization & Numerical Instability**

The negative binomial uses parameterization `mean = n*(1-p)/p`:

```python
# Current implementation
p_pred = torch.clamp(p_pred, min=eps, max=1 - eps)  # eps=1e-6
n_pred = torch.clamp(n_pred, min=1e-2, max=1e3)
```

**Problems**:
- **Hard to optimize**: The relationship between (n, p) and the mean is non-linear and complex
- **Numerical issues**: When `p` is close to 0 or 1, `log(p)` or `log(1-p)` become unstable
- **Gradient flow**: Changes in `n` or `p` don't smoothly translate to changes in predicted mean
- **Narrow clamping**: `n` clamped to [0.01, 1000] is very restrictive

---

### **3. Initialization Problems in NegativeBinomialLayer**

```python
# Current initialization
nn.init.constant_(self.linear_n.bias, 1.0)  # n starts around 1
nn.init.constant_(self.linear_p.bias, 0.0)  # p starts around 0.5
```

**Problems**:
- With `n=softplus(1.0) ≈ 1.31` and `p=sigmoid(0.0) = 0.5`, initial mean = `1.31 * 0.5 / 0.5 = 1.31`
- **Too small for ChIP-seq data** (typical counts: 5-50)
- Model needs to learn large changes from initialization
- Clamping logits before activation (`torch.clamp(p_logits, min=-10, max=10)`) **kills gradients**

---

### **4. Why Your Current Loss Formula Has Issues**

```python
nll = (
    torch.lgamma(n_pred + eps)
    + torch.lgamma(y_true + 1 + eps)
    - torch.lgamma(n_pred + y_true + eps)
    - n_pred * torch.log(p_pred + eps)
    - y_true * torch.log(1 - p_pred + eps)
)
```

**Issues**:
1. Adding `eps` inside `lgamma` changes the mathematical formula
2. `lgamma` can produce NaN/Inf for edge cases
3. No handling of zero counts properly
4. The robust clamping afterwards destroys the gradient signal

---

## **Why PyTorch Distributions is Better**

PyTorch's `torch.distributions.NegativeBinomial` handles:

### **1. Better Numerical Stability**
- Uses log-space computations internally
- Handles edge cases (zero counts, extreme parameters)
- Implements numerically stable `log_prob` method

### **2. Better Parameterization**
- Can use `logits` instead of `probs`: `logits = log(p/(1-p))`
- **Logits are unconstrained** → better gradient flow
- Avoids sigmoid saturation issues

### **3. Automatic Gradient Handling**
- PyTorch automatically computes stable gradients
- Built-in support for sampling, mean, variance computation
- Less prone to numerical errors

### **4. Mean-Dispersion Alternative**
You can also directly parameterize as:
- `mean` = μ (what you actually care about)
- `dispersion` = r (controls variance)

Then convert: `total_count = r`, `probs = r/(r+μ)`

This is **much easier to optimize** because the mean is directly predicted!

---

## **Recommended Solutions**

### **Solution 1: Use PyTorch Distributions with Logits (BEST)**

**Advantages**:
- Most numerically stable
- Unconstrained logits → better gradients
- Built-in PyTorch optimization

**Changes needed**:
1. Replace `NegativeBinomialLayer` output to return `(total_count, logits)`
2. Use `torch.distributions.NegativeBinomial(total_count, logits=logits)`
3. Compute loss with `.log_prob(y_true)`
4. Remove robust clamping entirely

See `improved_nb_loss.py` for implementation.

---

### **Solution 2: Mean-Dispersion Parameterization**

**Advantages**:
- Mean is directly interpretable
- Easier to initialize (target typical count values)
- More intuitive for biological data

**Changes needed**:
1. Predict `mean` and `dispersion` directly
2. Convert to `total_count = dispersion`, `probs = dispersion/(dispersion + mean)`
3. Use PyTorch NegativeBinomial with probs

See `MeanDispersionNBLayer` in `improved_nb_loss.py`.

---

### **Solution 3: Improve Current Implementation (NOT RECOMMENDED)**

If you must keep custom loss:

1. **Remove or reduce robust clamping**:
   ```python
   # Instead of delta=2.0, use delta=10.0 or remove entirely
   if robust:
       med = torch.median(nll).detach()
       lo, hi = med - 10.0, med + 10.0  # Much wider range
       nll = torch.clamp(nll, min=lo, max=hi)
   ```

2. **Remove logit clamping in forward pass**:
   ```python
   # Remove these lines:
   # p_logits = torch.clamp(p_logits, min=-10, max=10)
   # n_logits = torch.clamp(n_logits, min=-5)
   ```

3. **Better initialization**:
   ```python
   # Initialize to predict mean around 10, dispersion around 10
   nn.init.constant_(self.linear_n.bias, 2.3)  # softplus(2.3) ≈ 10
   nn.init.constant_(self.linear_p.bias, -1.5)  # gives mean ≈ 10 with n=10
   ```

4. **Widen parameter ranges**:
   ```python
   n_pred = torch.clamp(n_pred, min=0.1, max=1e4)  # Wider range
   ```

---

## **Implementation Steps**

### **Step 1: Test the Improved Loss**

```bash
cd /home/mforooz/projects/def-maxwl/mforooz/EpiDenoise
python improved_nb_loss.py
```

This will test:
- Both parameterizations
- Forward pass
- Loss computation
- Gradient flow

### **Step 2: Update Your Model**

In `model.py`:

```python
from improved_nb_loss import ImprovedNegativeBinomialLayer, ImprovedCANDI_LOSS

# Replace in CANDI.__init__:
self.neg_binom_layer = ImprovedNegativeBinomialLayer(self.f1, self.f1)

# Update CANDI.decode to return (total_count, logits) instead of (p, n)
total_count, logits = self.neg_binom_layer(count_decoded)
```

### **Step 3: Update Training Loop**

In `train.py`:

```python
from improved_nb_loss import ImprovedCANDI_LOSS, negative_binomial_loss_torch

# Replace criterion
criterion = ImprovedCANDI_LOSS(
    reduction='mean',
    count_weight=1.0,  # Try 1.0 instead of 2.0 initially
    pval_weight=1.0,
    peak_weight=1.0
)

# Update validation metrics to use torch distributions
```

### **Step 4: Retrain**

Start with a fresh training run:
- Use **lower learning rate** initially (1e-5) to ensure stability
- Monitor count R² closely - should improve within first few epochs
- If R² starts positive and increases, you're on the right track

---

## **Expected Results**

### **Before (Current)**
- Imputation Count R²: **-0.9 to -10** ❌
- Imputation Count PCC: **~0.001** ❌  
- Loss stable but not meaningful

### **After (With Fix)**
- Imputation Count R²: **0.3 to 0.7** ✅
- Imputation Count PCC: **0.5 to 0.85** ✅
- Loss continues to decrease
- Meaningful count predictions

---

## **Debugging Checklist**

If issues persist after implementing fixes:

### **1. Check Parameter Ranges**
```python
# Add logging in training loop
print(f"total_count: min={total_count.min():.2f}, max={total_count.max():.2f}")
print(f"logits: min={logits.min():.2f}, max={logits.max():.2f}")

probs = torch.sigmoid(logits)
mean = total_count * (1-probs) / probs
print(f"Predicted mean: min={mean.min():.2f}, max={mean.max():.2f}")
print(f"True counts: min={y_true.min():.2f}, max={y_true.max():.2f}")
```

**Expected**:
- `total_count`: 0.1 to 100
- `logits`: -5 to 5
- `mean`: Similar range to true counts (e.g., 1-50)

### **2. Check for NaN/Inf**
```python
if torch.isnan(loss) or torch.isinf(loss):
    print("NaN/Inf detected in loss!")
    # Log all intermediate values
```

### **3. Check Gradient Norms**
```python
for name, param in model.named_parameters():
    if 'neg_binom_layer' in name and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

**Expected**: Grad norms should be ~0.01 to 1.0, not 0 or >100

### **4. Verify Distribution is Valid**
```python
# In loss function
print(f"NB mean: {nb_dist.mean.mean():.2f}")
print(f"NB variance: {nb_dist.variance.mean():.2f}")
```

---

## **Additional Tips**

### **1. Loss Weighting**
Start with balanced weights:
```python
count_weight=1.0  # Not 2.0
pval_weight=1.0
peak_weight=1.0
```

Once count branch works, you can adjust.

### **2. Separate Decoders**
Your `separate_decoders=True` is good - keep it. This allows count and p-value branches to specialize.

### **3. Learning Rate Schedule**
Consider warmup for the count branch:
```python
# First 1000 steps: lr=1e-5
# Then ramp up to 1e-4
```

### **4. Data Preprocessing**
Verify your count data:
- Are there outliers? (counts > 1000)
- Are most counts small? (< 10)
- Consider log1p transformation for very large counts

---

## **References**

- [PyTorch Distributions Docs](https://pytorch.org/docs/stable/distributions.html)
- [Negative Binomial Distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
- [scVI uses this approach for scRNA-seq](https://github.com/scverse/scvi-tools)

---

## **Next Steps**

1. ✅ Review this analysis
2. ✅ Test `improved_nb_loss.py`
3. ✅ Choose parameterization (logits or mean-dispersion)
4. ✅ Update model and loss
5. ✅ Retrain and monitor count R²
6. ✅ Compare old vs new training curves

Let me know which approach you want to try first!

