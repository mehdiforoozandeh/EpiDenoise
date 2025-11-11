# Quick Fix Guide for Negative Binomial Count Loss

## **TL;DR - What's Wrong**

Your count loss isn't converging because:
1. **Overly aggressive robust loss clamping** (delta=2.0) kills gradients
2. **Poor parameterization** makes optimization difficult  
3. **Bad initialization** starts far from reasonable values
4. **Logit clamping** in the forward pass destroys gradient flow

**Result**: R² near 0, correlations near 0, loss plateaus early

---

## **Quick Fix Options**

### **Option A: Minimal Changes to Current Code** ⚡

**Time**: 10 minutes  
**Risk**: Low  
**Expected improvement**: R² from -0.9 → 0.2-0.4

#### Changes in `_utils.py`:

```python
# Line ~818-822: Reduce or remove robust clamping
def negative_binomial_loss(y_true, n_pred, p_pred, delta=10.0, robust=False):  # Changed!
    # ... keep rest of function
    if robust:
        med = torch.median(nll).detach()
        lo, hi = med - delta, med + delta  # Much wider range now
        nll = torch.clamp(nll, min=lo, max=hi)
```

**Why**: This gives gradients room to flow for difficult examples.

#### Changes in `model.py`:

```python
# Lines ~1886-1907: Remove logit clamping and better initialization
class NegativeBinomialLayer(nn.Module):
    def __init__(self, input_dim, output_dim, FF=False):
        super(NegativeBinomialLayer, self).__init__()
        # ... existing code ...
        
        # BETTER INITIALIZATION (Lines ~1888-1889)
        nn.init.xavier_normal_(self.linear_p.weight, gain=0.1)  # Smaller gain
        nn.init.constant_(self.linear_p.bias, -1.5)  # Better default
        nn.init.xavier_normal_(self.linear_n.weight, gain=0.1)
        nn.init.constant_(self.linear_n.bias, 2.3)  # softplus(2.3) ≈ 10

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)
        x = self.pre_norm(x)
        
        # REMOVE THESE CLAMPING LINES:
        # p_logits = torch.clamp(p_logits, min=-10, max=10)  # DELETE THIS
        # n_logits = torch.clamp(n_logits, min=-5)           # DELETE THIS
        
        # Just apply activations directly:
        p_logits = self.linear_p(x)
        p = torch.sigmoid(p_logits)
        
        n_logits = self.linear_n(x)
        n = F.softplus(n_logits) + 0.1  # Small offset for stability
        
        return p, n
```

**Why**: Removes gradient-killing clamps and starts from better initialization.

#### Changes in `train.py`:

```python
# Reduce count_weight from 2.0 to 1.0 initially
criterion = CANDI_LOSS(
    reduction='mean',
    count_weight=1.0,  # Changed from 2.0
    pval_weight=1.0,
    peak_weight=1.0
)
```

**Why**: Balanced weights let count branch learn properly first.

---

### **Option B: Use PyTorch Distributions** 🚀 (RECOMMENDED)

**Time**: 30-60 minutes  
**Risk**: Medium  
**Expected improvement**: R² from -0.9 → 0.5-0.7

This is the proper fix. See `improved_nb_loss.py` for full implementation.

#### Step 1: Copy the improved layer

```bash
# In model.py, add this import at top:
from improved_nb_loss import MeanDispersionNBLayer
```

#### Step 2: Replace in CANDI class

```python
# Line ~1070: Replace this line:
# self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1)

# With:
self.neg_binom_layer = MeanDispersionNBLayer(
    self.f1, self.f1, 
    min_dispersion=0.5, 
    max_dispersion=500.0
)
```

#### Step 3: Update decode method

```python
# Lines ~1091-1092: Update to handle new output format
# Old:
# p, n = self.neg_binom_layer(count_decoded)

# New:
total_count, probs = self.neg_binom_layer(count_decoded)

# For compatibility with rest of code, can still call them p, n:
p, n = probs, total_count  # Just swap names
```

#### Step 4: Update loss function

```python
# In train.py or model.py, replace the loss calculation:
from improved_nb_loss import negative_binomial_loss_mean_dispersion

# In CANDI_LOSS.forward():
observed_count_loss = negative_binomial_loss_mean_dispersion(
    ups_true_count, ups_n_pred, ups_p_pred, reduction=self.reduction
)
imputed_count_loss = negative_binomial_loss_mean_dispersion(
    imp_true_count, imp_n_pred, imp_p_pred, reduction=self.reduction
)
```

---

## **Verification Checklist**

After making changes, check these during training:

### **1. Parameter Ranges (First 10 batches)**

```python
# Add logging in training loop
print(f"Batch {batch_idx}:")
print(f"  n_pred: min={n_pred.min():.2f}, max={n_pred.max():.2f}, mean={n_pred.mean():.2f}")
print(f"  p_pred: min={p_pred.min():.4f}, max={p_pred.max():.4f}, mean={p_pred.mean():.4f}")

mean = n_pred * (1 - p_pred) / (p_pred + 1e-8)
print(f"  Predicted mean: min={mean.min():.2f}, max={mean.max():.2f}")
print(f"  True counts: min={y_true.min():.2f}, max={y_true.max():.2f}")
```

**Expected (Option A)**:
- n_pred: 1-100 (wider than before)
- p_pred: 0.1-0.9
- Predicted mean: Similar to true counts (e.g., 2-50)

**Expected (Option B)**:
- n_pred (dispersion): 0.5-50
- p_pred (probs): 0.3-0.9  
- Predicted mean: Very close to true counts

### **2. Loss Values (First 100 iterations)**

**Before fix**:
- Obs count loss: 1.1-1.3, stays flat
- Imp count loss: 1.3-1.8, stays flat
- No meaningful decrease

**After Option A**:
- Obs count loss: 2-8 → 1-3 (decreases gradually)
- Imp count loss: 3-10 → 1.5-4 (decreases gradually)
- Should see decrease over first 1000 iterations

**After Option B**:
- Obs count loss: 3-8 → 1-2 (decreases smoothly)
- Imp count loss: 4-10 → 1.5-3 (decreases smoothly)
- Faster convergence than Option A

### **3. Gradients (Check every 100 iterations)**

```python
for name, param in model.named_parameters():
    if 'neg_binom_layer' in name and param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm:.4f}")
```

**Before fix**: Gradients ~0.0001-0.001 (too small)  
**After fix**: Gradients ~0.01-1.0 (healthy range)

### **4. Metrics (After 5000 iterations)**

Track these in your validation:

| Metric | Before | After Option A | After Option B |
|--------|--------|----------------|----------------|
| Imp Count R² | -0.9 to -10 | 0.2 to 0.4 | 0.5 to 0.7 |
| Imp Count PCC | ~0.001 | 0.3 to 0.5 | 0.6 to 0.8 |
| Obs Count R² | -0.6 to -5 | 0.4 to 0.6 | 0.6 to 0.8 |

---

## **Troubleshooting**

### **Issue: Loss becomes NaN**

**Solution**: 
```python
# In loss function, add safety clamp:
n_pred = torch.clamp(n_pred, min=0.1, max=500.0)
p_pred = torch.clamp(p_pred, min=1e-4, max=1.0-1e-4)
```

### **Issue: Predicted means are way off (e.g., 1000+)**

**Solution**:
```python
# Reinitialize the layer with smaller bias:
self.linear_n.bias.data.fill_(1.0)  # Lower starting point
```

### **Issue: Still no improvement after Option A**

**Solution**: Go with Option B - the parameterization matters more than expected.

### **Issue: Count loss increases while p-value loss decreases**

**Solution**: Reduce `count_weight` to 0.5 temporarily:
```python
count_weight=0.5  # Let count branch learn slower
```

---

## **Expected Timeline**

### **With Option A**:
- Iterations 0-1000: Loss decreases slowly
- Iterations 1000-5000: R² becomes positive
- Iterations 5000-10000: R² reaches 0.2-0.4
- May plateau around 0.4

### **With Option B**:
- Iterations 0-500: Rapid loss decrease
- Iterations 500-2000: R² becomes positive and rises fast
- Iterations 2000-5000: R² reaches 0.5-0.7
- Continues improving

---

## **When to Try Each Option**

### **Use Option A if**:
- You want minimal code changes
- You're mid-training and don't want to restart
- You just need "good enough" performance
- You're unsure about changing architectures

### **Use Option B if**:
- You're starting a new training run anyway
- You want best possible performance
- You're willing to spend an hour on proper implementation
- You plan to use this long-term

---

## **Next Steps**

1. **Choose** Option A or B
2. **Make changes** as described above
3. **Start training** with fresh weights or checkpoint
4. **Monitor** the verification checklist metrics
5. **Adjust** hyperparameters if needed (learning rate, weights)
6. **Compare** old vs new training curves

**Questions?** Check `NEGATIVE_BINOMIAL_ANALYSIS.md` for deeper analysis.

**Implementation help?** See `improved_nb_loss.py` for working code.

---

## **Final Recommendation**

**For immediate improvement**: Do Option A (10 min fix)  
**For best results**: Do Option B after seeing Option A helps

Start with Option A to verify the diagnosis is correct. If you see R² improve from -0.9 to 0.2+, that confirms the issue. Then invest time in Option B for full fix.

Good luck! 🚀

