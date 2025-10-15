# Control Data Integration - Requirements

## Overview
Integrate ChIP-seq control experiment data as an additional always-available input channel to the CANDI model. 

**Key Requirement**: Controls must NEVER be masked during training - they serve as reference signals.

**Implementation Approach**: Load controls separately, apply masking to regular data only, then concatenate control after masking.

**Result**: Model receives F+1 features where feature F (control) is always available.

## Current State
- Control experiments are already downloaded and processed as separate "chipseq-control" experiments in biosample directories
- Each biosample with ChIP-seq experiments has a corresponding `chipseq-control/` directory with signal files
- Current data loading returns:
  - `x_data`: shape `(B, L, F)` - batch, sequence length, features
  - `x_meta`: shape `(B, 4, F)` - batch, metadata fields, features
  - `x_avail`: shape `(B, F)` - batch, features

## Requirements

### 1. Load Control Data Separately in `data.py`
- Modify `get_batch()` to load control signals when `side='x'`
- Control data returned separately (NOT concatenated with input data)
- Control data should have shape `(B, L, 1)` - single channel per biosample
- Control metadata should have shape `(B, 4, 1)` - metadata for the single control channel (matches x_meta pattern)
- Control availability should have shape `(B, 1)` - always 1 for available controls, 0 when missing
- If control doesn't exist for a biosample, fill with missing values (-1) and set availability to 0

### 2. Apply Masking in `train.py` (Before Concatenation)
- Apply random masking to regular input data `(B, L, F)` as usual
- Control data is NOT passed through masking - kept separate
- Controls remain fully available for the model

### 3. Concatenate After Masking in `train.py`
- After masking applied to input data, concatenate control as additional feature
- Concatenate control data to feature dimension: `(B, L, F)` → `(B, L, F+1)`
- Concatenate control metadata: `(B, 4, F)` → `(B, 4, F+1)` (transpose applied as needed)
- Concatenate control availability: `(B, F)` → `(B, F+1)` (always 1 for control)

### 4. Integration with Training Pipeline
- Model receives data with `F+1` features after concatenation
- Control channel is never masked - always available as reference
- No changes to model architecture needed - control is just another feature

## Success Criteria
- ✅ Control data successfully loaded separately from regular input data
- ✅ Controls are NEVER masked during training (verified by checking last feature != cloze_mask)
- ✅ Missing controls handled gracefully with missing value indicators
- ✅ Control concatenated after masking in training pipeline
- ✅ Training pipeline processes data with control channel without errors
- ✅ Model treats control as an additional (always available) input feature

## Non-Requirements
- ❌ No changes to model architecture (control is just another feature channel)
- ❌ No special handling or separate pathway for control data in the model
- ❌ No changes to loss computation or validation logic
- ❌ Control is not used in Y-side data (only X-side input)
- ❌ No need to add control to experiment aliases

## Summary

**What changes**:
1. `data.py`: Add 3 methods to load controls, return 7 values from `get_batch(side='x')`
2. `train.py`: Extract control, concatenate after masking, add +1 to signal_dim

**What stays the same**:
- Model architecture (handles variable feature dimensions)
- Loss computation (control participates like any other feature)
- Validation logic (no special handling needed)
- Y-side data (controls only in X-side)

**Core benefit**: Model always has access to control signal as reference, improving imputation quality for ChIP-seq experiments.

