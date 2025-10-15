# Control Data Integration - Design

## Architecture Overview
Integrate control signals by treating them as an additional feature channel in the existing data pipeline. **Critical Design Decision**: Load control data separately and concatenate AFTER masking to ensure controls are never masked. No model architecture changes required - control becomes feature `F+1`.

## Design Principles
1. **Minimal Changes**: Leverage existing infrastructure for loading and processing
2. **Backwards Compatible**: Gracefully handle biosamples without controls
3. **Treat as Feature**: Control is just another experiment/assay channel
4. **Never Mask Controls**: Controls must always remain available as reference signal

## Component Design

### 1. Data Loading (`data.py`)

#### Three New Methods (Follow Existing Hierarchical Pattern)

**Method 1: `load_bios_Control(bios_name, locus, DSF, f_format="npz")`**
- Similar to `load_bios_Counts()` but for control experiment
- Checks if `chipseq-control/` directory exists for biosample
- Loads from `chipseq-control/signal_DSF{DSF}_res{resolution}/{chr}.npz`
- Returns `(loaded_data_dict, loaded_metadata_dict)` or `(None, None)` if no control
- loaded_data format: `{"chipseq-control": numpy_array}`
- loaded_metadata format: `{"chipseq-control": {depth, platform, read_length, run_type}}`

**Method 2: `make_bios_tensor_Control(loaded_data, loaded_metadata, missing_value=-1)`**
- Similar to `make_bios_tensor_Counts()` but for single control channel
- Formats ONE biosample's control into tensors
- Returns `(data_tensor, metadata_tensor, availability)`:
  - `data_tensor`: shape `(L, 1)` - control signal values
  - `metadata_tensor`: shape `(4, 1)` - [log2_depth, platform_id, read_length, run_type]
  - `availability`: shape `(1,)` - scalar 1 if available, 0 if missing
- Handles None by filling with missing_value and availability=0

**Method 3: `make_region_tensor_Control(loaded_data_list, loaded_metadata_list)`**
- Similar to `make_region_tensor_Counts()` but for controls
- Stacks control tensors from multiple biosamples
- Returns `(data, metadata, availability)`:
  - `data`: shape `(B, L, 1)` 
  - `metadata`: shape `(B, 4, 1)`
  - `availability`: shape `(B, 1)`
- Exactly mirrors existing pattern for consistency

#### Modified Method: `get_batch()`

Update to load and return control data separately (not concatenated):

**Key Changes**:
1. Load control data for each biosample in the batch
2. Process through `make_region_tensor_Control()` for each locus
3. Stack across loci to get batch-level control tensors
4. Return 7 values instead of 4 (added 3 control outputs)

**Return Signature Change**:
```python
# OLD (4 return values):
return batch_data, batch_metadata, batch_availability, one_hot_sequences

# NEW (7 return values):
return batch_data, batch_metadata, batch_availability, one_hot_sequences, \
       batch_control_data, batch_control_metadata, batch_control_availability
```

**Shape Summary**:
- Regular data: `(B, L, F)`, `(B, 4, F)`, `(B, F)` - unchanged
- Control data: `(B, L, 1)`, `(B, 4, 1)`, `(B, 1)` - new outputs
- Control kept separate, NOT concatenated in data.py

### 2. Training Pipeline Integration (`train.py`)

#### Three-Step Process in `_process_batch()`

**Step 1: Extract Control Data Separately**

Extract control from batch dict (3 new lines):
```python
control_data = batch['control_data'].float()   # [B, L, 1]
control_meta = batch['control_meta'].float()   # [B, 4, 1]
control_avail = batch['control_avail']         # [B, 1]
```

**Step 2: Apply Masking (Control Stays Separate)**

Existing masking code runs on regular data only:
```python
# Masking applied to F features only (control not included)
x_data_masked, x_meta_masked, x_avail_masked = self.masker.mask_assays(
    x_data, x_meta, x_avail, num_mask
)
# Shapes after masking: (B, L, F), (B, 4, F), (B, F)
# Control data untouched: (B, L, 1), (B, 4, 1), (B, 1)
```

**Step 3: Concatenate Control After Masking**

Concatenate along feature dimension (3 lines):
```python
x_data_masked = torch.cat([x_data_masked, control_data], dim=2)      # [B, L, F+1]
x_meta_masked = torch.cat([x_meta_masked, control_meta], dim=2)      # [B, 4, F+1]
x_avail_masked = torch.cat([x_avail_masked, control_avail], dim=1)   # [B, F+1]
```

Now `x_data_masked` has F+1 features where the last feature (control) is never masked.

#### Update Signal Dimension

In `main()` function, add 1 to signal_dim:
```python
signal_dim = len(temp_dataset.aliases['experiment_aliases']) + 1  # +1 for control
```

**Why**: Control is not in experiment_aliases, so we manually add 1.

**Result**: Model initialized with correct input dimension (F+1 features).

## Complete Data Flow

### Phase 1: Data Loading (`data.py`)

```
get_batch(side='x')
    ↓
Load regular experiments for biosamples
    → x_data: (B, L, F)
    → x_meta: (B, 4, F)
    → x_avail: (B, F)
    ↓
Load control for each biosample
    → load_bios_Control() → (data_dict, meta_dict) or (None, None)
    → make_region_tensor_Control() → stack to (B, L, 1), (B, 4, 1), (B, 1)
    ↓
Return 7 separate tensors (control NOT concatenated):
    ✓ x_data, x_meta, x_avail, x_dna (regular - 4 values)
    ✓ control_data, control_meta, control_avail (control - 3 values)
```

### Phase 2: Batch Assembly (`CANDIIterableDataset`)

```
__iter__() method:
    ↓
Unpack 7 values from get_batch():
    x_data, x_meta, x_avail, x_dna, control_data, control_meta, control_avail
    ↓
Create sample dict with all 10 keys:
    {x_data, x_meta, x_avail, x_dna,           ← regular data
     control_data, control_meta, control_avail, ← control data (separate)
     y_data, y_meta, y_avail, y_pval, y_peaks, y_dna}  ← target data
    ↓
DataLoader batches samples → batch dict
```

### Phase 3: Training (`train.py`)

```
_process_batch(batch):
    ↓
Step 1 - Extract from batch:
    x_data:    (B, L, F)   ← regular
    x_meta:    (B, 4, F)   ← regular
    x_avail:   (B, F)      ← regular
    control_data:  (B, L, 1)  ← control (separate!)
    control_meta:  (B, 4, 1)  ← control (separate!)
    control_avail: (B, 1)     ← control (separate!)
    ↓
Step 2 - Apply masking to regular data ONLY:
    mask_assays(x_data, x_meta, x_avail)
    → x_data_masked:  (B, L, F)  ← some features masked
    → x_meta_masked:  (B, 4, F)  ← some metadata masked
    → x_avail_masked: (B, F)     ← some marked unavailable
    Control data stays untouched!
    ↓
Step 3 - Concatenate control AFTER masking:
    torch.cat([x_data_masked, control_data], dim=2)    → (B, L, F+1)
    torch.cat([x_meta_masked, control_meta], dim=2)    → (B, 4, F+1)
    torch.cat([x_avail_masked, control_avail], dim=1)  → (B, F+1)
    ↓
Step 4 - Forward pass:
    model(x_data_masked, x_dna, x_meta_masked, y_meta)
    → Model sees F+1 features
    → Last feature (control) is NEVER masked
    → Control always available as reference
```

**Key Insight**: By concatenating AFTER masking, the last feature (index F) is guaranteed to never have the cloze_mask token (-2), ensuring the model always has access to the control signal.

## Code Changes Summary

### `data.py` Changes
- **Add 3 new methods** (~110 lines total):
  1. `load_bios_Control()` - ~40 lines
  2. `make_bios_tensor_Control()` - ~40 lines  
  3. `make_region_tensor_Control()` - ~15 lines
- **Modify `get_batch()`** (~30 lines added):
  - Load and format control data
  - Return 7 values instead of 4
- **Modify `__iter__()`** (~3 lines):
  - Unpack 7 values from get_batch
  - Add control keys to sample dict

### `train.py` Changes
- **Modify `_process_batch()`** (~6 lines added):
  - Extract control data (3 lines)
  - Concatenate after masking (3 lines)
- **Modify `_validate_batch()`** (~1 line):
  - Add control keys to expected_keys
- **Modify `main()`** (~1 line):
  - Add `+ 1` to signal_dim

**Total**: ~160 lines of new code, ~40 lines modified

## Edge Cases Handled

1. **Biosample without control**:
   - `load_bios_Control()` returns `(None, None)`
   - `make_bios_tensor_Control()` fills with `missing_value=-1`
   - `availability = 0` for that biosample
   - Model sees -1 in control channel (same as missing experiments)

2. **Mixed batch** (some with controls, some without):
   - Each biosample handled independently
   - control_avail tensor shows which have controls: `[1, 0, 1, 0, ...]`
   - Model can use available controls, ignore missing ones

3. **Control file paths**:
   - EIC dataset: `DATA_CANDI_EIC/{bios}/chipseq-control/signal_DSF{dsf}_res25/`
   - Merged dataset: Uses navigation paths (handles merged cell types)
   - Same directory structure as regular experiments

## Implementation Strategy

**Reuse Existing Code Patterns**:
- `load_bios_Control` mirrors `load_bios_Counts` logic
- `make_bios_tensor_Control` mirrors `make_bios_tensor_Counts` logic
- `make_region_tensor_Control` mirrors `make_region_tensor_Counts` logic
- Use existing `_load_npz()`, `_select_region_from_loaded_data()`

**Why This Design is Minimal**:
- No new infrastructure needed - reuse existing loading/formatting
- No model changes - control is just feature F+1
- No loss changes - control participates in normal loss computation
- ~160 lines total, mostly by copying existing method patterns

