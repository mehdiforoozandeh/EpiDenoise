# Control Data Integration - Implementation Tasks

## Quick Overview

**Goal**: Add ChIP-seq controls as an always-available input feature (never masked).

**Strategy**: Load controls separately → Apply masking to regular data only → Concatenate control after masking

**Changes**: 
- `data.py`: Add 3 methods + modify `get_batch()` and `__iter__()` (~150 lines)
- `train.py`: Extract control, concatenate after masking (~8 lines)

---

## Task 1: Add `load_bios_Control()` Method

**File**: `data.py` | **Location**: After `load_bios_Peaks()` (line ~673) | **Lines**: ~40

```python
def load_bios_Control(self, bios_name, locus, DSF=1, f_format="npz"):
    """Load chipseq-control signal for a biosample."""
    control_path = os.path.join(self.base_path, bios_name, "chipseq-control")
    if not os.path.exists(control_path):
        return None, None
    
    loaded_data = {}
    loaded_metadata = {}
    
    signal_path = os.path.join(control_path, f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
    metadata_path_1 = os.path.join(control_path, f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
    metadata_path_2 = os.path.join(control_path, "file_metadata.json")
    
    if not os.path.exists(signal_path):
        return None, None
    
    try:
        result = self._load_npz(signal_path)
        if result is None:
            return None, None
        
        for key, data in result.items():
            if len(locus) == 1:
                loaded_data["chipseq-control"] = data.astype(np.int16)
            else:
                start_bin = int(locus[1]) // self.resolution
                end_bin = int(locus[2]) // self.resolution
                loaded_data["chipseq-control"] = data[start_bin:end_bin]
        
        with open(metadata_path_1, 'r') as f:
            md1 = json.load(f)
        with open(metadata_path_2, 'r') as f:
            md2 = json.load(f)
        
        loaded_metadata["chipseq-control"] = {
            "depth": md1["depth"],
            "sequencing_platform": md2.get("sequencing_platform", {"2": "unknown"}).get("2", "unknown"),
            "read_length": md2.get("read_length", {"2": None}).get("2", None),
            "run_type": md2.get("run_type", {"2": "single-ended"}).get("2", "single-ended")
        }
        
        return loaded_data, loaded_metadata
        
    except Exception as e:
        print(f"Error loading control for {bios_name}: {e}")
        return None, None
```

**Verify**: Load control for a biosample and check it returns data or None.

---

## Task 2: Add `make_bios_tensor_Control()` and `make_region_tensor_Control()`

**File**: `data.py` | **Location**: After `make_bios_tensor_Peaks()` (line ~821) | **Lines**: ~70

```python
def make_bios_tensor_Control(self, loaded_data, loaded_metadata, missing_value=-1):
    """Format control data for ONE biosample into tensors."""
    if loaded_data and "chipseq-control" in loaded_data:
        L = len(loaded_data["chipseq-control"])
        
        dtensor = loaded_data["chipseq-control"].reshape(-1, 1)  # (L, 1)
        
        meta = loaded_metadata["chipseq-control"]
        run_type_str = str(meta['run_type']).lower()
        runt = 0 if "single" in run_type_str else (1 if "pair" in run_type_str else 0)
        readl = meta['read_length'] if meta['read_length'] is not None else 50
        platform_id = self.sequencing_platform_to_id.get(str(meta['sequencing_platform']), 0)
        
        mdtensor = np.array([[np.log2(meta['depth'])], [platform_id], [readl], [runt]])  # (4, 1)
        availability = np.array([1])  # (1,)
        
    else:
        L = 100  # Will be overridden when stacked with actual data
        dtensor = np.full((L, 1), missing_value)
        mdtensor = np.full((4, 1), missing_value)
        availability = np.array([0])
    
    dtensor = torch.tensor(dtensor).float()
    mdtensor = torch.tensor(mdtensor).float()
    availability = torch.tensor(availability).float()
    
    return dtensor, mdtensor, availability


def make_region_tensor_Control(self, loaded_data_list, loaded_metadata_list):
    """Stack control tensors from MULTIPLE biosamples into a batch."""
    data, metadata, availability = [], [], []
    
    for i in range(len(loaded_data_list)):
        d, md, avl = self.make_bios_tensor_Control(loaded_data_list[i], loaded_metadata_list[i])
        data.append(d)
        metadata.append(md)
        availability.append(avl)
    
    data = torch.stack(data)          # (B, L, 1)
    metadata = torch.stack(metadata)  # (B, 4, 1)
    availability = torch.stack(availability)  # (B, 1)
    
    return data, metadata, availability
```

**Verify**: Test with mock data - shapes should be `(L,1)`, `(4,1)`, `(1,)` for single biosample and `(B,L,1)`, `(B,4,1)`, `(B,1)` for batch.

---

## Task 3: Update `get_batch()` to Return Control Data

**File**: `data.py` | **Location**: Modify `get_batch()` starting line 1255 | **Lines**: ~30 added

Find the `else: # side == 'x'` block (currently around line 1309) and replace it:

```python
    else: # side == 'x'
        # Load control data for each biosample in the batch
        batch_bios_list = list(self.navigation.keys())[self.bios_pointer: self.bios_pointer + self.bios_batchsize]
        current_dsf = self.dsf_list[self.dsf_pointer]
        
        # Load control data for each locus
        control_locus_tensors = []
        for locus in batch_loci_list:
            control_loaded_data = []
            control_loaded_metadata = []
            
            for bios in batch_bios_list:
                ctrl_data, ctrl_meta = self.load_bios_Control(bios, locus, DSF=current_dsf)
                control_loaded_data.append(ctrl_data)
                control_loaded_metadata.append(ctrl_meta)
            
            ctrl_tensor = self.make_region_tensor_Control(control_loaded_data, control_loaded_metadata)
            control_locus_tensors.append(ctrl_tensor)
        
        # Concatenate across loci
        batch_control_data = torch.cat([t[0] for t in control_locus_tensors], dim=0)        # (B, L, 1)
        batch_control_metadata = torch.cat([t[1] for t in control_locus_tensors], dim=0)    # (B, 4, 1)
        batch_control_availability = torch.cat([t[2] for t in control_locus_tensors], dim=0) # (B, 1)
        
        # Return with control data included (7 values instead of 4)
        return batch_data, batch_metadata, batch_availability, one_hot_sequences, \
               batch_control_data, batch_control_metadata, batch_control_availability
```

**Verify**: Call `get_batch(side='x')` and check it returns 7 values with correct shapes.

---

## Task 4: Update `CANDIIterableDataset.__iter__()`

**File**: `data.py` | **Location**: Line ~1428 | **Lines**: ~10 modified

Update unpacking and sample dict:

```python
# Around line 1428 - UPDATE THIS LINE:
x_batch = self.get_batch(side="x")
if x_batch is None: break
x_data, x_meta, x_avail, x_dna, control_data, control_meta, control_avail = x_batch  # Changed from 4 to 7 values

# Around line 1436 - UPDATE SAMPLE DICT:
sample = {
    "sample_id": sample_id,
    "x_data": x_data.squeeze(0), "x_meta": x_meta.squeeze(0),
    "x_avail": x_avail.squeeze(0), "x_dna": x_dna.squeeze(0),
    "control_data": control_data.squeeze(0),       # NEW
    "control_meta": control_meta.squeeze(0),       # NEW
    "control_avail": control_avail.squeeze(0),     # NEW
    "y_data": y_data.squeeze(0), "y_meta": y_meta.squeeze(0),
    "y_avail": y_avail.squeeze(0), "y_pval": y_pval.squeeze(0),
    "y_peaks": y_peaks.squeeze(0), "y_dna": y_dna.squeeze(0),
}
```

**Verify**: Check DataLoader batch has 13 keys (was 10).

---

## Task 5: Update `_process_batch()` in `train.py`

**File**: `train.py` | **Location**: Line ~352 | **Lines**: ~9 added

Add control extraction after line 356:

```python
# Around line 352-362 - ADD THESE 3 LINES:
x_data = batch['x_data'].float()
x_meta = batch['x_meta'].float()
x_avail = batch['x_avail']
x_dna = batch['x_dna'].float()

control_data = batch['control_data'].float()   # NEW
control_meta = batch['control_meta'].float()   # NEW
control_avail = batch['control_avail']         # NEW

y_data = batch['y_data'].float()
# ... rest of extraction ...
```

Add concatenation after masking (around line 396):

```python
# After this line: x_data_masked, x_meta_masked, x_avail_masked = self.masker.mask_assays(...)
# ADD THESE 3 LINES:
x_data_masked = torch.cat([x_data_masked, control_data], dim=2)      # (B, L, F+1)
x_meta_masked = torch.cat([x_meta_masked, control_meta], dim=2)      # (B, 4, F+1)
x_avail_masked = torch.cat([x_avail_masked, control_avail], dim=1)   # (B, F+1)
```

**Verify**: Add debug print after concatenation: `print(f"Shape after control concat: {x_data_masked.shape}")` - should show F+1 features.

---

## Task 6: Update `_validate_batch()` in `train.py`

**File**: `train.py` | **Location**: Line ~1394 | **Lines**: 1 modified

```python
# Update expected_keys to include control data:
expected_keys = {'x_data', 'x_meta', 'x_avail', 'x_dna',
                 'control_data', 'control_meta', 'control_avail',  # NEW
                 'y_data', 'y_meta', 'y_avail', 'y_pval', 'y_peaks', 'y_dna'}
```

**Verify**: No "Missing keys in batch" warnings during training.

---

## Task 7: Update Signal Dimension in `train.py`

**File**: `train.py` | **Location**: Line ~2054 | **Lines**: 2 modified

```python
# After this line: signal_dim = len(temp_dataset.aliases['experiment_aliases'])
# ADD THIS LINE:
signal_dim = signal_dim + 1  # +1 for chipseq-control
print(f"Signal dimension: {signal_dim} (F={signal_dim-1} experiments + 1 control)")
```

**Verify**: Check printed signal dimension is F+1 (e.g., 36 for EIC = 35+1).

---

## Task 8: Add Test Function (Optional)

**File**: `data.py` | **Location**: End of file before `if __name__` (line ~1664) | **Lines**: ~25

```python
def test_control_integration():
    """Test control data loading and shapes."""
    print("--- Testing Control Integration ---")
    set_global_seed(42)
    base_path = "/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"
    
    handler = CANDIDataHandler(base_path=base_path, resolution=25, dataset_type="merged")
    handler.setup_datalooper(m=5, context_length=600*25, bios_batchsize=2, loci_batchsize=1)
    handler.new_epoch()
    
    x_batch = handler.get_batch(side="x")
    x_data, x_meta, x_avail, x_dna, ctrl_data, ctrl_meta, ctrl_avail = x_batch
    
    print(f"✓ Regular data: {x_data.shape}, {x_meta.shape}, {x_avail.shape}")
    print(f"✓ Control data: {ctrl_data.shape}, {ctrl_meta.shape}, {ctrl_avail.shape}")
    
    x_concat = torch.cat([x_data, ctrl_data], dim=2)
    print(f"✓ After concat: {x_concat.shape} (expected: B, L, F+1)")
    print(f"✓ Controls available: {ctrl_avail.sum().item()}/{ctrl_avail.shape[0]}")
```

**Run**: `python -c "from data import test_control_integration; test_control_integration()"`

---

## Task 9: End-to-End Training Test

**Command**: 
```bash
python train.py --merged --epochs 1 --batch-size 4 --num-loci 10 --no-save
```

**Look for**:
1. No shape mismatch errors
2. Model initializes with signal_dim = F+1
3. Training completes first epoch
4. Batch logs show metrics without errors

---

## Implementation Checklist

- [ ] Task 1: `load_bios_Control()` method
- [ ] Task 2: `make_bios_tensor_Control()` and `make_region_tensor_Control()` methods
- [ ] Task 3: Update `get_batch()` to return 7 values
- [ ] Task 4: Update `__iter__()` to unpack 7 values
- [ ] Task 5: Extract control and concatenate in `_process_batch()`
- [ ] Task 6: Update `_validate_batch()` expected keys
- [ ] Task 7: Add +1 to signal_dim
- [ ] Task 8: Test with `test_control_integration()` (optional)
- [ ] Task 9: Run end-to-end training test

## Quick Verification

```bash
# Test data loading
python -c "from data import CANDIDataHandler; h = CANDIDataHandler('DATA_CANDI_MERGED/', dataset_type='merged'); h.setup_datalooper(m=5, context_length=15000, bios_batchsize=2, loci_batchsize=1); h.new_epoch(); x = h.get_batch('x'); print(f'Returns {len(x)} values'); print(f'Shapes: {[t.shape for t in x]}')"

# Test training
python train.py --merged --epochs 1 --batch-size 2 --num-loci 5 --no-save
```

## Critical Points

1. **Task 3 & 4 are paired** - `get_batch` returns 7 values, `__iter__` must unpack 7 values
2. **Task 5 is key** - Concatenation MUST happen after masking, not before
3. **Task 7 is simple** - Just add `+ 1` to one line
4. **Control shapes** - Metadata is `(B, 4, 1)` NOT `(B, 1, 4)` - matches x_meta pattern

## Rollback

Comment out in `get_batch()`:
```python
# batch_control_data = ...  # COMMENT THIS BLOCK
# Return original 4 values
return batch_data, batch_metadata, batch_availability, one_hot_sequences
```

Comment out in `_process_batch()`:
```python
# control_data = batch['control_data'].float()  # COMMENT OUT
# ... concatenation lines ...  # COMMENT OUT
```

Remove from signal_dim:
```python
signal_dim = len(temp_dataset.aliases['experiment_aliases'])  # Remove + 1
```
