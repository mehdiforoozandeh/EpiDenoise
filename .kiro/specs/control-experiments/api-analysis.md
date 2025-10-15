# ENCODE API Structure Analysis for Control Experiment Discovery

## Date: 2024
## Purpose: Understand ENCODE JSON API to implement smart control selection

---

## 1. Experiment JSON Structure

### Example: ENCSR000EVL (ChIP-seq experiment)

**Key Fields for Control Discovery:**

```json
{
  "assay_term_name": "ChIP-seq",
  "accession": "ENCSR000EVL",
  "assembly": ["hg19", "GRCh38"],
  "date_released": "2011-10-29",
  
  "possible_controls": [
    {
      "accession": "ENCSR000EZM",
      "assembly": ["hg19", "GRCh38"],
      "date_released": "2011-10-29",
      "control_type": "input library",
      "description": "Control ChIP-seq on human HeLa-S3",
      "files": ["/files/ENCFF000XGC/", ...],
      "replicates": ["/replicates/88c483cb-1ba6-4d79-bc1d-9b8892589f00/"]
    }
  ],
  
  "replicates": [
    {
      "biological_replicate_number": 1,
      "technical_replicate_number": 1,
      "library": {
        "biosample": {...}
      },
      "antibody": {...}
    }
  ]
}
```

### Key Observations:

1. **possible_controls** field:
   - List of control experiment objects (not just references)
   - Each control has full metadata including `assembly`, `date_released`
   - Multiple controls may be listed

2. **Assembly matching**:
   - `assembly` field is a list (e.g., ["hg19", "GRCh38"])
   - Need to check if target assembly (GRCh38) is IN this list

3. **Date fields for freshness**:
   - `date_released`: Release date (e.g., "2011-10-29")
   - Can use this to select newest control

4. **Replicate structure**:
   - Replicates are objects with detailed metadata
   - Each replicate has `biological_replicate_number` and `technical_replicate_number`
   - Replicates may have their own control references (need to investigate further)

---

## 2. Checking Replicate-Level Controls

**Investigation needed:** Do replicates have their own `possible_controls` field?

From the sample data, replicates have:
- `library` (with biosample info)
- `antibody` (for ChIP-seq)
- But no visible `possible_controls` field at replicate level

**Conclusion**: For simplicity, we'll focus on experiment-level controls from `possible_controls` field. If that's empty, return None.

**Updated approach** (per user's clarification):
1. Check experiment-level `possible_controls`
2. If empty, check each replicate for replicate-level controls
3. If still none, return None

---

## 3. Control Experiment JSON Structure

### Example: ENCSR000EZM (Control experiment)

**Key Fields:**

```json
{
  "accession": "ENCSR000EZM",
  "assembly": ["hg19", "GRCh38"],
  "date_released": "2011-10-29",
  "control_type": "input library",
  "files": [
    "/files/ENCFF000XGC/",  // References to files
    "/files/ENCFF000XGE/",
    ...
  ],
  "original_files": [
    "/files/ENCFF000XGC/",
    ...
  ]
}
```

### Key Observations:

1. **files** field contains file references (paths like "/files/ENCFF000XGC/")
2. Need to query each file individually to get full metadata
3. Files are NOT embedded as objects in the experiment JSON

---

## 4. File JSON Structure

### Example: Querying https://www.encodeproject.org/files/ENCFF000XGC/?format=json

**Key Fields:**

```json
{
  "accession": "ENCFF000XGC",
  "file_format": "bigWig",      // We need "bam"
  "output_type": "signal",      // We need "alignments"
  "assembly": "hg19",           // We need "GRCh38"
  "status": "archived",         // We need "released"
  "file_size": 381302983,
  "date_created": "2010-12-02T00:00:00.000000+00:00",
  "biological_replicates": []
}
```

### Filtering Criteria for BAM Files:

To find the correct BAM file in a control experiment:

1. ✅ `file_format == "bam"`
2. ✅ `assembly == "GRCh38"` (or target assembly)
3. ✅ `output_type == "alignments"` (preferred) or "unfiltered alignments"
4. ✅ `status == "released"`

### Additional considerations:
- `date_created` can be used if multiple BAMs match (choose newest)
- `file_size` helps validate downloads
- `biological_replicates` shows which replicates this file belongs to

---

## 5. Smart Control Selection Algorithm

Based on the API analysis, here's the selection logic:

### Step 1: Get Controls List
```python
possible_controls = experiment_json.get('possible_controls', [])

# If no experiment-level controls, check replicates
if not possible_controls:
    for replicate in experiment_json.get('replicates', []):
        # Query replicate JSON to check for replicate-level controls
        replicate_json = fetch_replicate(replicate_id)
        replicate_controls = replicate_json.get('possible_controls', [])
        possible_controls.extend(replicate_controls)

# If still no controls, return None
if not possible_controls:
    return None
```

### Step 2: Filter by Assembly
```python
target_assembly = "GRCh38"
matching_controls = []

for control in possible_controls:
    control_assemblies = control.get('assembly', [])
    if target_assembly in control_assemblies:
        matching_controls.append(control)

if not matching_controls:
    return None  # No controls match target assembly
```

### Step 3: Select Newest Control
```python
# Sort by date_released (newest first)
matching_controls.sort(
    key=lambda c: c.get('date_released', ''), 
    reverse=True
)

selected_control = matching_controls[0]
control_accession = selected_control['accession']
```

### Step 4: Find BAM File in Control
```python
# Get control experiment files
control_json = fetch_experiment(control_accession)
file_refs = control_json.get('files', [])

# Query each file to find matching BAM
for file_ref in file_refs:
    file_accession = file_ref.split('/')[-2]  # Extract from "/files/ENCFF.../
    file_json = fetch_file(file_accession)
    
    if (file_json.get('file_format') == 'bam' and
        file_json.get('assembly') == target_assembly and
        file_json.get('output_type') in ['alignments', 'unfiltered alignments'] and
        file_json.get('status') == 'released'):
        
        return {
            'control_exp_accession': control_accession,
            'control_bam_accession': file_accession
        }

return None  # No suitable BAM found
```

---

## 6. Date Comparison Logic

For selecting the newest control:

```python
from datetime import datetime

def parse_date(date_str):
    """Parse ENCODE date format: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS...'"""
    if not date_str:
        return datetime.min
    
    # Handle both date-only and datetime formats
    date_part = date_str.split('T')[0]
    return datetime.strptime(date_part, '%Y-%m-%d')

# Compare dates
date1 = parse_date(control1.get('date_released'))
date2 = parse_date(control2.get('date_released'))

if date1 > date2:
    # control1 is newer
    pass
```

---

## 7. API Endpoints Summary

1. **Experiment metadata:**
   ```
   https://www.encodeproject.org/experiments/{exp_accession}/?format=json
   ```

2. **Replicate metadata:**
   ```
   https://www.encodeproject.org/replicates/{replicate_id}/?format=json
   ```

3. **File metadata:**
   ```
   https://www.encodeproject.org/files/{file_accession}/?format=json
   ```

4. **File download URL:**
   ```
   https://www.encodeproject.org/files/{file_accession}/@@download/{file_accession}.bam
   ```

---

## 8. Error Handling Considerations

1. **API failures**: Use retry logic with exponential backoff
2. **Missing fields**: Check field existence before accessing
3. **Empty lists**: Handle empty `possible_controls`, `files`, etc.
4. **Invalid dates**: Use datetime.min for missing dates
5. **Network timeouts**: Set reasonable timeout (30s)

---

## 9. Implementation Notes

### Simplifications:
- Focus on experiment-level `possible_controls` first
- If needed, check replicate-level controls
- Select first control that matches criteria (assembly + newest)
- Prefer `output_type == "alignments"` over other types

### Not Implemented:
- Validation that control and experiment have compatible parameters
- Multiple controls support (just select one)
- Caching of API responses (do fresh queries each time)

---

## 10. Example Test Cases

### Test Case 1: Standard ChIP-seq with Control
- Experiment: ENCSR000EVL
- Expected control: ENCSR000EZM
- Expected BAM: Should find GRCh38 alignments BAM

### Test Case 2: No Controls Available
- Experiment: [Need to find example]
- Expected: Return None

### Test Case 3: Multiple Controls (Different Assemblies)
- Experiment: [Need to find example]
- Expected: Select control matching GRCh38

### Test Case 4: Multiple Controls (Same Assembly)
- Experiment: [Need to find example]
- Expected: Select newest by date_released

---

## Validation Checklist

- [x] Can navigate experiment JSON to find possible_controls
- [x] Can identify assembly field in controls
- [x] Can determine newest control by date_released
- [x] Can navigate control experiment JSON to find files
- [x] Can query file JSON to get file metadata
- [x] Can filter files by format, assembly, output_type, status
- [ ] Need to check replicate-level controls structure (TODO)
- [x] Understand date comparison logic

---

## Next Steps

1. Implement `find_control_for_chipseq()` function based on this analysis
2. Test with real ChIP-seq experiments from EIC and MERGED datasets
3. Handle edge cases (no controls, multiple controls, etc.)
4. Add logging for debugging

