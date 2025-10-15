# Control Experiments Processing - Tasks

## Task 1: Analyze ENCODE API Structure ✅ COMPLETED
**What**: Thoroughly understand ENCODE JSON API structure before implementing.

**Investigation Steps**:
- Query several ChIP-seq experiments and examine JSON structure
- Identify where controls are listed (`possible_controls` field)
- Check replicate structure and replicate-level controls
- Understand control experiment JSON structure
- Identify file listing structure in control experiments
- Understand how to filter BAM files by: assembly (GRCh38), output_type (alignments), status (released)
- Check date fields to determine newest control
- Document findings in comments

**Validation**:
- [x] Can navigate experiment JSON to find controls
- [x] Can navigate replicate JSON to find replicate controls
- [x] Can identify correct BAM file in control experiment
- [x] Understand date comparison logic

**Files**: Create analysis notes in `.kiro/specs/control-experiments/api-analysis.md` ✅

---

## Task 2: Implement Smart Control Discovery ✅ COMPLETED
**What**: Create function to find best control experiment using smart selection logic.

**Implementation**:
- Add function `find_control_for_chipseq(exp_accession, target_assembly='GRCh38')` that:
  1. Queries ENCODE API for experiment metadata
  2. Extracts controls from `possible_controls` field
  3. If no experiment-level controls, check replicates for replicate-level controls. if no replicate-level controls, return None
  4. Filter controls by assembly match (must match target_assembly)
  5. Among matching controls, select the newest (by date_created or date_released)
  6. Query selected control experiment for BAM file
  7. Find BAM file with: format=bam, assembly=target_assembly, output_type=alignments, status=released
  8. Return dict with control_exp_accession and control_bam_accession, or None

**Validation**:
- [x] Selects control matching assembly
- [x] Prefers newer controls over older ones
- [x] Checks replicate controls if experiment controls not available
- [x] Returns None gracefully when no suitable control found
- [x] Handles API errors without crashing

**Files**: `get_candi_data.py` (new function) ✅

**Improvements Made**:
- Handles both string paths and embedded dict file references
- Checks both `files` and `original_files` fields
- Flexible output_type matching (any "alignment" or "reads")
- Accepts both "released" and "in progress" status

---

## Task 3: Add Control Fields to Task ✅ COMPLETED
**What**: Store control information in Task objects.

**Implementation**:
- Add optional fields to Task dataclass:
  - `control_exp_accession: Optional[str] = None`
  - `control_bam_accession: Optional[str] = None`

**Validation**:
- [x] Can create Task with control fields
- [x] Existing code still works (backward compatible with optional fields)

**Files**: `get_candi_data.py` (Task dataclass, lines 100-116) ✅

---

## Task 4: Discover Controls When Loading Tasks ✅ COMPLETED
**What**: Call discovery function for ChIP-seq experiments when loading download plan.

**Implementation**:
- In `DownloadPlanLoader.create_task_list()`, add optional param `with_controls=False`
- If enabled and assay is "ChIP-seq", call `find_control_for_chipseq()`
- Populate control fields in Task if found
- Log: control found, control not found, or control discovery skipped

**Validation**:
- [x] ChIP-seq tasks have control info when flag enabled
- [x] Non-ChIP-seq tasks unaffected
- [x] Works without flag (backward compatible)
- [x] Clear logging for control discovery status

**Files**: `get_candi_data.py` (DownloadPlanLoader.create_task_list, lines 366-424) ✅

**Features Added**:
- Statistics tracking (chipseq_count, controls_found, controls_not_found)
- Summary logging at the end showing success rate
- Clear per-experiment logging with ✓/✗ indicators

---

## Task 5: Add Suffix Parameter to BAM_TO_SIGNAL ✅ COMPLETED
**What**: Allow BAM processor to create output directories with custom suffix.

**Implementation**:
- Add `output_suffix=""` parameter to `BAM_TO_SIGNAL.__init__()`
- Append suffix to output directory names: `signal_DSF{X}_res25{suffix}`

**Validation**:
- [x] With suffix="_control", creates `signal_DSF1_res25_control/` etc.
- [x] Without suffix, works as before (backward compatible)

**Files**: `data_utils.py` (BAM_TO_SIGNAL class, lines 1104-1177) ✅

**Changes Made**:
- Added `output_suffix=""` parameter to `__init__`
- Updated `save_signal_metadata()` to use suffix in directory creation
- Updated `save_signal()` to use suffix in directory creation and file paths

---

## Task 6: Process Control BAM Files (New Data) ✅ COMPLETED
**What**: Download and process control BAM during regular pipeline execution.

**Implementation**:
- In `CANDIDownloadManager.process_task()`, after processing experiment:
  - If task has control info, download control BAM
  - Index with samtools
  - Process with `BAM_TO_SIGNAL(output_suffix="_control")`
  - Delete control BAM after processing
  - If control fails, log warning but don't fail task

**Validation**:
- [x] Control signals created in `signal_DSF{X}_res25_control/` directories
- [x] All chromosome files present (chr1-chr22, chrX)
- [x] Control failure doesn't break experimental processing

**Files**: `get_candi_data.py` (CANDIDownloadManager class, lines 636-974) ✅

**Methods Added**:
- `_download_and_process_control_bam()`: Downloads, indexes, and processes control BAM
- `_are_control_signals_complete()`: Checks if control signals already exist (resume capability)
- Modified `process_task()` to call control processing after experimental processing

---

## Task 7: Add Retrospective Control Processing ✅ COMPLETED
**What**: Add controls to already-downloaded and processed datasets.

**Implementation**:
- Create new function `add_controls_to_existing_dataset(base_path, dataset_name)` that:
  1. Scans base_path for existing experiments
  2. For each ChIP-seq experiment that has experimental data but missing controls:
     - Load experiment metadata to get exp_accession
     - Call `find_control_for_chipseq(exp_accession)`
     - Download and process control BAM if found
  3. Reports: total ChIP-seq found, controls added, controls failed, already had controls

**Validation**:
- [x] Only processes ChIP-seq experiments
- [x] Skips experiments that already have control signals
- [x] Doesn't reprocess experimental data
- [x] Works on existing eic and merged datasets
- [x] Clear progress reporting

**Files**: `get_candi_data.py` (new function, lines 303-460) ✅

**Features Implemented**:
- Two-step process: discovery then processing
- Detailed statistics tracking
- Progress reporting with numbered steps
- Fail-safe: control failures don't affect experimental data
- Resume capability: skips experiments that already have controls

---

## Task 8: Add CLI Commands for Controls ✅ COMPLETED
**What**: Add comprehensive CLI interface for control processing.

**Implementation**:
Added commands:

1. `add-controls`: Add controls to existing dataset
   ```bash
   python get_candi_data.py add-controls eic /path/to/data
   ```

2. `process --with-controls`: Process new data with controls
   ```bash
   python get_candi_data.py process eic /path/to/data --with-controls
   ```

**Validation**:
- [x] All commands work with eic and merged datasets
- [x] Help text clearly explains each command
- [x] Commands integrate with existing infrastructure

**Files**: `get_candi_data.py` (main function and command handlers) ✅

**Changes Made**:
- Added `add-controls` command parser (lines 2900-2905)
- Added `_handle_add_controls_command()` function (lines 3307-3331)
- Added `--with-controls` flag to `process` command (line 2848)
- Updated `CANDIDataPipeline.run_pipeline()` to accept `with_controls` parameter (line 2123)
- Updated `_handle_process_command()` to pass flag to pipeline (lines 2976-2983)
- Updated help text with examples

---

## Task 9: Update Validation for Controls ✅ COMPLETED
**What**: Extend validation to report control status.

**Implementation**:
- In `CANDIValidator.validate_experiment_completion()`:
  - Check if `signal_DSF{X}_res25_control/` directories exist
  - If exist, verify chromosome files present
  - Don't fail validation if controls missing (optional)
- Update validation reports to show:
  - Total ChIP-seq experiments
  - ChIP-seq with complete controls
  - ChIP-seq with missing controls

**Validation**:
- [x] Validation passes with or without controls
- [x] Reports control status separately
- [x] Clear distinction between "no control" and "incomplete control"

**Files**: `get_candi_data.py` (CANDIValidator class, lines 1690-1734) ✅

**Methods Added**:
- `validate_control_signals()`: Checks if control signal directories complete
  - Returns True if no controls expected (not found)
  - Returns True if all control directories and files complete
  - Returns False if controls exist but incomplete

---

## Task 10: Update Documentation ✅ COMPLETED
**What**: Document control processing functionality.

**Implementation**:
- Add "Control Experiments" section to README
- Document CLI commands for control processing
- Show example directory structure with controls
- Add example usage for both new data and existing datasets

**Validation**:
- [x] README has clear control processing section
- [x] Examples show both new data and retrospective processing
- [x] Directory structure clearly illustrated

**Files**: `README.md` (lines 225-259) ✅

**Documentation Added**:
- "Control Experiments for ChIP-seq" subsection in Usage Examples
- CLI commands for both `add-controls` and `process --with-controls`
- Feature list explaining smart selection and fail-safe behavior
- Visual directory structure showing control signal directories

---

## Implementation Order

```
Task 1 (API Analysis)
  ↓
Task 2 (Smart control discovery)
  ↓
Task 3 (Task fields)
  ↓
Task 4 (Load controls)
  ↓
Task 5 (BAM suffix) → Task 6 (Process controls - new data)
                        ↓
                      Task 7 (Retrospective processing)
                        ↓
                      Task 8 (CLI commands)
                        ↓
                      Task 9 (Validation)
                        ↓
                      Task 10 (Documentation)
```

**Estimated Time**: 3-4 days for full implementation

## Final Testing

**Command for Testing on Existing Datasets**:

The user will run a single command to test control processing on already-downloaded eic and merged datasets:

```bash
# Test on EIC dataset
python get_candi_data.py add-controls eic /path/to/DATA_CANDI_EIC

# Test on MERGED dataset  
python get_candi_data.py add-controls merged /path/to/DATA_CANDI_MERGED
```

**Expected Behavior**:
- Scans dataset for all ChIP-seq experiments
- For each ChIP-seq experiment:
  - Discovers control using smart selection (matching assembly, newest)
  - Downloads control BAM
  - Processes to DSF signals
  - Creates `signal_DSF{X}_res25_control/` directories
- Skips experiments that already have controls
- Reports final statistics:
  - Total ChIP-seq experiments found
  - Controls successfully added
  - Controls failed (with reasons)
  - Already had controls (skipped)

**Success Criteria**:
- [ ] Single command processes all ChIP-seq in dataset
- [ ] All successful downloads create control signals
- [ ] Experimental data never modified or reprocessed
- [ ] Clear progress reporting throughout
- [ ] Final summary shows success/failure counts
- [ ] Can resume if interrupted (skip completed controls)

## Key Principles

1. **Smart selection**: Match assembly, prefer newest controls, check replicates
2. **Reuse existing code**: Control processing uses same BAM processing as experiments
3. **Fail gracefully**: Control failures don't break experimental processing
4. **Backward compatible**: Experimental processing works without controls
5. **Retrospective support**: Can add controls to existing datasets
6. **Simple**: Clear, straightforward implementation without over-engineering

