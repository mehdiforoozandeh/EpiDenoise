# Biosample-Level Control Processing - Tasks

## Task 1: Remove Experiment-Level Control Code ✅ COMPLETED
**What**: Remove all experiment-level control processing code.

**Implementation**:
- Remove `find_control_for_chipseq()` function
- Remove control fields from `Task` class
- Remove `_download_and_process_control_bam()` method
- Remove experiment-level control validation logic
- Clean up any experiment-level control references

**Validation**:
- [ ] No experiment-level control code remains
- [ ] Task class has no control fields
- [ ] No experiment-level control processing functions
- [ ] Code compiles without errors

**Files**: Modify `get_candi_data.py`

---

## Task 2: Implement Biosample Grouping ✅ COMPLETED
**What**: Group ChIP-seq experiments by biosample_name and select primary bios_accession.

**Implementation**:
- Create `group_chipseq_by_biosample(all_tasks)` function
- Count ChIP-seq experiments per bios_accession within each biosample
- Select bios_accession with most ChIP-seq experiments
- If tied, select most recent by date_released

**Validation**:
- [ ] Correctly groups experiments by biosample_name
- [ ] Selects bios_accession with most ChIP-seq experiments
- [ ] Handles ties by selecting most recent
- [ ] Logs selection reasoning

**Files**: Modify `get_candi_data.py`

---

## Task 3: Implement Biosample Control Discovery ✅ COMPLETED
**What**: Create biosample-level control discovery function.

**Implementation**:
- Create `find_control_for_biosample(biosample_name, primary_bios_accession)` function
- Use primary biosample's first ChIP-seq experiment for control discovery
- Validate control is suitable for all ChIP-seq experiments in biosample
- Return biosample-level control info

**Validation**:
- [ ] Discovers control using primary biosample
- [ ] Validates control compatibility
- [ ] Returns proper biosample control info
- [ ] Handles missing controls gracefully

**Files**: Modify `get_candi_data.py`

---

## Task 4: Implement Control as Separate Experiment Processing ✅ COMPLETED
**What**: Process controls as separate "chipseq-control" experiments.

**Implementation**:
- Create `process_biosample_control(task, base_path)` function
- Download control BAM to `biosample/chipseq-control/`
- Process control with all DSF values (1,2,4,8)
- Create control experiment directory structure
- Treat as regular experiment in dataset

**Validation**:
- [ ] Downloads control BAM to correct location
- [ ] Processes with all DSF values
- [ ] Creates proper directory structure
- [ ] Treats as separate experiment
- [ ] Handles processing failures

**Files**: Modify `get_candi_data.py`

---

## Task 5: Update Process Command ✅ PENDING
**What**: Modify process command to include ChIP-seq controls as separate experiments.

**Implementation**:
- Modify `_handle_process_command()` function
- Add biosample-level control discovery and processing
- Process regular experiments first, then add controls
- Update progress tracking and statistics
- Return combined results

**Validation**:
- [ ] Processes regular experiments
- [ ] Adds ChIP-seq controls as separate experiments
- [ ] Maintains progress tracking
- [ ] Returns correct statistics
- [ ] Handles errors gracefully

**Files**: Modify `get_candi_data.py`

---

## Task 6: Update Add-Controls Command ✅ COMPLETED
**What**: Modify add-controls command to only add ChIP-seq controls as separate experiments.

**Implementation**:
- Modify `add_controls_to_existing_dataset()` function
- Skip regular experiments (already processed)
- Only add ChIP-seq controls as separate experiments
- Update progress tracking and statistics
- Return control processing results

**Validation**:
- [ ] Skips regular experiments
- [ ] Only processes ChIP-seq controls
- [ ] Creates separate control experiments
- [ ] Maintains progress tracking
- [ ] Returns correct statistics

**Files**: Modify `get_candi_data.py`

---

## Task 7: Update Validation Logic ✅ PENDING
**What**: Update validation to include "chipseq-control" as separate experiments.

**Implementation**:
- Modify validation functions to recognize "chipseq-control" experiments
- Update completion checking for control experiments
- Update statistics and reporting
- Handle control experiments in validation pipeline

**Validation**:
- [ ] Recognizes chipseq-control experiments
- [ ] Validates control experiment completeness
- [ ] Updates statistics correctly
- [ ] Handles validation errors

**Files**: Modify `get_candi_data.py`

---

## Task 8: Update Shell Scripts ✅ PENDING
**What**: Update SLURM scripts for optimized resource allocation.

**Implementation**:
- Reduce time estimates due to faster processing
- Update CPU/memory allocation if needed
- Update job names and descriptions
- Test with sample datasets

**Validation**:
- [ ] Reduced time estimates
- [ ] Appropriate resource allocation
- [ ] Updated job descriptions
- [ ] Tested successfully

**Files**: Update `jobs/add_controls_eic.sh`, `jobs/add_controls_merged.sh`

---

## Task 9: Testing and Validation ✅ PENDING
**What**: Test biosample-level control processing with sample datasets.

**Implementation**:
- Test with small subset of EIC dataset
- Validate control discovery and selection
- Validate control processing as separate experiments
- Compare performance with original approach
- Test both process and add-controls commands

**Validation**:
- [ ] Correct control discovery
- [ ] Proper control processing as separate experiments
- [ ] Performance improvement
- [ ] Both commands work correctly
- [ ] No data corruption

**Files**: Create test scripts, validate results

---

## Success Metrics

- **API Calls**: 5x reduction (200 vs 1000)
- **Downloads**: 5x reduction (200 vs 1000)
- **Processing Time**: 5x reduction
- **Storage**: Controls as separate experiments
- **Commands**: Two distinct CLI commands working correctly