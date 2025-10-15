# Control Experiments Processing - Design

## Architecture Overview

The control experiments feature extends the existing CANDI pipeline with minimal changes to the core architecture. The design follows these principles:

1. **Reuse existing infrastructure**: Leverage `Task`, `CANDIDownloadManager`, and `BAM_TO_SIGNAL` components
2. **Separation of concerns**: Keep control logic separate but integrated with experimental processing
3. **Optional feature**: Control processing is opt-in via command-line flag
4. **Fail-safe**: Experimental processing succeeds even if control processing fails

### High-Level Architecture

```
Download Plan JSON
       ↓
Task Creation (with control info)
       ↓
┌──────────────────────────────┐
│  CANDIDownloadManager        │
│  ├─ Process Experiment BAM   │
│  └─ Process Control BAM ←NEW │
└──────────────────────────────┘
       ↓
Signal Directories
├─ signal_DSF{X}_res25/        (experiment)
└─ signal_DSF{X}_res25_control/ (control) ←NEW
```

## Component Design

### 1. Enhanced Task Data Model

**Location**: `get_candi_data.py` (lines 160-189)

**Changes**: Add control-related fields to `Task` dataclass

```python
@dataclass
class Task:
    # Existing fields
    celltype: str
    assay: str
    bios_accession: str
    exp_accession: str
    file_accession: str
    bam_accession: Optional[str]
    # ... existing fields ...
    
    # NEW: Control experiment fields
    control_exp_accession: Optional[str] = None
    control_bam_accession: Optional[str] = None
    has_control: bool = False
    control_status: TaskStatus = TaskStatus.PENDING
    control_error_message: Optional[str] = None
    
    @property
    def is_chipseq(self) -> bool:
        """Check if task is ChIP-seq experiment."""
        return self.assay == "ChIP-seq"
```

**Rationale**: Storing control information in the Task object keeps all experiment-related data together and allows parallel tracking of experimental and control processing status.

### 2. Control Discovery Module

**Location**: New class `ControlExperimentDiscovery` in `get_candi_data.py`

```python
class ControlExperimentDiscovery:
    """Discover and extract control experiment information from ENCODE API."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_base = "https://www.encodeproject.org"
    
    def find_control_for_experiment(self, exp_accession: str) -> Optional[Dict]:
        """
        Query ENCODE API to find control experiment.
        
        Args:
            exp_accession: Experiment accession (e.g., "ENCSR000ABC")
            
        Returns:
            Dict with control info or None if no control found
            {
                'control_exp_accession': str,
                'control_bam_accession': str,
                'control_metadata': dict
            }
        """
        # Query experiment metadata
        exp_metadata = self._fetch_experiment_metadata(exp_accession)
        
        if not exp_metadata:
            return None
        
        # Extract possible_controls field
        possible_controls = exp_metadata.get('possible_controls', [])
        
        if not possible_controls:
            self.logger.warning(f"No controls found for {exp_accession}")
            return None
        
        # Select first available control
        control_exp_path = possible_controls[0]
        control_exp_accession = control_exp_path.split('/')[-2]
        
        # Query control experiment to find BAM file
        control_bam = self._find_bam_in_experiment(control_exp_accession)
        
        if not control_bam:
            self.logger.warning(f"No BAM found in control {control_exp_accession}")
            return None
        
        return {
            'control_exp_accession': control_exp_accession,
            'control_bam_accession': control_bam['accession'],
            'control_metadata': control_bam
        }
    
    def _fetch_experiment_metadata(self, exp_accession: str) -> Dict:
        """Fetch experiment metadata from ENCODE API with retry."""
        url = f"{self.api_base}/experiments/{exp_accession}/?format=json"
        headers = {'accept': 'application/json'}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch {exp_accession}: {e}")
                    return {}
        
        return {}
    
    def _find_bam_in_experiment(self, exp_accession: str) -> Optional[Dict]:
        """
        Find appropriate BAM file in control experiment.
        
        Filters:
        - file_format: "bam"
        - output_type: "alignments" (prefer) or "unfiltered alignments"
        - assembly: "GRCh38"
        - status: "released"
        """
        exp_metadata = self._fetch_experiment_metadata(exp_accession)
        
        if not exp_metadata or 'files' not in exp_metadata:
            return None
        
        # Filter BAM files
        bam_files = []
        for file_obj in exp_metadata['files']:
            # Handle both string paths and dict objects
            if isinstance(file_obj, str):
                file_accession = file_obj.split('/')[-2]
                file_metadata = self._fetch_file_metadata(file_accession)
            else:
                file_metadata = file_obj
            
            # Filter criteria
            if (file_metadata.get('file_format') == 'bam' and
                file_metadata.get('assembly') == 'GRCh38' and
                file_metadata.get('status') == 'released'):
                bam_files.append(file_metadata)
        
        if not bam_files:
            return None
        
        # Prefer "alignments" output type
        for bam in bam_files:
            if bam.get('output_type') == 'alignments':
                return bam
        
        # Otherwise return first available
        return bam_files[0]
    
    def _fetch_file_metadata(self, file_accession: str) -> Dict:
        """Fetch file metadata from ENCODE API."""
        url = f"{self.api_base}/files/{file_accession}/?format=json"
        headers = {'accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch file {file_accession}: {e}")
            return {}
```

**Rationale**: Separate class for control discovery keeps API logic isolated and testable. The discovery process mirrors the approach used for experimental data.

### 3. Enhanced Download Plan Loader

**Location**: Modify `DownloadPlanLoader.create_task_list()` (lines 254-273)

**Changes**: Optionally enrich tasks with control information

```python
class DownloadPlanLoader:
    # ... existing methods ...
    
    def create_task_list(self, discover_controls: bool = False) -> List[Task]:
        """
        Convert download plan to list of Task objects.
        
        Args:
            discover_controls: If True, query ENCODE API for ChIP-seq controls
        """
        tasks = []
        control_discovery = ControlExperimentDiscovery() if discover_controls else None
        
        for celltype, assays in self.download_plan.items():
            for assay, files in assays.items():
                task = Task(
                    celltype=celltype,
                    assay=assay,
                    bios_accession=files['bios_accession'],
                    exp_accession=files['exp_accession'],
                    file_accession=files['file_accession'],
                    bam_accession=files['bam_accession'],
                    tsv_accession=files['tsv_accession'],
                    signal_bigwig_accession=files['signal_bigwig_accession'],
                    peaks_bigbed_accession=files['peaks_bigbed_accession']
                )
                
                # Discover control if ChIP-seq and discovery enabled
                if discover_controls and task.is_chipseq:
                    control_info = control_discovery.find_control_for_experiment(
                        task.exp_accession
                    )
                    if control_info:
                        task.control_exp_accession = control_info['control_exp_accession']
                        task.control_bam_accession = control_info['control_bam_accession']
                        task.has_control = True
                
                tasks.append(task)
        
        return tasks
```

**Rationale**: Control discovery at task creation time allows early failure detection and better progress reporting. The `discover_controls` flag maintains backward compatibility.

### 4. Control BAM Processing

**Location**: New methods in `CANDIDownloadManager` class

```python
class CANDIDownloadManager:
    # ... existing methods ...
    
    def process_task(self, task: Task) -> Task:
        """Enhanced task processing with optional control processing."""
        try:
            # ... existing experimental processing ...
            
            # NEW: Process control if available
            if task.has_control and task.control_bam_accession:
                self.logger.info(f"Processing control for {task.task_id}")
                control_success = self._process_control_bam(task, exp_path)
                
                if control_success:
                    task.control_status = TaskStatus.COMPLETED
                    self.logger.info(f"Control processing completed for {task.task_id}")
                else:
                    task.control_status = TaskStatus.FAILED
                    task.control_error_message = "Control processing failed"
                    # Don't fail entire task if only control fails
                    self.logger.warning(f"Control failed for {task.task_id}, continuing")
            
            # ... existing validation ...
            
        except Exception as e:
            # ... existing error handling ...
        
        return task
    
    def _process_control_bam(self, task: Task, exp_path: str) -> bool:
        """
        Download and process control BAM file.
        
        Args:
            task: Task with control information
            exp_path: Experiment directory path
            
        Returns:
            bool: True if successful
        """
        try:
            control_bam_file = os.path.join(
                exp_path, 
                f"{task.control_bam_accession}_control.bam"
            )
            download_url = (
                f"https://www.encodeproject.org/files/"
                f"{task.control_bam_accession}/@@download/"
                f"{task.control_bam_accession}.bam"
            )
            
            # Check if already processed
            if self._are_control_signals_complete(exp_path):
                self.logger.info(f"Control signals already exist for {task.task_id}")
                return True
            
            # Download control BAM
            self.logger.info(f"Downloading control BAM for {task.task_id}")
            if not download_save(download_url, control_bam_file):
                raise Exception("Failed to download control BAM")
            
            # Index control BAM
            self.logger.info(f"Indexing control BAM for {task.task_id}")
            index_result = os.system(f"samtools index {control_bam_file}")
            if index_result != 0:
                raise Exception(f"Control BAM indexing failed: return code {index_result}")
            
            # Process control BAM to signals
            self.logger.info(f"Processing control BAM to signals for {task.task_id}")
            control_bam_processor = BAM_TO_SIGNAL(
                bam_file=control_bam_file,
                chr_sizes_file=self.chr_sizes_file,
                output_suffix="_control"  # NEW: Add suffix for control outputs
            )
            control_bam_processor.full_preprocess()
            
            # Save control metadata
            self._save_control_metadata(task, exp_path)
            
            # Clean up control BAM files
            os.remove(control_bam_file)
            if os.path.exists(f"{control_bam_file}.bai"):
                os.remove(f"{control_bam_file}.bai")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process control BAM for {task.task_id}: {e}")
            # Clean up on failure
            if os.path.exists(control_bam_file):
                os.remove(control_bam_file)
            if os.path.exists(f"{control_bam_file}.bai"):
                os.remove(f"{control_bam_file}.bai")
            return False
    
    def _are_control_signals_complete(self, exp_path: str) -> bool:
        """Check if all control DSF signal directories are complete."""
        for dsf in self.dsf_list:
            control_dsf_path = os.path.join(
                exp_path, 
                f"signal_DSF{dsf}_res{self.resolution}_control"
            )
            if not self._is_signal_complete(control_dsf_path):
                return False
        return True
    
    def _save_control_metadata(self, task: Task, exp_path: str) -> None:
        """Save control experiment metadata to control_metadata.json."""
        control_metadata = {
            "control_exp_accession": {"2": task.control_exp_accession},
            "control_bam_accession": {"2": task.control_bam_accession},
            "experiment_accession": {"2": task.exp_accession},
            "download_url": {
                "2": f"https://www.encodeproject.org/files/{task.control_bam_accession}/@@download/{task.control_bam_accession}.bam"
            },
            "assay": {"2": task.assay},
            "biosample": {"2": task.bios_accession},
            "assembly": {"2": "GRCh38"}
        }
        
        # Convert to pandas Series for consistent format
        series_data = pd.Series(control_metadata)
        
        metadata_path = os.path.join(exp_path, "control_metadata.json")
        with open(metadata_path, "w") as f:
            f.write(series_data.to_json(indent=4))
```

**Rationale**: Control processing follows the exact same pattern as experimental BAM processing, with a suffix to differentiate outputs. Control failures don't fail the entire task.

### 5. BAM_TO_SIGNAL Modification

**Location**: `data.py` - Modify `BAM_TO_SIGNAL` class

**Changes**: Add optional output suffix parameter

```python
class BAM_TO_SIGNAL:
    def __init__(self, bam_file, chr_sizes_file="data/hg38.chrom.sizes", 
                 output_suffix=""):
        """
        Initialize BAM to signal processor.
        
        Args:
            bam_file: Path to BAM file
            chr_sizes_file: Path to chromosome sizes file
            output_suffix: Suffix to add to output directories (e.g., "_control")
        """
        self.bam_file = bam_file
        self.chr_sizes_file = chr_sizes_file
        self.output_suffix = output_suffix  # NEW
        # ... rest of initialization ...
    
    def full_preprocess(self):
        """Process BAM file to all DSF levels."""
        for dsf in [1, 2, 4, 8]:
            output_dir = os.path.join(
                os.path.dirname(self.bam_file),
                f"signal_DSF{dsf}_res{self.resolution}{self.output_suffix}"  # Use suffix
            )
            # ... rest of processing ...
```

**Rationale**: Adding an optional suffix parameter is the minimal change needed to support control outputs without duplicating the entire BAM processing logic.

### 6. Enhanced Validation

**Location**: Modify `CANDIValidator` class (lines 1091-1302)

**Changes**: Add control signal validation

```python
class CANDIValidator:
    def validate_experiment_completion(self, exp_path):
        """Enhanced validation including control signals."""
        validation_results = {
            "dsf_signals": self.validate_dsf_signals(exp_path),
            "bw_signals": self.validate_bw_signals(exp_path),
            "peaks": self.validate_peaks(exp_path),
            "metadata": self.validate_metadata(exp_path),
            "control_signals": self.validate_control_signals(exp_path),  # NEW
            "overall": False
        }
        
        # Overall validation requires DSF signals and metadata
        # Control signals are optional
        validation_results["overall"] = (
            validation_results["dsf_signals"] and 
            validation_results["metadata"]
        )
        
        return validation_results
    
    def validate_control_signals(self, exp_path):
        """
        Check control DSF signal directories and files.
        
        Returns:
            bool: True if control signals complete, or if no control expected
        """
        # Check if control metadata exists
        control_metadata_path = os.path.join(exp_path, "control_metadata.json")
        if not os.path.exists(control_metadata_path):
            # No control expected
            return True
        
        # If control metadata exists, validate control signals
        for dsf in self.required_dsf:
            dsf_path = os.path.join(
                exp_path, 
                f"signal_DSF{dsf}_res{self.resolution}_control"
            )
            
            if not os.path.exists(dsf_path):
                print(f"Missing control DSF directory: {dsf_path}")
                return False
            
            # Check metadata.json exists
            metadata_file = os.path.join(dsf_path, "metadata.json")
            if not os.path.exists(metadata_file):
                print(f"Missing control metadata file: {metadata_file}")
                return False
            
            # Check all chromosome files exist
            for chr_name in self.main_chrs:
                chr_file = os.path.join(dsf_path, f"{chr_name}.npz")
                if not os.path.exists(chr_file):
                    print(f"Missing control chromosome file: {chr_file}")
                    return False
        
        return True
```

**Rationale**: Control validation is separate from experimental validation. Missing controls don't cause validation failure, but incomplete controls do.

## Data Model

### Task Object Extensions

```python
Task:
  # Existing fields
  - celltype: str
  - assay: str
  - exp_accession: str
  - bam_accession: str
  
  # NEW: Control fields
  - control_exp_accession: Optional[str]
  - control_bam_accession: Optional[str]
  - has_control: bool
  - control_status: TaskStatus
  - control_error_message: Optional[str]
```

### Directory Structure

```
{base_path}/{celltype}/{assay}/
├── file_metadata.json                    # Existing
├── control_metadata.json                 # NEW
├── signal_DSF1_res25/                    # Existing
├── signal_DSF2_res25/                    # Existing
├── signal_DSF4_res25/                    # Existing
├── signal_DSF8_res25/                    # Existing
├── signal_DSF1_res25_control/            # NEW
├── signal_DSF2_res25_control/            # NEW
├── signal_DSF4_res25_control/            # NEW
└── signal_DSF8_res25_control/            # NEW
```

### control_metadata.json Schema

```json
{
  "control_exp_accession": {"2": "ENCSR000XYZ"},
  "control_bam_accession": {"2": "ENCFF123ABC"},
  "experiment_accession": {"2": "ENCSR000ABC"},
  "download_url": {"2": "https://..."},
  "assay": {"2": "ChIP-seq"},
  "biosample": {"2": "ENCBS000AAA"},
  "assembly": {"2": "GRCh38"},
  "file_size": {"2": 12345678},
  "date_created": {"2": 1234567890}
}
```

## API Integration

### ENCODE API Endpoints

1. **Experiment Metadata**
   - URL: `https://www.encodeproject.org/experiments/{exp_accession}/?format=json`
   - Response includes: `possible_controls` array

2. **Control Experiment Metadata**
   - URL: `https://www.encodeproject.org/experiments/{control_exp_accession}/?format=json`
   - Response includes: `files` array with file objects

3. **File Metadata**
   - URL: `https://www.encodeproject.org/files/{file_accession}/?format=json`
   - Response includes: file details, download URL

### Rate Limiting Strategy

```python
class APIRateLimiter:
    """Simple rate limiter for ENCODE API calls."""
    
    def __init__(self, calls_per_second=10):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            min_interval = 1.0 / self.calls_per_second
            
            if time_since_last_call < min_interval:
                time.sleep(min_interval - time_since_last_call)
            
            self.last_call_time = time.time()
```

## Processing Workflow

### Overall Flow

```
1. Load download plan
   ↓
2. Create tasks (optional: discover controls)
   ↓
3. For each ChIP-seq task with control:
   ├─ Download & process experiment BAM
   └─ Download & process control BAM
   ↓
4. Validate outputs (including controls)
   ↓
5. Report completion status
```

### Parallel Processing Strategy

Control processing happens **sequentially** after experimental processing within each task, but tasks themselves are processed in parallel.

**Rationale**: This approach:
- Simplifies error handling (experiment always processes first)
- Avoids disk I/O contention (only one BAM being processed per task)
- Maintains reasonable parallelism (across different tasks)

### Error Handling Hierarchy

```
Level 1: Control Discovery Failure
  → Log warning, mark control as unavailable, continue

Level 2: Control Download Failure
  → Retry 3 times, then mark control as failed, continue

Level 3: Control Processing Failure
  → Log error, mark control as failed, continue
  → Experimental data remains valid

Level 4: Experimental Processing Failure
  → Mark entire task as failed
  → Skip control processing
```

## Command-Line Interface

### New Command: process-with-controls

```bash
# Process datasets with controls
python get_candi_data.py process-with-controls eic /path/to/data

# Options
python get_candi_data.py process-with-controls eic /path/to/data \
    --max-workers 8 \
    --skip-existing \
    --controls-only  # Only process controls for existing experiments
```

### Modified Existing Commands

```bash
# Add --with-controls flag to process command
python get_candi_data.py process eic /path/to/data --with-controls

# Validation includes controls
python get_candi_data.py validate eic /path/to/data
# Output shows:
#   Experiments: 100/100 complete
#   Controls: 85/100 complete (ChIP-seq only)
```

## Implementation Considerations

### Phase 1: Control Discovery (Read-only)
- Implement `ControlExperimentDiscovery` class
- Add control fields to `Task` dataclass
- Test control discovery without processing
- Generate report of available controls

### Phase 2: Single Task Processing
- Modify `BAM_TO_SIGNAL` to support output suffix
- Implement `_process_control_bam` method
- Test on a single ChIP-seq experiment
- Validate control signals are correct

### Phase 3: Parallel Processing Integration
- Integrate control processing into main pipeline
- Add command-line flags
- Test on small dataset (5-10 experiments)
- Verify no regression in experimental processing

### Phase 4: Validation and Retry
- Extend validation to include controls
- Implement retry logic for failed controls
- Test on full dataset
- Performance benchmarking

### Phase 5: Production Deployment
- Update documentation
- Add monitoring and logging
- Deploy to production environment
- Monitor first production run

## Testing Strategy

### Unit Tests

```python
# test_control_discovery.py
def test_find_control_for_chipseq():
    """Test control discovery for ChIP-seq experiment."""
    discovery = ControlExperimentDiscovery()
    control_info = discovery.find_control_for_experiment("ENCSR000EVL")
    assert control_info is not None
    assert 'control_exp_accession' in control_info
    assert 'control_bam_accession' in control_info

def test_no_control_for_atac():
    """Test that non-ChIP-seq experiments don't require controls."""
    # Test implementation
    pass

def test_control_bam_processing():
    """Test control BAM processing creates correct outputs."""
    # Test implementation
    pass
```

### Integration Tests

```python
# test_control_pipeline.py
def test_end_to_end_chipseq_with_control():
    """Test complete pipeline for ChIP-seq with control."""
    # Setup test data
    # Run pipeline
    # Verify experimental and control signals exist
    # Verify validation passes
    pass
```

### Manual Test Cases

1. **ChIP-seq with valid control**: Full processing succeeds
2. **ChIP-seq with no control**: Experimental processing succeeds, control skipped
3. **ChIP-seq with invalid control**: Experimental processing succeeds, control fails gracefully
4. **Non-ChIP-seq assay**: No control processing attempted
5. **Parallel processing**: Multiple tasks process correctly

## Performance Analysis

### Expected Performance Impact

**Control Discovery Phase** (one-time per run):
- API calls: ~1-2 per ChIP-seq experiment
- Time: ~0.5s per experiment (with rate limiting)
- For 100 ChIP-seq experiments: ~50 seconds

**Control Processing Phase** (per experiment):
- Download: Same as experimental BAM (~10-30 min for large files)
- Processing: Same as experimental BAM (~15-45 min)
- Total: ~1.5-2x experimental processing time per ChIP-seq experiment

**Overall Pipeline Impact**:
- For dataset with 50% ChIP-seq: ~1.25x baseline time
- For dataset with 100% ChIP-seq: ~1.5x baseline time
- Meets NFR-1 requirement (<1.5x baseline)

### Optimization Opportunities

1. **Parallel control processing**: Process control in parallel with experiment (complex)
2. **Cached control discovery**: Store control mappings in JSON (simple)
3. **Shared controls**: Reuse control data for experiments with same control (medium)

## Security Considerations

1. **API Authentication**: ENCODE API is public, no authentication required
2. **Input Validation**: Validate all API responses before processing
3. **File System**: Ensure control files written to correct locations
4. **Denial of Service**: Rate limiting prevents API abuse

## Migration and Rollout

### Backward Compatibility

- Default behavior unchanged (controls disabled)
- Existing experiments not affected
- Validation passes for experiments without controls
- No changes to existing download plan JSON files

### Rollout Plan

1. **Week 1**: Deploy to development environment, test on small dataset
2. **Week 2**: Process controls for EIC test dataset, validate results
3. **Week 3**: Process controls for MERGED test dataset, compare with EIC
4. **Week 4**: Full deployment, monitor performance and errors
5. **Week 5+**: Ongoing monitoring, optimization based on real usage

## Open Design Decisions

### Decision 1: Multiple Controls
**Question**: Should we support experiments with multiple controls?

**Options**:
- A: Select first control only (simple, matches current design)
- B: Download all controls (comprehensive, more complex)
- C: Let user specify which control to use (flexible, requires UI changes)

**Recommendation**: Start with A, add B if needed

### Decision 2: Control Reuse
**Question**: Should we reuse control data when multiple experiments share the same control?

**Options**:
- A: Download control once, symlink to other experiments (efficient, complex)
- B: Download control for each experiment (simple, wasteful)
- C: Hybrid: detect duplicates, offer to reuse (balanced)

**Recommendation**: Start with B, implement C in Phase 6

### Decision 3: Control Storage
**Question**: Should we keep control BAM files after processing?

**Options**:
- A: Delete after processing (saves space, matches experimental behavior)
- B: Keep permanently (easier debugging, wastes space)
- C: Configurable via flag (flexible, more complex)

**Recommendation**: A (delete), with option to keep via flag

## Success Metrics

1. **Control Discovery Rate**: >90% of ChIP-seq experiments have controls found
2. **Control Processing Success**: >85% of discovered controls process successfully
3. **Performance**: Pipeline with controls completes in <1.5x baseline time
4. **Validation**: 100% of successful control processing passes validation
5. **Error Recovery**: 95%+ of control failures don't affect experimental data

## Dependencies and Risks

### External Dependencies
- ENCODE API availability and stability
- Network connectivity for API calls and downloads
- Samtools, pysam, pyBigWig libraries

### Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ENCODE API changes | Low | High | Version API calls, monitor ENCODE announcements |
| Control metadata missing | Medium | Medium | Graceful degradation, clear error messages |
| Disk space exhaustion | Medium | High | Monitor usage, cleanup controls after processing |
| Performance degradation | Low | Medium | Benchmark early, optimize if needed |
| Control-experiment mismatch | Low | High | Validate compatibility before processing |

## Future Enhancements

1. **Shared control optimization**: Detect and reuse shared controls
2. **Control quality metrics**: Add QC checks for control data quality
3. **Alternative control sources**: Support user-provided controls
4. **Control-experiment validation**: Verify compatibility of control and experiment
5. **Differential analysis pipeline**: Use controls for peak calling and normalization

