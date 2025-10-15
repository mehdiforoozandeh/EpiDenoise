# Biosample-Level Control Processing - Design

## Architecture Overview

Replace experiment-level control processing with biosample-centric approach. Controls are treated as separate "chipseq-control" experiments.

## Core Components

### 1. Biosample Control Discovery
```python
@dataclass
class BiosampleControlTask:
    biosample_name: str
    primary_bios_accession: str        # Selected bios_accession with most ChIP-seq
    control_exp_accession: str
    control_bam_accession: str
    chipseq_experiments: List[str]     # All ChIP-seq assays that will use this control
    status: TaskStatus = TaskStatus.PENDING
```

### 2. Control Selection Algorithm
```python
def select_primary_bios_accession(biosample_name: str, bios_accessions: List[str]) -> str:
    """
    Select bios_accession with most ChIP-seq experiments.
    If tied, select most recent by date_released.
    """
    # Group experiments by bios_accession
    # Count ChIP-seq experiments per bios_accession
    # Return bios_accession with highest count
    # If tied, select by date_released
```

### 3. Control as Separate Experiment Processing
```python
def process_biosample_control(task: BiosampleControlTask, base_path: str):
    """
    Process control as separate experiment in biosample/chipseq-control/.
    """
    control_path = f"{base_path}/{task.biosample_name}/chipseq-control"
    
    # Download control BAM to control_path
    # Process with all DSF values (1,2,4,8)
    # Create control experiment directory structure
    # Treat as regular experiment in dataset
```

### 4. CLI Command Structure
```python
# Command 1: Process full dataset including controls
def process_dataset_with_controls(dataset_name: str, directory: str):
    """
    Process all experiments + add ChIP-seq controls as separate experiments.
    """
    # Process all regular experiments
    # Add ChIP-seq controls as separate experiments
    # Return combined results

# Command 2: Add controls to existing dataset
def add_controls_to_existing_dataset(dataset_name: str, directory: str):
    """
    Add ChIP-seq controls to already processed dataset.
    """
    # Skip regular experiments (already processed)
    # Only add ChIP-seq controls as separate experiments
    # Return control processing results
```

## Processing Pipeline

### Command 1: Process Full Dataset
```
Download Plan → Process Regular Experiments → Group ChIP-seq by Biosample → Select Primary Biosample → Discover Controls → Process Controls as Separate Experiments
```

### Command 2: Add-Controls Only
```
Download Plan → Skip Regular Experiments → Group ChIP-seq by Biosample → Select Primary Biosample → Discover Controls → Process Controls as Separate Experiments
```

### Detailed Phases
1. **Grouping**: Group ChIP-seq experiments by `biosample_name`
2. **Selection**: Select primary `bios_accession` per biosample (most ChIP-seq experiments)
3. **Discovery**: Use primary biosample's first ChIP-seq experiment for control discovery
4. **Processing**: Download control BAM to `biosample/chipseq-control/` and process as separate experiment

## Data Flow

### Process Command
```
Regular Experiments → Biosample Grouping → Control Discovery → Control Processing (as separate experiments)
```

### Add-Controls Command
```
Biosample Grouping → Control Discovery → Control Processing (as separate experiments)
```

## Error Handling

- **Missing Control**: Log warning, skip biosample
- **Download Failure**: Retry once, then skip
- **Processing Failure**: Clean up partial files, skip

## Performance Optimization

- **Parallel Discovery**: Process multiple biosamples simultaneously
- **Parallel Processing**: Process multiple control downloads simultaneously
- **Progress Tracking**: tqdm progress bars for all phases
- **Resource Management**: Limit concurrent downloads to avoid overwhelming ENCODE servers
