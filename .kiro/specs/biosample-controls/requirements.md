# Biosample-Level Control Processing - Requirements

## Overview

Replace experiment-level control processing with biosample-level control processing. Controls are treated as separate "chipseq-control" experiments that are shared across all ChIP-seq experiments within the same biosample.

## Problem Statement

**Current Inefficiency:**
- Multiple ChIP-seq experiments per biosample download the same control BAM file
- Redundant API calls for control discovery
- Wasted bandwidth and storage
- Example: `right_lobe_of_liver_nonrep` has 7 ChIP-seq experiments but downloads the same control 7 times

**Solution:**
- Remove all experiment-level control code
- Treat controls as separate "chipseq-control" experiments
- One control per biosample, shared across all ChIP-seq experiments

## Key Requirements

### R1: Two CLI Commands
1. **`process`**: Process full dataset including ChIP-seq controls as separate experiments
2. **`add-controls`**: Add ChIP-seq controls to existing dataset

### R2: Biosample-Level Control Discovery
- Group ChIP-seq experiments by `biosample_name`
- Handle multiple `bios_accession` per biosample
- Select best control using priority: (1) most ChIP-seq experiments, (2) most recent

### R3: Control Selection Logic
When multiple `bios_accession` exist for same biosample:
1. Count ChIP-seq experiments per bios_accession
2. Select bios_accession with most ChIP-seq experiments
3. If tied, select most recent control by `date_released`

### R4: Control as Separate Experiment
- Download control BAM to `biosample/chipseq-control/`
- Process control with all DSF values (1,2,4,8)
- Treat "chipseq-control" as regular experiment in dataset

### R5: Directory Structure
```
biosample/
├── H3K4me2/                    # ChIP-seq experiment
├── H3K9me3/                    # ChIP-seq experiment
├── chipseq-control/            # Control experiment (separate)
│   ├── signal_DSF1_res25/
│   ├── signal_DSF2_res25/
│   ├── signal_DSF4_res25/
│   ├── signal_DSF8_res25/
│   └── file_metadata.json
└── ATAC-seq/                   # Non-ChIP-seq (no control)
```

## Success Criteria

- **5x reduction** in control API calls (200 vs 1000)
- **5x reduction** in control downloads (200 vs 1000)
- **5x reduction** in control processing time
- Controls treated as separate experiments in dataset
- Clean separation between dataset processing and control addition

## Out of Scope

- Experiment-level control processing (removed entirely)
- Symlinks or backward compatibility (controls are separate experiments)
- Changes to existing download plan JSON structure
- Processing controls for non-ChIP-seq assays
