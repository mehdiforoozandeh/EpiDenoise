# Control Experiments Processing - Requirements

## Overview

Extend the CANDI data pipeline to automatically discover, download, and process control experiments for ChIP-seq assays. Control experiments are essential for ChIP-seq analysis as they provide background signal measurements.

## Background

Currently, `get_candi_data.py` downloads and processes experimental BAM files for various assays (ChIP-seq, ATAC-seq, DNase-seq, RNA-seq) from ENCODE. However, ChIP-seq experiments require corresponding control experiments (typically input DNA or IgG controls) for proper peak calling and signal normalization.

## Objectives

1. Identify ChIP-seq experiments in download plan JSON files (EIC and MERGED datasets)
2. Query ENCODE API to find corresponding control experiments for each ChIP-seq experiment
3. Extract control BAM file accessions from control experiments
4. Download control BAM files to the same directory as the experimental data
5. Process control BAM files using the identical pipeline as experimental BAM files (indexing, DSF signal generation)
6. Maintain backward compatibility with existing pipeline functionality

## Scope

### In Scope
- ChIP-seq experiments from both EIC and MERGED datasets
- Control experiment discovery via ENCODE API
- Control BAM file download and processing
- Integration with existing `CANDIDownloadManager` and `Task` infrastructure
- Validation of control experiment completeness
- Metadata tracking for control experiments

### Out of Scope
- Processing controls for non-ChIP-seq assays (ATAC-seq, DNase-seq have different control requirements)
- Reprocessing existing experiments that already have controls
- Advanced control-experiment pairing logic beyond ENCODE's designated controls
- Peak calling or differential analysis (processing only)

## User Stories

### US-1: As a data scientist
I want control experiments automatically downloaded for ChIP-seq data so that I can perform proper peak calling and background normalization without manual intervention.

**Acceptance Criteria:**
- Control experiments are identified for all ChIP-seq experiments
- Control BAM files are downloaded to the experiment directory
- Control BAM files are processed into DSF signals (DSF1, DSF2, DSF4, DSF8)

### US-2: As a pipeline operator
I want the control processing to be optional and configurable so that I can run the pipeline with or without controls based on my needs.

**Acceptance Criteria:**
- Command-line flag to enable/disable control processing
- Default behavior maintains backward compatibility (controls disabled by default)
- Clear logging indicates when controls are being processed

### US-3: As a data validator
I want to verify that control experiments are complete so that I can ensure data quality before analysis.

**Acceptance Criteria:**
- Validation checks include control experiment completeness
- Missing controls are reported in validation summaries
- Retry mechanism works for failed control downloads

## Functional Requirements

### FR-1: Control Experiment Discovery
- Parse download plan JSON files to identify ChIP-seq experiments
- Query ENCODE API for each ChIP-seq experiment's metadata
- Extract `possible_controls` field from experiment metadata
- Select the first available control experiment (or implement selection logic)
- Handle cases where no control is specified

### FR-2: Control BAM File Identification
- Query ENCODE API for control experiment metadata
- Find BAM files in control experiment's file list
- Filter for appropriate file types (alignments, BAM format, GRCh38 assembly)
- Prefer specific output types (e.g., "alignments" over "redacted alignments")

### FR-3: Control BAM Download
- Download control BAM files to experiment directory with naming convention: `{control_accession}_control.bam`
- Use existing `download_save()` function with retry logic
- Validate file size against metadata
- Handle network failures gracefully

### FR-4: Control BAM Processing
- Index control BAM files using samtools
- Process control BAM files using `BAM_TO_SIGNAL` class
- Generate DSF signals at all required levels (1, 2, 4, 8)
- Store in directories named: `signal_DSF{X}_res25_control/`
- Generate metadata files for control signals

### FR-5: Control Metadata Management
- Store control experiment accession in experiment metadata
- Create `control_metadata.json` file alongside `file_metadata.json`
- Track control file accession, experiment accession, and download URL
- Include control-specific fields (read_length, run_type, platform, lab)

### FR-6: Validation and Completeness Checking
- Extend validation to check for control signal directories
- Report control completion status separately from experimental data
- Support partial completion (experiment complete, control missing)

## Non-Functional Requirements

### NFR-1: Performance
- Control processing should run in parallel with experimental processing when possible
- Should not significantly increase total pipeline runtime (target: <20% increase)
- Memory usage should remain within acceptable limits

### NFR-2: Reliability
- Robust error handling for API failures
- Retry logic for failed downloads
- Graceful degradation when controls are unavailable

### NFR-3: Maintainability
- Minimal code duplication (reuse existing BAM processing functions)
- Clear separation of control vs. experimental processing logic
- Comprehensive logging for debugging

### NFR-4: Compatibility
- Works with existing download plan JSON structure (no breaking changes)
- Compatible with existing validation tools
- Supports both EIC and MERGED datasets

## Data Requirements

### Input Data
1. **Download Plan JSON Files**
   - Location: `data/download_plan_eic.json`, `data/download_plan_merged.json`
   - Contains: experiment accessions, assay types, BAM accessions

2. **ENCODE API**
   - Endpoint: `https://www.encodeproject.org/experiments/{exp_accession}/?format=json`
   - Provides: control experiment links, file metadata

3. **Chromosome Sizes**
   - Location: `data/hg38.chrom.sizes`
   - Used for: BAM processing and binning

### Output Data
1. **Control BAM Files** (temporary)
   - Location: `{base_path}/{celltype}/{assay}/{control_accession}_control.bam`
   - Deleted after processing

2. **Control Signal Directories**
   - `signal_DSF1_res25_control/`
   - `signal_DSF2_res25_control/`
   - `signal_DSF4_res25_control/`
   - `signal_DSF8_res25_control/`
   - Each contains chromosome .npz files and metadata.json

3. **Control Metadata**
   - `control_metadata.json` - stores control experiment information
   - Format similar to `file_metadata.json`

## Constraints

### Technical Constraints
- Must use samtools for BAM indexing (external dependency)
- Must use pyBigWig and pysam libraries (existing dependencies)
- Limited by ENCODE API rate limits (handle 429 responses)
- Disk space requirements double for ChIP-seq experiments

### Business Constraints
- Should not modify existing processed experiments unless explicitly requested
- Must maintain audit trail of control processing
- Should work within HPC resource constraints (CPU, memory, storage)

## Success Criteria

1. **Functionality**: 90%+ of ChIP-seq experiments have successfully downloaded and processed controls
2. **Performance**: Pipeline completes within 1.5x the time of baseline (no controls)
3. **Quality**: Control signals pass validation checks (all chromosomes present, correct format)
4. **Usability**: Single command-line flag enables control processing
5. **Reliability**: Retry mechanism recovers from 95%+ of transient failures

## Open Questions

1. Should we support multiple controls per experiment, or just select one?
2. How should we handle experiments with no designated control?
3. Should control processing be a separate command or integrated into existing commands?
4. Do we need to validate that control and experiment have compatible parameters (same sequencing platform, etc.)?
5. Should we store control BAM files permanently or only keep processed signals?

## Dependencies

- Existing `get_candi_data.py` module
- ENCODE API availability
- `data.py` module (BAM_TO_SIGNAL class)
- samtools, pysam, pyBigWig libraries
- Download plan JSON files

## Assumptions

1. ENCODE metadata correctly identifies control experiments via `possible_controls` field
2. Control experiments have BAM files available for download
3. Control BAM files are in GRCh38 assembly (matching experimental data)
4. Control processing uses the same resolution (25bp) as experimental data
5. Users have sufficient disk space for control data (~2x current requirements)

