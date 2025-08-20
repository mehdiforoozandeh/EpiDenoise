# Enhanced Retry System for Failed CANDI Experiments

## Overview

The enhanced retry system addresses the small percentage of experiments that failed during the main CANDI data processing pipeline. This document summarizes the failure analysis, root causes, and implemented solutions.

## Failure Analysis Results

### Summary Statistics
- **EIC Dataset**: 1 failed experiment out of 363 total (0.28% failure rate)
- **MERGED Dataset**: 25 failed experiments out of 2,684 total (0.93% failure rate)
- **Total**: 26 failed experiments out of 3,047 total (0.85% failure rate)

### Failure Types Identified

#### 1. BigBed Processing Failures (17 experiments)
**Error**: `"Invalid interval bounds!"`
**Root Cause**: Issues in `get_binned_bigBed_peaks()` function when processing BigBed files with:
- Invalid intervals where `peak_start >= peak_end`
- Intervals extending beyond chromosome boundaries
- Malformed coordinate data in source files

**Affected Experiments**:
- T_testis-H3K9me3 (EIC)
- Various H3K9me2, H3K9me3, EZH2, and H3K27me3 assays from multiple cell lines

#### 2. BAM Indexing Failures (8 experiments)
**Error**: `"fetch called on bamfile without index"`
**Root Cause**: Silent failure of `samtools index` command, followed by attempts to process unindexed BAM files
- All failures from `Peyer's_patch_nonrep` celltype suggest systematic issue with these specific BAM files
- Files may be corrupted or have unusual formatting

**Affected Experiments**:
- All 8 experiments from `Peyer's_patch_nonrep` celltype (H3K27ac, H3K4me3, H3K4me1, CTCF, H3K36me3, H3K27me3, H3K9me3, DNase-seq)

## Enhanced Solutions Implemented

### 1. Robust BigBed Processor (`EnhancedBigBedProcessor`)
**Key Improvements**:
- **Interval Validation**: Check `peak_start < peak_end` before processing
- **Boundary Checks**: Ensure intervals are within chromosome boundaries
- **Graceful Error Handling**: Skip invalid intervals and continue processing valid ones
- **Comprehensive Logging**: Report counts of valid vs. invalid intervals

**Technical Details**:
```python
# Validate interval bounds
if peak_start >= peak_end:
    invalid_intervals += 1
    continue

if peak_start < 0 or peak_end > chr_size:
    invalid_intervals += 1
    continue

# Validate bin indices
if start_bin < 0 or end_bin >= num_bins or start_bin > end_bin:
    invalid_intervals += 1
    continue
```

### 2. Enhanced BAM Processor (`EnhancedBAMProcessor`)
**Key Improvements**:
- **Pre-validation**: Test BAM file header before attempting indexing
- **Retry Logic**: Multiple attempts with error checking for `samtools index`
- **Index Verification**: Test that created index actually works
- **Timeout Protection**: 5-minute timeout for indexing operations

**Technical Details**:
```python
# Verify index is functional
try:
    with pysam.AlignmentFile(bam_file, 'rb') as test_bam:
        for read in test_bam.fetch('chr1', 0, 1000):
            break  # Just test that fetch works
except Exception as e:
    # Index not functional, retry
```

### 3. Enhanced Download Manager (`EnhancedCANDIDownloadManager`)
**Integration Features**:
- Uses robust processors for BigBed and BAM files
- Enhanced error reporting with specific failure reasons
- Maintains all existing functionality while adding robustness
- Proper cleanup on both success and failure

## Retry System Components

### 1. Failure Analyzer (`FailureAnalyzer`)
- Parses log files to extract failed experiments
- Classifies failures by type (BigBed bounds, BAM indexing, download, other)
- Maps failed tasks back to their original Task objects

### 2. Task Recreation
- Loads original download plans (EIC and MERGED)
- Creates Task objects for failed experiments
- Resets task status to PENDING for retry processing

### 3. Enhanced Processing Pipeline
- Uses the same task-based architecture as main pipeline
- Applies enhanced error handling specifically for identified failure types
- Provides detailed logging of retry attempts and outcomes

## Usage

### Command Line Interface
```bash
python retry_failed_experiments.py \
    --download-directory /project/compbio-lab/encode_data \
    --max-workers 6 \
    --eic-log eic_complete_processing.log \
    --merged-log merged_complete_processing.log
```

### SLURM Submission
```bash
sbatch job_retry_failed.sh
```

## Expected Outcomes

### BigBed Failures
- **High Success Rate Expected**: Enhanced bounds checking should resolve most "Invalid interval bounds" errors
- **Partial Recovery**: Some files may have too many invalid intervals to be useful, but most should recover
- **Detailed Reporting**: Clear indication of which intervals were skipped and why

### BAM Failures
- **Variable Success Rate**: Depends on whether files are corrupted or just had indexing issues
- **Systematic Investigation**: Will reveal if `Peyer's_patch_nonrep` files are fundamentally problematic
- **Fallback Strategy**: Corrupted files may need to be excluded from the dataset

## Monitoring and Validation

### Real-time Monitoring
- Detailed logging in `retry_processing.log`
- Progress updates for each retry attempt
- Clear success/failure indicators with specific error messages

### Post-Processing Validation
- Verification that retry-processed experiments meet all quality standards
- Integration with existing CANDIValidator for completeness checking
- Summary statistics on recovery rates by failure type

## Files Created

1. **`retry_failed_experiments.py`**: Main retry script with enhanced error handling
2. **`test_retry_fixes.py`**: Test script for validation
3. **`job_retry_failed.sh`**: SLURM submission script
4. **`RETRY_SYSTEM_SUMMARY.md`**: This documentation

## Future Considerations

### Monitoring Recommendations
1. **Track Failure Patterns**: Monitor if certain file types or cell lines are more prone to failures
2. **Source Data Quality**: Report systematic issues back to ENCODE for investigation
3. **Pipeline Robustness**: Integrate enhanced error handling into main pipeline for future runs

### Potential Improvements
1. **Automatic Retry**: Integrate retry logic directly into main pipeline
2. **Quality Scoring**: Develop metrics for data quality based on processing success
3. **Alternative Sources**: For systematically failing experiments, identify alternative data sources

This enhanced retry system provides a robust solution for recovering failed experiments while maintaining data quality and providing detailed diagnostics for any remaining issues.

