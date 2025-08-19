# CANDI Production Jobs - Complete Dataset Processing

## Overview

This document describes the production-ready SLURM jobs for processing complete EIC and MERGED datasets using the optimized CANDI pipeline.

## Job Files Created

### 1. EIC Dataset Processing
- **SLURM Script**: `job_eic_complete.sh`
- **Python Script**: `process_eic_complete.py`
- **Log File**: `eic_complete_processing.log`

### 2. MERGED Dataset Processing  
- **SLURM Script**: `job_merged_complete.sh`
- **Python Script**: `process_merged_complete.py`
- **Log File**: `merged_complete_processing.log`

## Dataset Statistics

| Dataset | Experiments | Biosamples | Est. Time (16 workers) | Storage Est. |
|---------|-------------|------------|------------------------|--------------|
| EIC     | 363         | 89         | ~2 hours               | ~150 GB      |
| MERGED  | 2,684       | 361        | ~15 hours              | ~1.1 TB      |

## Resource Allocation

### Per Job Configuration
- **CPUs**: 16 (16 parallel experiments)
- **Memory**: 80 GB (5 GB per experiment)
- **Time Limits**: 
  - EIC: 8 hours (generous buffer)
  - MERGED: 24 hours (generous buffer)

### Environment & Modules
- `samtools/1.22.1` (for BAM processing)
- `bedtools/2.31.0` (for BigBed processing)
- Uses existing Python environment with pandas, torch, etc.

## Features

### ✅ Detailed Progress Logging
- Real-time experiment completion status
- Remaining experiments and biosamples count
- Progress percentage tracking
- Processing time and throughput metrics

### ✅ Resume Capability
- Automatically detects completed experiments
- Only processes missing/incomplete experiments
- Safe to restart after interruption

### ✅ Error Handling
- Comprehensive error logging
- Failed experiment tracking
- Retry mechanisms for network issues
- Safe cleanup of incomplete downloads

### ✅ Resource Monitoring
- Memory usage optimization
- Disk space monitoring
- CPU utilization tracking
- Final storage usage reporting

## Usage Instructions

### Submit EIC Dataset Job
```bash
sbatch job_eic_complete.sh
```

### Submit MERGED Dataset Job
```bash
job_merged_complete.sh
```

### Monitor Job Progress
```bash
# Check job status
squeue -u $USER

# Follow live logs
tail -f eic_complete_processing.log
tail -f merged_complete_processing.log

# Check SLURM output
tail -f eic_complete_<JOBID>.out
tail -f merged_complete_<JOBID>.out
```

## Output Structure

### Directory Layout
```
DATA_CANDI_EIC/
├── {biosample_1}/
│   ├── {assay_1}/
│   │   ├── file_metadata.json
│   │   ├── signal_DSF1_res25/
│   │   │   ├── chr1.npz
│   │   │   ├── chr2.npz
│   │   │   └── ...
│   │   ├── signal_DSF2_res25/
│   │   ├── signal_DSF4_res25/
│   │   ├── signal_DSF8_res25/
│   │   ├── signal_BW_res25/
│   │   └── peaks_res25/
│   └── {assay_2}/
└── {biosample_2}/
```

### Generated Files Per Experiment
- **Processed signals**: DSF1, DSF2, DSF4, DSF8, BW signals (`.npz` files per chromosome)
- **Peak data**: Processed BigBed peaks (`.npz` files per chromosome)  
- **Metadata**: Experiment metadata in required format (`file_metadata.json`)
- **Raw files**: BAM files are deleted after successful processing to save space

## Log File Format

### Progress Messages
```
✅ COMPLETED: T_adrenal_gland - ATAC-seq
📊 PROGRESS SUMMARY:
   🧬 Experiments: 45/363 completed (318 remaining)
   🏷️  Biosamples: 12/89 completed (77 remaining)
   📈 Progress: 12.4%
```

### Final Summary
```
📊 FINAL RESULTS
✅ Successful experiments: 363
❌ Failed experiments: 0
📈 Success rate: 100.0%
⚡ Throughput: 3.02 experiments/minute
💾 Total disk usage: 145.2 GB
```

## Validation

Each processed experiment includes:
- ✅ File size validation against ENCODE metadata
- ✅ Signal processing verification
- ✅ Metadata format compliance
- ✅ Complete directory structure
- ✅ All chromosomes processed

## Recovery & Troubleshooting

### If a Job Fails
1. Check error logs: `{dataset}_complete_<JOBID>.err`
2. Review detailed log: `{dataset}_complete_processing.log`
3. Restart job - it will resume from where it left off
4. Failed downloads are logged to `failed_downloads.log`

### Common Issues
- **Out of memory**: Reduce max_workers in Python script
- **Network timeouts**: Automatic retry with exponential backoff
- **Disk space**: Monitor with `df -h` and clean temporary files
- **Module loading**: Ensure samtools/bedtools modules are available

## Performance Optimization

Based on comprehensive testing:
- **Optimal parallelization**: 16 workers provides best throughput/resource balance
- **Memory efficiency**: 5 GB per experiment handles largest files safely
- **Network resilience**: 10 retry attempts with exponential backoff
- **Storage optimization**: Large BAM files deleted after processing

## Contact & Support

For issues or questions:
- Check log files for detailed error messages
- Review this README for common troubleshooting
- Validate environment setup matches successful test configurations
