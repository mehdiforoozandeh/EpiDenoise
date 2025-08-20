# CANDI Data Pipeline - Unified Interface

A comprehensive, unified pipeline for downloading, processing, validating, and visualizing ENCODE data for the CANDI dataset. This tool combines all functionality into a single, easy-to-use command-line interface.

## Features

- **🔄 Complete Data Processing**: Download and process EIC/MERGED datasets with parallel execution
- **✅ Comprehensive Validation**: Validate dataset completeness and integrity 
- **📊 Smart Visualization**: Create availability heatmaps for dataset overview
- **🔧 Enhanced Retry System**: Retry failed experiments with improved error handling
- **⚡ Parallel Processing**: CPU-aware parallel execution with memory optimization
- **📝 Detailed Logging**: Complete logging with progress tracking
- **🎯 Resume Capability**: Skip already processed data automatically

## Installation & Setup

### Prerequisites

```bash
# Required bioinformatics tools (available as modules on most HPC systems)
module load samtools/1.22.1
module load bedtools/2.31.0

# Required Python packages
pip install pandas numpy requests matplotlib seaborn tqdm pysam pyBigWig torch
```

### Directory Structure

```
EpiDenoise/
├── get_candi_data.py          # Main unified pipeline script
├── data.py                    # Core bioinformatics functions
├── data/
│   ├── download_plan_eic.json      # EIC dataset experiment definitions
│   ├── download_plan_merged.json   # MERGED dataset experiment definitions
│   ├── merged_train_va_test_split.json  # MERGED dataset splits
│   └── hg38.chrom.sizes            # Chromosome sizes file
└── README_CANDI_DATA.md       # This documentation
```

## Command Overview

The pipeline provides a unified command-line interface with the following commands:

```bash
python get_candi_data.py <command> [options]
```

### Available Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `process` | Download and process datasets | Main data processing |
| `validate` | Comprehensive dataset validation | Quality assurance |
| `quick-validate` | Quick validation check | Rapid status check |
| `analyze-missing` | Analyze missing experiments | Identify gaps |
| `create-plots` | Create availability visualizations | Dataset overview |
| `retry` | Retry failed experiments | Fix processing failures |
| `run-complete` | Complete processing with detailed logging | Production runs |

## Quick Start

### 1. Basic Dataset Processing

```bash
# Process EIC dataset
python get_candi_data.py process eic /path/to/data

# Process MERGED dataset with custom settings
python get_candi_data.py process merged /path/to/data --max-workers 8 --resolution 25
```

### 2. Validation and Quality Checks

```bash
# Quick validation
python get_candi_data.py quick-validate eic /path/to/data

# Comprehensive validation with CSV output
python get_candi_data.py validate merged /path/to/data --save-csv results.csv

# Analyze missing experiments
python get_candi_data.py analyze-missing eic /path/to/data --save-json missing.json
```

### 3. Visualization

```bash
# Create availability plots for both datasets
python get_candi_data.py create-plots /path/to/data

# Custom output prefix
python get_candi_data.py create-plots /path/to/data --output-prefix my_analysis
```

### 4. Production Processing

```bash
# Complete processing with detailed logging (recommended for large datasets)
python get_candi_data.py run-complete eic /path/to/data --max-workers 16 --log-file eic_processing.log

# Retry failed experiments
python get_candi_data.py retry merged /path/to/data --max-workers 6
```

## Detailed Command Reference

### `process` - Main Data Processing

Downloads and processes ENCODE data files for the specified dataset.

```bash
python get_candi_data.py process <dataset> <directory> [options]
```

**Arguments:**
- `dataset`: Dataset type (`eic`, `merged`)
- `directory`: Output directory for processed data

**Options:**
- `--resolution INT`: Resolution in base pairs (default: 25)
- `--max-workers INT`: Maximum parallel workers (default: CPU count - 1)
- `--validate-only`: Only validate existing data, don't download
- `--verbose`: Enable verbose logging

**Example:**
```bash
python get_candi_data.py process eic /data/CANDI_EIC --max-workers 12 --verbose
```

**What it does:**
1. Creates experiment directories (`{celltype}/{assay}/`)
2. Downloads BAM/BigWig/BigBed/TSV files from ENCODE
3. Processes BAM files to DSF signals (DSF1, DSF2, DSF4, DSF8)
4. Processes BigWig files to binned signals
5. Processes BigBed files to peak signals
6. Creates comprehensive metadata files
7. Validates processed data
8. Cleans up large raw files after processing

### `validate` - Comprehensive Validation

Performs detailed validation of dataset completeness and integrity.

```bash
python get_candi_data.py validate <dataset> <directory> [options]
```

**Options:**
- `--resolution INT`: Resolution for validation (default: 25)
- `--save-csv FILE`: Save detailed results to CSV file

**Example:**
```bash
python get_candi_data.py validate eic /data/CANDI_EIC --save-csv validation_results.csv
```

**Output:**
- Total experiments and biosamples
- Completion percentages
- Detailed missing experiment lists
- Per-experiment validation status

### `quick-validate` - Quick Status Check

Provides a rapid overview of dataset completion status.

```bash
python get_candi_data.py quick-validate <dataset> <directory>
```

**Example:**
```bash
python get_candi_data.py quick-validate merged /data/CANDI_MERGED
```

**Output:**
```
=== MERGED Dataset Validation Results ===
Experiments: 1820/1846 (98.6% complete)
Biosamples: 87/89 (97.8% coverage)
```

### `analyze-missing` - Missing Data Analysis

Analyzes patterns in missing experiments to identify common issues.

```bash
python get_candi_data.py analyze-missing <dataset> <directory> [options]
```

**Options:**
- `--save-json FILE`: Save analysis results to JSON file

**Example:**
```bash
python get_candi_data.py analyze-missing eic /data/CANDI_EIC --save-json missing_analysis.json
```

**Output:**
- Top missing celltypes
- Top missing assays
- Pattern analysis by experiment type

### `create-plots` - Dataset Visualization

Creates comprehensive availability heatmaps for both EIC and MERGED datasets.

```bash
python get_candi_data.py create-plots <directory> [options]
```

**Options:**
- `--output-prefix PREFIX`: Filename prefix (default: 'candi')

**Example:**
```bash
python get_candi_data.py create-plots /data --output-prefix analysis_2024
```

**Output Files:**
- `{prefix}_eic_availability.png`: EIC dataset heatmap
- `{prefix}_merged_availability.png`: MERGED dataset heatmap

**Plot Features:**
- Color-coded by train/validation/test splits
- Sorted by data availability
- Union logic for EIC celltypes (T_/V_/B_ prefixes)
- Consistent color scheme across datasets

### `retry` - Enhanced Retry System

Retries failed experiments with enhanced error handling.

```bash
python get_candi_data.py retry <dataset> <directory> [options]
```

**Options:**
- `--max-workers INT`: Maximum parallel workers (default: 6)
- `--log-file FILE`: Log file for retry operations

**Example:**
```bash
python get_candi_data.py retry eic /data/CANDI_EIC --max-workers 4 --log-file retry.log
```

**Enhanced Features:**
- Improved BAM indexing with validation
- BigBed processing with invalid interval handling
- Per-chromosome error handling for robustness
- Detailed error classification and logging

### `run-complete` - Production Processing

Complete dataset processing with detailed progress logging and monitoring.

```bash
python get_candi_data.py run-complete <dataset> <directory> [options]
```

**Options:**
- `--max-workers INT`: Maximum parallel workers (default: 16)
- `--log-file FILE`: Custom log file name

**Example:**
```bash
python get_candi_data.py run-complete merged /data/CANDI_MERGED --max-workers 16 --log-file merged_complete.log
```

**Features:**
- Detailed progress logging with experiment/biosample counts
- Memory usage optimization
- Resume capability (skips completed experiments)
- Comprehensive error reporting
- Production-ready resource management

## Data Processing Details

### File Types Processed

1. **BAM Files**: Aligned sequencing reads
   - Indexed with `samtools index`
   - Processed to DSF signals (1x, 2x, 4x, 8x downsampling)
   - Cleaned up after processing to save space

2. **BigWig Files**: Signal tracks
   - Binned to specified resolution (default: 25bp)
   - Stored as compressed numpy arrays

3. **BigBed Files**: Peak/region annotations
   - Converted to binary peak calls per bin
   - Handles invalid intervals gracefully

4. **TSV Files**: RNA-seq quantification
   - Downloaded and stored for RNA-seq experiments

### Output Structure

```
{base_directory}/
├── {celltype}/
│   └── {assay}/
│       ├── file_metadata.json           # ENCODE metadata
│       ├── signal_DSF1_res25/          # 1x downsampled signals
│       │   ├── metadata.json
│       │   ├── chr1.npz
│       │   ├── chr2.npz
│       │   └── ...
│       ├── signal_DSF2_res25/          # 2x downsampled signals
│       ├── signal_DSF4_res25/          # 4x downsampled signals
│       ├── signal_DSF8_res25/          # 8x downsampled signals
│       ├── signal_BW_res25/            # BigWig signals (if available)
│       └── peaks_res25/                # Peak calls (if available)
```

### Metadata Format

Each experiment includes comprehensive metadata:

```json
{
  "assay": {"2": "ATAC-seq"},
  "accession": {"2": "ENCFF803TLA"},
  "biosample": {"2": "ENCBS063ABX"},
  "file_format": {"2": "bam"},
  "output_type": {"2": "alignments"},
  "experiment": {"2": "/experiments/ENCSR113MBR/"},
  "bio_replicate_number": {"2": [1]},
  "file_size": {"2": 9617182802},
  "assembly": {"2": "GRCh38"},
  "read_length": {"2": 101},
  "run_type": {"2": "paired-ended"},
  "download_url": {"2": "https://www.encodeproject.org/files/..."},
  "date_created": {"2": 1472601600},
  "status": {"2": "released"}
}
```

## Performance Optimization

### CPU and Memory Settings

The pipeline automatically limits parallel workers to available CPU cores:

```bash
# Automatic CPU detection (recommended)
python get_candi_data.py process eic /data --max-workers auto

# Manual CPU limiting for shared systems
python get_candi_data.py process eic /data --max-workers 8

# Production settings for dedicated systems
python get_candi_data.py run-complete eic /data --max-workers 16
```

### Memory Considerations

- **Single experiment**: ~1-2GB memory
- **Parallel processing**: ~(max_workers × 2GB) total memory
- **Large BAM files**: Up to 10GB during processing (cleaned up automatically)

**Recommended Settings:**
- **Small system (32GB RAM)**: `--max-workers 8`
- **Medium system (64GB RAM)**: `--max-workers 16`
- **Large system (128GB+ RAM)**: `--max-workers 24`

## Error Handling and Troubleshooting

### Common Issues and Solutions

1. **"samtools: command not found"**
   ```bash
   module load samtools/1.22.1
   ```

2. **"ModuleNotFoundError: No module named 'pysam'"**
   ```bash
   pip install pysam pyBigWig
   ```

3. **"Invalid interval bounds!" (BigBed processing)**
   - Automatically handled by enhanced retry system
   - Skips invalid intervals, continues with valid data

4. **"fetch called on bamfile without index"**
   - Enhanced BAM indexing with validation
   - Automatic retry for failed indexing

5. **Out of Memory errors**
   - Reduce `--max-workers`
   - Check available system memory

### Log File Analysis

Each command creates detailed logs. Key patterns to look for:

```bash
# Success indicators
grep "✅ Successfully" logfile.log

# Error patterns
grep "❌\|ERROR\|FAILED" logfile.log

# Progress tracking
grep "📊 PROGRESS SUMMARY" logfile.log
```

## SLURM Integration

For HPC clusters, the pipeline integrates well with SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=candi_eic
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --account=your_account

# Load required modules
module load samtools/1.22.1
module load bedtools/2.31.0

# Set working directory
cd /path/to/EpiDenoise

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run complete processing
python get_candi_data.py run-complete eic /path/to/data \\
  --max-workers 16 \\
  --log-file ${SLURM_JOB_ID}_eic_processing.log
```

## Best Practices

### Development and Testing

1. **Start small**: Use test datasets for development
2. **Validate frequently**: Run quick-validate after changes
3. **Monitor resources**: Check memory and CPU usage
4. **Review logs**: Always check log files for issues

### Production Usage

1. **Use run-complete**: For full dataset processing
2. **Set appropriate resources**: Match CPU/memory to system capabilities  
3. **Enable logging**: Always specify log files for production runs
4. **Plan for retries**: Budget time for retry operations if needed
5. **Validate results**: Run comprehensive validation after processing

### Data Management

1. **Plan storage**: ~50-100GB per complete dataset
2. **Backup metadata**: Save validation results and logs
3. **Monitor disk space**: BAM files are large during processing
4. **Clean up**: Pipeline automatically removes large intermediate files

## Dataset Information

### EIC Dataset
- **Experiments**: 363 total
- **Biosamples**: 89 unique celltypes
- **Assays**: 35 types (ATAC-seq, DNase-seq, 33 histone marks)
- **Splits**: Train (T_), Validation (V_), Blind (B_) by celltype prefix

### MERGED Dataset  
- **Experiments**: 1,846 total
- **Biosamples**: 89 unique celltypes
- **Assays**: Same 35 types as EIC
- **Splits**: Train/Validation/Test by celltype assignment

### Expected Output Sizes
- **Complete EIC dataset**: ~45GB processed data
- **Complete MERGED dataset**: ~85GB processed data
- **Single experiment**: ~50-200MB processed data

## Support and Development

This unified pipeline replaces the following individual scripts:
- `process_eic_complete.py` / `process_merged_complete.py` 
- `validate_candi_datasets.py` / `quick_validate_candi.py`
- `analyze_missing_experiments.py`
- `retry_eic_failed.py` / `retry_merged_failed.py`
- `create_final_clean_plots.py`

All functionality has been integrated into the single `get_candi_data.py` script with a comprehensive CLI interface.

For questions or issues, check the log files first, then review this documentation for troubleshooting guidance.
