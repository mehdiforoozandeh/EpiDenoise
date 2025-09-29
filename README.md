# CANDI: Confidence-Aware Neural Denoising Imputer

[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.01.23.634626v1-blue.svg)](https://www.biorxiv.org/content/10.1101/2025.01.23.634626v1)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org)

## Overview

CANDI (Confidence-Aware Neural Denoising Imputer) is a self-supervised deep learning framework for epigenomic data denoising and imputation. CANDI addresses key limitations in existing epigenome imputation methods by predicting raw counts, handling experiment-specific covariates, and providing calibrated uncertainty estimates.

This implementation is based on the research presented in the preprint: **"CANDI: self-supervised, confidence-aware denoising imputation of genomic data"** ([biorxiv.org/10.1101/2025.01.23.634626v1](https://www.biorxiv.org/content/10.1101/2025.01.23.634626v1)).

## Key Technical Innovations

CANDI addresses three major limitations in existing epigenome imputation methods:

1. **Raw Count Prediction**: Unlike existing methods that operate on processed signals, CANDI predicts raw read counts and handles experiment-specific covariates (sequencing depth, read length, coverage, run type)

2. **Zero-Shot Denoising**: CANDI can incorporate information from low-quality existing experiments when predicting targets without retraining, enabling practical denoising applications

3. **Calibrated Uncertainty**: CANDI outputs calibrated measures of uncertainty for all predictions, providing confidence intervals that help researchers assess prediction reliability

## Key Features

- **🔬 Self-Supervised Learning**: Transformer-based architecture with SSL training paradigm
- **📊 Raw Count Prediction**: Predicts raw read counts and handles experiment-specific covariates
- **🎯 Zero-Shot Imputation**: Generalizes to new cell types without retraining
- **📈 Calibrated Uncertainty**: Provides confidence estimates for all predictions
- **🧬 Multi-Assay Support**: ATAC-seq, DNase-seq, and 33+ histone modifications
- **⚡ High-Performance Computing**: Optimized for multi-GPU training and inference
- **📊 Comprehensive Data Pipeline**: Complete ENCODE data processing and validation

## Project Structure

```
CANDI/
├── 📁 Core Components
│   ├── data.py                    # Data loading and preprocessing utilities
│   ├── model.py                   # Deep learning model architectures
│   ├── train_candi.py            # Training pipeline and model definitions
│   ├── eval.py                   # Evaluation and benchmarking tools
│   ├── inference.py              # Model inference and prediction
│   └── benchmark.py              # Performance comparison utilities
│
├── 📁 Data Processing
│   ├── get_candi_data.py         # Unified ENCODE data pipeline
│   ├── candi.py                  # CANDI framework integration
│   └── data/                     # Processed datasets and metadata
│       ├── download_plan_*.json  # Experiment definitions
│       ├── *_metadata.csv       # Dataset metadata
│       └── *.bed, *.fa          # Reference genome files
│
├── 📁 Analysis Tools
│   ├── SAGA.py                   # Segmentation and genome annotation
│   ├── baselines.py              # Baseline method implementations
│   └── unified_benchmark.py      # Comprehensive evaluation suite
│
├── 📁 Legacy & Utilities
│   ├── legacy/                   # Deprecated components
│   ├── _utils.py                 # Utility functions
│   └── hybrid_ddp_approach*.py   # Distributed training implementations
│
└── 📁 Documentation
    ├── README.md                 # This file
    ├── README_CANDI_DATA.md      # Data pipeline documentation
    └── final_comprehensive_analysis.md  # Performance analysis
```

## Installation

### Prerequisites

```bash
# Required system tools
module load samtools/1.22.1
module load bedtools/2.31.0

# Python dependencies
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn scipy
pip install pysam pyBigWig
pip install intervaltree hmmlearn
pip install umap-learn tqdm
```

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd CANDI

# Set up environment
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Verify installation
python -c "import data, model, train_candi; print('Installation successful!')"
```

## Quick Start

### 1. Data Processing

Process ENCODE datasets using the unified pipeline:

```bash
# Process EIC dataset
python get_candi_data.py process eic /path/to/output --max-workers 16

# Process MERGED dataset
python get_candi_data.py process merged /path/to/output --max-workers 16

# Validate processed data
python get_candi_data.py validate eic /path/to/output --save-csv results.csv
```

### 2. Model Training

Train denoising models on processed data:

```bash
# Single GPU training
python train_candi.py --data_path /path/to/data --output_dir models/

# Multi-GPU training
python train_candi_multiGPU.py --data_path /path/to/data --num_gpus 4

# Hybrid DDP approach (recommended for large datasets)
python hybrid_ddp_approach_clean.py --data_path /path/to/data
```

### 3. Model Evaluation

Evaluate trained models:

```bash
# Comprehensive evaluation
python eval.py --model_path models/best_model.pt --data_path /path/to/data

# Benchmark against baselines
python benchmark.py --model_path models/best_model.pt --data_path /path/to/data

# Generate predictions
python inference.py --model_path models/best_model.pt --output_dir predictions/
```

## Core Components

### 1. Data Processing Pipeline (`get_candi_data.py`)

A unified command-line interface for ENCODE data processing:

- **Download**: Automated ENCODE data retrieval
- **Process**: BAM → DSF signals, BigWig → binned signals
- **Validate**: Comprehensive data quality checks
- **Visualize**: Availability heatmaps and statistics

**Key Features:**
- Parallel processing with CPU-aware worker management
- Resume capability for interrupted downloads
- Comprehensive error handling and retry mechanisms
- Memory-optimized processing for large datasets

### 2. CANDI Model Architecture (`model.py`, `train_candi.py`)

**CANDI Neural Network:**
- **Encoder**: Conv1D towers for epigenomic signals and DNA sequence processing
- **Transformer**: 4-layer encoder with relative positional encoding and 9 attention heads
- **Decoder**: Separate deconvolution towers for count and signal prediction
- **Distribution Layers**: Negative binomial (counts) and Gaussian (signals) output distributions

**Key Innovations:**
- **Self-Supervised Learning**: Masking and upsampling objectives
- **Raw Count Prediction**: Handles sequencing depth and experimental covariates
- **Uncertainty Quantification**: Calibrated confidence intervals for all predictions
- **Zero-Shot Generalization**: Works on new cell types without retraining

### 3. Evaluation Framework (`eval.py`, `benchmark.py`)

**Metrics:**
- Genome-wide correlation (Pearson/Spearman)
- Mean squared error (MSE)
- Signal-to-noise ratio improvements
- Cross-assay prediction accuracy


## Usage Examples

### Basic Denoising

```python
from train_candi import CANDI_LOADER, CANDI
from eval import EVAL_CANDI

# Load data
loader = CANDI_LOADER(data_path="/path/to/data", batch_size=32)

# Load trained model
model = CANDI.load_from_checkpoint("models/best_model.pt")

# Generate denoised predictions
evaluator = EVAL_CANDI(model, loader)
predictions = evaluator.predict_denoised_signals()
```

### Custom Analysis

```python
from SAGA import SoftMultiAssayHMM
from data import ExtendedEncodeDataHandler

# Load processed data
dataset = ExtendedEncodeDataHandler("/path/to/data")
dataset.init_eval(context_length=1600, split="test")

# Perform segmentation
hmm = SoftMultiAssayHMM(n_components=15)
states = hmm.fit_predict(dataset.get_chromosome_data("chr1"))

# Export results
write_bed(states, "chr1", 0, 25, "output/chr1_states.bed")
```

## Benchmarking Results

CANDI has been extensively evaluated on multiple datasets:

- **ENCODE Imputation Challenge (EIC)**: 35 assays across 50 biosamples
- **Extended ENCODE Dataset**: 361 merged cell types, 35 assays
- **Performance**: Comparable to challenge-winning methods (Avocado, Guacamole, Lavawizard)
- **Uncertainty Calibration**: Near-perfect calibration for high confidence levels (≥90%)
- **Zero-Shot Generalization**: Effective prediction on unseen cell types without retraining


## Citation

If you use CANDI in your research, please cite:

```bibtex
@article{candi2025,
  title={CANDI: self-supervised, confidence-aware denoising imputation of genomic data},
  author={Mehdi Foroozandeh and Abdul Rahman Diab and Maxwell Libbrecht},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.01.23.634626v1}
}
```

## Support

For questions, issues, or feature requests:

1. Check the [documentation](README_CANDI_DATA.md) for detailed usage instructions
2. Review [performance analysis](final_comprehensive_analysis.md) for optimization tips
3. Open an issue on the project repository
4. Contact the development team

---

**CANDI** - Confidence-Aware Neural Denoising Imputer for epigenomic data analysis.