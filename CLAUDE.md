# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sleep stage classification ML project for rodent EEG data. The system classifies Wake/NREM/REM states from rodent EEG and EMG signals using deep learning models. The goal is near-real-time classification with macro F1 ≥ 0.82 and per-class F1(REM) ≥ 0.75.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv for package management)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test types
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Run only unit tests
pytest -m integration # Run only integration tests

# Run single test file
pytest tests/test_specific.py
```

### Training
```bash
# Train with default configuration
python train.py

# Train with specific model type
python train.py --model-type cnn_bilstm --epochs 50 --batch-size 32

# Available model types: small_cnn, cnn_bilstm, implicit_cnn, simple_dense
```

### Data Processing Scripts
```bash
# Generate HDF5 dataset from EDF files
python scripts/generate_dataset_hdf5.py

# Preprocess EDF directory
python scripts/preprocess_edf_dir.py

# Downsample existing dataset
python scripts/downsample_dataset.py

# Process MPSD data
python scripts/process_mpsd.py
```

## Code Architecture

### Package Structure
- `src/rembler/` - Main package
  - `dataio/` - Data loading and preprocessing
    - `datasets.py` - PyTorch Dataset classes (SleepStageDataset, CustomDataset)
    - `edf_reader.py` - EDF file reading utilities
  - `models/` - Neural network architectures
    - `basic_cnn.py` - SmallCNN model
    - `cnn_bilstm.py` - CNN + BiLSTM model
    - `implicit_cnn.py` - Implicit frequency CNN
    - `basic_dense.py` - Simple dense model
  - `utils/` - Utility modules
    - `sleep_utils.py` - Sleep stage mapping and utilities
    - `dataset_utils.py` - Dataset processing helpers (CustomDataset)
    - `hardware_utils.py` - Device resolution
    - `config_utils.py` - Configuration handling
  - `analysis/metrics/` - Custom evaluation metrics
    - `custom_metrics.py` - PerClassMetric for sleep stage evaluation

### Data Flow
1. **Raw Data**: EDF files containing EEG/EMG signals
2. **Preprocessing**: Scripts convert EDF → HDF5 format with epoching (typically 10-50s windows)
3. **Training Dataset**: HDF5 files with signals and labels, loaded via CustomDataset
4. **Models**: Various architectures (CNN, CNN+BiLSTM, etc.) for sequence classification
5. **Evaluation**: Per-class metrics optimized for sleep stage imbalance (REM is minority class)

### Key Design Patterns
- **Signal Processing**: 50s context windows centered on 10s classification epochs
- **Class Imbalance Handling**: Weighted loss functions (1/√frequency) for rare REM stages
- **Model Registry**: `build_model()` function in train.py supports multiple architectures
- **Cross-Validation**: Animal-wise splits to prevent subject leakage
- **Temporal Context**: Models use multi-epoch context for transition modeling

### Configuration
- Uses `pyproject.toml` for dependencies and project metadata
- Training parameters configurable via command line arguments
- Test coverage target: 80% minimum (`--cov-fail-under=80`)
- Default output directory: `artifacts/` for checkpoints and metrics

### Data Expectations
- **Input Signals**: EEG (500 Hz) + optional EMG (500 Hz)
- **Labels**: 0=Wake, 1=NREM, 2=REM (see `sleep_utils.py` for mappings)
- **Format**: HDF5 with signals (N,C,T) and labels (N,) arrays
- **Context**: Models expect temporal context around each epoch for better transition modeling