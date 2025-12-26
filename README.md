# Capacity-SEM Project

Structural Equation Model analysis of government capacity effects on disaster recovery outcomes.

## Overview

This project analyzes how government administrative capacity affects the timeliness and effectiveness of CDBG-DR disaster recovery fund expenditure using Structural Equation Modeling.

## Setup

### Prerequisites

- Python 3.10+
- pip or conda for package management

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

- pandas, numpy - Data manipulation
- semopy - SEM estimation
- matplotlib - Visualization
- quarto - Manuscript rendering (optional)

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run complete pipeline
python src/pipeline.py run_all

# Or run individual stages
python src/pipeline.py ingest_data
python src/pipeline.py build_panel
python src/pipeline.py compute_features
python src/pipeline.py run_estimation
python src/pipeline.py make_figures
```

## Project Structure

```
capacity-sem-project/
├── src/
│   ├── pipeline.py           # Main CLI entry point
│   ├── config.py             # Configuration constants
│   ├── stages/               # Pipeline stages
│   │   ├── s00_ingest.py     # Data ingestion
│   │   ├── s01_link.py       # Panel construction
│   │   ├── s02_features.py   # Feature engineering
│   │   ├── s03_estimation.py # SEM estimation
│   │   ├── s04_robustness.py # Robustness checks
│   │   └── s05_figures.py    # Figure generation
│   └── capacity_sem/         # Core analysis modules
│       ├── data/             # Data loading functions
│       ├── features/         # Feature computation
│       ├── models/           # SEM specifications
│       └── utils/            # Utilities
├── manuscript_quarto/        # Quarto manuscript (outputs to _output/)
├── data_raw/                 # Raw data (not tracked)
├── data_work/                # Working data
├── figures/                  # Output figures
├── doc/                      # Documentation
└── tests/                    # Test suite
```

## Pipeline Commands

| Command | Description |
|---------|-------------|
| `ingest_data` | Load QPR and external data |
| `build_panel` | Construct analysis panel |
| `compute_features` | Calculate indicators |
| `run_estimation` | Fit SEM models |
| `run_robustness` | Run robustness checks |
| `make_figures` | Generate figures |
| `run_all` | Run complete pipeline |
| `list_models` | List available SEM specifications |

## Key Results

[Summary of key findings to be added after analysis]

## Documentation

- [CLAUDE.md](CLAUDE.md) - AI agent instructions
- [doc/PIPELINE.md](doc/PIPELINE.md) - Pipeline documentation
- [doc/METHODOLOGY.md](doc/METHODOLOGY.md) - SEM methodology
- [doc/DATA_DICTIONARY.md](doc/DATA_DICTIONARY.md) - Variable definitions

## Citation

[Citation information to be added]

## License

[License information to be added]
