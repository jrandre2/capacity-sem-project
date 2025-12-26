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
- quarto - Manuscript rendering (HTML/PDF/DOCX article; optional)

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
├── manuscript_quarto/        # Quarto article manuscript (outputs to _output/)
├── data_raw/                 # Raw data (not tracked)
├── data_work/                # Working data
├── figures/                  # Output figures
├── doc/                      # Documentation
└── tests/                    # Test suite
```

## Pipeline Commands

| Command | Description |
|---------|-------------|
| `ingest_data` | Load QPR data, build quarterly rollups, and external covariates |
| `build_panel` | Construct analysis panel |
| `compute_features` | Calculate indicators |
| `run_estimation` | Fit SEM models |
| `run_robustness` | Run robustness checks |
| `make_figures` | Generate figures |
| `run_all` | Run complete pipeline |
| `list_models` | List available SEM specifications |

## Manuscript Rendering

```bash
cd manuscript_quarto
./render_all.sh
```

`render_all.sh` clears `_output/` before rendering to avoid stale files.

## Data Quality Notes

- Location coverage is limited to `Grantee State` (no county/city/FIPS/lat-lon fields in the QPR export).
- Some records lack `QPR Actual Quarter`, so they are excluded from quarterly rollups.
- Negative dollar values appear in the raw export (adjustments); they are flagged but not modified.
- Cumulative totals can decrease within a grantee-disaster series (revisions); flagged in the quarterly quality report.
- `Grantee State` is imputed from the grant code when missing; see `data_work/qpr_clean.parquet`.

Quality reports: `data_work/quality/qpr_quality_report.csv` and `data_work/quality/qpr_quarterly_quality_report.csv`.
Re-run `python src/pipeline.py ingest_data` to refresh these summaries after updating `qpr_data.csv`.

## Key Results

This project analyzes how government administrative capacity affects CDBG-DR disaster recovery outcomes using Structural Equation Modeling. Key findings include:

- **Capacity-Outcome Relationship**: The canonical pipeline (grantee-disaster level analysis) shows weak/non-significant capacity effects on recovery duration
- **Methodological Sensitivity**: Results are highly sensitive to analytical choices (unit of analysis, variable construction, duration censoring)
- **Measurement Challenges**: 73.7% of observations are right-censored at the 95% completion threshold
- **Data Quality**: 58% of grantee-disaster pairs show cumulative decrease anomalies due to legitimate adjustments

### Kaifa's Models (Experimental)

An experimental replication of Kaifa's original manuscript methodology is available for verification. Key differences include grantee-level aggregation and duration right-censoring. See `CLAUDE.md` for usage.

## Documentation

- [CLAUDE.md](CLAUDE.md) - AI agent instructions
- [doc/PIPELINE.md](doc/PIPELINE.md) - Pipeline documentation
- [doc/METHODOLOGY.md](doc/METHODOLOGY.md) - SEM methodology
- [doc/DATA_DICTIONARY.md](doc/DATA_DICTIONARY.md) - Variable definitions

## Citation

If you use this code or methodology, please cite:

```
Andrews, J. & Kaifa, [Year]. Modeling State and Local Governmental Capacity
in Managing CDBG-DR Funds: A Structural Equation Modeling (SEM) Approach.
[Journal/Conference information to be added upon publication]
```

## License

This project is provided for academic and research purposes. Please contact the authors for licensing information.
