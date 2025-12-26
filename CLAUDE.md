# Capacity-SEM Project - Claude Code Instructions

## Quick Start

```bash
source .venv/bin/activate  # REQUIRED for all scripts
```

### Common Commands

```bash
# Pipeline
python src/pipeline.py ingest_data
python src/pipeline.py build_panel
python src/pipeline.py compute_features
python src/pipeline.py run_estimation --model exp_optimal_v1
python src/pipeline.py run_robustness
python src/pipeline.py make_figures

# Run complete pipeline
python src/pipeline.py run_all

# Manuscript
cd manuscript_quarto && ./render_all.sh
```

## Key Concepts

### SEM Model

We use Structural Equation Modeling to analyze how government capacity affects disaster recovery outcomes:

- **Latent Capacity**: Measured by disbursement and expenditure ratios
- **Latent Outcome**: Measured by log duration and spending CV
- **Structural Path**: Capacity → Outcome relationship

### Key Model Specifications

| Model | Description |
|-------|-------------|
| `exp_optimal_v1` | Recommended: 2x2 factor structure, log duration, CV |
| `full` | Original 3x3 with Timeliness (has redundancy issues) |
| `reduced` | Model without Duration indicator (for robustness) |
| `improved_3x3` | Enhanced 3x3 addressing measurement concerns |

### Government Subsets

- `all` - Full sample
- `state` - State governments only (37 grantees)
- `local` - Local governments only (40 grantees)

## Data Pipeline

```
data_raw/qpr_data.csv
    ↓
data_work/qpr_raw.parquet        (s00_ingest)
    ↓
data_work/panel.parquet          (s01_link)
    ↓
data_work/panel_features.parquet (s02_features)
    ↓
data_work/diagnostics/           (s03_estimation)
    ↓
figures/*.png                    (s05_figures)
```

## Critical Constraints

### DO NOT

- Modify raw data in `data_raw/`
- Commit QPR data to git (contains sensitive information)
- Overwrite working parquet files manually

### ALWAYS

- Activate `.venv` before running scripts
- Use the pipeline CLI for data processing
- Run diagnostics after estimation changes
- Re-render Quarto after modifying `.qmd` files

## Data Files

| File | Purpose |
|------|---------|
| `data_work/qpr_raw.parquet` | Ingested QPR data |
| `data_work/panel.parquet` | Analysis panel |
| `data_work/panel_features.parquet` | Panel with computed features |
| `data_work/diagnostics/*.csv` | Estimation results |

## Manuscript

Location: `manuscript_quarto/`

### Rendering

```bash
cd manuscript_quarto
./render_all.sh                  # All formats (HTML, PDF, DOCX)
```

Output in `manuscript_quarto/_output/`

## Documentation

- `doc/README.md` - Documentation index
- `doc/PIPELINE.md` - Pipeline stages
- `doc/METHODOLOGY.md` - SEM methodology
- `doc/DATA_DICTIONARY.md` - Variable definitions

## Project Structure

```
capacity-sem-project/
├── CLAUDE.md              # This file
├── README.md              # Project overview
├── src/
│   ├── pipeline.py        # Main CLI
│   ├── config.py          # Configuration
│   ├── stages/            # Pipeline stages (s00-s05)
│   └── capacity_sem/      # Core analysis modules
├── manuscript_quarto/     # Quarto manuscript
├── data_raw/              # Raw data (gitignored)
├── data_work/             # Working data
├── figures/               # Output figures
└── doc/                   # Documentation
```

## Troubleshooting

**semopy not found**: `pip install semopy`

**Missing data**: Run `python src/pipeline.py run_all --demo` to use demo data

**Git issues**: Check that data files are properly gitignored
