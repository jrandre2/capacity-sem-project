# Pipeline Documentation

## Overview

The Capacity-SEM analysis pipeline consists of six sequential stages that process raw QPR data through to publication figures.

## Pipeline Stages

### Stage 0: Data Ingestion (`s00_ingest.py`)

**Command**: `python src/pipeline.py ingest_data`

**Purpose**: Load QPR data and external covariates.

**Inputs**:
- `data_raw/qpr_data.csv` - Raw QPR export

**Outputs**:
- `data_work/qpr_raw.parquet` - Validated QPR data
- `data_work/covariates.parquet` - Combined external covariates (population, severity, employment)

**Key Functions**:
- `ingest_qpr_data()` - Load and validate QPR CSV
- `ingest_covariates()` - Load external datasets

---

### Stage 1: Panel Construction (`s01_link.py`)

**Command**: `python src/pipeline.py build_panel`

**Purpose**: Link data sources and construct analysis panel.

**Inputs**:
- `data_work/qpr_raw.parquet`
- `data_work/*.parquet` (covariates)

**Outputs**:
- `data_work/panel.parquet` - Grantee-disaster level panel

**Key Functions**:
- `create_grantee_disaster_panel()` - Aggregate to analysis unit
- `merge_population()` - Add population covariates
- `merge_severity()` - Add disaster severity
- `compute_ratios()` - Calculate financial ratios
- `scale_covariates()` - Z-score standardization

---

### Stage 2: Feature Engineering (`s02_features.py`)

**Command**: `python src/pipeline.py compute_features`

**Purpose**: Compute timeliness metrics, experience indicators, and program stratification.

**Inputs**:
- `data_work/panel.parquet`
- `data_work/qpr_raw.parquet`

**Outputs**:
- `data_work/panel_features.parquet` - Panel with all computed features

**Key Functions**:
- `compute_timeliness_features()` - Duration, timeliness metrics
- `compute_additional_timeliness_metrics()` - Progress rate, CV, log duration
- `build_experience_dataset()` - Prior grant experience
- `add_program_type_column()` - Activity type classification

---

### Stage 3: SEM Estimation (`s03_estimation.py`)

**Command**: `python src/pipeline.py run_estimation [--model MODEL] [--subset SUBSET]`

**Purpose**: Fit structural equation models.

**Inputs**:
- `data_work/panel_features.parquet`

**Outputs**:
- `data_work/diagnostics/estimates_*.csv` - Parameter estimates
- `data_work/diagnostics/fit_stats_*.csv` - Fit statistics

**Options**:
- `--model, -m`: Model specification (default: `exp_optimal_v1`)
- `--subset, -s`: Government type (`all`, `state`, `local`)

**Key Functions**:
- `run_estimation()` - Fit and evaluate model
- `run_model_comparison()` - Compare specifications
- `run_subset_comparison()` - Compare government types

---

### Stage 4: Robustness Checks (`s04_robustness.py`)

**Command**: `python src/pipeline.py run_robustness`

**Purpose**: Run alternative specifications and sensitivity analyses.

**Inputs**:
- `data_work/panel_features.parquet`

**Outputs**:
- `data_work/diagnostics/robustness_specifications.csv`
- `data_work/diagnostics/robustness_subsets.csv`
- `data_work/diagnostics/robustness_sample_sensitivity.csv`
- `data_work/diagnostics/robustness_covariates.csv`

**Key Functions**:
- `run_alternative_specifications()` - Test model variants
- `run_subset_robustness()` - State vs. local comparison
- `run_sample_sensitivity()` - Vary minimum quarters
- `run_covariate_robustness()` - Test covariate inclusion

---

### Stage 5: Figure Generation (`s05_figures.py`)

**Command**: `python src/pipeline.py make_figures`

**Purpose**: Generate publication-ready figures.

**Inputs**:
- `data_work/panel_features.parquet`
- `data_work/diagnostics/*.csv`

**Outputs**:
- `figures/fig_descriptive.png`
- `figures/fig_model_comparison.png`
- `figures/fig_subset_comparison.png`
- `figures/fig_sensitivity.png`
- `figures/fig_path_diagram.png`

**Options**:
- `--style, -s`: Figure style (`publication`, `presentation`)

---

## Full Pipeline Execution

```bash
# Run all stages sequentially
python src/pipeline.py run_all

# With options
python src/pipeline.py run_all --model exp_optimal_v1 --subset all
```

---

## Model Specifications

### Primary Models

| Name | Description | Indicators |
|------|-------------|------------|
| `exp_optimal_v1` | Recommended 2x2 model | Ratios → Capacity; Log Duration, CV → Outcome |
| `full` | Original 3x3 model | Includes Timeliness (has redundancy) |
| `improved_3x3` | Enhanced 3x3 | Adds Startup Lag, Time to 50% |

### List Available Models

```bash
python src/pipeline.py list_models
```

---

## Troubleshooting

### Missing semopy

```bash
pip install semopy
```

### Missing data files

Run ingestion stage first:
```bash
python src/pipeline.py ingest_data
```

### Empty panel

Check that QPR data file exists in `data_raw/` and has expected columns.
