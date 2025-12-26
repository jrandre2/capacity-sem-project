# Documentation Index

**Project**: Capacity-SEM Analysis
**Purpose**: Structural Equation Model analysis of government capacity effects on disaster recovery

---

## Quick Reference

| Document | Location | Purpose |
|----------|----------|---------|
| **CLAUDE.md** | Root | AI agent project instructions |
| **README.md** | Root | Project overview, setup, key commands |

---

## Core References

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [PIPELINE.md](PIPELINE.md) | Pipeline stages and CLI commands | Running the pipeline |
| [METHODOLOGY.md](METHODOLOGY.md) | SEM methodology and equations | Understanding the models |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Variable definitions | Variable lookups |
| [CHANGELOG.md](CHANGELOG.md) | Change history | Tracking changes |

---

## Pipeline Commands

### Data Processing

```bash
python src/pipeline.py ingest_data     # Load QPR and external data
python src/pipeline.py build_panel     # Construct analysis panel
python src/pipeline.py compute_features # Calculate indicators
```

### Analysis

```bash
python src/pipeline.py run_estimation --model exp_optimal_v1
python src/pipeline.py run_robustness
```

### Output

```bash
python src/pipeline.py make_figures
```

### Complete Pipeline

```bash
python src/pipeline.py run_all
```

---

## Source Code

| Directory | Purpose |
|-----------|---------|
| `src/pipeline.py` | Main CLI entry point |
| `src/config.py` | Configuration constants |
| `src/stages/` | Pipeline stage modules (s00-s05) |
| `src/capacity_sem/` | Core analysis modules |

---

## Key Data Files

| File | Purpose |
|------|---------|
| `data_work/qpr_raw.parquet` | Ingested QPR data |
| `data_work/panel.parquet` | Analysis panel |
| `data_work/panel_features.parquet` | Panel with computed features |
| `data_work/diagnostics/` | Estimation results |
