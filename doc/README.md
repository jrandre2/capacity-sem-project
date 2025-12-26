# Documentation Index

**Project**: Capacity-SEM Analysis
**Purpose**: Survival analysis of government capacity effects on disaster recovery completion timing

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
| [METHODOLOGY.md](METHODOLOGY.md) | Survival analysis and SEM methodology | Understanding the models |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Variable definitions | Variable lookups |
| [MANUSCRIPT_GUIDE.md](MANUSCRIPT_GUIDE.md) | PAR formatting and writing rules | Writing the manuscript |
| [CHANGELOG.md](CHANGELOG.md) | Change history | Tracking changes |

---

## Pipeline Commands

### Data Processing

```bash
python src/pipeline.py ingest_data     # Load QPR, build quarterly rollups, and external data
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

### Extended Analysis Modules

| Module | Purpose |
|--------|---------|
| `src/stages/s03_manuscript_replication.py` | Kaifa's Models replication (experimental) |
| `src/capacity_sem/models/sem_multigroup.py` | Multi-group SEM and invariance testing |
| `src/capacity_sem/models/sem_mediation.py` | Indirect effect computation |
| `src/capacity_sem/models/sem_manuscript_replication.py` | Kaifa replication utilities |
| `src/capacity_sem/models/sem_specifications.py` | 51+ model specifications |

---

## Key Data Files

| File | Purpose |
|------|---------|
| `data_work/qpr_raw.parquet` | Ingested QPR data |
| `data_work/qpr_clean.parquet` | Cleaned QPR data with QA flags and imputed grantee state |
| `data_work/qpr_quarterly.parquet` | Quarterly rollup with flows and cumulative totals |
| `data_work/quality/qpr_quality_report.csv` | Row-level QPR quality summary |
| `data_work/quality/qpr_quarterly_quality_report.csv` | Quarterly rollup quality summary |
| `data_work/panel.parquet` | Analysis panel |
| `data_work/panel_features.parquet` | Panel with computed features |
| `data_work/diagnostics/` | Estimation results |
| `data_work/population.parquet` | Population covariates |
| `data_work/grantee_disaster_population.parquet` | Grantee-disaster population covariates |
| `data_work/severity.parquet` | Disaster severity covariates |
| `data_work/employment.parquet` | Employment covariates |

---

## Manuscript

### Location

| File | Purpose |
|------|---------|
| `manuscript_quarto/index.qmd` | Main manuscript |
| `manuscript_quarto/appendix-*.qmd` | Appendices A, B, C |
| `manuscript_quarto/references.bib` | BibTeX references |

### Rendering

```bash
cd manuscript_quarto
./render_all.sh                           # Full render
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh  # Re-render only
```

Output: `manuscript_quarto/_output/` (HTML, PDF, DOCX)

See [MANUSCRIPT_GUIDE.md](MANUSCRIPT_GUIDE.md) for PAR formatting and writing rules.

---

## SEM Infrastructure (Sensitivity Analysis)

The SEM codebase remains for robustness checks in Appendix C. For primary analysis, use survival analysis.

### Kaifa's Models Replication

For verifying the original (archived) manuscript methodology:

```bash
PYTHONPATH=src python3 src/stages/s03_manuscript_replication.py --subset state --model kaifa_3x3_full
```

**WARNING**: Experimental only. The original SEM approach had issues with right-censoring. See [METHODOLOGY.md](METHODOLOGY.md).

### Extended Robustness

```bash
python src/pipeline.py run_robustness --extended  # Multi-group and mediation
```

---

## Data Quality Notes

- Location coverage is limited to `Grantee State`; no county/city/FIPS/lat-lon fields exist in the QPR export.
- Some records lack `QPR Actual Quarter`, so they are excluded from quarterly rollups; see `QA_missing_qpr_actual_quarter`.
- Negative dollar values appear in the raw export (adjustments); they are flagged but not modified.
- Cumulative totals can decrease within a grantee-disaster series (revisions); flagged in the quarterly quality report.
- `Grantee State` is imputed from the grant code when missing (see `Grantee State Source`).

Use the quality reports to track current counts: `data_work/quality/qpr_quality_report.csv` and `data_work/quality/qpr_quarterly_quality_report.csv`.
Re-run `python src/pipeline.py ingest_data` to refresh the quality summaries after updating `qpr_data.csv`.
