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
| [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md) | Synthetic review methodology | Pre-submission review |
| [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md) | Revision tracking | Review status |
| [reviews/README.md](reviews/README.md) | Review cycle index | Review history |
| [CHANGELOG.md](CHANGELOG.md) | Change history | Tracking changes |

---

## Pipeline Commands

### Data Processing (Standardized - RECOMMENDED)

```bash
python src/pipeline.py ingest_data              # Load QPR and external data
python src/pipeline.py standardize_data         # Standardize with fixed denominators
python src/pipeline.py build_panel              # Construct analysis panel
python src/pipeline.py build_features_std       # Build standardized features
python src/pipeline.py aggregate_program_types  # Aggregate program types
```

### Legacy Data Processing (DEPRECATED)

```bash
python src/pipeline.py compute_features  # OLD: Uses dynamic denominators
```

### Analysis

```bash
python src/pipeline.py run_survival      # Time-varying survival analysis
python src/pipeline.py run_alternatives  # Alternative modeling approaches
python src/pipeline.py run_estimation --model exp_optimal_v1  # SEM estimation
python src/pipeline.py run_robustness    # Robustness checks
```

### Output

```bash
python src/pipeline.py make_figures      # Publication figures
python src/pipeline.py capacity_summary  # Corrected capacity summary
```

### Complete Pipeline

```bash
python src/pipeline.py run_all
```

### Review Management

```bash
python src/pipeline.py review_status        # Check current review status
python src/pipeline.py review_new --focus par_general
python src/pipeline.py review_verify        # PAR compliance checks
python src/pipeline.py review_archive
python src/pipeline.py review_report
```

---

## Source Code

| Directory | Purpose |
|-----------|---------|
| `src/pipeline.py` | Main CLI entry point |
| `src/config.py` | Configuration constants |
| `src/stages/` | Pipeline stage modules (s00-s07) |
| `src/capacity_sem/` | Core analysis modules |

### Extended Analysis Modules

| Module | Purpose |
|--------|---------|
| `src/stages/s00b_standardize.py` | Data standardization with fixed denominators |
| `src/stages/s01b_features.py` | Build features from standardized data |
| `src/stages/s01c_program_types.py` | Aggregate program type features |
| `src/stages/s03b_survival_estimation.py` | Time-varying survival analysis |
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
| `data_work/qpr_clean.parquet` | Cleaned QPR data with QA flags |
| `data_work/qpr_quarterly.parquet` | Quarterly rollup with flows and cumulative totals |
| `data_work/qpr_standardized.parquet` | Standardized QPR with fixed denominators (NEW) |
| `data_work/panel.parquet` | Analysis panel |
| `data_work/panel_features.parquet` | Panel with computed features (legacy) |
| `data_work/panel_features_std.parquet` | Panel with standardized features (RECOMMENDED) |
| `data_work/panel_program_types.parquet` | Program type aggregations (NEW) |
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

## Synthetic Review System

The project includes a systematic peer review system for pre-submission manuscript validation using LLM-generated synthetic reviews.

### Overview

- **Purpose**: Identify methodological gaps, strengthen robustness, anticipate reviewer concerns
- **Focus Areas**: par_general (comprehensive), methods (methodology), policy (practitioner relevance), clarity (writing)
- **Documentation**: See [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md)

### Quick Workflow

1. **Generate Review**: `python src/pipeline.py review_new --focus par_general`
2. **Obtain LLM Review**: Send manuscript + embedded prompt to Claude/GPT-4
3. **Triage Comments**: Classify as VALID/ADDRESSED/SCOPE/INVALID in `manuscript_quarto/REVISION_TRACKER.md`
4. **Implement Changes**: Address valid concerns, update manuscript
5. **Verify**: `python src/pipeline.py review_verify` (includes PAR compliance checks)
6. **Archive**: `python src/pipeline.py review_archive`

### PAR Compliance Checks

The `review_verify` command automatically checks:

- Word count ≤ 8,000 (currently ~7,851)
- No "this study" self-references (currently 0)
- Evidence for Practice section present
- Abstract ≤ 150 words

### Review History

All completed reviews are archived in `reviews/archive/` with format:
`review_NN_YYYY-MM-DD_FOCUS.md`

Track all cycles: `python src/pipeline.py review_report`

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

---

## Analysis Reports

| Report | Purpose |
|--------|---------|
| [reports/VELOCITY_DIAGNOSTICS_REPORT.md](reports/VELOCITY_DIAGNOSTICS_REPORT.md) | Velocity calculation diagnostics |
| [reports/MEASUREMENT_VALIDATION_REPORT.md](reports/MEASUREMENT_VALIDATION_REPORT.md) | Measurement validation results |
| [reports/PREDICTOR_DISCOVERY_REPORT.md](reports/PREDICTOR_DISCOVERY_REPORT.md) | Predictor analysis findings |
| [reports/DATA_QUALITY_FIXES.md](reports/DATA_QUALITY_FIXES.md) | Data quality issues and fixes |
| [RESEARCH_SYNTHESIS_REPORT.md](RESEARCH_SYNTHESIS_REPORT.md) | Comprehensive research synthesis |

---

## Standalone Analysis Scripts

Extended analysis scripts are in `scripts/`:

```bash
python scripts/run_multistage_analysis.py     # Multi-stage efficiency
python scripts/run_trajectory_clustering.py   # Velocity trajectory clustering
python scripts/run_meta_analysis.py           # Meta-analysis of velocity effects
```

---

## Archive

Historical documentation is preserved in `archive/`:

| File | Original Name |
|------|---------------|
| [archive/analysis_logs/01_INITIAL_SETUP.md](archive/analysis_logs/01_INITIAL_SETUP.md) | PHASE1_WEEK1_SUMMARY |
| [archive/analysis_logs/02_MULTISTAGE_EFFICIENCY.md](archive/analysis_logs/02_MULTISTAGE_EFFICIENCY.md) | PHASE2_WEEK3_SUMMARY |
| [archive/analysis_logs/03_BOTTLENECK_ANALYSIS.md](archive/analysis_logs/03_BOTTLENECK_ANALYSIS.md) | PHASE2_WEEK4_SUMMARY |
| [archive/analysis_logs/04_VELOCITY_TRAJECTORIES.md](archive/analysis_logs/04_VELOCITY_TRAJECTORIES.md) | PHASE2_WEEK5_SUMMARY |
| [archive/analysis_logs/05_LEARNING_CURVES.md](archive/analysis_logs/05_LEARNING_CURVES.md) | PHASE2_WEEK6_SUMMARY |
| [archive/ETL_STANDARDIZATION_PROPOSAL.md](archive/ETL_STANDARDIZATION_PROPOSAL.md) | Original proposal (superseded) |
| [archive/ANALYSIS_COMPARISON_REPORT.md](archive/ANALYSIS_COMPARISON_REPORT.md) | Kaifa vs survival comparison |
