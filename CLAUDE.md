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

## Current Methodology: Survival Analysis

The manuscript uses **survival analysis** (Cox Proportional Hazards, Accelerated Failure Time models) to analyze disaster recovery completion timing. SEM infrastructure remains for sensitivity analysis but is not the primary methodology.

### Why Survival Analysis?

With 73.7% of CDBG-DR programs incomplete at the 95% threshold, standard regression approaches face a censoring problem. Survival analysis properly handles right-censored observations while utilizing the full sample.

### Key Results

| Model | Disbursement HR/TR | p-value | Expenditure HR/TR | p-value |
|-------|-------------------|---------|-------------------|---------|
| Cox PH | 4.367 | 0.006 | 0.958 | 0.626 |
| AFT Lognormal | 0.157 | <0.001 | 1.008 | 0.954 |

- **Disbursement ratio**: Significant predictor (HR = 4.37, p = 0.006)
- **Expenditure ratio**: Not significant
- **Sample**: N=152 grantee-disaster pairs (with proper censoring)

### Capacity Indicators

- `Ratio_disbursed_to_obligated`: Cumulative mean ratio of disbursed to obligated funds
- `Ratio_expended_to_disbursed`: Cumulative mean ratio of expended to disbursed funds

---

## Manuscript

### Location and Rendering

- **Primary manuscript**: `manuscript_quarto/index.qmd`
- **Archived Kaifa SEM manuscript**: `manuscript_kaifa_archive/`
- **Output**: `manuscript_quarto/_output/`

```bash
cd manuscript_quarto
./render_all.sh                      # All formats (HTML, PDF, DOCX)
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh  # Skip pipeline re-run
```

### Target Journal: Public Administration Review (PAR)

| Requirement | Value |
|-------------|-------|
| Word limit | 8,000 (including abstract, endnotes, references) |
| Abstract | 150 words maximum |
| Font | 12-point Times New Roman |
| Spacing | Double-spaced |
| Margins | 1 inch |
| Citations | Chicago Author-Date (16th ed.) |
| Reference names | Full first names required |
| Special section | Evidence for Practice (3-5 bullet points) |
| Review type | Blind (no author identification) |

### Manuscript Writing Rules

#### DO NOT

- Use "this study" self-references — present findings directly
- Compare to internal prior work (Kaifa SEM manuscript)
- Use metacommentary ("advances the literature", "first application", "most robust estimates")
- Reference "latent constructs" or "complex measures" comparatively
- Include "why survival analysis succeeds where SEM fails" framing
- Add comparisons to "prior approaches" when meaning internal work

#### DO

- Present findings directly without self-referential framing
- Reference legitimate external literature appropriately (GAO, HUD, academic publications)
- Let the methodology speak for itself
- Keep robustness comparisons in appendices (appendix-c)

#### Examples

| Avoid | Use Instead |
|-------|-------------|
| "This study examines..." | "Government administrative capacity affects..." |
| "This approach advances the literature..." | [Simply present the analysis] |
| "Prior latent variable approaches may overcomplicate..." | [Remove or cite external literature] |
| "Unlike traditional SEM approaches..." | [Present survival analysis on its own merits] |

### Legitimate External References

Citing published research is appropriate:
- GAO reports: `[@gao2019]`
- HUD evaluations: `[@hud2020]`
- Academic literature: `[@gerber2022; @peacock2022]`

**What to avoid**: References to "prior SEM approaches" when meaning our internal Kaifa manuscript (now archived).

---

## Data Pipeline

```
data_raw/qpr_data.csv
    ↓
data_work/qpr_raw.parquet        (s00_ingest)
    ↓
data_work/qpr_clean.parquet      (s00_ingest)
    ↓
data_work/qpr_quarterly.parquet  (s00_ingest)
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

---

## SEM Infrastructure (Sensitivity Analysis Only)

The SEM codebase remains for robustness checks in appendix-c. For primary analysis, use survival analysis.

### Available SEM Models

Run `python src/pipeline.py list_models` for complete list (51+ specifications).

| Category | Count | Description |
|----------|-------|-------------|
| Core Models | 10 | Primary analysis specifications |
| Experimental | 15 | Alternative indicator combinations |
| Covariates | 6 | Models with control variables |
| Multi-Group | 3 | State vs. local comparison |

### Government Subsets

- `all` - Full sample
- `state` - State governments only (37 grantees)
- `local` - Local governments only (40 grantees)

---

## Archived Kaifa Manuscript

**Location**: `manuscript_kaifa_archive/`

This contains the original SEM-based manuscript with known methodological issues:

1. **Right-censoring**: 73.7% of observations lack valid Duration at 95% threshold
2. **Mathematical circularity**: Timeliness = 1/Duration as capacity indicator
3. **Grantee-level aggregation**: Averaging across disasters reduces variance

See `doc/ANALYSIS_COMPARISON_REPORT.md` for detailed comparison of methodologies.

---

## Key Data Files

| File | Purpose |
|------|---------|
| `data_work/panel_features.parquet` | Analysis-ready panel with all features |
| `data_work/qpr_quarterly.parquet` | Quarterly rollup with flows and cumulative totals |
| `data_work/diagnostics/*.csv` | Estimation results |
| `data_work/quality/*.csv` | Data quality reports |

See [doc/DATA_DICTIONARY.md](doc/DATA_DICTIONARY.md) for complete variable definitions.

---

## Documentation

| File | Content |
|------|---------|
| `doc/README.md` | Documentation index |
| `doc/PIPELINE.md` | Pipeline stages |
| `doc/METHODOLOGY.md` | Survival analysis and SEM methodology |
| `doc/DATA_DICTIONARY.md` | Variable definitions |
| `doc/MANUSCRIPT_GUIDE.md` | PAR formatting and writing rules |
| `doc/ANALYSIS_COMPARISON_REPORT.md` | Kaifa vs. survival analysis comparison |

---

## Troubleshooting

**semopy not found**: `pip install semopy`

**lifelines not found**: `pip install lifelines`

**Missing data**: Run `python src/pipeline.py run_all --demo` to use demo data

**Git issues**: Check that data files are properly gitignored
