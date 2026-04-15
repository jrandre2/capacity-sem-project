# Capacity-SEM Project

Analysis of administrative capacity and completion timing in HUD CDBG-DR disaster recovery programs.

## Current Status

- **Primary manuscript**: `manuscript_quarto/index.qmd` — "A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"
- **Contribution**: A six-item measurement-sensitivity audit protocol whose deliverable is a specification-curve dashboard, demonstrated on a national CDBG-DR dataset (543 local administering jurisdictions primary; N=573 with state agencies supplementary; 151 grantee-disaster pairs for cross-framework comparison)
- **Headline finding**: The capacity-timeliness coefficient is *not stably identified* under principled measurement perturbations; the dashboard spans positive (β = +0.266 reference; +0.297 residualized; +0.257 local-only), near-zero (β ≈ 0 reconstructed-panel fixed-horizon; HR ≈ 1.0 time-varying Cox; β = 0.132 n.s. non-suppressed-QCEW), and negative (β = −0.244 mature-only; −0.443 raw counts; −0.600 financial-ratio bridge) estimates. Current capacity findings in CDBG-DR are not robust enough for benchmarking or policy anchoring.
- **Target journal**: Public Administration Review (PAR)
- **Review cycles completed**: Ten synthetic peer reviews (R1–R10), all closed; structural pivot at R6 from "instability narrative" to "audit-protocol contribution"; central claim recast at R10 from "near zero" to "not stably identified". Response letters and revision tracker in `doc/reviews/quarto/`
- **Archived manuscripts**: `manuscript_velocity/` (survival-only draft, superseded) and `manuscript_kaifa_archive/` (original SEM draft)

See [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md) for the current analytical state and [manuscript_quarto/REVISION_TRACKER.md](manuscript_quarto/REVISION_TRACKER.md) for the current review cycle disposition.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quarto is an external CLI. Use a system install or the vendored wrapper in `tools/bin/quarto`.

## Recommended Workflow

Run the active pipeline:

```bash
python src/pipeline.py run_all
```

Or run the current stages explicitly:

```bash
python src/pipeline.py ingest_data
python src/pipeline.py standardize_data
python src/pipeline.py build_panel
python src/pipeline.py build_features_std
python src/pipeline.py aggregate_program_types
python src/pipeline.py run_survival
```

Useful supporting commands:

```bash
python src/pipeline.py run_survival_threshold_sensitivity
python src/pipeline.py run_alternatives
python src/pipeline.py make_figures
python src/pipeline.py capacity_summary
```

Kaifa-style SEM analysis (generates the primary-manuscript sensitivity results):

```bash
python src/pipeline.py run_kaifa_recovered_analysis
```

Legacy SEM replication pipeline:

```bash
python src/pipeline.py run_all_legacy
python src/pipeline.py list_models
```

## Manuscript and Review

Primary manuscript in `manuscript_quarto/`:

```bash
cd manuscript_quarto
./render_all.sh                                  # all formats
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh     # skip pipeline re-run
quarto render index.qmd --to docx                # DOCX only
```

Review management:

```bash
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py review_new --manuscript quarto --focus par_general
python src/pipeline.py review_archive --manuscript quarto
python src/pipeline.py review_report
```

Review artifacts live in `doc/reviews/quarto/`:

- `INDEX.md` — review log (R1–R10)
- `archive/review_0N_*.md` — reviewer text per cycle
- `response_0N_*.md` / `.docx` — point-by-point response letters

## CENTAUR Integration

The vendored CENTAUR framework is namespaced under `src/centaur/` and does not replace the Capacity-SEM workflow. See [doc/centaur/README.md](doc/centaur/README.md).

```bash
python src/pipeline.py centaur --help
python src/pipeline.py centaur list_stages
python src/pipeline.py centaur review_status
```

## Repository Layout

```text
src/
  pipeline.py                 Host CLI
  stages/                     Capacity-SEM pipeline stages
  capacity_sem/               Core analysis modules (including kaifa_recovered_analysis.py)
  centaur/                    Vendored CENTAUR framework
doc/
  PROJECT_STATUS.md           Current state and next steps
  PIPELINE.md                 Current workflow guide
  METHODOLOGY.md              Analysis methods
  DATA_DICTIONARY.md          Variable definitions
  SYNTHETIC_REVIEW_PROCESS.md Review workflow
  reviews/quarto/             R1–R10 review archive, responses, tracker
manuscript_quarto/            Primary manuscript (PAR target)
manuscript_velocity/          Archived (superseded by manuscript_quarto/)
manuscript_kaifa_archive/     Archived original SEM draft
data_raw/                     Source datasets (read-only)
  svi_historical/             CDC/ATSDR SVI vintages 2000–2022 (downloaded 2026-04-14)
data_work/                    Derived data and diagnostics
  jurisdiction_disaster_year_svi.parquet   Per-jurisdiction disaster-year SVI assignments
  state_earliest_disaster_year.parquet     State → earliest disaster year (for vintage selection)
  sem_input_disaster_year_svi.parquet      SEM-ready data with disaster-year SVI
  fixed_horizon_outcomes.parquet           Quarter-8/12/16 expenditure shares per grantee
figures/                      Analysis figures
tests/                        Regression tests
```

## Documentation Map

- [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md) — current analytical state
- [doc/PIPELINE.md](doc/PIPELINE.md) — current commands and outputs
- [doc/METHODOLOGY.md](doc/METHODOLOGY.md) — SEM and survival methods
- [doc/DATA_DICTIONARY.md](doc/DATA_DICTIONARY.md) — variable definitions
- [doc/ETL_STANDARDIZATION.md](doc/ETL_STANDARDIZATION.md) — fixed-denominator standardization
- [doc/ANALYSIS_JOURNEY.md](doc/ANALYSIS_JOURNEY.md) — methodological history
- [doc/MANUSCRIPT_GUIDE.md](doc/MANUSCRIPT_GUIDE.md) — manuscript locations and writing rules
- [doc/SYNTHETIC_REVIEW_PROCESS.md](doc/SYNTHETIC_REVIEW_PROCESS.md) — review workflow
- [doc/CHANGELOG.md](doc/CHANGELOG.md) — cycle-by-cycle revision log

Historical reports and archived analyses remain in the repo for provenance; consult [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md) before citing any file predating the current revision.
