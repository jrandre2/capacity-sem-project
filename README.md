# Capacity-SEM Project

Analysis of administrative capacity and completion timing in HUD CDBG-DR disaster recovery programs.

## Current Status

- Canonical workflow: standardized, quarter-corrected survival analysis
- Legacy workflow: SEM pipeline retained for replication and comparison only
- Main finding status: the earlier positive velocity results were invalidated by the December 27, 2025 duration bug
- Active manuscript: `manuscript_velocity/` in major revision
- Vendored framework: CENTAUR is available under `src/centaur/` as a separate toolchain

Start with [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md). That file is the source of truth for what is currently trusted, what is historical, and what should be run next.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes the core Capacity-SEM stack plus the optional CENTAUR GUI, spatial, and LLM dependencies.
- Quarto is still an external CLI. Use a system install or the vendored wrapper in `tools/bin/quarto`.

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

Legacy SEM workflow:

```bash
python src/pipeline.py run_all_legacy
python src/pipeline.py list_models
```

## Manuscripts And Review

- Active research manuscript: `manuscript_velocity/`
- Vendored CENTAUR scaffold: `manuscript_quarto/`

Review management now uses the unified CENTAUR review engine through the host CLI.
The active rewrite lives in `manuscript_velocity/`, and the archived Kaifa SEM
lineage can be attached to the same workflow through DOCX import.

```bash
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_diff --manuscript velocity
python src/pipeline.py review_response --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_report
python src/pipeline.py review_ingest_docx --manuscript kaifa
```

Render the active manuscript:

```bash
cd manuscript_velocity
./render_all.sh
```

## CENTAUR Integration

The imported CENTAUR framework is namespaced and does not replace the Capacity-SEM workflow.

Examples:

```bash
python src/pipeline.py centaur --help
python src/pipeline.py centaur list_stages
python src/pipeline.py centaur validate_submission --journal jeem
python src/pipeline.py centaur review_status
python src/pipeline.py centaur review_diff
python src/pipeline.py centaur review_ingest_docx --manuscript kaifa --dry-run
python src/pipeline.py centaur analyze_project --path /path/to/project
```

See [doc/centaur/README.md](doc/centaur/README.md).

## Repository Layout

```text
src/
  pipeline.py                 Host CLI
  stages/                     Capacity-SEM pipeline stages
  capacity_sem/               Core analysis modules
  centaur/                    Vendored CENTAUR framework
doc/
  PROJECT_STATUS.md           Current state and next steps
  PIPELINE.md                 Current workflow guide
  METHODOLOGY.md              Analysis methods
  centaur/                    Vendored CENTAUR docs
manuscript_velocity/          Active manuscript draft
manuscript_quarto/            Vendored CENTAUR scaffold
data_work/                    Derived data and diagnostics
figures/                      Analysis figures
tests/                        Regression tests
```

## Documentation Map

- [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md): current analytical state
- [doc/PIPELINE.md](doc/PIPELINE.md): current commands and outputs
- [doc/METHODOLOGY.md](doc/METHODOLOGY.md): survival and SEM methods
- [doc/ETL_STANDARDIZATION.md](doc/ETL_STANDARDIZATION.md): fixed-denominator standardization
- [doc/ANALYSIS_JOURNEY.md](doc/ANALYSIS_JOURNEY.md): methodological history and bug discovery
- [doc/MANUSCRIPT_GUIDE.md](doc/MANUSCRIPT_GUIDE.md): manuscript locations and writing rules
- [doc/SYNTHETIC_REVIEW_PROCESS.md](doc/SYNTHETIC_REVIEW_PROCESS.md): review workflow

Historical reports and archived analyses remain in the repo, but many predate the duration bug fix. Treat them as historical unless [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md) says otherwise.
