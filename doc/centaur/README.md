# CENTAUR Import

Last updated: 2026-04-09

## Purpose

This repository now vendors the reusable core of the CENTAUR framework from:

`/Volumes/T9/Projects/Research Project Management Software`

The import is intentionally namespaced under `src/centaur/` so the working
Capacity-SEM analysis pipeline remains the authoritative project workflow.

## What Is Imported

- `src/centaur/analysis/`
  - analysis engine protocol and result container
  - engine registry
  - Python and R estimation engines
  - specification helpers
- `src/centaur/agents/`
  - project analyzer
  - structure mapper
  - migration planner
  - migration executor
- `src/centaur/utils/`
  - cache manager
  - validation helpers
  - synthetic data generator
  - generic path and formatting helpers
  - figure styling helpers
  - spatial cross-validation helpers
  - DOCX feedback parsing helpers
- `src/centaur/gui/`
  - FastAPI dashboard package
  - stage, QA, cache, review, and supervision services
- `src/centaur/llm/`
  - provider interface
  - Anthropic and OpenAI adapters
  - manuscript drafting prompts
- `src/centaur/spatial/`
  - spatial I/O helpers
  - CRS utilities
  - distance and neighborhood utilities
- `src/centaur/stages/`
  - ingestion through estimation
  - robustness, figures, manuscript validation
  - reviews, journal parsing, AI-assisted writing
- `manuscript_quarto/`
  - Quarto manuscript scaffold
  - journal profiles and variants
  - review tracker and draft workspace
- `tools/bin/quarto`
  - vendored Quarto launcher used by the scaffold render scripts

## Current Scope

The previously deferred GUI, LLM stack, spatial module, and full
stage/manuscript scaffold are now imported and wired into the host CLI.

Remaining limitations are operational rather than structural:

- optional web dependencies are still required to launch `centaur gui`
- LLM drafting commands still require provider SDKs and credentials
- spatial workflows still require the geospatial Python stack
- the active Capacity-SEM analysis pipeline remains separate from the vendored CENTAUR workflow

## Isolation Rules

- The current project pipeline remains in `src/pipeline.py` and `src/stages/`.
- Vendored framework outputs are isolated under:
  - `data_work/centaur/`
  - `figures/centaur/`
  - `doc/centaur/`
- CENTAUR tooling is exposed through the host CLI as a separate command group.
- The vendored manuscript scaffold lives in `manuscript_quarto/`.
- The active Capacity-SEM paper remains in `manuscript_velocity/`.

## CLI Usage

Analyze a project:

```bash
python src/pipeline.py centaur analyze_project --path /path/to/project
```

Map a project to the CENTAUR template:

```bash
python src/pipeline.py centaur map_project --path /path/to/project
```

Generate a migration plan without executing it:

```bash
python src/pipeline.py centaur plan_migration --path /path/to/project --target /path/to/target
```

Inspect vendored analysis engines:

```bash
python src/pipeline.py centaur engines list
python src/pipeline.py centaur engines check
```

Discover and run vendored stages:

```bash
python src/pipeline.py centaur list_stages
python src/pipeline.py centaur run_stage s03_estimation
```

Use the manuscript and review scaffold:

```bash
python src/pipeline.py centaur journal_list
python src/pipeline.py centaur validate_submission --journal jeem
python src/pipeline.py centaur review_status
python src/pipeline.py centaur draft_abstract --dry-run
```

Launch the GUI:

```bash
python src/pipeline.py centaur gui
```

## Dependency Notes

The base analysis environment in this repo does not guarantee every optional
CENTAUR dependency.

- Install the combined project environment with:

```bash
pip install -r requirements.txt
```

- GUI: `fastapi`, `uvicorn`
- LLM drafting: `anthropic` or `openai`, plus provider credentials
- Spatial workflows: `geopandas`, `shapely`, `pyproj`, `fiona`
- Quarto rendering: system Quarto or the vendored wrapper in `tools/bin/quarto`

Non-GUI commands such as `engines`, `journal_list`, `review_status`, and
`list_stages` are kept import-safe so they still work when the web stack is not
installed.

## Working Rule

Use the vendored CENTAUR package for framework capabilities and project
analysis, stage orchestration, manuscript validation, and drafting support.
Use the top-level Capacity-SEM pipeline for actual disaster-recovery analysis
until the project intentionally migrates the research workflow into CENTAUR.
