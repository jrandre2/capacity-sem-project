# Project Status

Last updated: 2026-04-12

## Current Position

This repository has undergone two critical bug fixes (December 2025 duration calculation, April 2026 ratio aggregation) that independently invalidated prior positive velocity findings. Both fixes confirmed the same null result.

- Active branch: `analysis/alternative-capacity-measures`
- Active analytical baseline: standardized, fixed-denominator survival workflow with ratio clipping
- Legacy workflow status: preserved for replication only
- Manuscript status: `manuscript_velocity/` remains in major revision and should not be treated as publication-ready

## Trusted Findings

These are the findings that should guide new analysis unless superseded by later verified runs:

- The original positive velocity results are not reliable (invalidated by two independent bugs).
- The standardized time-varying survival workflow is the current reference path.
- Overall velocity effects are null in the corrected time-varying models: Disb HR=1.001 [0.821, 1.221], p=0.991.
- Concordance is 0.723 (model discriminates via covariates, not capacity ratios).
- Bootstrap clustered SEs (1,000 iterations) confirm null finding with properly sized uncertainty.
- All ratio variables now use SUM aggregation across activities, $1K minimum denominator, and [0, 2] clipping.
- Any remaining subgroup effects should be treated as exploratory until revalidated on the cleaned pipeline.

## What Changed In This Cleanup Pass

This pass establishes a cleaner base for continuing analysis:

1. `python src/pipeline.py run_all` now reflects the active standardized workflow:
   - `ingest_data`
   - `standardize_data`
   - `build_panel`
   - `build_features_std`
   - `aggregate_program_types`
   - `run_survival`
2. The old SEM-oriented end-to-end flow is preserved as:
   - `python src/pipeline.py run_all_legacy`
3. Quarter-based feature construction now collapses mixed activity rows to one row per grantee-disaster-quarter before computing:
   - early-window velocity features
   - phase-specific velocity features
   - standardized survival reshaping inputs
4. Reusable CENTAUR framework modules are now vendored under `src/centaur/`:
   - analysis engines
   - project analysis and migration-planning agents
   - generic validation, caching, and synthetic-data utilities
   - host CLI access through `python src/pipeline.py centaur ...`
5. The deferred CENTAUR subsystems are now also imported and wired:
   - FastAPI GUI under `src/centaur/gui/`
   - LLM drafting stack under `src/centaur/llm/`
   - spatial module under `src/centaur/spatial/`
   - full stage stack under `src/centaur/stages/`
   - manuscript and journal scaffold under `manuscript_quarto/`

## Recommended Commands

### Active workflow

```bash
python src/pipeline.py run_all
```

### Active workflow, step-by-step

```bash
python src/pipeline.py ingest_data
python src/pipeline.py standardize_data
python src/pipeline.py build_panel
python src/pipeline.py build_features_std
python src/pipeline.py aggregate_program_types
python src/pipeline.py run_survival
```

### Legacy replication workflow

```bash
python src/pipeline.py run_all_legacy
```

## Current Risks / Follow-Up Work

These items still need cleanup after this pass:

- Active documentation is now aligned around the corrected null-finding story, but historical reports remain in the repo for provenance and should continue to be clearly labeled when cited.
- `run_alternatives`, figures, and manuscript render paths still need a full audit to ensure they all follow the active standardized workflow.
- The velocity manuscript is now a rewrite draft, not a stale results narrative, but it still needs a full substantive rewrite before submission.
- Output management is still analysis-workspace style rather than run-manifest style.
- The new combined `requirements.txt` is not yet CI-tested as a fresh environment build for the full optional CENTAUR stack.

## Working Rule

For new analysis, treat the standardized survival path as canonical and the SEM path as historical/replication infrastructure.
