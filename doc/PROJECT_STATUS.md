# Project Status

Last updated: 2026-04-13

## Current Position

The project now centers on a cross-sectional SEM manuscript (`manuscript_quarto/index.qmd`) analyzing administrative capacity across N=573 jurisdictions, with a complementary time-varying survival analysis (N=142-151 grantee-disaster pairs) providing a longitudinal comparison.

- Active branch: `analysis/alternative-capacity-measures`
- Primary methodology: cross-sectional SEM (N=573, primary finding: burden->timeliness beta=0.266)
- Complementary analysis: time-varying survival (HR~1.0, null finding)
- Primary manuscript: `manuscript_quarto/`
- Archived velocity manuscript: `manuscript_velocity/` (survival-only draft, superseded)
- Legacy workflow status: preserved for replication only

## Trusted Findings

These are the findings that should guide new analysis unless superseded by later verified runs:

- **Primary (SEM)**: Cross-sectional SEM finds burden->timeliness beta=0.266 (N=573 jurisdictions). Administrative burden is a significant predictor of recovery timeliness.
- **Complementary (Survival)**: Time-varying survival analysis finds null velocity effects: Disb HR=1.001 [0.821, 1.221], p=0.991 (N=142-151). Concordance is 0.723 (model discriminates via covariates, not capacity ratios).
- The original positive velocity results are not reliable (invalidated by two independent bugs).
- Bootstrap clustered SEs (1,000 iterations) confirm null survival finding with properly sized uncertainty.
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
- The velocity manuscript (`manuscript_velocity/`) is archived; the primary manuscript is now `manuscript_quarto/`.
- Output management is still analysis-workspace style rather than run-manifest style.
- The new combined `requirements.txt` is not yet CI-tested as a fresh environment build for the full optional CENTAUR stack.

## Working Rule

For new analysis, treat the cross-sectional SEM as the primary methodology and the survival analysis as a complementary longitudinal comparison. The standardized data pipeline supports both.
