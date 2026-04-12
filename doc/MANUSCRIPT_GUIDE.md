# Manuscript Guide

## Which Manuscript Is Active

- Active research manuscript: `manuscript_velocity/`
- Vendored CENTAUR scaffold: `manuscript_quarto/`

The active Capacity-SEM paper is still in `manuscript_velocity/`. The CENTAUR scaffold is available for validation, drafting, and future migration work, but it is not yet the authoritative paper.

## Current Writing Rule

The manuscript is in rewrite after the December 27, 2025 duration bug invalidated the earlier positive velocity narrative.

Do not treat the following as current findings unless you are explicitly discussing superseded historical results:
- contingent-capacity headline claims
- strong late-phase velocity effects
- wildfire or administration subgroup claims as settled results
- meta-analytic summaries built from pre-fix heterogeneity estimates

Current framing should align with [PROJECT_STATUS.md](PROJECT_STATUS.md):
- overall velocity effects are null or near-null in corrected models
- subgroup signals are exploratory
- the manuscript is not submission-ready

## File Map

| Path | Purpose |
|------|---------|
| `manuscript_velocity/index.qmd` | Main paper draft |
| `manuscript_velocity/appendix-a-data.qmd` | Data and sample appendix |
| `manuscript_velocity/appendix-b-methods.qmd` | Methods appendix |
| `manuscript_velocity/appendix-c-heterogeneity.qmd` | Heterogeneity appendix |
| `manuscript_velocity/appendix-d-meta-analysis.qmd` | Historical meta-analysis appendix; needs caution |
| `manuscript_velocity/REVISION_TRACKER.md` | Review and verification tracker |
| `manuscript_velocity/render_all.sh` | Multi-format render script |

## Rendering

```bash
cd manuscript_velocity
./render_all.sh
```

Re-render without changing upstream analysis inputs:

```bash
cd manuscript_velocity
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh
```

## Review Commands

```bash
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_new --manuscript velocity --focus par_general
```

Current tracker status should be read from:
- `manuscript_velocity/REVISION_TRACKER.md`
- `doc/MANUSCRIPT_REVISION_CHECKLIST.md`

## Writing Guardrails

- Present corrected findings directly.
- Keep legacy positive results explicitly labeled as invalidated when mentioned.
- Avoid internal “SEM vs survival” victory framing; describe the bug and correction plainly.
- Separate confirmed results from exploratory subgroup patterns.
- Treat `manuscript_quarto/` as a separate scaffold unless the project intentionally migrates the live paper.

## If You Need The CENTAUR Scaffold

Use `manuscript_quarto/` for:
- journal profile validation
- AI-assisted draft generation
- review-cycle experiments in the vendored framework

See:
- [manuscript_quarto/README.md](../manuscript_quarto/README.md)
- [centaur/README.md](centaur/README.md)
