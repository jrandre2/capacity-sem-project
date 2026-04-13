# Manuscript Guide

## Which Manuscript Is Active

- Primary research manuscript: `manuscript_quarto/` (cross-sectional SEM, N=573)
- Archived survival draft: `manuscript_velocity/` (superseded)

The active Capacity-SEM paper is `manuscript_quarto/index.qmd`. The velocity manuscript in `manuscript_velocity/` documents the complementary survival analysis but is no longer the primary manuscript.

## Current Writing Rule

The manuscript is in rewrite after the December 27, 2025 duration bug invalidated the earlier positive velocity narrative.

Do not treat the following as current findings unless you are explicitly discussing superseded historical results:
- contingent-capacity headline claims
- strong late-phase velocity effects
- wildfire or administration subgroup claims as settled results
- meta-analytic summaries built from pre-fix heterogeneity estimates

Current framing should align with [PROJECT_STATUS.md](PROJECT_STATUS.md):
- primary SEM finding: burden->timeliness beta=0.266 (N=573)
- complementary survival finding: velocity effects are null or near-null
- subgroup signals are exploratory

## File Map

| Path | Purpose |
|------|---------|
| `manuscript_quarto/index.qmd` | Main paper (cross-sectional SEM) |
| `manuscript_quarto/appendix-a-data.qmd` | Data appendix |
| `manuscript_quarto/appendix-b-methods.qmd` | Methods appendix |
| `manuscript_quarto/appendix-c-robustness.qmd` | Robustness appendix |
| `manuscript_quarto/REVISION_TRACKER.md` | Review and verification tracker |
| `manuscript_quarto/render_all.sh` | Multi-format render script |
| `manuscript_velocity/index.qmd` | Archived survival draft |

## Rendering

```bash
cd manuscript_quarto
./render_all.sh
```

Re-render without changing upstream analysis inputs:

```bash
cd manuscript_quarto
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh
```

## Review Commands

```bash
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py review_new --manuscript quarto --focus par_general
```

Current tracker status should be read from:
- `manuscript_quarto/REVISION_TRACKER.md`
- `doc/MANUSCRIPT_REVISION_CHECKLIST.md`

## Writing Guardrails

- Present corrected findings directly.
- Keep legacy positive results explicitly labeled as invalidated when mentioned.
- Avoid internal “SEM vs survival” victory framing; present each method on its own merits.
- Separate confirmed results from exploratory subgroup patterns.
- `manuscript_quarto/` is the primary manuscript; `manuscript_velocity/` is archived.
