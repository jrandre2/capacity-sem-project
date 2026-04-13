# Velocity Manuscript (Archived)

> **Note**: This manuscript has been superseded by `manuscript_quarto/`, which presents the primary cross-sectional SEM analysis. This directory is retained as documentation of the complementary survival analysis (time-varying Cox PH, N=142-151, null findings).

## Status

Archived. Superseded by `manuscript_quarto/`.

This directory contains the survival-analysis-focused manuscript draft. It documents the complementary longitudinal analysis but is not the primary research manuscript.

## Current Position

The manuscript is being rewritten around the corrected post-bug results:

- overall velocity effects are null or near-null in the corrected time-varying survival models
- earlier strong heterogeneity claims were generated before the duration bug was fixed
- remaining subgroup signals are exploratory

The main manuscript and appendices now function as rewrite scaffolding rather than a stale pre-fix paper.

## Key Files

| File | Purpose |
|------|---------|
| `index.qmd` | Main manuscript draft |
| `appendix-a-data.qmd` | Data and sample appendix |
| `appendix-b-methods.qmd` | Corrected methods appendix |
| `appendix-c-heterogeneity.qmd` | Rewrite placeholder for heterogeneity results |
| `appendix-d-meta-analysis.qmd` | Rewrite placeholder for meta-analysis |
| `REVISION_TRACKER.md` | Review and verification tracker |
| `render_all.sh` | Render helper |

## Rendering

```bash
./render_all.sh
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh
```

## Review

```bash
python ../src/pipeline.py review_status --manuscript velocity
python ../src/pipeline.py review_verify --manuscript velocity
```

## Rule

The primary manuscript is in `../manuscript_quarto/`. For the current trusted analytical position, use:
- `../doc/PROJECT_STATUS.md`
- `../doc/PIPELINE.md`

Do not treat this manuscript as the primary analysis; it documents the complementary survival findings only.
