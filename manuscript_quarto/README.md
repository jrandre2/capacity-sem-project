# Primary Manuscript: Administrative Throughput in Disaster Recovery

This directory contains the primary research manuscript for the Capacity-SEM project. The manuscript presents a cross-sectional SEM analysis (N=573 jurisdictions) of how administrative capacity affects disaster recovery timeliness, with a complementary survival analysis comparison.

**Target journal**: Public Administration Review (PAR), 8,000-word limit.

## What Lives Here

- `_quarto.yml` and `_quarto-*.yml`: base Quarto book config and journal-specific profiles
- `index.qmd` and `appendix-*.qmd`: manuscript content
- `journal_configs/`: parsed journal requirement YAML files
- `drafts/`: AI-generated draft fragments from `python src/pipeline.py centaur draft_*`
- `variants/`: manuscript variant metadata and helpers
- `data/` and `figures/`: manuscript-side inputs
- `render_all.sh`: multi-format render helper

## Working Rules

- This is the primary Capacity-SEM manuscript.
- `manuscript_velocity/` is archived (superseded survival-only draft).
- Treat `drafts/` as generated working material that still requires human review.

## Key Commands

```bash
cd manuscript_quarto && ./render_all.sh
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py centaur validate_submission --journal par
```
