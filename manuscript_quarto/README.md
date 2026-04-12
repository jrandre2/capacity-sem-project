# CENTAUR Manuscript Scaffold

This directory is the vendored CENTAUR manuscript scaffold. It exists to support:

- Quarto manuscript rendering
- journal profile validation and comparison
- review tracking
- AI-assisted drafting experiments

It is not the authoritative research manuscript for the Capacity-SEM analysis. The active project manuscript remains in `manuscript_velocity/`.

## What Lives Here

- `_quarto.yml` and `_quarto-*.yml`: base Quarto book config and journal-specific profiles
- `index.qmd` and `appendix-*.qmd`: scaffold manuscript content
- `journal_configs/`: parsed journal requirement YAML files
- `drafts/`: AI-generated draft fragments from `python src/pipeline.py centaur draft_*`
- `variants/`: manuscript variant metadata and helpers
- `data/` and `figures/`: manuscript-side inputs used by the scaffold
- `render_all.sh`: multi-format render helper

## Working Rules

- Use this directory for CENTAUR validation, review, and drafting workflows.
- Use `manuscript_velocity/` for the active Capacity-SEM paper unless and until the project intentionally migrates the real manuscript into this scaffold.
- Treat `drafts/` as generated working material that still requires human review.

## Key Commands

```bash
python src/pipeline.py centaur journal_list
python src/pipeline.py centaur validate_submission --journal jeem
python src/pipeline.py centaur review_status
python src/pipeline.py centaur draft_abstract --dry-run
./render_all.sh --profile jeem
```
