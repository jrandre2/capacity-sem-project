# Documentation Index

Start here: [PROJECT_STATUS.md](PROJECT_STATUS.md)

That file is the current source of truth for:
- trusted findings
- invalidated findings
- canonical workflow
- active manuscript status
- immediate follow-up work

## Current Project Docs

| Document | Purpose |
|----------|---------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Current state, trusted findings, next steps |
| [PIPELINE.md](PIPELINE.md) | Active pipeline commands and outputs |
| [METHODOLOGY.md](METHODOLOGY.md) | Cross-sectional SEM as primary method; survival analysis as complementary comparison |
| [TIME_VARYING_SURVIVAL.md](TIME_VARYING_SURVIVAL.md) | Technical details for the time-varying survival implementation |
| [ETL_STANDARDIZATION.md](ETL_STANDARDIZATION.md) | Fixed-denominator standardization and quarter-level cleanup |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Variable and output definitions |

## Manuscript And Review Docs

| Document | Purpose |
|----------|---------|
| [MANUSCRIPT_GUIDE.md](MANUSCRIPT_GUIDE.md) | Active manuscript locations, render commands, writing guardrails |
| [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md) | Current manuscript rewrite checklist |
| [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md) | Review workflow for `manuscript_quarto/` |
| [reviews/README.md](reviews/README.md) | Review index and current manuscript coverage |
| [reviews/kaifa/README.md](reviews/kaifa/README.md) | Kaifa DOCX import and archived-manuscript review workspace |
| [reviews/kaifa/CODE_BUNDLE_AUDIT.md](reviews/kaifa/CODE_BUNDLE_AUDIT.md) | Audit of the recovered Kaifa SEM notebook ZIP and its reproducibility gaps |

## Project-Local Codex Skills

Committed under `.codex/skills/` (mirrored as slash commands in `~/.claude/commands/`):

| Skill | Purpose |
|-------|---------|
| [`manuscript-review-triage`](../.codex/skills/manuscript-review-triage/SKILL.md) | Turn reviews into tracker-ready comment sets, statuses, and revision plans |
| [`sem-manuscript-audit`](../.codex/skills/sem-manuscript-audit/SKILL.md) | Audit SEM manuscripts for reporting, design, timing, and interpretation problems |
| [`data-provenance-appendix`](../.codex/skills/data-provenance-appendix/SKILL.md) | Draft and audit appendix text for sources, linkage, harmonization, and data availability |
| [`docx-quarto-manuscript-bridge`](../.codex/skills/docx-quarto-manuscript-bridge/SKILL.md) | Convert manuscript structure between DOCX and Quarto, or reapply supplied Word-manuscript formatting to rendered DOCX outputs |

Manuscript-facing skills require standalone article prose and forbid revision-memo metacommentary inside manuscript text. For user-scope discovery outside this repo, copy or symlink into `~/.codex/skills/`.

## Vendored CENTAUR Docs

| Document | Purpose |
|----------|---------|
| [centaur/README.md](centaur/README.md) | What was imported and how it is exposed in this repo |
| [centaur/PIPELINE.md](centaur/PIPELINE.md) | Vendored CENTAUR stage/manuscript workflow |
| [centaur/TUTORIAL.md](centaur/TUTORIAL.md) | CENTAUR walkthrough |
| [centaur/GETTING_STARTED.md](centaur/GETTING_STARTED.md) | CENTAUR entry points |

Repo-specific rule: all vendored CENTAUR CLI examples run through `python src/pipeline.py centaur ...`, not a standalone CENTAUR binary.

## Historical Or Superseded Docs

These files are useful for provenance, but they are not authoritative for current analysis unless explicitly referenced by [PROJECT_STATUS.md](PROJECT_STATUS.md):

- `doc/archive/*` — including `RESEARCH_SYNTHESIS_REPORT.md`, `ANALYSIS_JOURNEY.md`, `STANDARDIZED_PIPELINE_TEST_RESULTS.md`, and prior-phase reports under `doc/archive/reports/`
- `manuscript_kaifa_archive/`
- `manuscript_velocity/` — archived survival-only draft
- `manuscript_predictors_archive/` — archived predictors draft
  The archive includes the received Word source manuscript at
  `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-07.docx`
  and the current standalone review target at
  `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_full_revision.docx`

## Quick Commands

```bash
python src/pipeline.py run_all
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_ingest_docx --manuscript kaifa
python src/pipeline.py run_kaifa_recovered_analysis
python src/pipeline.py centaur list_stages
```
