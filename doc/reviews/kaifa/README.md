# Kaifa Manuscript Review Notes

**Manuscript**: `manuscript_kaifa_archive/`
**Status**: DOCX-linked review workspace with exploratory redraft
**Last Updated**: 2026-04-09

## Purpose

This review workspace attaches Kaifa Lu's updated SEM manuscript source DOCX to
the same unified review and triage system used by the active project manuscript.

Configured source document:

- `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_full_revision.docx`

## Commands

```bash
python src/pipeline.py review_ingest_docx --manuscript kaifa
python src/pipeline.py review_status --manuscript kaifa
python src/pipeline.py review_response --manuscript kaifa
python src/pipeline.py review_diff --manuscript kaifa
python src/pipeline.py review_archive --manuscript kaifa
```

Planning artifacts:

- `doc/reviews/kaifa/REVISION_PLAN.md`
- `doc/reviews/kaifa/REVIEW_02_PLAN.md`
- `doc/reviews/kaifa/REVIEW_03_PLAN.md`
- `doc/reviews/kaifa/REVIEW_04_PLAN.md` (pending)
- `doc/reviews/kaifa/QUANTITATIVE_AUDIT.md`
- `doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md`
- `doc/reviews/kaifa/REDESIGN_DECISION_MEMO.md`
- `manuscript_kaifa_archive/data/external_sem_exports/README.md`
- `manuscript_kaifa_archive/data/recovered_code_bundle/README.md`
- recovered Kaifa notebook analysis command:
  `python src/pipeline.py run_kaifa_recovered_analysis`
- provisional external SEM reconstruction command:
  `python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem`

## Review History

| Cycle | Focus | Status | Key Change |
|-------|-------|--------|------------|
| #1 | Initial synthetic review | Archived | Baseline feedback |
| #2 | Cleanup and transparency | Archived | Metacommentary removal, formatting, diagnostics |
| #3 | Structural credibility | Archived | Claim narrowing, terminology, measurement framing |
| #4 | Full journal-style (reject) | **Active** | Analytic provenance, design fork point |

## Current Import State

The original 2026-04-07 Word file seeded this review workflow, but the
current review target is the standalone `2026-04-09_full_revision.docx`
manuscript. The source Word file did not contain embedded comments or tracked
changes, so the tracker began as a manual AI-review scaffold. The current
workspace now also contains:

- a quantitative audit that reconciles the archived SEM outputs;
- a recovered-code audit that reconciles the later SEM notebook ZIP against
  the saved SEM export bundle;
- an exploratory Quarto redraft aligned to those outputs;
- a redesign memo that distinguishes the archival manuscript path from a
  full empirical rebuild.
- imported external SEM export bundles for the later Word manuscript:
  `baseline_sem/` as the primary expanded cross-sectional SEM bundle and
  `baseline_sem_admin_4/` as a smaller robustness bundle.
- an imported recovered SEM notebook bundle that now provides the actual
  notebook source for the later Kaifa analysis, albeit with a remaining
  573-row versus 577-row input discrepancy.
- a narrowed reconstruction of that discrepancy: the missing 573-row sample
  membership is now identified as the 577-row recovered CSV minus four
  grantees, with a smaller remaining drift in several outcome columns.
- a temporary `PROVISIONAL_EXTERNAL_RECONSTRUCTION` workflow that refits the
  inferred one-factor and two-factor SEMs from those imported bundles while
  the original analysis notebook remains unavailable.

That is still useful because it:

- records the source file in review metadata
- gives the manuscript an auditable review home
- preserves the recovered notebook and its reproducibility gaps explicitly
- allows re-import without changing the workflow if a commented DOCX arrives later

## Working Rule

Treat this manuscript as archived lineage material. Use this workspace for
provenance, audit, comparison, and comment tracking, not as the canonical
analysis manuscript.
