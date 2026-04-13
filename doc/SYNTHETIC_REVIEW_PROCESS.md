# Synthetic Review Process

**Status**: Active
**Last Updated**: 2026-04-09

This workflow is currently used for the primary manuscript in
`manuscript_quarto/`. The same review engine also supports actual-review and
DOCX-import workflows for archived manuscripts such as `kaifa` and `velocity`.

## Current Scope

| Manuscript | Directory | Tracker | Status |
|------------|-----------|---------|--------|
| Quarto (primary) | `manuscript_quarto/` | `manuscript_quarto/REVISION_TRACKER.md` | Pre-submission |
| Velocity (archived) | `manuscript_velocity/` | `manuscript_velocity/REVISION_TRACKER.md` | Archived |

The purpose of synthetic review is to:
- pressure-test the SEM framing and burden->timeliness findings
- verify that the complementary survival null is presented clearly
- catch stale pre-fix claims that remain in the manuscript
- improve clarity, caveats, and evidentiary discipline

## Core Commands

```bash
python src/pipeline.py review_new --manuscript quarto --focus par_general
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_diff --manuscript quarto
python src/pipeline.py review_response --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py review_archive --manuscript quarto
python src/pipeline.py review_report
python src/pipeline.py review_ingest_docx --manuscript kaifa
```

## Workflow

1. Generate a review cycle with `review_new`.
2. Obtain the external or synthetic review text.
3. Triage comments in `manuscript_quarto/REVISION_TRACKER.md`.
4. Make manuscript and analysis changes.
5. Run `review_verify`.
6. Archive the cycle when complete.

For actual-review intake from Word files:

1. Place the source `.docx` under the manuscript archive.
2. Run `review_ingest_docx --manuscript kaifa`.
3. Triage imported items in `manuscript_kaifa_archive/REVISION_TRACKER.md`.
4. Use `review_response` and `review_diff` as needed.

## Status Labels

| Status | Meaning |
|--------|---------|
| `VALID - ACTION NEEDED` | Legitimate concern that requires changes |
| `ALREADY ADDRESSED` | Concern is already covered in the manuscript |
| `BEYOND SCOPE` | Valid concern, deferred with explicit reason |
| `INVALID` | Concern is based on a misunderstanding or false premise |

## Review Focus Guidance

| Focus | Use It For |
|-------|------------|
| `par_general` | Overall contribution, framing, and journal readiness |
| `methods` | Correctness of the standardized survival workflow and claims |
| `policy` | Practitioner relevance without overclaiming |
| `clarity` | Readability and removal of stale pre-fix narrative |

## What Reviewers Should Be Checking Now

- Does the manuscript clearly present the SEM findings (beta=0.266) alongside the complementary survival null?
- Are the cross-sectional SEM results and their limitations presented honestly?
- Are subgroup findings treated as exploratory rather than definitive?
- Do tables, figures, and appendices align with the current SEM-primary framing?
- Does the discussion avoid overclaiming what capacity measures can explain?

## Related Files

- [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md)
- [reviews/README.md](reviews/README.md)
- [reviews/velocity/README.md](reviews/velocity/README.md)
- [reviews/quarto/README.md](reviews/quarto/README.md)
