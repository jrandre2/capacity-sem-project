# Synthetic Review Process

**Status**: Active
**Last Updated**: 2026-04-09

This workflow is currently used for the velocity manuscript in
`manuscript_velocity/`. The same review engine also supports actual-review and
DOCX-import workflows for archived manuscripts such as `kaifa`.

## Current Scope

| Manuscript | Directory | Tracker | Status |
|------------|-----------|---------|--------|
| Velocity | `manuscript_velocity/` | `manuscript_velocity/REVISION_TRACKER.md` | Active rewrite |

The purpose of synthetic review here is no longer to stress-test the old contingent-capacity claims. It is to:
- pressure-test the null-finding reframing
- catch stale pre-fix claims that remain in the manuscript
- improve clarity, caveats, and evidentiary discipline

## Core Commands

```bash
python src/pipeline.py review_new --manuscript velocity --focus par_general
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_diff --manuscript velocity
python src/pipeline.py review_response --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_archive --manuscript velocity
python src/pipeline.py review_report
python src/pipeline.py review_ingest_docx --manuscript kaifa
```

## Workflow

1. Generate a review cycle with `review_new`.
2. Obtain the external or synthetic review text.
3. Triage comments in `manuscript_velocity/REVISION_TRACKER.md`.
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

- Does the manuscript clearly distinguish trusted corrected findings from invalidated historical ones?
- Are the null results presented honestly and usefully?
- Are subgroup findings treated as exploratory rather than definitive?
- Do tables, figures, and appendices still contain stale positive-velocity wording?
- Does the discussion avoid overclaiming what velocity measures can explain?

## Related Files

- [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md)
- [reviews/README.md](reviews/README.md)
- [reviews/velocity/README.md](reviews/velocity/README.md)
