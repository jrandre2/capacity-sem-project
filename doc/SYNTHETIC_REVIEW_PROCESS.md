# Synthetic Review Process

**Status**: Active (10 cycles closed for primary manuscript)
**Last Updated**: 2026-04-15

This workflow is currently used for the primary manuscript in
`manuscript_quarto/`. The same review engine also supports actual-review and
DOCX-import workflows for archived manuscripts such as `kaifa` and `velocity`.

## Current Scope

| Manuscript | Directory | Tracker | Status |
|------------|-----------|---------|--------|
| Quarto (primary) | `manuscript_quarto/` | `manuscript_quarto/REVISION_TRACKER.md` | R1–R10 closed; pre-submission |
| Velocity (archived) | `manuscript_velocity/` | `manuscript_velocity/REVISION_TRACKER.md` | Archived |

The purpose of synthetic review is to:
- pressure-test the audit-protocol contribution and the specification-curve dashboard
- verify the headline coefficient is presented as not-stably-identified, not as a single substantive estimate
- catch stale pre-pivot claims that remain in the manuscript
- catch internal inconsistencies (table↔prose, class taxonomy propagation, cross-references)
- enforce PAR compliance (≤ 8,000 prose words, ≤ 150 abstract, Chicago Author-Date, no self-references)

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

- Is the audit-protocol contribution (six items + dashboard) clearly the unit of contribution, with the SEM and Cox as demonstrations rather than the headline?
- Does the abstract / EfP / §6.1 / Conclusion consistently state the central claim as "not stably identified," not as "near zero" or as a single coefficient?
- Is the class taxonomy (Ia / Ib / II-C / II-O / III) used consistently across §4.1 prose, @tbl-robustness-summary, @tbl-sensitivity, and @fig-spec-curve?
- Are stability flags presented as text labels (Stable / Attenuated / Reversed), not emojis?
- Is QCEW treated as a first-order measurement problem (suppression terminology, ε-sensitivity, transportability comparison) rather than as one robustness item among many?
- Does the discussion avoid privileging any single slice of the dashboard (positive ε-offset, near-zero non-suppressed, negative bridge) as the "true" effect?
- Are tables and prose internally consistent? Do cross-references (@tbl-*, @fig-*, §) all resolve?

## Related Files

- [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md)
- [reviews/README.md](reviews/README.md)
- [reviews/velocity/README.md](reviews/velocity/README.md)
- [reviews/quarto/README.md](reviews/quarto/README.md)
