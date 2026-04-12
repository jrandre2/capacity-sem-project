# Velocity Manuscript Review Notes

**Manuscript**: `manuscript_velocity/`
**Status**: Active rewrite
**Last Updated**: 2026-04-09

## Current Review Objective

The review target has changed.

This manuscript should now be reviewed as a corrected, null-results paper in revision, not as a finished contingent-capacity manuscript. Reviewers should focus on:
- whether the manuscript clearly explains the duration bug and its implications
- whether corrected null findings are presented clearly and defensibly
- whether any remaining subgroup language is appropriately cautious
- whether stale pre-fix claims still survive in appendices, tables, or prose

## Current Commands

```bash
python src/pipeline.py review_new --manuscript velocity --focus par_general
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_diff --manuscript velocity
python src/pipeline.py review_response --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_archive --manuscript velocity
```

## Current Snapshot

| Metric | Value |
|--------|-------|
| Major comments | 0 |
| Minor comments | 0 |
| Verification progress | 4 / 13 complete |

## Reviewer Prompt Guidance

Use prompts that assume:
- the earlier positive velocity results were invalidated
- the manuscript may still contain stale text from the old framing
- the main task is analytical honesty, coherence, and usefulness of the corrected narrative

Avoid prompts that assume:
- a valid contingent-capacity framework has already been established
- wildfire, administration, or late-phase effects are settled findings
- the meta-analysis appendix reflects current trusted evidence
