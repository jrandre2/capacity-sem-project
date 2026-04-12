# Manuscript Revision Checklist

**Manuscript**: `manuscript_velocity/`
**Status**: Active rewrite
**Last Updated**: 2026-04-09

## Current Review Snapshot

Derived from `python src/pipeline.py review_status --manuscript velocity`:

| Metric | Value |
|--------|-------|
| Major comments | 0 |
| Minor comments | 0 |
| Verification progress | 4 / 13 complete |
| Active tracker | `manuscript_velocity/REVISION_TRACKER.md` |

## Current Priorities

- [ ] Rewrite headline framing around corrected null findings
- [ ] Remove or clearly label pre-fix contingent-capacity claims
- [ ] Re-check abstract, introduction, discussion, and conclusion for stale positive-velocity language
- [ ] Audit appendices for subgroup results that should now be treated as exploratory or historical
- [ ] Re-render the manuscript after text cleanup
- [ ] Re-run review verification after each substantial revision

## File Focus

- [ ] `manuscript_velocity/index.qmd`
- [ ] `manuscript_velocity/appendix-a-data.qmd`
- [ ] `manuscript_velocity/appendix-b-methods.qmd`
- [ ] `manuscript_velocity/appendix-c-heterogeneity.qmd`
- [ ] `manuscript_velocity/appendix-d-meta-analysis.qmd`

## Verification Commands

```bash
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_report
```

## Readiness Checks

- [ ] Main text matches [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [ ] No section treats invalidated heterogeneity claims as settled findings
- [ ] Figures and tables align with corrected diagnostics
- [ ] Review tracker reflects current manuscript state
- [ ] Rendered outputs in `manuscript_velocity/_output/` are current

## Notes

- `manuscript_quarto/` is a vendored CENTAUR scaffold and is not the active Capacity-SEM manuscript.
- Historical notes and old results can remain for provenance, but they should be explicitly labeled as superseded.
