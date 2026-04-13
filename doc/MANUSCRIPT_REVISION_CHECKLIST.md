# Manuscript Revision Checklist

**Manuscript**: `manuscript_quarto/`
**Status**: Pre-submission
**Last Updated**: 2026-04-13

## Current Review Snapshot

Derived from `python src/pipeline.py review_status --manuscript quarto`:

| Metric | Value |
|--------|-------|
| Major comments | 0 |
| Minor comments | 0 |
| Verification progress | 4 / 13 complete |
| Active tracker | `manuscript_quarto/REVISION_TRACKER.md` |

## Current Priorities

- [ ] Verify SEM framing (burden->timeliness beta=0.266) is clearly presented
- [ ] Ensure complementary survival null (HR~1.0) is properly contextualized
- [ ] Re-check abstract, introduction, discussion, and conclusion for stale velocity-primary language
- [ ] Audit appendices for alignment with SEM-primary framing
- [ ] Re-render the manuscript after text cleanup
- [ ] Re-run review verification after each substantial revision

## File Focus

- [ ] `manuscript_quarto/index.qmd`
- [ ] `manuscript_quarto/appendix-a-data.qmd`
- [ ] `manuscript_quarto/appendix-b-methods.qmd`
- [ ] `manuscript_quarto/appendix-c-robustness.qmd`

## Verification Commands

```bash
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py review_report
```

## Readiness Checks

- [ ] Main text matches [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [ ] No section treats invalidated heterogeneity claims as settled findings
- [ ] Figures and tables align with corrected diagnostics
- [ ] Review tracker reflects current manuscript state
- [ ] Rendered outputs in `manuscript_quarto/_output/` are current

## Notes

- `manuscript_quarto/` is the active Capacity-SEM manuscript (cross-sectional SEM, N=573).
- `manuscript_velocity/` is archived (superseded survival-only draft).
- Historical notes and old results can remain for provenance, but they should be explicitly labeled as superseded.
