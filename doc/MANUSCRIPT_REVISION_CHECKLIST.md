# Manuscript Revision Checklist

**Manuscript**: `manuscript_quarto/`
**Title**: A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery
**Status**: R1–R10 CLOSED; ready for PAR editorial submission
**Last Updated**: 2026-04-15

## Current Review Snapshot

All ten synthetic peer review cycles completed. See `doc/reviews/quarto/INDEX.md` for the full cycle-by-cycle log.

| Metric | Value |
|--------|-------|
| Review cycles closed | R1–R10 |
| Latest response letter | `doc/reviews/quarto/response_10_2026-04-15.md` |
| Main body word count | 7,996 / 8,000 (PAR compliant) |
| Abstract word count | 141 / 150 (PAR compliant) |
| Citation style | Chicago Author-Date (PAR requirement) |
| Output | `_output/A-Measurement-Sensitivity-Audit-Protocol-...docx` (656 KB) |

## Pre-Submission Verification Checklist

- [x] Title leads with audit-protocol contribution (post-R6 pivot)
- [x] Abstract states central claim as "not stably identified" (post-R10 recast), not "near zero"
- [x] Evidence for Practice formatted as 6-item operational checklist
- [x] Six-item protocol specified in §4.1 with worked CDBG-DR examples
- [x] Class taxonomy (Ia / Ib / II-C / II-O / III) introduced in §4.1 and propagated through tables and figure
- [x] Specification-curve dashboard (@tbl-robustness-summary, @fig-spec-curve) is the headline result, not a single point estimate
- [x] Stability flags use text labels (Stable / Attenuated / Reversed) consistently across §4.1, @tbl-robustness-summary, @tbl-sensitivity, and @fig-spec-curve
- [x] State-clustered bootstrap (1,000 iterations) CI [−0.129, +0.531] in main @tbl-structural and Appendix C.6
- [x] Complementary survival null (HR ≈ 1.0) presented with within-Cox divergence (HR = 1.46–2.58 baseline) — framework divergence not single direction
- [x] QCEW first-order treatment: ε-sensitivity (Appendix C.10), suppression terminology block (Appendix A.3), transportability comparison (@tbl-zero-nonzero-comparison)
- [x] Temporal-capacity caveat in §4.3.2 main text (QCEW is contemporaneous, not pre-existing)
- [x] Included-vs-excluded ARO sample comparison (Appendix A.5, @tbl-incl-excl)
- [x] No "this study" / "advances the literature" / "first application" / "novel contribution" phrases
- [x] No emoji flags in body or tables
- [x] Cross-references all resolve
- [x] Appendix single-source via `{{< include >}}` in `index.qmd`
- [x] Bibliography in Chicago Author-Date with full first names
- [x] Re-rendered DOCX after each revision cycle (final 656 KB)
- [ ] Final pre-submission bibliography pass (optional polish)
- [ ] Confirm reference numbers per target journal template

## File-Level Status

- [x] `manuscript_quarto/index.qmd` — R1–R10 addressed; 7,996 prose words; 141-word abstract
- [x] `manuscript_quarto/appendix-a-data.qmd` — A.1–A.5 post-R10 (transportability comparison @tbl-zero-nonzero-comparison in A.3; included-vs-excluded @tbl-incl-excl + reproducibility boundary in A.5)
- [x] `manuscript_quarto/appendix-b-methods.qmd` — B.1–B.5
- [x] `manuscript_quarto/appendix-c-robustness.qmd` — C.1–C.10 (C.10 adds ε-sensitivity scan + imputation-range units justification; C.8 adds reconstructed-panel fixed-horizon @tbl-fixed-horizon-full)
- [x] `manuscript_quarto/references.bib` — Chicago Author-Date normalized
- [x] `manuscript_quarto/_quarto.yml` — CSL chicago-author-date, appendices block removed
- [x] `manuscript_quarto/figures/fig_02_specification_curve.png` — regenerated R10 with II-C (red) / II-O (orange) distinct colors
- [x] `manuscript_quarto/figures/fig_03_qcew_denominator.png` — QCEW denominator panel

## Rendering Verification

```bash
cd manuscript_quarto
CAPACITY_SEM_SKIP_PIPELINE=1 quarto render . --to docx
# Expected: _output/A-Measurement-Sensitivity-Audit-Protocol-...docx (~656 KB)
```

Word count check:

```bash
cat index.qmd | sed '/^```/,/^```/d' | sed '/^---$/,/^---$/d' | sed '/^|/d' | sed '/^\$/d' | sed '/^#|/d' | grep -v '^\s*$' | wc -w
# Target: ≤ 8,000
```

PAR compliance:

```bash
python src/pipeline.py review_verify --manuscript quarto
```

## Post-Submission Tasks (Not Yet Started)

- [ ] Tag release `v0.3.0-r10-closed` in git
- [ ] Copy final DOCX to submission system
- [ ] File cover letter referencing R1–R10 synthetic review history

## Related Documentation

- `doc/reviews/quarto/INDEX.md` — review log and archive index (R1–R10)
- `doc/reviews/quarto/response_{01,03,04,05,06,08,09,10}_*.md` — response letters
- `doc/reviews/quarto/triage_02_*.md` — R2 triage
- `doc/PROJECT_STATUS.md` — current analytical state
- `doc/MANUSCRIPT_GUIDE.md` — writing rules and manuscript locations
- `doc/CHANGELOG.md` — cycle-by-cycle revision log
