# Revision Tracker: A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery

**Manuscript**: `manuscript_quarto/`
**Target**: Public Administration Review (PAR)
**Status**: R1–R10 CLOSED; ready for editorial submission
**Last Updated**: 2026-04-15

---

## Cycle History

| Cycle | Date | Focus | Recommendation | Key Outcome |
|-------|------|-------|----------------|-------------|
| R1 | 2026-04-13 | par_general | Major revision | Reframe around measurement sensitivity (abstract / EfP / Discussion / Conclusion); QCEW suppression documented (Appendix A.3); SEM described as exploratory decomposition |
| R2 | 2026-04-13 | par_general | Major revision | Triaged; design-change items deferred to R3 |
| R3 | 2026-04-14 | par_general | Major revision | Nonzero-QCEW SEM; within-SEM financial-ratio bridge; centerpiece-formula bug fixed; geography contradiction reconciled; appendix double-render resolved; title made sensitivity-forward |
| R4 | 2026-04-14 | par_general | Major revision | Nine new analyses: local-only primary, state-clustered bootstrap (n=200), Cox staffing bridge, Cox threshold sensitivity, observed-composite, fixed-horizon q=8/12/16, SVI vintage 2010–2022, per-jurisdiction disaster-year SVI |
| R5 | 2026-04-14 | par_general | Major revision | Cox threshold bug fix; design matrix added (§4.3); construct-contamination paragraph in C.1; covariate-set divergence in §6.3; causal language tightened; Workload Manageability terminology unified |
| R6 | 2026-04-14 | par_general | Major revision | **Structural pivot to audit-protocol paper**: title rewritten; 6-item protocol in §4.1; EfP rewritten as protocol checklist; Results relabeled by audit item; SEM repositioned as reference specification |
| R7 | 2026-04-14 | par_general | Major revision | Decision rule + estimand taxonomy in §4.1; bootstrap n=200 → n=1000 (CI [−0.129, +0.531] still crosses zero); NDR/MIT exclusion; @fig-spec-curve and @fig-qcew added to main text; observed-composite promoted to §5.4 |
| R8 | 2026-04-15 | par_general | Major revision | Heuristic reframe with Simonsohn/Steegen citations; **Class Ia/Ib split**; NAICS-925110-specific language; exact upstream counts (`scripts/rebuild_upstream_sample_flow.py`); local-only equal billing in @tbl-structural; within-Cox divergence in §5.5 main text; AdminResources reconciled |
| R9 | 2026-04-15 | par_general | Major revision | **Pass/fail verdict demoted to dashboard**; QCEW elevated to first-order measurement problem; pooled ε-coded SEM reframed; pooled-construct qualification; 573-jurisdiction crosswalk deposited; Reproducibility Boundary subsection; abstract → 146 words |
| R10 | 2026-04-15 | par_general | Major revision | **Central claim recast to "not stably identified"**; **Class II-O / II-C split**; ε-sensitivity scan; zero-vs-nonzero-QCEW transportability comparison; cluster-bootstrap CI in main @tbl-structural; included-vs-excluded ARO comparison; temporal-capacity caveat in §4.3.2; emoji flags → text labels; Cox abstract symmetric |

For full text and response letters, see [`doc/reviews/quarto/INDEX.md`](../doc/reviews/quarto/INDEX.md).

---

## Final Manuscript State (post-R10)

| Metric | Value |
|--------|-------|
| Title | A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery |
| Prose word count | 7,996 / 8,000 (PAR compliant) |
| Abstract | 141 / 150 (PAR compliant) |
| Citation style | Chicago Author-Date |
| Output | `_output/A-Measurement-Sensitivity-Audit-Protocol-...docx` (656 KB) |

### Structural Architecture

- **Contribution**: 6-item measurement-sensitivity audit protocol (§4.1)
- **Deliverable**: Specification-curve dashboard (@tbl-robustness-summary, @fig-spec-curve)
- **Demonstrations**: Cross-sectional SEM (N=573) and Cox survival (N=151) on CDBG-DR DRGR/QPR data
- **Headline finding**: Capacity-timeliness coefficient is *not stably identified* under principled measurement perturbations

### Class Taxonomy (final, post-R10)

| Class | Description | Example perturbations |
|-------|-------------|----------------------|
| Ia | Measurement-preserving (same estimand, same population) | State-clustered bootstrap, weak-indicator drop, residualization, maturity-band controls, portfolio-scale controls, QCEW imputation bounds |
| Ib | Sample-scope (same construct, different population) | Local-only (PRIMARY), nonzero-QCEW, local-AND-nonzero-QCEW, mature-only, NDR/MIT-excluded |
| II-C | Capacity-operationalization change (same outcome, reoperationalized capacity) | Raw-portfolio-counts; financial-ratio bridge (II-C\*) |
| II-O | Outcome change (same capacity, reoperationalized outcome) | Reconstructed-panel fixed-horizon q=8/12/16 |
| III | Simultaneous design-dimension changes | SEM vs. Cox |

---

## Related Documentation

- [`doc/reviews/quarto/INDEX.md`](../doc/reviews/quarto/INDEX.md) — full review log + archive index
- [`doc/reviews/quarto/response_*.md`](../doc/reviews/quarto/) — per-cycle response letters
- [`doc/PROJECT_STATUS.md`](../doc/PROJECT_STATUS.md) — current analytical state
- [`doc/MANUSCRIPT_REVISION_CHECKLIST.md`](../doc/MANUSCRIPT_REVISION_CHECKLIST.md) — pre-submission verification
- [`doc/CHANGELOG.md`](../doc/CHANGELOG.md) — cycle-by-cycle revision log
