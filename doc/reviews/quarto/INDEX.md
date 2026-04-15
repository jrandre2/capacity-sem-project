# Review Index: manuscript_quarto

**Manuscript**: A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery
**Target**: Public Administration Review (PAR)
**Status**: Ten synthetic review cycles closed (R1–R10); audit-protocol framing established in R6; decision rule demoted to specification-curve dashboard in R9; central claim recast from "near zero" to "not stably identified" in R10; Class II-O / II-C split; reference SEM prose-demoted; ready for editorial submission

## Review Cycles

| Cycle | Date | Focus | Source | Recommendation | Status |
|-------|------|-------|--------|----------------|--------|
| #1 | 2026-04-13 | par_general | Synthetic | Major revision | CLOSED |
| #2 | 2026-04-13 | par_general | Synthetic | Major revision | TRIAGED |
| #3 | 2026-04-14 | par_general | Synthetic | Major revision | CLOSED |
| #4 | 2026-04-14 | par_general | Synthetic | Major revision (bordering on reject/resubmit) | CLOSED |
| #5 | 2026-04-14 | par_general | Synthetic | Major revision (constructive) | CLOSED |
| #6 | 2026-04-14 | par_general | Synthetic | Major revision (audit-protocol pivot) | CLOSED |
| #7 | 2026-04-14 | par_general | Synthetic | Major revision (operationalization: decision rule, estimand taxonomy, n=1000 bootstrap) | CLOSED |
| #8 | 2026-04-15 | par_general | Synthetic | Major revision (heuristic reframe + Class Ia/Ib split + exact upstream counts + local-only equal billing) | CLOSED |
| #9 | 2026-04-15 | par_general | Synthetic | Major revision (demote verdict to dashboard + QCEW first-order + SEM prose-demoted + pooled construct narrowed + crosswalk deposited) | CLOSED |
| #10 | 2026-04-15 | par_general | Synthetic | Major revision (recast claim to indeterminacy; II-O/II-C taxonomy; ε-sensitivity + transportability comparison + included-vs-excluded; bootstrap CIs in main table; temporal caveat) | CLOSED |

## Archive

| File | Contents |
|------|----------|
| `archive/review_01_2026-04-13_par_general.md` | Review #1 text and summary |
| `response_01_2026-04-13.md` | Review #1 CENTAUR response letter |
| `archive/review_02_2026-04-13_par_general.md` | Review #2 text and summary |
| `triage_02_2026-04-13.md` | Review #2 triage with classifications |
| `archive/review_03_2026-04-14_par_general.md` | Review #3 text and summary |
| `response_03_2026-04-14.md` | Review #3 response letter with re-analysis results |
| `archive/review_04_2026-04-14_par_general.md` | Review #4 text and summary |
| `response_04_2026-04-14.md` | Review #4 response letter (Markdown source) |
| `response_04_2026-04-14.docx` | Review #4 response letter (rendered DOCX) |
| `response_03_2026-04-14.docx` | Review #3 response letter (rendered DOCX) |
| `archive/review_05_2026-04-14_par_general.md` | Review #5 text and summary |
| `response_05_2026-04-14.md` | Review #5 response letter with Cox threshold bug fix + construct-contamination paragraph |
| `archive/review_06_2026-04-14_par_general.md` | Review #6 text and summary |
| `response_06_2026-04-14.md` | Review #6 response letter documenting structural pivot to audit-protocol paper |
| `archive/review_08_2026-04-15_par_general.md` | Review #8 text and summary |
| `response_08_2026-04-15.md` | Review #8 response letter: decision-rule heuristic reframe, Class Ia/Ib split, exact upstream counts, local-only equal billing, within-Cox divergence to main text, AdminResources reconciliation |
| `archive/review_09_2026-04-15_par_general.md` | Review #9 text and summary |
| `response_09_2026-04-15.md` | Review #9 response letter: verdict demoted to dashboard, QCEW first-order, SEM prose-demoted, pooled construct narrowed, 573-jurisdiction crosswalk deposited |
| `archive/review_10_2026-04-15_par_general.md` | Review #10 text and summary |
| `response_10_2026-04-15.md` | Review #10 response letter: central claim recast to indeterminacy, Class II-O/II-C split, ε-sensitivity + transportability + included-vs-excluded comparison, bootstrap CIs in main table, temporal caveat elevated, Cox abstract symmetric |

## R1–R4 Cycle Summary

| Cycle | Key asks | Actions taken |
|-------|----------|---------------|
| R1 | Reframe around instability; fix QCEW proxy documentation; validate SEM latent structure | Abstract/EfP/Discussion/Conclusion reframed around measurement sensitivity; Appendix A.3 (QCEW suppression documented); exploratory-decomposition framing adopted |
| R2 | Run nonzero-QCEW sensitivity; remove weak indicator; add cross-framework bridges; local-only primary | Nonzero-QCEW SEM added; 3-indicator Recovery Performance confirmed identical; within-SEM financial-ratio bridge added |
| R3 | Confirm previous asks implemented; fix specific document bugs | Centerpiece-formula bug fixed; geography contradiction reconciled; appendix double-render resolved; title changed to sensitivity-forward |
| R4 | Local-only nonzero-QCEW; Cox-with-staffing bridge; clustered SEs; survival diagnostics; fixed-horizon outcomes; SVI vintage | 9 new analyses executed: nonzero-QCEW local-only, Cox with staffing covariates, state-clustered bootstrap SEs (CI crosses zero), Cox threshold-sensitivity 20%–100%, observed-composite regressions, fixed-horizon outcomes (q=8/12/16), SVI vintage sensitivity across 6 vintages, per-jurisdiction disaster-year SVI re-estimation (N=572) |
| R5 | SEM too fragile as primary; timeliness-sample selection; survival internal inconsistency; construct contamination; tighten causal language; design matrix | **Critical bug fixed**: Cox threshold table rebuilt with correct event coding (previous version used raw `Completion_Pct` as if fraction); design matrix added (§4.3); construct-contamination paragraph added to C.1; Appendix C.7 self-undermining language softened; covariate-set divergence added to §6.3; causal/mechanism language tightened throughout; QCEW zero rates clarified; second-factor terminology unified to Workload Manageability |
| R6 | Reframe around proxy instability; SEM too fragile as primary; QCEW proxy compromised; cross-framework comparison too many things; sample-selection prominence; uncertainty understated; **propose 5-item audit protocol** | **STRUCTURAL PIVOT**: title and abstract rewritten to lead with audit protocol as contribution; new §4.1 specifies 6-item protocol (extending R6's 5 by separating cluster-appropriate inference); EfP rewritten as operational protocol checklist with worked examples from CDBG-DR; Introduction reframed; Literature Review consolidated; Results sections relabeled by audit item; Discussion §6.1 renamed "What the Audit Protocol Surfaces"; Conclusion emphasizes portable protocol. SEM is now a "reference specification" exercising audit items, not primary inferential engine. All previous analyses retained as protocol demonstrations. |
| R7 | Define a decision rule + "benchmark-ready" rubric; classify robustness exercises by estimand (I/II/III); reduce bootstrap CI pre-§5.4 to n=1000; add NDR/Mitigation exclusion sensitivity; add specification-curve and QCEW-denominator figures; promote observed-composite to main text; narrow normative claim | Added decision rule + estimand taxonomy §4.1; bootstrap n=200 → n=1000 (95% CI [−0.129, 0.531] still crosses zero); NDR/Mitigation sensitivity added (β=0.255 on N=512); @fig-spec-curve and @fig-qcew added to main text; observed-composite §5.4 subsection; "not benchmark-ready" narrowed to specific ingredient combination |
| R8 | Decision-rule cutoffs lack justification (either simulate or reframe as heuristic); Class I conflates measurement with sample-scope; narrow QCEW claim to NAICS 925110 proxy; exact upstream counts required; local-only should be main or equal-billed; within-Cox divergence to main text; AdminResources "benchmark-ready" internal contradiction; EfP methodology-first | **Heuristic reframe** with Simonsohn + Steegen citations; **Class Ia/Ib split** (measurement vs. sample-scope); abstract/EfP/§6.1/Conclusion narrowed to NAICS-925110-specific language; rebuilt upstream linker script (`scripts/rebuild_upstream_sample_flow.py`) producing exact per-stage counts → @tbl-upstream-geo rewritten; local-only (N=543) added to @tbl-structural alongside pooled (β=0.257 vs 0.266); within-Cox divergence paragraph added to §5.5 main text; AdminResources reconciled as "benchmark-ready as stable association, not policy-ready without portfolio-complexity identification"; EfP reordered with explicit causal-caveat closing |
| R9 | Decision rule's 50% magnitude pegged to unstable reference (circular); QCEW suppression is first-order, not one audit item; reverse hierarchy — SEM secondary; pooled-construct interpretation over-claims; cross-framework illustrative not adjudicative; deposit final crosswalk; abstract repeats; vulnerability tone | **Pass/fail "benchmark-ready" verdict demoted to specification-curve dashboard** with 🟢/🟡/🔴 flags in @tbl-robustness-summary and @tbl-sensitivity; **QCEW elevated to first-order measurement problem** — §6.1 now leads with nonzero-QCEW (N=111, β=0.132 n.s.) + time-varying Cox (N=151, null) + fixed-horizon (attenuates by q=16); pooled ε-coded SEM reframed as "illustration of what happens when BLS zeros are handled by ε-offset"; §5 opening explicitly demotes SEM to "most specification-sensitive illustration"; §5.3 adds pooled-construct qualification paragraph (robustness to population ≠ construct invariance); `scripts/export_jurisdiction_crosswalk.py` produces `data_work/replication/jurisdiction_crosswalk.csv` (573 rows); Appendix A.3 adds QCEW terminology block (literal-zero / BLS-suppressed / non-suppressed); Appendix A.5 adds Reproducibility Boundary subsection; 95% threshold reframed as analytical approximation; vulnerability-firmer-ground softened; abstract compressed to 146 words |

## Tracker

See `manuscript_quarto/REVISION_TRACKER.md` for item-level disposition.
