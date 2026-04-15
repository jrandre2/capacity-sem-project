# Project Status

Last updated: 2026-04-15

## Current Position

Primary manuscript (`manuscript_quarto/index.qmd`) has completed ten synthetic peer review cycles (R1–R10). The R6 cycle pivoted the manuscript structurally from "instability narrative around the SEM" to "audit-protocol contribution with a specification-curve dashboard." The R10 cycle recast the central substantive claim from "near zero" to "not stably identified." Manuscript is at 7,996 prose words / 141-word abstract; PAR-compliant; ready for editorial submission.

- Primary manuscript: `manuscript_quarto/`
- Current title: *"A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"*
- Target journal: Public Administration Review (PAR)
- Active branch: `analysis/alternative-capacity-measures`
- Archived velocity manuscript: `manuscript_velocity/` (superseded by quarto rewrite)
- Archived Kaifa SEM: `manuscript_kaifa_archive/` (source draft)
- Review cycles completed: R1–R10 (all CLOSED; see `doc/reviews/quarto/INDEX.md`)

## Contribution

A **six-item measurement-sensitivity audit protocol** for administrative-capacity studies in CDBG-DR. The deliverable is a **specification-curve dashboard**, not a pass/fail verdict. The six audit items:

1. **Proxy transparency** — disaggregated suppression rates + non-suppressed-subsample sensitivity
2. **Sample-selection sensitivity** — maturity-stratified or fixed-horizon outcome
3. **Within-framework operationalization bridge** — alternative capacity operationalization in same framework
4. **Cross-framework triangulation** — alternative analytical framework on same data
5. **Vintage and granularity documentation** — vintage sensitivity for time-varying contextual covariates
6. **Cluster-appropriate inference** — state- or disaster-clustered bootstrap

## Trusted Findings (Specification-Curve Dashboard)

The capacity-timeliness coefficient is **not stably identified** under principled measurement perturbations.

- **Primary analytical specification**: local-only (N=543), β = +0.257 (Stable), p < 0.001
- **Pooled supplementary**: N=573, β = +0.266 (conventional ML p<0.001); 95% CI [−0.129, +0.531] under state-clustered bootstrap (1,000 iterations, 35 clusters) **crosses zero**
- **Class Ia (measurement-preserving)**: residualized burden +0.297 (Stable); maturity-band controls +0.105 (Attenuated); portfolio-scale controls +0.127 (Attenuated); QCEW imputation bounds [+0.024, +0.267] across 0.25–10 emp./K pop (Attenuated)
- **Class Ib (sample-scope)**: nonzero-QCEW subsample N=111: β=+0.132 n.s.; local-AND-nonzero-QCEW N=81: β=+0.145 n.s.; mature-only N=275: **β=−0.244 (Reversed)**; NDR/MIT-excluded N=512: β=+0.255 (Stable)
- **Class II-C (capacity-operationalization change)**: raw portfolio counts: **β=−0.443 (Reversed)**; within-SEM financial-ratio bridge: **β=−0.600 (Reversed, II-C\* construct reassignment)**
- **Class II-O (outcome change)**: reconstructed-panel fixed-horizon q=8/12/16 (N=102–128): β ≈ 0 across horizons (Attenuated)
- **Class III (multiple dimensions)**: time-varying Cox (N=151): HR ≈ 1.0, null
- **Within-Cox divergence**: single-spell baseline Cox: HR=1.46–2.58 across 20–100% completion thresholds, p<0.01. Same data, different specification, opposite answer.
- **ε-sensitivity (Appendix C.10)**: SEM coefficient invariant to ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²} — what matters is the *treatment* of suppressed zeros, not the ε value
- **Transportability (Appendix A.3)**: non-suppressed-QCEW subsample is **not representative** — 40× population mean, 27% state agencies vs. 0%, 5× more programs, *lower* completion ratios. Substantive findings on N=111 do not generalize to the suppressed N=462 majority.
- **AdminResources direction not stable**: −0.170 in ε-offset family but reverses to +0.249 at QCEW imputation = 10. No CDBG-DR capacity-specific structural path is direction-stable across principled suppression-handling alternatives.
- **SVI vintage** (Audit Item 5, Appendix C.9): Theme 1 (socioeconomic) flips sign between 2010 (β=−0.215) and 2014+ vintages (β=+0.167 to +0.460) due to documented CDC methodological revision; per-jurisdiction disaster-year re-estimation (572/573 match) reconfigures theme-to-outcome assignments while leaving capacity-timeliness intact (β=+0.286)
- **Sample composition (Appendix A.5)**: SEM N=995 included AROs (county/city-resolvable + all states); 444 institutional excluded; 1,708 unresolved excluded

## Historical Pipeline Context

- The original positive velocity results from the 2025 velocity manuscript were invalidated by two independent data-handling bugs. The archived `manuscript_velocity/` reflects a post-bugfix null-findings analysis, now superseded by the broader SEM+survival framing in `manuscript_quarto/`.
- All ratio variables use SUM aggregation across activities, $1K minimum denominator, and [0, 2] clipping.
- The primary capacity-SEM data is the Kaifa recovered-analysis bundle (`manuscript_kaifa_archive/data/recovered_code_bundle/`), processed through `src/capacity_sem/models/kaifa_recovered_analysis.py`.

## What Changed In Each Review Cycle

### R1 (2026-04-13): reframe around measurement sensitivity

- Abstract, Evidence for Practice, Discussion opening, and Conclusion reframed to foreground instability
- Appendix A.3 now documents QCEW NAICS 925110 suppression (80.6% zero-rate overall; 85.1% zero among locals)
- SEM described as exploratory decomposition, not latent-variable validation

### R2 (2026-04-13): triaged, partial action

- Demands escalated to design changes; many items deferred

### R3 (2026-04-14): run R2 deferrals + fix document bugs

- Nonzero-QCEW subsample SEM added
- Within-SEM financial-ratio capacity bridge added
- Title changed to sensitivity-forward
- Centerpiece-variable formula fixed in main text
- Geography-matching contradiction reconciled
- Appendix double-render resolved

### R4 (2026-04-14): nine additional analyses

- Local-only primary SEM
- Local-only nonzero-QCEW
- State-clustered bootstrap SEs (CI crosses zero)
- Cox with staffing/workload indicators (one-dimension bridge)
- Cox threshold-sensitivity 20%–100% as rendered table
- Observed-composite regressions across all six outcome indicators
- Fixed-horizon outcomes at q = 8, 12, 16
- CDC/ATSDR SVI historical vintages 2010–2022 downloaded + state-level vintage sensitivity
- Per-jurisdiction disaster-year SVI re-estimation (N = 572 match)

### R5 (2026-04-14): Cox threshold bug fix + construct contamination

- **Critical bug fixed**: Cox threshold table rebuilt with correct event coding (previous version used raw `Completion_Pct` as if fraction)
- Design matrix added (§4.3) cataloging unit/outcome/censoring/operationalization/covariate-set differences across designs
- Construct-contamination paragraph added to Appendix C.1 explaining why financial-ratio bridge has dual SEM role
- Appendix C.7 self-undermining language softened
- Covariate-set divergence paragraph added to §6.3
- Causal/mechanism language tightened throughout
- Second-factor terminology unified to "Workload Manageability"

### R6 (2026-04-14): structural pivot to audit-protocol paper

- **Title rewritten**: "Measurement Sensitivity in..." → "A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"
- Abstract rewritten to lead with audit protocol as contribution
- New §4.1 specifies six-item protocol (proxy / sample-selection / within-framework bridge / cross-framework / vintage / cluster-appropriate inference)
- EfP rewritten as operational protocol checklist with worked CDBG-DR examples
- Introduction reframed; Literature Review consolidated; Results sections relabeled by audit item
- Discussion §6.1 renamed "What the Audit Protocol Surfaces"
- Conclusion emphasizes portable protocol
- SEM is now a "reference specification" exercising audit items, not the primary inferential engine

### R7 (2026-04-14): operationalization of decision rule + estimand taxonomy

- Added decision rule + estimand taxonomy in §4.1
- Bootstrap iterations: n=200 → n=1000 (95% CI [−0.129, 0.531] still crosses zero)
- NDR/Mitigation exclusion sensitivity added (β=0.255 on N=512)
- Specification-curve figure added to main text
- QCEW-denominator figure added to main text
- Observed-composite promoted to §5.4 main-text subsection
- "Not benchmark-ready" narrowed to specific ingredient combination

### R8 (2026-04-15): heuristic reframe + Class Ia/Ib split + exact upstream counts

- **Heuristic reframe** of decision rule with Simonsohn (2020) and Steegen et al. (2016) citations
- **Class Ia/Ib split** in taxonomy (measurement-preserving vs. sample-scope perturbations)
- Abstract / EfP / §6.1 / Conclusion narrowed to NAICS-925110-specific language (not "QCEW writ large")
- Rebuilt upstream linker script (`scripts/rebuild_upstream_sample_flow.py`) producing exact per-stage counts → @tbl-upstream-geo rewritten with authoritative N at each stage
- Local-only (N=543) added to @tbl-structural alongside pooled (β=0.257 vs. 0.266) with equal billing
- Within-Cox divergence paragraph promoted from appendix to §5.5 main text
- AdminResources reconciled as "benchmark-ready as stable association, not policy-ready without portfolio-complexity identification"
- EfP reordered: methodology first, causal-caveat closing

### R9 (2026-04-15): verdict demoted to dashboard + QCEW first-order + SEM prose-demoted

- **Pass/fail "benchmark-ready" verdict demoted** to specification-curve dashboard with stability flags in @tbl-robustness-summary and @tbl-sensitivity
- **QCEW elevated to first-order measurement problem**: §6.1 leads with nonzero-QCEW (N=111, β=0.132 n.s.) + time-varying Cox (N=151, null) + fixed-horizon (attenuates by q=16)
- Pooled ε-coded SEM reframed as "illustration of what happens when BLS zeros are handled by ε-offset"
- §5 opening explicitly demotes SEM to "most specification-sensitive illustration"
- §5.3 adds pooled-construct qualification paragraph (robustness to population ≠ construct invariance)
- 573-jurisdiction crosswalk deposited (`scripts/export_jurisdiction_crosswalk.py` → `data_work/replication/jurisdiction_crosswalk.csv`)
- Appendix A.3 adds QCEW terminology block (literal-zero / BLS-suppressed / non-suppressed)
- Appendix A.5 adds Reproducibility Boundary subsection
- 95% threshold reframed as analytical approximation
- Vulnerability-firmer-ground language softened
- Abstract compressed to 146 words

### R10 (2026-04-15): central claim recast to indeterminacy + Class II-O/II-C + transportability

- **Central claim recast** from "near zero" to **"not stably identified / not benchmark-ready"** — abstract, §6.1, Conclusion all rewritten
- **Class II split** into II-C (capacity-operationalization change) and II-O (outcome change) with relabeling propagated through @tbl-robustness-summary, @tbl-sensitivity, @fig-spec-curve, §5.4 prose
- **ε-sensitivity scan** (Appendix C.10): SEM invariant across ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²}
- **Zero-vs-nonzero-QCEW transportability comparison** (Appendix A.3, @tbl-zero-nonzero-comparison): non-suppressed subsample dramatically non-representative; R9 normative claim that "substantive claims should rest on the non-suppressed subsample" walked back
- **Cluster-bootstrap CI column** added to @tbl-structural (main table)
- **Included-vs-excluded comparison table** added (Appendix A.5, @tbl-incl-excl)
- **Temporal-capacity caveat** elevated to main text §4.3.2 (QCEW is contemporaneous, not pre-existing/baseline)
- **Cox abstract symmetric**: "null under time-varying specification, positive and significant under baseline specification"
- **Reconstructed-panel fixed-horizon** added to Appendix C.8 at N=102–128 (alongside original grantee-level N=32–36)
- Emoji stability flags 🟢/🟡/🔴 → text labels (Stable / Attenuated / Reversed) throughout tables
- Figure regenerated with II-C (red) and II-O (orange) as distinct colors
- Final manuscript: 7,996 prose words; 141-word abstract

## Appendix Structure (Post-R10)

- **Appendix A (5 sections)**: Data flow, variable dictionary, QCEW proxy distributions + transportability comparison (@tbl-zero-nonzero-comparison), SVI vintage specification, geography matching + reproducibility boundary + included-vs-excluded comparison (@tbl-incl-excl)
- **Appendix B (5 sections)**: Model specifications, estimation details, full parameter tables, measurement discussion, survival analysis specification
- **Appendix C (10 sections)**: Sensitivity across specifications (master table with stability flags), proxy validation, maturity composition, ratio artifact summary, Cox model sensitivity and staffing bridge, state-clustered bootstrap SEs (n=1000), observed-variable regressions, fixed-horizon outcomes (grantee-level + reconstructed-panel), SVI vintage and disaster-year sensitivity, QCEW imputation bounds + ε-sensitivity scan

## Recommended Commands

### Active workflow

```bash
python src/pipeline.py run_all
```

### Active workflow, step-by-step

```bash
python src/pipeline.py ingest_data
python src/pipeline.py standardize_data
python src/pipeline.py build_panel
python src/pipeline.py build_features_std
python src/pipeline.py aggregate_program_types
python src/pipeline.py run_survival
```

### SEM + sensitivity pipeline

```bash
python src/pipeline.py run_kaifa_recovered_analysis
python src/pipeline.py run_robustness
```

### Manuscript

```bash
cd manuscript_quarto
./render_all.sh
CAPACITY_SEM_SKIP_PIPELINE=1 quarto render index.qmd --to docx
```

### Review management

```bash
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
```

## Current Risks / Follow-Up Work

- **Pipeline integration of cycle scripts**: scripts added during R5–R10 (`run_state_clustered_bootstrap_sem.py`, `run_qcew_bounds_sensitivity.py`, `run_epsilon_sensitivity.py`, `build_zero_nonzero_comparison.py`, `build_sample_selection_comparison.py`, `build_full_fixed_horizon_panel.py`, `rebuild_upstream_sample_flow.py`, `export_jurisdiction_crosswalk.py`, `make_audit_figures.py`) currently run standalone against the Kaifa SEM data. Promoting them to `src/capacity_sem/models/` helpers and registering them as pipeline stages would make the full audit reproducible from `run_all`.
- **Fixed-horizon outcomes at full N = 573**: reconstructed-panel achieves N = 102–128 (Appendix C.8); full coverage of all 573 jurisdictions still requires either better F31 coverage or a different QPR-vintage reconstruction.
- **SVI vintage automation**: the CDC historical CSVs live in `data_raw/svi_historical/` but aren't part of the pipeline; adding a `load_historical_svi` stage would make them first-class.
- **Censored-data SEM estimator**: ε-offset treatment of suppressed zeros is the dominant measurement exposure (R10 M3); a Tobit-style censored-data SEM would resolve it but is not currently supported by `semopy`. Flagged as future work.
- **Baseline/lagged QCEW staffing analysis**: per R10 temporal-capacity caveat, baseline-year QCEW substitution would test whether the proxy reflects pre-existing capacity. Inherits the same suppression structure for the suppressed majority, so does not resolve M3 alone but provides additional triangulation.

## Working Rule

For new analysis, treat the **six-item audit protocol** as the unit of contribution. The cross-sectional SEM and the time-varying Cox are demonstrations of audit items, not authoritative findings in themselves. Default to the local-only specification (N=543) as primary; report pooled (N=573) as supplementary. Use cluster-appropriate inference (1,000-iteration state-clustered bootstrap) as the default lens on any reported headline coefficient. Distinguish Class Ia (measurement-preserving), Class Ib (sample-scope), Class II-C (capacity-operationalization change), and Class II-O (outcome change) when reporting robustness results.
