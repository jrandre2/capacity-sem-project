# Changelog

All notable changes to this project will be documented in this file.

## [2026-04-15] — Synthetic Review Cycles R5–R10 Closed

Six additional synthetic review cycles closed across two days. Net effect: the manuscript pivoted structurally at R6 from "instability narrative around the SEM" to "audit-protocol contribution with a specification-curve dashboard," and the central substantive claim was recast at R10 from "near zero" to "not stably identified."

### Changed

#### Manuscript Title (R6 pivot)

- From: "Measurement Sensitivity in Administrative-Throughput Analysis of CDBG-DR Disaster Recovery: Evidence from a Cross-Sectional SEM and Complementary Survival Model"
- To: "A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"

#### Central Claim (R10 recast)

- From: "the capacity-timeliness coefficient is near zero in measurement-appropriate specifications"
- To: "the capacity-timeliness coefficient is not stably identified under these measurement choices"
- Rationale: dashboard contains positive (+0.266 reference, +0.297 residualized, +0.257 local-only), near-zero (+0.132 n.s., HR ≈ 1.0), and negative (−0.244 mature-only, −0.443 raw counts, −0.600 financial-ratio bridge) estimates — privileging any single slice was selective

#### Class Taxonomy (R8 + R10 refinement)

- R8: split Class I into Ia (measurement-preserving) and Ib (sample-scope perturbations)
- R10: split Class II into II-C (capacity-operationalization change) and II-O (outcome change)
- Final 5-class taxonomy: Ia / Ib / II-C / II-O / III propagated through @tbl-robustness-summary, @tbl-sensitivity, @fig-spec-curve, §4.1 prose, §5.4 dashboard prose

#### Bootstrap Iterations (R7)

- 200 → 1,000 iterations
- Updated CI: [−0.119, +0.542] → [−0.129, +0.531] (still crosses zero)

#### Stability Flags (R10)

- Emoji 🟢 / 🟡 / 🔴 → text labels (Stable / Attenuated / Reversed) throughout tables and §4.1 description

### Added

#### New Analyses (across R5–R10)

- **Decision rule + estimand taxonomy** (R7) in §4.1, reframed as heuristic with Simonsohn (2020) and Steegen et al. (2016) citations (R8)
- **NDR/Mitigation-excluded sensitivity** (R7): N=512, β=+0.255 (Stable)
- **Specification-curve dashboard figure** (R7) added to main text
- **QCEW-denominator distribution figure** (R7) added to main text
- **Observed-composite regressions** (R7): promoted from Appendix C.7 to main-text §5.4 subsection
- **Local-only primary specification** (R8): N=543, β=+0.257 added to @tbl-structural with equal billing alongside pooled
- **Within-Cox divergence** (R8): single-spell baseline Cox HR=1.46–2.58 vs. time-varying HR ≈ 1.0; promoted to §5.5 main text
- **Exact upstream sample-flow counts** (R8): `scripts/rebuild_upstream_sample_flow.py` produces per-stage counts → @tbl-upstream-geo rewritten with authoritative N
- **573-jurisdiction crosswalk deposited** (R9): `scripts/export_jurisdiction_crosswalk.py` → `data_work/replication/jurisdiction_crosswalk.csv`
- **QCEW terminology block** (R9) in Appendix A.3: literal-zero / BLS-suppressed / non-suppressed
- **Reproducibility Boundary subsection** (R9) in Appendix A.5
- **ε-sensitivity scan** (R10): `scripts/run_epsilon_sensitivity.py` refits SEM at ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²} → invariant; Appendix C.10
- **Zero-vs-nonzero-QCEW transportability comparison** (R10): `scripts/build_zero_nonzero_comparison.py` → @tbl-zero-nonzero-comparison in Appendix A.3
- **Included-vs-excluded ARO comparison** (R10): `scripts/build_sample_selection_comparison.py` → @tbl-incl-excl in Appendix A.5
- **Cluster-bootstrap CI column** (R10) added to main @tbl-structural
- **Reconstructed-panel fixed-horizon** (R10): `scripts/build_full_fixed_horizon_panel.py` → @tbl-fixed-horizon-full at N=102–128, alongside grantee-level N=32–36
- **Temporal-capacity caveat** (R10) elevated to §4.3.2 main text
- **Cox abstract symmetric language** (R10): null under time-varying, positive under baseline

#### Construct Contamination + Bug Fix (R5)

- Cox threshold table rebuilt with correct event coding (previous version used raw `Completion_Pct` as if fraction)
- Construct-contamination paragraph added to Appendix C.1 explaining financial-ratio bridge dual SEM role
- Design matrix added (§4.3) cataloging unit/outcome/censoring/operationalization/covariate-set differences across designs

#### Response Letters and Archive

- `doc/reviews/quarto/archive/review_{05..10}_2026-04-1{4,5}_par_general.md`
- `doc/reviews/quarto/response_{05,06,08,09,10}_2026-04-1{4,5}.md`
- `doc/reviews/quarto/INDEX.md` updated through R10

### Removed

- Pass/fail "benchmark-ready" verdict (R9): demoted to specification-curve dashboard with stability flags
- Emoji stability flags (R10) replaced with text labels throughout

### Verification (post-R10)

- Prose: 7,996 / 8,000 words ✓
- Abstract: 141 / 150 words ✓
- Self-references: 0 ✓
- Emoji in body: 0 ✓
- Output: `_output/A-Measurement-Sensitivity-Audit-Protocol-...docx` (656 KB)

## [2026-04-14] — Synthetic Review Cycle R4 Closed

### Added

#### CDC/ATSDR SVI Historical Vintages

- Downloaded 6 SVI vintages (2010, 2014, 2016, 2018, 2020, 2022) from ATSDR Feature Services to `data_raw/svi_historical/SVI{YYYY}_US_COUNTY.csv`
- State-level vintage-sensitivity SEM run across all 6 vintages (Appendix C.9, @tbl-svi-vintage)
- Per-jurisdiction disaster-year SVI re-estimation with 99.8% match rate (572/573); capacity-timeliness β = +0.286 robust to disaster-year substitution
- New derived data files: `state_earliest_disaster_year.parquet`, `jurisdiction_disaster_year_svi.parquet`, `sem_input_disaster_year_svi.parquet`

#### R4-Cycle New Analyses (Appendix C)

- **C.1 consolidated sensitivity table** — 15 rows grouped by perturbation type; merged from former C.1, C.4, C.5, C.6
- **C.5 Cox model sensitivity** — threshold sensitivity 20%–100% as rendered table; staffing-bridge Cox with programs/staff and disasters/staff as static covariates (N = 100); merged from former C.8 and C.9
- **C.6 state-clustered bootstrap SEs** — 200 iterations, 35 clusters; primary β = 0.266 95% CI [−0.119, 0.542] crosses zero
- **C.7 observed-variable regressions** — 6-outcome OLS (matches SEM β = +0.266 at +0.276); financial-ratio-only OLS as subset; merged from former C.11 and C.12
- **C.8 fixed-horizon outcomes** — expenditure shares at q = 8, 12, 16 (N = 32–36)
- **C.9 SVI vintage and disaster-year** — state-level vintage sensitivity + per-jurisdiction disaster-year re-estimation; merged from former C.14 and C.15

#### Response Letters and Archive

- `doc/reviews/quarto/archive/review_04_2026-04-14_par_general.md`
- `doc/reviews/quarto/response_04_2026-04-14.md` + `.docx`
- `doc/reviews/quarto/INDEX.md` updated to show R1–R4 closure

### Changed

#### Manuscript Title

- From: "Administrative Throughput in Disaster Recovery: Evidence from Cross-Sectional SEM of CDBG-DR Fund Management"
- To: "Measurement Sensitivity in Administrative-Throughput Analysis of CDBG-DR Disaster Recovery: Evidence from a Cross-Sectional SEM and Complementary Survival Model"

#### Manuscript Framing (post-R1)

- Abstract, Evidence for Practice, Discussion opening, and Conclusion all reframed to foreground measurement sensitivity over the primary β = 0.266
- SEM reframed as exploratory decomposition rather than latent-variable validation
- Robustness section restructured so state-clustered bootstrap CI crossing zero is the first pattern discussed

#### Appendix Structure (post-R4)

- Appendix A: 9 subsections → 8 subsections (A.1 data-sources paragraph trimmed to preamble; A.5 SVI vintage spec reduced to pointer)
- Appendix B: 6 subsections → 5 subsections (B.5 fit verification deleted; parsimony-penalty discussion merged into B.4)
- Appendix C: 15 subsections → 9 subsections (consolidated master tables; no substantive content lost)
- Total appendix line count: ~595 → ~489 (~18% reduction)

#### Bibliography

- `manuscript_quarto/references.bib`: 36 entries normalized
- Full first names expanded (Cox, David R.; Geddam, Sheshadri Mohan; Gendeshmin, Saeed Bagherzadeh; etc.)
- IMF and CRS reports converted from `@article` to `@techreport`
- URL citations given structured access-date metadata
- Uncited entries removed (annanprah2023, hudoig2021, jaroscak2020, khorrammanesh2020)
- CSL switched from `apa.csl` to `chicago-author-date.csl` (PAR target requirement)

#### Data Quality and Documentation

- Centerpiece workload-indicator formula corrected in main text: now `Num_Program / (avg_employment × E_TOTPOP + ε)` matching code
- Geography-matching narrative reconciled between Methods section (upstream workflow) and Appendix A.5 (final 573-profile match rate)
- Appendix double-rendering resolved: `appendices:` block removed from `_quarto.yml`; single-source via `{{< include >}}` in `index.qmd`
- Survival sample-size inconsistencies resolved: authoritative N = 151 (70 events, 81 censored); documented smaller subsamples (142, 100) in Appendix A.7

### Removed

- `annanprah2023`, `hudoig2021`, `jaroscak2020`, `khorrammanesh2020` from bibliography (uncited)
- Stale textual cross-references ("Appendix Table A1", "Appendix Tables B1–B3", etc.) replaced with proper `@tbl-*` Quarto references

## [2026-04-13] — Synthetic Review Cycles R1, R2, R3

### Added

- Synthetic peer review process documented in `doc/SYNTHETIC_REVIEW_PROCESS.md`
- R1 review archive and response (all 10 major comments addressed)
- R2 triage (items escalated beyond R1; some deferred)
- R3 review archive and response with 3 re-analyses (nonzero-QCEW SEM, three-indicator primary check, cross-sectional financial-ratio bridge)

### Changed

- Primary manuscript switched from `manuscript_velocity/` to `manuscript_quarto/`
- `manuscript_velocity/` and `manuscript_kaifa_archive/` marked as archived

## [Unreleased]

### Added

#### Kaifa's Models Replication (Experimental)
- `s03_manuscript_replication.py` - Full pipeline replicating Kaifa's original manuscript analysis
- `sem_manuscript_replication.py` - Core functions for Kaifa's methodology
- Kaifa's model specifications: `kaifa_3x3_full`, `kaifa_3x3_no_duration`, `kaifa_2x2_minimal`
- Right-censoring implementation for Duration (incomplete programs use observation time)
- Grantee-level aggregation option (vs. grantee-disaster pairs)
- Methodology critique documentation embedded in code

#### Extended SEM Analysis Infrastructure
- `sem_multigroup.py` - Multi-group SEM fitting and measurement invariance testing
- `sem_mediation.py` - Indirect effect computation and bootstrap CIs
- `sem_longitudinal.py` - Cross-lagged panel model infrastructure (stub)
- `sem_bayesian.py` - Bayesian SEM estimation wrapper (stub)
- 51+ model specifications in `sem_specifications.py` (up from 24)

#### Multi-Threshold Duration Analysis
- Duration variables computed at 15 thresholds (30%, 35%, ..., 100%)
- `DURATION_THRESHOLDS` configuration in `config.py`
- Log-transformed duration columns for each threshold

#### Data Quality Improvements
- Adjustment tracking columns (positive flows vs. negative adjustments)
- "Clean" monotonic cumulative series for ratio calculation
- Enhanced QA flags: `QA_ratio_exceeds_one`, `QA_duration_censored`, `QA_adjustment_detected`
- Quality report includes adjustment/anomaly/censoring statistics

#### Analysis Enhancements
- Quartile ratio interaction cutoffs/knots (q25/q75) for velocity interaction tests
- Penalized stratified Cox fits and pooled/stratified interaction models for ratio strata
- New diagnostics output: `alternatives_survival_velocity_strata_models.csv`
- Cluster-robust standard errors for grantee-disaster analysis
- Standardized coefficient reporting
- Composite reliability (CR) and Average Variance Extracted (AVE) computation
- Bootstrap standard errors (1000 iterations)

#### Documentation
- Project restructuring to follow Research Project Management template
- New pipeline CLI (`src/pipeline.py`) with subcommand pattern
- Stage-based module organization (s00-s05)
- Configuration module (`src/config.py`)
- Quarto manuscript system (`manuscript_quarto/`)
- Comprehensive documentation (`doc/`)
- This changelog
- Quarterly QPR rollup output (`data_work/qpr_quarterly.parquet`)
- QPR cleaning step with QA flags (`data_work/qpr_clean.parquet`)
- Quality summaries for QPR raw and quarterly data (`data_work/quality/*.csv`)
- Configurable ratio construction (`RATIO_DEFINITION`) and QPR flow handling (`QPR_DOLLAR_FIELDS_ARE_FLOW`)
- QPR column mappings for alternate export labels and special year mappings

### Changed
- `s03_estimation.py`: Added grantee-level analysis option
- `s04_robustness.py`: Added multi-group and mediation analyses
- `config.py`: Added `DURATION_THRESHOLDS` configuration (15 levels)
- `sem_specifications.py`: Expanded from 24 to 51+ model specifications
- Renamed directories:
  - `data/raw/` → `data_raw/`
  - `data/processed/` → `data_work/`
  - `docs/` → `doc/`
  - `outputs/figures/` → `figures/`
- Reorganized source code into stages pattern
- Unified imports through config module
- Quarto manuscript renders as a single-article output; `render_all.sh` clears `_output/` before rendering
- Timeliness metrics and ratios now use quarterly rollups with cumulative series
- Feature engineering prefers cleaned QPR data when available

### Fixed

#### Ratio Aggregation Bug in Time-Varying Panel (April 12, 2026)
- **BREAKING**: Fixed `collapse_to_quarterly_panel()` in `src/utils/quarterly_panel.py`
  - Dollar columns (`QPR Fund Obligated $`, `QPR Fund Disbursed $`, `QPR Fund Expended $`) were aggregated with MAX instead of SUM across activities
  - Picked one activity's value instead of grant-level total, producing ratios up to 38.8 million
  - Cox model coefficients were driven to machine-zero (HR≈1.0000000000, concordance=0.171)
- **Fix details**:
  - Dollar columns now use SUM aggregation across activities
  - Ratios recomputed from summed grant-level totals after collapse
  - Added $1,000 minimum denominator to avoid division-by-near-zero
  - Added [0, 2] clipping to ratio outputs in both `quarterly_panel.py` and `time_varying_survival.py`
- **Impact**: Concordance corrected from 0.171 to 0.723; HR estimates now properly scaled (1.001 vs 1.0000000000)
- **Substantive conclusion unchanged**: Null finding confirmed with clean data (HR=1.001, p=0.991)
- Regenerated `panel_time_varying.parquet` and re-ran 1000-iteration bootstrap SEs
- See `doc/ANALYSIS_JOURNEY.md` Phase 8 for complete narrative

#### Critical Duration Calculation Bug (December 27, 2025)
- **BREAKING**: Fixed Duration calculation in `s01b_features.py`
  - `compute_timeliness_features_std()` was counting activity rows (~9 per quarter) instead of unique quarters
  - Duration=326 "quarters" was actually 326 rows → ~36 actual quarters
  - All velocity findings now NULL (HR≈1.00) with correct calculation
  - See `doc/ANALYSIS_JOURNEY.md` Phase 5 for complete narrative
- **Impact on prior results**:
  - Original velocity HR=4.37 (p=0.006) was artifact of bug
  - Corrected velocity HR≈1.00 (p≈1.00) - null effect
  - All phase-specific and heterogeneity effects also null
- **Fix details**:
  - Aggregate by quarter before computing duration
  - Use `group[quarter_col].nunique()` instead of `len(group)`
  - Added sanity checks for impossible Duration values

#### Data Quality Fixes (December 26, 2025)
- **Prior Grant Experience Integration**: Integrated `build_experience_dataset()` into s01b_features.py
  - Resolves zero-variance errors in full covariate survival models
  - 73/156 grantee-disasters (47%) now have non-zero prior experience
  - Adds Prior_Grant_Count, Prior_Grant_Dollars, Years_Experience, Experience_Index
  - Panel features increased from 177 to 182 columns
- **Government Classification**: Added 'rogco' (Northern Mariana Islands) to STATE_GOVERNMENTS list
  - Eliminates "Unknown grantee 'rogco'" warning
  - Corrects classification from Local to State
- See `doc/DATA_QUALITY_FIXES.md` for complete documentation

#### Earlier Fixes
- Duration calculation now handles incomplete programs correctly (censoring option)
- Ratio anomalies flagged instead of silently removed
- Zero-denominator handling documented (4 affected grantees)

### Removed
- Empty directories (`spatial/`, `visualization/`, `scripts/`)

---

## [0.1.0] - Initial Development

### Added
- Core SEM analysis modules:
  - `capacity_sem.data.loader` - QPR data loading
  - `capacity_sem.data.external_data` - External covariates
  - `capacity_sem.features.timeliness` - Duration metrics
  - `capacity_sem.features.experience_indicators` - Experience measures
  - `capacity_sem.features.program_stratification` - Activity classification
  - `capacity_sem.models.sem_specifications` - 24 model specifications
  - `capacity_sem.models.sem_fitting` - Model estimation
  - `capacity_sem.models.sem_diagnostics` - Fit evaluation
- Embedded population, severity, and employment datasets

---

## Format

Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Categories:
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerabilities
