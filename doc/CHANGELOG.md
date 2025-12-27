# Changelog

All notable changes to this project will be documented in this file.

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
