# Capacity-SEM Project - Claude Code Instructions

## Quick Start

```bash
source .venv/bin/activate  # REQUIRED for all scripts
```

### Common Commands

```bash
# Standardized Pipeline (RECOMMENDED) ✨
python src/pipeline.py ingest_data          # Stage 0: Ingest raw data
python src/pipeline.py standardize_data     # Stage 0b: Standardize with fixed denominators
python src/pipeline.py build_panel          # Stage 1: Create grantee-disaster panel
python src/pipeline.py build_features_std   # Stage 1b: Build features from standardized data
python src/pipeline.py aggregate_program_types  # Stage 1c: Aggregate program type features
python src/pipeline.py run_survival         # Stage 3b: Time-varying survival analysis

# Legacy Pipeline (DEPRECATED - for replication only)
python src/pipeline.py compute_features     # Stage 2: OLD - uses dynamic denominators

# SEM Models (for sensitivity analysis)
python src/pipeline.py run_estimation --model exp_optimal_v1
python src/pipeline.py run_robustness
python src/pipeline.py make_figures

# Additional Analysis Commands
python src/pipeline.py run_alternatives             # Stage 6: Alternative modeling approaches
python src/pipeline.py run_survival_threshold_sensitivity  # Threshold sensitivity (20-100%)
python src/pipeline.py capacity_summary             # Stage 7: Corrected capacity summary
python src/pipeline.py list_models                  # List available SEM specifications

# Run complete pipeline
python src/pipeline.py run_all

# Manuscript
cd manuscript_quarto && ./render_all.sh

# Synthetic Peer Review (Multi-Manuscript)
python src/pipeline.py review_status --manuscript quarto     # Check review status (primary)
python src/pipeline.py review_new --manuscript quarto --focus par_general  # New review
python src/pipeline.py review_verify --manuscript quarto     # PAR compliance checks
python src/pipeline.py review_archive --manuscript quarto    # Archive completed review
python src/pipeline.py review_report        # Summary across all manuscripts
```

### Manuscript Slash Commands

General-purpose commands available in `~/.claude/commands/`:

| Command | When to use |
|---------|-------------|
| `/manuscript-audit` | Before submission: audit tables vs. prose, check writing rules, verify robustness honesty |
| `/review-triage` | After receiving reviews: classify comments, update REVISION_TRACKER.md, prioritize revisions |
| `/manuscript-convert` | When converting between DOCX and Quarto or rendering output |
| `/data-provenance` | When writing or updating data appendix descriptions |

These commands read CLAUDE.md for project-specific context (target journal, writing rules, file locations). Project-specific Codex skills with reference checklists are also in `.codex/skills/`.

## Project Branching Strategy

This project uses **git branches** to manage alternative analytical approaches while preserving the main analysis.

### Current Branches

| Branch | Purpose | Status | Key Files |
|--------|---------|--------|-----------|
| `main` | Cross-sectional SEM with survival comparison | Active (PAR submission target) | manuscript_quarto/ |
| `analysis/alternative-capacity-measures` | Explore non-ratio capacity operationalizations | Active | src/stages/s01b_features.py, scripts/ |

### Branch Workflow

1. **Preserve main**: All commits to main are tagged at major milestones
2. **Create analysis branch**: `git checkout -b analysis/[approach-name]`
3. **Develop alternative**: Modify capacity measure calculation, re-run analyses
4. **Document findings**: Update branch-specific documentation
5. **Decision point**: Merge to main if superior, archive if not

### Archived Manuscripts

| Directory | Method | Finding | Date Archived |
|-----------|--------|---------|---------------|
| `manuscript_kaifa_archive/` | Cross-sectional SEM (N=573 jurisdictions) | Burden→Timeliness β=0.266, p<0.001 | Apr 2026 |
| `manuscript_velocity/` | Time-varying survival (null-results draft) | HR≈1.0, p≈0.99 (null) | Apr 2026 |

**Note**: `manuscript_kaifa_archive/` preserves Kaifa's polished SEM draft that served as the basis for the current `manuscript_quarto/` rewrite. The rewrite retains Kaifa's core SEM framing, adds robustness caveats and a survival analysis comparison.

### Tags for Milestone Tracking

- `v0.1.0`: Initial commit with SEM infrastructure
- `v0.2.0-time-varying-null-findings`: Time-varying survival complete, null findings documented
- Future: `v0.3.0-alternative-capacity-[result]`

---

## Current Methodology: Measurement-Sensitivity Audit Protocol

**Manuscript title** (current): *"A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"*

The primary manuscript (`manuscript_quarto/`) is structured around a **six-item measurement-sensitivity audit protocol** whose deliverable is a **specification-curve dashboard**, not a single point estimate. The protocol items are: (1) proxy transparency, (2) sample-selection sensitivity, (3) within-framework operationalization bridge, (4) cross-framework triangulation, (5) vintage and granularity documentation, (6) cluster-appropriate inference. The cross-sectional SEM on N=573 administering jurisdictions and the complementary Cox survival analysis on N=151 grantee-disaster pairs serve as **demonstrations** of the protocol, not as the primary inferential engine.

Post-R6 structural pivot: the contribution is the protocol; the substantive headline is that the capacity-timeliness coefficient is *not stably identified* under principled perturbations. Class taxonomy refined at R8 (Ia measurement-preserving / Ib sample-scope) and R10 (II-C capacity-operationalization change / II-O outcome change). Central claim recast at R10 from "near zero" to "not stably identified" after the dashboard was shown to span positive, near-zero, and negative estimates.

### SEM Demonstration Results (Audit Item 1, 2, 3, 6)

- **Primary analytical specification**: Local-only (N=543), β=+0.257, Stable
- **Pooled supplementary**: N=573 administering-jurisdiction profiles (30 state, 543 local), β=+0.266 (conventional ML p<0.001), but **95% CI [−0.129, 0.531] under state-clustered bootstrap (n=1000, 35 clusters)** crosses zero
- **Specification-curve dashboard** (capacity-timeliness path):
  - Class Ia (measurement-preserving): residualized burden +0.297 (Stable); maturity-band controls +0.105 (Attenuated); portfolio-scale controls +0.127 (Attenuated); QCEW imputation bounds [+0.024, +0.267] (Attenuated)
  - Class Ib (sample-scope): local-only (PRIMARY) +0.257; nonzero-QCEW N=111 +0.132 n.s. (Attenuated); local-AND-nonzero-QCEW N=81 +0.145 n.s.; mature-only N=275 **−0.244 Reversed**; NDR/MIT-excluded N=512 +0.255 (Stable)
  - Class II-C (capacity-operationalization change): raw portfolio counts **−0.443 Reversed**; within-SEM financial-ratio bridge **−0.600 Reversed (II-C\*)**
  - Class II-O (outcome change): reconstructed-panel fixed-horizon q=8/12/16 (N=102–128) ≈ 0 (Attenuated)
  - Class III (multiple dimensions): time-varying Cox HR ≈ 1.0, null
- **ε-sensitivity**: SEM coefficient invariant to ε∈{10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²} (Appendix C.10) — what matters is the *treatment* of suppressed zeros, not the ε value
- **Transportability**: non-suppressed-QCEW subsample is *not* representative of broader population (40× population, 27% state vs. 0%, 5× more programs, lower completion ratios; Appendix A.3, @tbl-zero-nonzero-comparison)
- **Model fit**: Two-factor reference specification (CFI=0.915, RMSEA=0.081); AIC/BIC prefer one-factor, so the two-factor choice is theoretically motivated rather than statistically compelled

### Survival Analysis Results (Audit Item 4)

- **Sample**: N=151 grantee-disaster pairs (70 events, 81 right-censored)
- **Time-varying specification**: Null capacity-outcome association (Disbursement ratio HR=1.001, p=0.991); concordance 0.723 (covariates discriminate via channels other than capacity ratios)
- **Within-Cox divergence**: Single-spell Cox at baseline (q=3) is positive and significant (HR=1.46 at 20% threshold rising to HR=2.58 at 100%, p<0.01) — same data, different specification, different answer
- **Interpretation**: Cross-framework divergence from SEM cannot be attributed to a single design dimension; the within-Cox instability shows some of what looks like framework-level divergence is within-framework measurement sensitivity

### Bridge Analyses (One-Dimension-at-a-Time)

- **Within-SEM capacity bridge**: β = −0.600 (financial-flow capacity indicators in same framework) — sign reversal isolates operationalization from framework
- **Within-survival staffing bridge**: HR = 1.52 and 0.61 (staffing-scaled workload indicators in same survival framework, N=100, null) — survival null holds across capacity operationalizations

### Capacity Indicators

- SEM primary: staffing-scaled workload ratios `programs/staff`, `disasters/staff` with population-interaction denominator (`avg_employment × E_TOTPOP + ε`) + QCEW employment/payroll proxies
- SEM bridge: financial flow ratios (disbursed/obligated, expended/disbursed) as reflective capacity indicators
- Survival primary: financial flow ratios as time-varying covariates with 1-quarter lag, clipped [0, 2], $1K min denominator
- Survival bridge: staffing-scaled workload ratios as static covariates

### SVI Vintage and Disaster-Year Sensitivity (Appendix C.9)

- CDC/ATSDR historical SVI vintages 2010–2022 downloaded to `data_raw/svi_historical/`
- State-level vintage sensitivity: capacity-timeliness coefficient stable (0.148–0.239); Theme 1 (socioeconomic) flips sign between 2010 and 2014+ vintages
- Per-jurisdiction disaster-year SVI re-estimation: 572/573 match rate, β(Burden → Timeliness) = +0.286 (robust to SVI substitution)

---

## Standardized ETL Pipeline

**Status**: Production-ready (December 2025)

**Purpose**: Eliminate computational artifacts in velocity calculations through fixed-denominator approach.

### Problem Solved

Time-varying velocity calculations produced extreme outliers (±1,933 pp/quarter) due to **dynamic denominators**:

```
# BEFORE (Dynamic denominators):
Velocity_t = (Disbursed_t / Obligated_t) - (Disbursed_{t-1} / Obligated_{t-1})
# When Obligated changes → spurious velocity swings

# AFTER (Fixed denominators):
Velocity_t^std = (Disbursed_t / Obligated_final) - (Disbursed_{t-1} / Obligated_final)
# Stable denominator → only numerator changes create velocity
```

**Impact**: Extreme velocity reduced from 0.6% to 0.24%; velocity std dev reduced 68%

### New Pipeline Stages

| Stage | Command | Purpose |
|-------|---------|---------|
| **0b** | `standardize_data` | Standardize with fixed denominators + winsorization |
| **1b** | `build_features_std` | Aggregate standardized velocity to grantee-disaster level |

**Output files**:
- `data_work/qpr_standardized.parquet` - Standardized quarterly data (130,605 rows)
- `data_work/panel_features_std.parquet` - Standardized features (156 rows, 177 columns)

### Usage

**Always use standardized pipeline for new analyses**:

```bash
python src/pipeline.py ingest_data           # 0: Ingest
python src/pipeline.py standardize_data      # 0b: Standardize
python src/pipeline.py build_panel           # 1: Panel
python src/pipeline.py build_features_std    # 1b: Features
python src/pipeline.py run_survival          # 3b: Analysis (auto-uses standardized data)
```

**Legacy pipeline** (s02_features.py) deprecated - use only for replication.

### Key Features

- **Fixed denominators**: Uses final obligated amount across all quarters
- **Winsorization**: Caps velocity at 1%/99% percentiles
- **QA flags**: Tracks extreme velocity, obligated jumps, negative adjustments
- **Backward compatible**: Adds Duration_of_completion, N_Quarters aliases
- **Single source of truth**: Pre-computed velocity eliminates inconsistencies

### Documentation

- **Methodology**: `doc/ETL_STANDARDIZATION.md`
- **Test results**: `doc/STANDARDIZED_PIPELINE_TEST_RESULTS.md`
- **Column definitions**: `doc/DATA_DICTIONARY.md` (Standardized QPR Variables section)
- **Pipeline stages**: `doc/PIPELINE.md` (Stages 0b and 1b)

---

## Research Extension: Velocity Mechanisms & Heterogeneity

**Purpose**: Investigate whether spending velocity predicts CDBG-DR program completion through mechanistic analysis and heterogeneity testing.

**Documentation**: See `doc/RESEARCH_SYNTHESIS_REPORT.md` for complete findings.

### Analysis Scripts

Extended analysis scripts are in `scripts/`:

```bash
python scripts/run_multistage_analysis.py     # Multi-stage bottleneck identification
python scripts/run_trajectory_clustering.py   # Velocity trajectory clustering
python scripts/run_meta_analysis.py           # Aggregate all velocity estimates
```

### New Pipeline Stage: Stage 1c

**Command**: `python src/pipeline.py aggregate_program_types`

**Purpose**: Aggregate activity-level data to grantee-disaster level to create program portfolio features.

**Inputs**:
- `data_work/qpr_standardized.parquet` (quarterly data with Activity Type)
- `data_work/panel_features_std.parquet` (base panel)

**Outputs**:
- `data_work/panel_program_types.parquet` (156 records, 18 columns)
  - Primary_Program_Type (Housing, Infrastructure, Administration, etc.)
  - Program_Diversity_Index (Herfindahl index)
  - Category percentages (Housing_Pct, Infrastructure_Pct, etc.)

### New Features in Panel

**Phase-Specific Velocity** (added to panel_features_std.parquet, 202 columns total):
- `Velocity_Early` - Mean velocity in first third of program duration
- `Velocity_Mid` - Mean velocity in middle third
- `Velocity_Late` - Mean velocity in final third
- `Velocity_Acceleration` - Change from early to late phase (Late - Early)
- Median versions: `Velocity_Early_median`, etc.

### Outputs

- **Synthesis**: `doc/RESEARCH_SYNTHESIS_REPORT.md`
- **Diagnostics**: `data_work/diagnostics/*.csv`
- **Figures**: `figures/*.png`
- **Analysis logs**: `doc/archive/analysis_logs/`

---

## Manuscript

### Location and Rendering

- **Primary manuscript**: `manuscript_quarto/index.qmd`
- **Title**: *"A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"*
- **Archived Kaifa SEM manuscript**: `manuscript_kaifa_archive/` (source for ~70% of current content; archived after reframing)
- **Archived velocity manuscript**: `manuscript_velocity/` (survival-only draft, superseded)
- **Output**: `manuscript_quarto/_output/`
- **Current state**: Ten synthetic review cycles (R1–R10) closed; structural pivot to audit-protocol framing at R6; central claim recast to indeterminacy at R10. Response letters and INDEX in `doc/reviews/quarto/`; CSL Chicago Author-Date (PAR); 7,996 prose words; 141-word abstract

```bash
cd manuscript_quarto
./render_all.sh                      # All formats (HTML, PDF, DOCX)
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh  # Skip pipeline re-run
```

**Worktree warning**: When rendering inside a git worktree (`.claude/worktrees/*/`), Quarto writes output relative to the worktree copy, not the main repo. After rendering in a worktree, copy the output to the main repo:

```bash
# After rendering in a worktree, copy output to main repo
cp manuscript_quarto/_output/*.docx /Volumes/T9/Projects/capacity-sem-project/manuscript_quarto/_output/
```

Or use absolute paths to render directly into the main repo:

```bash
quarto render /Volumes/T9/Projects/capacity-sem-project/manuscript_quarto/ --to docx
```

### Target Journal: Public Administration Review (PAR)

| Requirement | Value |
|-------------|-------|
| Word limit | 8,000 (including abstract, endnotes, references) |
| Abstract | 150 words maximum |
| Font | 12-point Times New Roman |
| Spacing | Double-spaced |
| Margins | 1 inch |
| Citations | Chicago Author-Date (16th ed.) |
| Reference names | Full first names required |
| Special section | Evidence for Practice (3-5 bullet points) |
| Review type | Blind (no author identification) |

### Manuscript Writing Rules

#### DO NOT

- Use "this study" self-references — present findings directly
- Compare to internal prior work or pit the SEM against the survival analysis
- Use metacommentary ("advances the literature", "first application", "most robust estimates")
- Frame one method as superior to the other — present divergence as informative, not as one method "winning"
- Add comparisons to "prior approaches" when meaning internal archived work

#### DO

- Present findings directly without self-referential framing
- Reference legitimate external literature appropriately (GAO, HUD, academic publications)
- Let the methodology speak for itself
- Present a robustness summary table in the main text (key results from Appendix C)
- Present the SEM–survival divergence as evidence of framework sensitivity, not as invalidation

#### Examples

| Avoid | Use Instead |
|-------|-------------|
| "This study examines..." | "Government administrative capacity affects..." |
| "This approach advances the literature..." | [Simply present the analysis] |
| "The survival analysis proves the SEM wrong..." | "The divergence demonstrates sensitivity to analytical framework" |
| "Our novel contribution..." | [State the contribution without metacommentary] |

### Legitimate External References

Citing published research is appropriate:
- GAO reports: `[@gao2019]`
- HUD evaluations: `[@hud2026]`
- Academic literature: `[@peacock2014; @miao2025; @martin2022]`

**What to avoid**: References to "prior approaches" when meaning internal archived work. The Kaifa archive is a source document, not a comparison target.

---

## Data Pipeline

### Standardized Pipeline (Current)

```
data_raw/qpr_data.csv
    ↓
data_work/qpr_raw.parquet              (s00_ingest)
    ↓
data_work/qpr_clean.parquet            (s00_ingest)
    ↓
data_work/qpr_quarterly.parquet        (s00_ingest)
    ↓
data_work/qpr_standardized.parquet     (s01a_standardize)
    ↓
data_work/panel.parquet                (s01_link)
    ↓
data_work/panel_features_std.parquet   (s01b_features_std)
    ↓
data_work/panel_program_types.parquet  (s01c_program_types)
    ↓
data_work/diagnostics/                 (s03_survival / SEM)
    ↓
figures/*.png                          (s05_figures)
```

### Legacy Pipeline (Deprecated)

```
data_work/panel_features.parquet       (s02_features — dynamic denominators, for replication only)
```

## Critical Constraints

### DO NOT

- Modify raw data in `data_raw/`
- Commit QPR data to git (contains sensitive information)
- Overwrite working parquet files manually

### ALWAYS

- Activate `.venv` before running scripts
- Use the pipeline CLI for data processing
- Run diagnostics after estimation changes
- Re-render Quarto after modifying `.qmd` files

---

## SEM Infrastructure

The SEM codebase provides the primary analysis for `manuscript_quarto/`. The two-factor model (Administrative Resources vs. Administrative Burden Capacity) is estimated on 573 administering-jurisdiction profiles with robustness checks reported in Appendix C.

### Available SEM Models

Run `python src/pipeline.py list_models` for complete list (51+ specifications).

| Category | Count | Description |
|----------|-------|-------------|
| Core Models | 10 | Primary analysis specifications |
| Experimental | 15 | Alternative indicator combinations |
| Covariates | 6 | Models with control variables |
| Multi-Group | 3 | State vs. local comparison |

### Government Subsets

- `all` - Full sample
- `state` - State governments only (37 grantees)
- `local` - Local governments only (40 grantees)

---

## Archived Kaifa Manuscript

**Location**: `manuscript_kaifa_archive/`

This contains Kaifa's polished SEM draft (`kaifa_r3_response_2026-04-11.qmd`) which served as the source for ~70% of the current `manuscript_quarto/` rewrite. The archived version includes:

- Two-factor SEM (Administrative Resources vs. Administrative Burden Capacity, N=573)
- Main finding: Burden Capacity → Recovery Timeliness (β=0.266, p<0.001)
- Comprehensive appendices (data, methods, 7 robustness checks)
- Spatial analysis sections (removed in the rewrite)

**Known limitations acknowledged in appendices**: maturity confounding (C.4 sign reversal), staffing-ratio artifact (C.6 sign reversal), QCEW proxy noise (85.1% zero-value rate for local jurisdictions).

See `doc/archive/ANALYSIS_COMPARISON_REPORT.md` for historical comparison of methodologies.

---

## Velocity Manuscript (manuscript_velocity/)

**Location**: `manuscript_velocity/`
**Status**: Archived — superseded by `manuscript_quarto/` rewrite. Retained as reference for the null survival-analysis findings incorporated into the primary manuscript's cross-framework comparison.

### Structure

| File | Purpose |
|------|---------|
| `index.qmd` | Main manuscript |
| `appendix-a-data.qmd` | Data appendix |
| `appendix-b-methods.qmd` | Methods appendix |
| `appendix-c-heterogeneity.qmd` | Heterogeneity analysis |
| `appendix-d-meta-analysis.qmd` | Effect heterogeneity summary |

### Rendering

```bash
cd manuscript_velocity
./render_all.sh                           # All formats (HTML, PDF, DOCX)
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh  # Skip pipeline re-run
```

---

## Key Data Files

| File | Purpose |
|------|---------|
| `data_work/panel_features.parquet` | Analysis-ready panel with all features |
| `data_work/qpr_quarterly.parquet` | Quarterly rollup with flows and cumulative totals |
| `data_work/diagnostics/*.csv` | Estimation results |
| `data_work/quality/*.csv` | Data quality reports |

See [doc/DATA_DICTIONARY.md](doc/DATA_DICTIONARY.md) for complete variable definitions.

---

## Synthetic Peer Review System

A systematic approach to stress-testing manuscripts before PAR submission using LLM-generated synthetic reviews.

### Overview

- **Purpose**: Identify methodological gaps, strengthen robustness, and anticipate reviewer concerns
- **Focus Areas**: par_general (comprehensive), methods (methodology), policy (practitioner relevance), clarity (writing)
- **Documentation**: See [doc/SYNTHETIC_REVIEW_PROCESS.md](doc/SYNTHETIC_REVIEW_PROCESS.md) for full methodology

### Multi-Manuscript Architecture

The review system supports multiple manuscript approaches:

| Manuscript | Directory | Reviews | Status |
|------------|-----------|---------|--------|
| `quarto` | `manuscript_quarto/` | `doc/reviews/quarto/` | Primary |
| `velocity` | `manuscript_velocity/` | `doc/reviews/velocity/` | Archived |

Each manuscript has its own:
- `REVISION_TRACKER.md` - Current review tracking
- `doc/reviews/{name}/` - Review-specific index and archive
- Focus-specific prompts tailored to the manuscript's methodology

### Workflow

1. **Generate Review**: `python src/pipeline.py review_new --manuscript velocity --focus par_general`
2. **Obtain LLM Review**: Send manuscript + embedded prompt to Claude/GPT-4
3. **Triage Comments**: Classify as VALID/ADDRESSED/SCOPE/INVALID in `manuscript_velocity/REVISION_TRACKER.md`
4. **Implement Changes**: Address valid concerns, update manuscript, re-render
5. **Verify**: `python src/pipeline.py review_verify --manuscript velocity` (includes PAR compliance checks)
6. **Archive**: `python src/pipeline.py review_archive --manuscript velocity` when complete

### PAR Compliance Checks

The `review_verify` command automatically checks:

- Word count ≤ 8,000
- No "this study" self-references
- Evidence for Practice section present
- Abstract ≤ 150 words

### Manuscript Word Count

**Important**: PAR's 8,000-word limit applies to **prose text only** in the main body, including abstract, endnotes, and references. It does **NOT** include:

- Tables and their contents
- Code blocks
- YAML front matter
- Appendices (these are supplementary)
- Figure captions (usually)

**How to count words accurately**:

```bash
# Count prose words in a .qmd file (excludes code blocks, tables, YAML)
cat manuscript.qmd | \
  sed '/^```/,/^```/d' | \      # Remove code blocks
  sed '/^---$/,/^---$/d' | \    # Remove YAML front matter
  sed '/^|/d' | \               # Remove markdown tables
  sed '/^\$/d' | \              # Remove LaTeX equations
  sed '/^#|/d' | \              # Remove Quarto chunk options
  grep -v '^\s*$' | \           # Remove blank lines
  wc -w

# Quick one-liner version:
cat index.qmd | sed '/^```/,/^```/d' | sed '/^---$/,/^---$/d' | sed '/^|/d' | sed '/^\$/d' | sed '/^#|/d' | grep -v '^\s*$' | wc -w
```

**Target word counts for PAR**:

- **Abstract**: 150 words maximum
- **Main body**: 6,000-7,500 words typical
- **Total with references**: ≤8,000 words

**Note**: A manuscript with ~3,000 prose words is substantially under-length for PAR and likely needs expansion. Typical full-length PAR articles run 6,000-7,500 words of prose.

### Review History

All completed reviews are archived in `doc/reviews/{manuscript}/archive/` with the format:
`review_NN_YYYY-MM-DD_FOCUS.md`

Track all review cycles: `python src/pipeline.py review_report`

### Adding a New Manuscript

When creating a new analytical approach:

1. Create manuscript directory: `manuscript_{name}/`
2. Add entry to `MANUSCRIPTS` dict in `src/review_management.py`
3. Create review subdirectory: `doc/reviews/{name}/`
4. Create `manuscript_{name}/REVISION_TRACKER.md`
5. Update pipeline.py choices list for `--manuscript` argument

---

## Documentation

| File | Content |
|------|---------|
| `doc/README.md` | Documentation index |
| `doc/PIPELINE.md` | Pipeline stages |
| `doc/METHODOLOGY.md` | Survival analysis and SEM methodology |
| `doc/DATA_DICTIONARY.md` | Variable definitions |
| `doc/ETL_STANDARDIZATION.md` | Fixed-denominator methodology |
| `doc/RESEARCH_SYNTHESIS_REPORT.md` | Research findings synthesis |
| `doc/reports/` | Analysis reports |
| `doc/archive/` | Historical documentation |

---

## Troubleshooting

**semopy not found**: `pip install semopy`

**lifelines not found**: `pip install lifelines`

**Missing data**: Run `python src/pipeline.py run_all --demo` to use demo data

**Git issues**: Check that data files are properly gitignored
