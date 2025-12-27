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
python src/pipeline.py review_status --manuscript velocity   # Check review status
python src/pipeline.py review_new --manuscript velocity --focus par_general  # New review
python src/pipeline.py review_verify --manuscript velocity   # PAR compliance checks
python src/pipeline.py review_archive --manuscript velocity  # Archive completed review
python src/pipeline.py review_report        # Summary across all manuscripts
```

## Project Branching Strategy

This project uses **git branches** to manage alternative analytical approaches while preserving the main analysis.

### Current Branches

| Branch | Purpose | Status | Key Files |
|--------|---------|--------|-----------|
| `main` | Time-varying survival with capacity ratios | Complete (null findings) | manuscript_quarto/ |
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
| `manuscript_kaifa_archive/` | SEM with latent constructs | β=71.02, p=0.01 (N=36-40) | Dec 2024 |
| TBD | Time-varying with ratios | HR=1.10, p>0.05 (null) | TBD |

### Tags for Milestone Tracking

- `v0.1.0`: Initial commit with SEM infrastructure
- `v0.2.0-time-varying-null-findings`: Time-varying survival complete, null findings documented
- Future: `v0.3.0-alternative-capacity-[result]`

---

## Current Methodology: Survival Analysis

The manuscript uses **survival analysis** (Cox Proportional Hazards, Accelerated Failure Time models) to analyze disaster recovery completion timing. SEM infrastructure remains for sensitivity analysis but is not the primary methodology.

### Why Survival Analysis?

With 73.7% of CDBG-DR programs incomplete at the 95% threshold, standard regression approaches face a censoring problem. Survival analysis properly handles right-censored observations while utilizing the full sample.

### Key Results

- **Sample**: N=143-151 grantee-disaster pairs
- **Events**: 71 at 95% completion threshold
- See `doc/RESEARCH_SYNTHESIS_REPORT.md` for detailed findings

### Capacity Indicators

- `Ratio_disbursed_to_obligated`: Cumulative mean ratio of disbursed to obligated funds
- `Ratio_expended_to_disbursed`: Cumulative mean ratio of expended to disbursed funds

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
- **Archived Kaifa SEM manuscript**: `manuscript_kaifa_archive/`
- **Output**: `manuscript_quarto/_output/`

```bash
cd manuscript_quarto
./render_all.sh                      # All formats (HTML, PDF, DOCX)
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh  # Skip pipeline re-run
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
- Compare to internal prior work (Kaifa SEM manuscript)
- Use metacommentary ("advances the literature", "first application", "most robust estimates")
- Reference "latent constructs" or "complex measures" comparatively
- Include "why survival analysis succeeds where SEM fails" framing
- Add comparisons to "prior approaches" when meaning internal work

#### DO

- Present findings directly without self-referential framing
- Reference legitimate external literature appropriately (GAO, HUD, academic publications)
- Let the methodology speak for itself
- Keep robustness comparisons in appendices (appendix-c)

#### Examples

| Avoid | Use Instead |
|-------|-------------|
| "This study examines..." | "Government administrative capacity affects..." |
| "This approach advances the literature..." | [Simply present the analysis] |
| "Prior latent variable approaches may overcomplicate..." | [Remove or cite external literature] |
| "Unlike traditional SEM approaches..." | [Present survival analysis on its own merits] |

### Legitimate External References

Citing published research is appropriate:
- GAO reports: `[@gao2019]`
- HUD evaluations: `[@hud2020]`
- Academic literature: `[@gerber2022; @peacock2022]`

**What to avoid**: References to "prior SEM approaches" when meaning our internal Kaifa manuscript (now archived).

---

## Data Pipeline

```
data_raw/qpr_data.csv
    ↓
data_work/qpr_raw.parquet        (s00_ingest)
    ↓
data_work/qpr_clean.parquet      (s00_ingest)
    ↓
data_work/qpr_quarterly.parquet  (s00_ingest)
    ↓
data_work/panel.parquet          (s01_link)
    ↓
data_work/panel_features.parquet (s02_features)
    ↓
data_work/diagnostics/           (s03_estimation)
    ↓
figures/*.png                    (s05_figures)
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

## SEM Infrastructure (Sensitivity Analysis Only)

The SEM codebase remains for robustness checks in appendix-c. For primary analysis, use survival analysis.

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

This contains the original SEM-based manuscript with known methodological issues:

1. **Right-censoring**: 73.7% of observations lack valid Duration at 95% threshold
2. **Mathematical circularity**: Timeliness = 1/Duration as capacity indicator
3. **Grantee-level aggregation**: Averaging across disasters reduces variance

See `doc/archive/ANALYSIS_COMPARISON_REPORT.md` for detailed comparison of methodologies.

---

## Velocity Manuscript (manuscript_velocity/)

**Location**: `manuscript_velocity/`
**Status**: Development

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
| `velocity` | `manuscript_velocity/` | `doc/reviews/velocity/` | Active |

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
