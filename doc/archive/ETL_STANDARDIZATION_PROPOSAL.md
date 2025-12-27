# ETL Standardization Proposal

**Date**: December 26, 2024
**Purpose**: Create standardized, validated data pipeline upstream of all analyses

---

## Problem Statement

Current issues identified:
1. **Dynamic denominator artifacts**: Velocity calculations use changing obligated amounts, creating extreme "outliers" that are computational artifacts
2. **Branch-specific data processing**: Each analysis (main, experimental branches) may process data differently
3. **No centralized data validation**: Quality checks scattered across pipeline stages
4. **Unclear data lineage**: Hard to trace which "clean" version is authoritative

---

## Proposed ETL Architecture

### **Level 1: Raw Data (Unchanged)**
```
data_raw/
  qpr_data.csv              # Original HUD QPR export (never modified)
```

**Principle**: Never touch raw data.

---

### **Level 2: Standardized Data (Project-Level)**
```
data_work/
  qpr_standardized.parquet  # NEW: Single source of truth
```

**Stage**: `s00b_standardize_data.py` (NEW)

**Processing**:
1. ✅ **Parse and validate** all fields
2. ✅ **Handle missing values** with documented rules
3. ✅ **Detect and flag adjustments** (negative flows, large swings)
4. ✅ **Compute stable denominators** (final or max obligated amount)
5. ✅ **Winsorize extreme values** at configurable percentiles
6. ✅ **Create QA flags** for downstream filtering
7. ✅ **Generate data quality report**

**Output columns**:
- All original QPR columns
- `Obligated_Final`: Maximum obligated amount for this grantee-disaster (stable denominator)
- `Obligated_Clean`: Monotonic obligated series (for ratio calculation)
- `Disbursed_Clean`, `Expended_Clean`: Monotonic cumulative series
- `QA_Flag_Extreme_Change`: Boolean flag for >100 pp/quarter changes
- `QA_Flag_Negative_Adjustment`: Boolean flag for retroactive corrections
- `QA_Flag_Missing_Quarters`: Boolean flag for reporting gaps

---

### **Level 3: Analysis-Ready Features (Project-Level)**
```
data_work/
  panel_standardized.parquet  # Grantee-disaster panel with stable features
```

**Stage**: `s01b_standardized_features.py` (NEW)

**Feature engineering with data quality rules**:

#### **1. Ratio Calculation (Fixed Denominator)**
```python
# Use FINAL obligated amount as denominator for all quarters
final_obligated = group['Obligated_Final'].iloc[0]

ratio_disbursed = group['Disbursed_Clean'] / final_obligated
ratio_expended = group['Expended_Clean'] / final_obligated
```

**Rationale**: Eliminates denominator-change artifacts.

#### **2. Velocity Calculation (Winsorized)**
```python
# Calculate quarter-to-quarter change
velocity_raw = ratio.diff()

# Winsorize at 1st/99th percentile by default
# Configurable in config.py: VELOCITY_WINSOR_PERCENTILES = (0.01, 0.99)
velocity_clean = winsorize(velocity_raw, limits=VELOCITY_WINSOR_PERCENTILES)
```

**Rationale**: Caps extreme values while preserving variation.

#### **3. Quality-Aware Aggregation**
```python
# For static features, exclude flagged observations
velocity_static = velocity_clean[~QA_Flag_Extreme_Change].mean()

# Time-varying features include QA flags for optional filtering
tv_panel['Velocity_Clean'] = velocity_clean
tv_panel['Velocity_QA_Flag'] = QA_Flag_Extreme_Change
```

**Rationale**: Analysts can choose whether to include flagged observations.

---

### **Level 4: Branch-Specific Analysis**
```
Branches (main, experimental):
  - Read from data_work/panel_standardized.parquet
  - Apply branch-specific models/methods
  - No data cleaning/transformation at this level
```

**Principle**: All branches start from the same clean data.

---

## Implementation Plan

### **Phase 1: Create Standardized ETL (1-2 days)**

1. **Create `s00b_standardize_data.py`**:
   - Parse QPR data
   - Compute stable denominators (Obligated_Final, Obligated_Clean)
   - Flag extreme changes, adjustments, missing data
   - Winsorize flows and ratios
   - Output: `data_work/qpr_standardized.parquet`

2. **Create `s01b_standardized_features.py`**:
   - Compute ratios using fixed denominators
   - Compute winsorized velocity
   - Create both static (mean) and time-varying (panel) features
   - Output: `data_work/panel_standardized.parquet`

3. **Add configuration** to `src/config.py`:
   ```python
   # ETL Configuration
   VELOCITY_WINSOR_PERCENTILES = (0.01, 0.99)  # Winsorize at 1%/99%
   VELOCITY_EXTREME_THRESHOLD = 100  # Flag changes >100 pp/quarter
   RATIO_DENOMINATOR = "final"  # "final", "max", or "quarter"
   ```

4. **Update `src/pipeline.py`**:
   ```bash
   python src/pipeline.py standardize_data  # Run s00b
   python src/pipeline.py build_features    # Run s01b
   ```

### **Phase 2: Validate Against Existing Results (Half day)**

1. **Regression test**: Ensure static velocity results still significant after fixing denominators
2. **Compare outlier distributions**: Verify winsorization reduces extreme values
3. **Check correlation**: Confirm new standardized features correlate with old features (should be r>0.90 after outlier removal)

### **Phase 3: Update Existing Stages (Half day)**

1. **Modify `s02_features.py`**: Read from `panel_standardized.parquet` instead of computing features
2. **Modify `s03b_survival_estimation.py`**: Read from `panel_standardized.parquet` for time-varying
3. **Remove redundant feature calculations** from downstream stages

### **Phase 4: Documentation and Testing (Half day)**

1. **Create data lineage diagram** in `doc/DATA_LINEAGE.md`
2. **Update** `doc/DATA_DICTIONARY.md` with QA flags and clean columns
3. **Add unit tests** for standardization logic
4. **Generate** standardized quality report

---

## Outlier Handling Strategy

### **Current Outliers (Before Fix)**

| Grantee | Quarter | Type | Raw Value | Issue |
|---------|---------|------|-----------|-------|
| Joplin, MO | Q5-Q6 | Expenditure | +1,933 / -1,019 | Dynamic denominator |
| Mississippi | Q2, Q5 | Disbursement | +819 / -785 | Dynamic denominator |

### **After Standardization (Expected)**

With **fixed denominators** and **winsorization at 1%/99%**:

| Measure | Before (99th pct) | After (estimated) | Reduction |
|---------|------------------|-------------------|-----------|
| Disbursement velocity | 33.9 pp/q | ~15 pp/q | 56% |
| Expenditure velocity | 34.6 pp/q | ~15 pp/q | 57% |

**Rationale**: Most extreme values will disappear when denominators are fixed. Remaining outliers (legitimate large jumps) will be capped at 99th percentile.

---

## Benefits

### **For Reproducibility**
✅ Single source of truth for all analyses
✅ Clear data lineage from raw → standardized → features
✅ Version-controlled transformation rules

### **For Analysis Quality**
✅ Eliminates computational artifacts (dynamic denominator issue)
✅ Reduces noise from extreme outliers
✅ Provides QA flags for sensitivity analyses

### **For Collaboration**
✅ All branches work with same clean data
✅ Data quality decisions documented centrally
✅ Easy to update/revise standardization rules

### **For Manuscript**
✅ Can clearly state data processing steps
✅ Robust to reviewer requests for alternative specifications
✅ Easy to generate supplementary data quality tables

---

## Configuration Options

### **`src/config.py` Additions**

```python
# =============================================================================
# ETL Standardization Configuration
# =============================================================================

# Ratio denominator choice
RATIO_DENOMINATOR = "final"  # Options: "final", "max", "quarter"
# - "final": Use last reported obligated amount (default)
# - "max": Use maximum obligated amount across all quarters
# - "quarter": Use quarter-specific obligated (current behavior, NOT recommended)

# Velocity winsorization
VELOCITY_WINSOR_PERCENTILES = (0.01, 0.99)  # (lower, upper)
# Set to None to disable winsorization

# Extreme change flagging
VELOCITY_EXTREME_THRESHOLD = 100  # Flag changes >100 pp/quarter as extreme
# These will be QA-flagged but not removed (analyst decision)

# Missing data handling
MAX_MISSING_QUARTERS_ALLOWED = 4  # Grantees with >4 missing quarters get QA flag

# Adjustment detection
NEGATIVE_ADJUSTMENT_THRESHOLD = -10  # Flag negative changes <-10 pp as adjustments
```

---

## Alternative Approaches Considered

### **❌ Branch-Specific Cleaning**
- Each branch implements own cleaning rules
- **Problem**: Inconsistent results, hard to compare across branches

### **❌ No Winsorization (Remove Outliers)**
- Simply drop observations >3 std
- **Problem**: Loses information, creates sample selection bias

### **❌ Post-Model Adjustment**
- Fix outliers after seeing model results
- **Problem**: Data-dependent cleaning creates p-hacking risk

### **✅ Upstream Standardization (Proposed)**
- Clean data once, centrally, before any analysis
- **Benefit**: Transparent, reproducible, branch-agnostic

---

## Migration Path

### **For Current Experimental Branch**

1. **Keep existing results** as-is for comparison
2. **Run standardized ETL** to create `panel_standardized.parquet`
3. **Re-run survival models** using standardized data
4. **Compare results** (expect same directional findings, different magnitudes)
5. **Document** in `doc/VELOCITY_DIAGNOSTICS_REPORT.md` that outliers were artifacts

### **For Future Branches**

- Always start from `panel_standardized.parquet`
- No need to re-implement data cleaning
- Focus on methodological innovation, not data wrangling

---

## Success Criteria

✅ **Outlier reduction**: <1% of observations >50 pp/quarter change
✅ **Result stability**: Static velocity remains significant (HR>1.3, p<0.05)
✅ **Time-varying improvement**: Null results remain, but with smaller std errors
✅ **Documentation complete**: Data lineage and QA report generated
✅ **Tests passing**: Unit tests for standardization logic pass

---

## Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| 1. Standardized ETL | 1-2 days | s00b, s01b scripts + parquet files |
| 2. Validation | 0.5 days | Regression test results |
| 3. Update stages | 0.5 days | Modified s02, s03b stages |
| 4. Documentation | 0.5 days | Updated docs + tests |
| **Total** | **2-3 days** | Production-ready standardized pipeline |

---

## Questions for Discussion

1. **Winsorization thresholds**: Use 1%/99% (conservative) or 5%/95% (aggressive)?
2. **QA flag handling**: Should models auto-exclude flagged observations or leave to analyst?
3. **Denominator choice**: Use final obligated (last reported) or max obligated (highest ever)?
4. **Backward compatibility**: Keep old feature columns for comparison or clean break?

---

## Recommendation

**Implement immediately** - This fixes a fundamental data quality issue affecting all velocity-based analyses. The dynamic denominator problem explains why time-varying velocity appears so noisy.

**Priority**: HIGH - Core infrastructure that improves all downstream analyses.
