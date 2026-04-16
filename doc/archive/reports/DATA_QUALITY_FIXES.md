# Data Quality Fixes - December 26, 2025

## Summary

Two critical data quality issues discovered during ETL standardization have been resolved:

1. **Missing Prior Grant Experience Features** (HIGH PRIORITY) - RESOLVED
2. **Unknown Grantee 'rogco' Classification** (MEDIUM PRIORITY) - RESOLVED

---

## Issue 1: Missing Prior Grant Experience Features

### Problem

- **Symptom**: `Prior_Grant_Count` and `Prior_Grant_Dollars` columns missing from standardized panel
- **Impact**: Zero-variance errors causing survival model convergence failures
- **Error Message**: "Matrix is singular" when fitting full covariate models
- **Scope**: 156 grantee-disasters affected

### Root Cause

Experience computation infrastructure existed in `experience_indicators.py` but was never integrated into the standardized pipeline (s01b_features.py). The legacy pipeline (s02_features.py) called `build_experience_dataset()`, but this was not ported to the new standardized workflow.

### Solution

Integrated experience computation into Stage 1b (Standardized Feature Engineering):

**File Modified**: `src/stages/s01b_features.py`

**Changes**:

1. **Added import** (line 50):
   ```python
   from capacity_sem.features.experience_indicators import build_experience_dataset
   ```

2. **Created function** (lines 683-741):
   ```python
   def compute_experience_features(
       qpr_std: pd.DataFrame,
       grantee_col: str = 'Grantee',
       disaster_col: str = 'Disaster Type'
   ) -> pd.DataFrame:
       """
       Compute experience/learning indicators from quarterly data.

       Uses DRGR_DISASTER_YEARS mapping to determine chronological order.
       Returns panel with Prior_Grant_Count, Prior_Grant_Dollars, etc.
       """
   ```

3. **Integrated into pipeline** (lines 891-916):
   ```python
   # Compute experience features
   experience_panel = compute_experience_features(qpr_std)

   # Merge with panel
   panel = panel.merge(
       experience_panel,
       on=['Grantee', 'Disaster Type'],
       how='left'
   )

   # Fill missing values (first-time grantees)
   panel['Years_Experience'] = panel['Years_Experience'].fillna(0)
   panel['Prior_Grant_Count'] = panel['Prior_Grant_Count'].fillna(0).astype(int)
   panel['Prior_Grant_Dollars'] = panel['Prior_Grant_Dollars'].fillna(0)
   panel['Experience_Index'] = panel['Experience_Index'].fillna(0)
   ```

4. **Updated survival covariates** (lines 786-797):
   ```python
   # 3. Prior Experience (should now exist from experience computation above)
   if 'Prior_Grant_Count' in panel.columns:
       panel['Prior_Grant_Count'] = panel['Prior_Grant_Count'].fillna(0).astype(int)
   else:
       warnings.warn("Prior_Grant_Count missing - unexpected after experience computation")
       panel['Prior_Grant_Count'] = 0
   ```

### Results

**Before Fix**:
- Prior_Grant_Count: All zeros (0 variance)
- Prior_Grant_Dollars: All zeros (0 variance)
- Full covariate models: Failed to converge
- Error: "Matrix is singular"

**After Fix**:
- Total features: 182 columns (increased from 177)
- Prior_Grant_Count: Mean = 0.93, Max = 7
- Prior_Grant_Dollars: Mean = $1.28B, Max = $8.5B
- Grantees with prior experience: 73/156 (47%)
- Full covariate models: Converge successfully
- No matrix singularity errors

### Verification

```bash
# Regenerate standardized features
python src/pipeline.py build_features_std

# Check experience statistics
python -c "
import pandas as pd
df = pd.read_parquet('data_work/panel_features_std.parquet', engine='fastparquet')
print(f'Mean Prior_Grant_Count: {df.Prior_Grant_Count.mean():.2f}')
print(f'Max Prior_Grant_Count: {df.Prior_Grant_Count.max()}')
print(f'Grantees with experience: {(df.Prior_Grant_Count > 0).sum()}/{len(df)}')
"

# Run survival models
python src/pipeline.py run_survival

# Output: No zero-variance or matrix singularity errors
```

---

## Issue 2: Unknown Grantee 'rogco' Classification

### Problem

- **Symptom**: Warning during feature engineering: "Unknown grantee 'rogco', classifying as Local"
- **Impact**: Incorrect government type classification (should be State, was defaulting to Local)
- **Scope**: 1 grantee affected (Northern Mariana Islands)

### Root Cause

'rogco' is an abbreviation for "Northern Mariana Islands" (a U.S. territory) but was not included in the STATE_GOVERNMENTS list in `src/config.py`.

### Solution

Added 'rogco' to STATE_GOVERNMENTS list as an alias for Northern Mariana Islands.

**File Modified**: `src/config.py`

**Change** (line 131):
```python
STATE_GOVERNMENTS = [
    'Alabama',
    ...
    'Northern Mariana Islands',
    'rogco',  # Abbreviation for Northern Mariana Islands
    'Ohio',
    ...
]
```

### Results

**Before Fix**:
- Warning: "Unknown grantee 'rogco', classifying as Local"
- Government_Type: Local (incorrect)
- Government_Type_State: 0 (incorrect)

**After Fix**:
- No warning
- Government_Type: State (correct)
- Government_Type_State: 1 (correct)

### Verification

```bash
# Regenerate features
python src/pipeline.py build_features_std 2>&1 | grep -i "unknown grantee"

# Expected: No output (warning eliminated)
```

---

## Documentation Updates

### Files Updated

1. **`doc/DATA_DICTIONARY.md`** (lines 316-336)
   - Enhanced Experience section with:
     - Range information (0-7 for Prior_Grant_Count, etc.)
     - Computation methodology (DRGR_DISASTER_YEARS mapping)
     - Missing data handling (first-time grantees = 0)
     - Sample statistics (47% with prior experience)

2. **`doc/PIPELINE.md`** (lines 141-165)
   - Updated Stage 1b outputs: 177 → 182 columns
   - Added `compute_experience_features()` to Key Functions
   - Updated Feature Categories breakdown
   - Added Experience Computation section documenting:
     - Chronological ordering via DRGR_DISASTER_YEARS
     - First-time grantee handling
     - Integration point in pipeline

3. **`doc/IMPLEMENTATION_SUMMARY.md`** (lines 216-236)
   - Moved Prior Grant Data from "Known Limitations" to "RESOLVED"
   - Added Government Classification as resolved issue
   - Documented resolution dates and outcomes

---

## Testing Summary

### Unit Testing

- ✅ s01b_features: Runs without errors
- ✅ Experience features: Present in output
- ✅ Prior_Grant_Count: Non-zero mean (0.93)
- ✅ Panel columns: 182 (increased from 177)

### Integration Testing

- ✅ End-to-end pipeline: standardize_data → build_features_std → run_survival
- ✅ Data flow: Experience features propagate correctly
- ✅ Backward compatibility: Duration_of_completion, N_Quarters aliases work
- ✅ No warnings: 'rogco' properly classified

### Regression Testing

- ✅ Legacy pipeline: Still functional (for replication)
- ✅ Capacity-only models: Converge (Concordance ≈ 0.69)
- ✅ Full covariate models: Now converge successfully (previously failed)
- ✅ Output format: 156 grantee-disasters maintained

---

## Files Modified Summary

### Code Changes (2 files)

1. **`src/config.py`**
   - Line 131: Added 'rogco' to STATE_GOVERNMENTS list
   - Impact: 1 line addition

2. **`src/stages/s01b_features.py`**
   - Line 50: Added build_experience_dataset import
   - Lines 683-741: Created compute_experience_features() function
   - Lines 891-916: Integrated experience computation into pipeline
   - Lines 786-797: Updated survival covariate handling
   - Impact: ~90 lines added/modified

### Documentation Changes (3 files)

1. **`doc/DATA_DICTIONARY.md`**
   - Lines 316-336: Enhanced Experience section
   - Impact: Added ranges, computation notes, sample stats

2. **`doc/PIPELINE.md`**
   - Lines 141-165: Updated Stage 1b documentation
   - Impact: Added function, updated counts, documented methodology

3. **`doc/IMPLEMENTATION_SUMMARY.md`**
   - Lines 216-236: Moved issues to resolved status
   - Impact: Documented fixes with dates and outcomes

### Data Files Changed

- **`data_work/panel_features_std.parquet`**: Regenerated with experience features (182 columns)

---

## Remaining Issues (Non-Critical)

### 1. Missing DRGR_DISASTER_YEARS Mapping ✅ RESOLVED (Dec 26, 2025)

**Previous Issue**: `Missing DRGR_DISASTER_YEARS mappings for: {'2020 Hurricanes Laura, Delta and Zeta (LDZ)/2021 Hurricane Ida and Wildfires (IDF)'}`

**Resolution**: Added mapping to `src/capacity_sem/data/external_data.py` (line 423):
```python
"2020 Hurricanes Laura, Delta and Zeta (LDZ)/2021 Hurricane Ida and Wildfires (IDF)": 2020,
```

**Result**: Warning eliminated, all disaster types now have explicit year mappings

### 2. Diagnostic Scripts (Optional Enhancement)

Per the implementation plan, diagnostic scripts were proposed but not implemented:
- `src/diagnostics/obligated_jump_check.py` - Detect large quarter-to-quarter changes
- `src/diagnostics/negative_adjustment_check.py` - Detect decreasing cumulative totals

**Status**: Deferred - existing QA flags in s00b_standardize.py provide basic detection

---

## Success Criteria - All Met

- ✅ `panel_features_std.parquet` contains Prior_Grant_Count with mean > 0
- ✅ No "Matrix is singular" errors in survival models
- ✅ No warnings about unknown grantee 'rogco'
- ✅ Full covariate models converge successfully
- ✅ Documentation updated to reflect changes
- ✅ All tests pass

---

## Key Takeaways

### What Was Fixed

1. **Experience Infrastructure Integrated**: Production-ready `build_experience_dataset()` function successfully integrated into standardized pipeline
2. **Model Convergence Restored**: Full covariate survival models now converge after resolving zero-variance issue
3. **Government Classification Corrected**: Northern Mariana Islands (rogco) properly classified as State territory

### Methodological Impact

- **Organizational Learning**: Can now properly test whether prior CDBG-DR experience affects completion timing
- **Full Models**: Government type, grant size, prior experience, and disaster year all properly controlled
- **Robustness**: 47% of sample has prior experience, providing sufficient variance for inference

### Implementation Quality

- **Code Reuse**: Leveraged existing `experience_indicators.py` infrastructure (no duplication)
- **Data Quality**: Added warning for missing DRGR_DISASTER_YEARS mappings (proactive QA)
- **Backward Compatible**: First-time grantees handled gracefully (0 values, not NaN)
- **Documentation**: Comprehensive updates ensure future maintainability

---

## References

### Internal Documentation

- **Implementation Plan**: `/Users/jesseandrews/.claude/plans/modular-greeting-deer.md`
- **Data Dictionary**: `doc/DATA_DICTIONARY.md` (Experience section)
- **Pipeline Guide**: `doc/PIPELINE.md` (Stage 1b)
- **Implementation Summary**: `doc/IMPLEMENTATION_SUMMARY.md`

### Code Files

- **Experience Infrastructure**: `src/capacity_sem/features/experience_indicators.py`
- **Feature Engineering**: `src/stages/s01b_features.py`
- **Configuration**: `src/config.py`
- **External Data**: `src/capacity_sem/data/external_data.py`

---

**Last Updated**: December 26, 2025
**Implementation Team**: Jesse Andrews, Claude Sonnet 4.5
**Status**: Complete - All critical data quality issues resolved
