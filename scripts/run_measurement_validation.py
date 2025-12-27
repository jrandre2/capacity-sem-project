#!/usr/bin/env python3
"""
Quick script to run Phase 1 measurement validation analyses.
Standalone version to avoid import issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter

# Paths
DATA_WORK_DIR = Path("../data_work")
DIAG_DIR = DATA_WORK_DIR / "diagnostics"
DIAG_DIR.mkdir(exist_ok=True)

# Load panel
panel = pd.read_parquet(DATA_WORK_DIR / "panel_features_std.parquet", engine='fastparquet')

print("="*80)
print("PHASE 1: MEASUREMENT VALIDATION ANALYSES")
print("="*80)

# =============================================================================
# Analysis 1.1: QA Flag Sensitivity
# =============================================================================

print("\n" + "="*80)
print("Analysis 1.1: QA Flag Sensitivity Analysis")
print("="*80)

# Prepare survival data
panel_surv = panel.copy()
panel_surv['Event'] = panel_surv['Duration'].notna() & (panel_surv['Duration'] > 0)
panel_surv['Duration_Surv'] = panel_surv['Duration'].fillna(panel_surv['N_Quarters'])

# Scale velocity by 100 for proper pp/quarter interpretation
for vel_col in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
    if vel_col in panel_surv.columns:
        panel_surv[f'{vel_col}_scaled'] = panel_surv[vel_col] * 100

results_qa = []

# 1. Baseline model
print(f"\n1. Baseline (All observations, N={len(panel_surv)}, Events={panel_surv['Event'].sum()})")

for vel_var in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
    vel_var_scaled = f'{vel_var}_scaled'
    subset = panel_surv[['Duration_Surv', 'Event', vel_var_scaled, 'Government_Type_State']].dropna()

    if len(subset) < 20:
        continue

    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(subset, duration_col='Duration_Surv', event_col='Event')
        hr = np.exp(cph.params_[vel_var_scaled])
        p = cph.summary.loc[vel_var_scaled, 'p']
        ci_lower = np.exp(cph.confidence_intervals_[vel_var_scaled][0])
        ci_upper = np.exp(cph.confidence_intervals_[vel_var_scaled][1])

        results_qa.append({
            'Model': 'Baseline',
            'Velocity_Measure': vel_var.replace('_pp', ''),
            'N': len(subset),
            'Events': int(subset['Event'].sum()),
            'Velocity_HR': hr,
            'Velocity_CI_lower': ci_lower,
            'Velocity_CI_upper': ci_upper,
            'Velocity_p': p,
        })
        print(f"  {vel_var}: HR={hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p:.4f}")
    except Exception as e:
        print(f"  {vel_var} failed: {e}")

# 2. Exclude high-flag programs
panel_clean = panel_surv[panel_surv['QA_High_Flag_Program'] == False].copy()
print(f"\n2. Excluding high-flag programs (N={len(panel_clean)}, {len(panel_clean)/len(panel_surv)*100:.1f}%)")

for vel_var in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
    vel_var_scaled = f'{vel_var}_scaled'
    subset = panel_clean[['Duration_Surv', 'Event', vel_var_scaled, 'Government_Type_State']].dropna()

    if len(subset) < 10:
        print(f"  {vel_var}: insufficient sample (N={len(subset)})")
        continue

    cph = CoxPHFitter(penalizer=0.05)
    try:
        cph.fit(subset, duration_col='Duration_Surv', event_col='Event')
        hr = np.exp(cph.params_[vel_var_scaled])
        p = cph.summary.loc[vel_var_scaled, 'p']
        ci_lower = np.exp(cph.confidence_intervals_[vel_var_scaled][0])
        ci_upper = np.exp(cph.confidence_intervals_[vel_var_scaled][1])

        results_qa.append({
            'Model': 'Exclude_High_Flag',
            'Velocity_Measure': vel_var.replace('_pp', ''),
            'N': len(subset),
            'Events': int(subset['Event'].sum()),
            'Velocity_HR': hr,
            'Velocity_CI_lower': ci_lower,
            'Velocity_CI_upper': ci_upper,
            'Velocity_p': p,
        })
        print(f"  {vel_var}: HR={hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p:.4f}")
    except Exception as e:
        print(f"  {vel_var} failed: {e}")

# 3. Exclude ANY extreme velocity flags
panel_no_extreme = panel_surv[panel_surv['Flag_Count_Extreme_Velocity'] == 0].copy()
print(f"\n3. Excluding ANY extreme velocity flags (N={len(panel_no_extreme)}, {len(panel_no_extreme)/len(panel_surv)*100:.1f}%)")

for vel_var in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
    vel_var_scaled = f'{vel_var}_scaled'
    subset = panel_no_extreme[['Duration_Surv', 'Event', vel_var_scaled, 'Government_Type_State']].dropna()

    if len(subset) < 10:
        print(f"  {vel_var}: insufficient sample (N={len(subset)})")
        continue

    cph = CoxPHFitter(penalizer=0.05)
    try:
        cph.fit(subset, duration_col='Duration_Surv', event_col='Event')
        hr = np.exp(cph.params_[vel_var_scaled])
        p = cph.summary.loc[vel_var_scaled, 'p']
        ci_lower = np.exp(cph.confidence_intervals_[vel_var_scaled][0])
        ci_upper = np.exp(cph.confidence_intervals_[vel_var_scaled][1])

        results_qa.append({
            'Model': 'Exclude_Any_Extreme',
            'Velocity_Measure': vel_var.replace('_pp', ''),
            'N': len(subset),
            'Events': int(subset['Event'].sum()),
            'Velocity_HR': hr,
            'Velocity_CI_lower': ci_lower,
            'Velocity_CI_upper': ci_upper,
            'Velocity_p': p,
        })
        print(f"  {vel_var}: HR={hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p:.4f}")
    except Exception as e:
        print(f"  {vel_var} failed: {e}")

# Save results
results_qa_df = pd.DataFrame(results_qa)
output_qa = DIAG_DIR / "measurement_validation_qa_flags.csv"
results_qa_df.to_csv(output_qa, index=False)
print(f"\n✓ Saved QA sensitivity results: {output_qa}")

# =============================================================================
# Analysis 1.2: Velocity Operationalization Comparison
# =============================================================================

print("\n" + "="*80)
print("Analysis 1.2: Velocity Operationalization Comparison")
print("="*80)

# Test different velocity operationalizations
velocity_variants = {
    'Expenditure_pp_mean': 'Expenditure_Velocity_pp',
    'Expenditure_pp_median': 'Expenditure_Velocity_median',
    'Disbursement_pp_mean': 'Disbursement_Velocity_pp',
    'Disbursement_pp_median': 'Disbursement_Velocity_median',
    'Index_pp_mean': 'Capacity_Velocity_Index_pp',
    'Index_pp_median': 'Capacity_Velocity_Index_median',
    'Expenditure_early_2q': 'Expenditure_Velocity_early_2q_pp',
    'Expenditure_early_4q': 'Expenditure_Velocity_early_4q_pp',
    'Expenditure_early_6q': 'Expenditure_Velocity_early_6q_pp',
    'Expenditure_fixed_12m': 'Expenditure_Velocity_fixed_12m_pp',
    'Expenditure_fixed_18m': 'Expenditure_Velocity_fixed_18m_pp',
}

results_vel = []

for variant_name, vel_col in velocity_variants.items():
    if vel_col not in panel_surv.columns:
        continue

    # Scale by 100 for this variant
    panel_surv[f'{vel_col}_temp_scaled'] = panel_surv[vel_col] * 100

    subset = panel_surv[['Duration_Surv', 'Event', f'{vel_col}_temp_scaled', 'Government_Type_State']].dropna()

    if len(subset) < 20:
        continue

    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(subset, duration_col='Duration_Surv', event_col='Event')

        vel_col_scaled = f'{vel_col}_temp_scaled'
        hr = np.exp(cph.params_[vel_col_scaled])
        log_hr = cph.params_[vel_col_scaled]
        se = cph.standard_errors_[vel_col_scaled]
        p = cph.summary.loc[vel_col_scaled, 'p']
        ci_lower = np.exp(cph.confidence_intervals_[vel_col_scaled][0])
        ci_upper = np.exp(cph.confidence_intervals_[vel_col_scaled][1])

        results_vel.append({
            'Operationalization': variant_name,
            'Column': vel_col,
            'N': len(subset),
            'Events': int(subset['Event'].sum()),
            'Velocity_HR': hr,
            'Velocity_log_HR': log_hr,
            'Velocity_SE': se,
            'Velocity_CI_lower': ci_lower,
            'Velocity_CI_upper': ci_upper,
            'Velocity_p': p,
        })
        print(f"{variant_name:30s}: HR={hr:.3f} ({ci_lower:.3f}-{ci_upper:.3f}), p={p:.4f}, N={len(subset)}")
    except Exception as e:
        print(f"{variant_name:30s}: FAILED - {e}")

# Meta-analysis
results_vel_df = pd.DataFrame(results_vel)
if len(results_vel_df) > 0:
    results_vel_df['Weight'] = 1 / (results_vel_df['Velocity_SE'] ** 2)
    meta_log_hr = (results_vel_df['Velocity_log_HR'] * results_vel_df['Weight']).sum() / results_vel_df['Weight'].sum()
    meta_se = np.sqrt(1 / results_vel_df['Weight'].sum())
    meta_hr = np.exp(meta_log_hr)
    meta_ci_lower = np.exp(meta_log_hr - 1.96 * meta_se)
    meta_ci_upper = np.exp(meta_log_hr + 1.96 * meta_se)

    print("\n" + "-"*80)
    print("META-ANALYSIS (Inverse-variance weighted):")
    print(f"  Average HR = {meta_hr:.3f} (95% CI: {meta_ci_lower:.3f}-{meta_ci_upper:.3f})")
    print(f"  Range: {results_vel_df['Velocity_HR'].min():.3f} - {results_vel_df['Velocity_HR'].max():.3f}")
    print(f"  Std dev: {results_vel_df['Velocity_HR'].std():.3f}")
    print("-"*80)

# Save results
output_vel = DIAG_DIR / "measurement_validation_velocity_variants.csv"
results_vel_df.to_csv(output_vel, index=False)
print(f"\n✓ Saved velocity operationalization results: {output_vel}")

print("\n" + "="*80)
print("MEASUREMENT VALIDATION COMPLETE")
print("="*80)
print(f"\nOutputs:")
print(f"  - {output_qa}")
print(f"  - {output_vel}")
