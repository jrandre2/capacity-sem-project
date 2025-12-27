#!/usr/bin/env python3
"""
Phase 2 Week 4: Stratified Analysis by Stage1_Efficiency

Tests if velocity effects differ for programs with high vs low disbursement capacity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from stages._io_utils import safe_read_parquet

# Load panel
panel_path = Path("../data_work/panel_features_std.parquet")
panel = safe_read_parquet(panel_path)

print("=" * 80)
print("Phase 2 Week 4: Stratified Analysis by Stage1_Efficiency")
print("=" * 80)
print()

# Prepare survival data
panel['Duration_Surv'] = panel['Duration'].fillna(panel['N_Quarters'])
panel['Velocity_scaled'] = panel['Expenditure_Velocity_pp'] * 100
panel['Event'] = (panel['Duration'].notna() & (panel['Duration'] > 0)).astype(int)

# Create Stage1_Efficiency quartiles
panel['Stage1_Quartile'] = pd.qcut(
    panel['Stage1_Efficiency'],
    q=4,
    labels=['Q1_Low', 'Q2_Mid-Low', 'Q3_Mid-High', 'Q4_High'],
    duplicates='drop'
)

print("Stage1_Efficiency quartile distribution:")
print(panel['Stage1_Quartile'].value_counts().sort_index())
print()

print("Stage1_Efficiency summary by quartile:")
print(panel.groupby('Stage1_Quartile')['Stage1_Efficiency'].describe())
print()

# Run stratified Cox PH by quartile
results = []

for quartile in panel['Stage1_Quartile'].dropna().unique():
    print(f"\nFitting Cox PH for {quartile}:")

    # Subset to quartile
    subset_panel = panel[panel['Stage1_Quartile'] == quartile].copy()

    # Drop missing
    subset = subset_panel[
        ['Duration_Surv', 'Event', 'Velocity_scaled', 'Government_Type_State']
    ].dropna()

    n_events = subset['Event'].sum()
    print(f"  Sample: N={len(subset)}, Events={n_events}")

    if n_events < 5:
        print(f"  ⚠ Too few events ({n_events}), skipping")
        continue

    # Fit Cox PH
    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(
            subset,
            duration_col='Duration_Surv',
            event_col='Event'
        )

        hr = np.exp(cph.params_['Velocity_scaled'])
        p = cph.summary.loc['Velocity_scaled', 'p']
        ci_lower = np.exp(cph.confidence_intervals_.loc['Velocity_scaled', '95% lower-bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc['Velocity_scaled', '95% upper-bound'])

        # Get quartile bounds
        quartile_min = subset_panel['Stage1_Efficiency'].min()
        quartile_max = subset_panel['Stage1_Efficiency'].max()
        quartile_median = subset_panel['Stage1_Efficiency'].median()

        results.append({
            'Stage1_Quartile': quartile,
            'Stage1_Min': quartile_min,
            'Stage1_Median': quartile_median,
            'Stage1_Max': quartile_max,
            'N': len(subset),
            'N_Events': int(n_events),
            'Velocity_HR': hr,
            'Velocity_CI_lower': ci_lower,
            'Velocity_CI_upper': ci_upper,
            'Velocity_p': p,
        })

        print(f"  Velocity HR: {hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p:.4f}")

    except Exception as e:
        print(f"  ✗ Model failed: {e}")
        import traceback
        traceback.print_exc()

# Save results
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/stage1_stratified_analysis.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved results: {output_path}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

print("\n" + "=" * 80)
print("Stratified Analysis Complete")
print("=" * 80)
