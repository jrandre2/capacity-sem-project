#!/usr/bin/env python3
"""
Phase 2 Week 5: Velocity Trajectory Clustering

Uses K-means clustering to identify distinct velocity patterns across program timelines.
Tests if different trajectory types (fast-start, slow-ramp, stalled, late-surge) have
different completion outcomes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from stages._io_utils import safe_read_parquet

print("=" * 80)
print("Phase 2 Week 5: Velocity Trajectory Clustering")
print("=" * 80)
print()

# Load data
qpr_path = Path("../data_work/qpr_standardized.parquet")
panel_path = Path("../data_work/panel_features_std.parquet")

print("Loading data...")
qpr_std = safe_read_parquet(qpr_path)
panel = safe_read_parquet(panel_path)
print(f"  Loaded {len(qpr_std):,} quarterly observations")
print(f"  Loaded {len(panel):,} grantee-disaster pairs")
print()

# Select velocity column
velocity_col = 'Velocity_Exp_Std_pp_winsor'
if velocity_col not in qpr_std.columns:
    # Try alternatives
    for alt_col in ['Expenditure_Velocity_pp_Std', 'Velocity_Exp_Std_pp']:
        if alt_col in qpr_std.columns:
            velocity_col = alt_col
            print(f"  Using velocity column: {velocity_col}")
            break

# Create Quarter_Index for each grantee-disaster (sequential integer starting from 0)
print("Creating quarter index for each grantee-disaster...")
qpr_std = qpr_std.sort_values(['Grantee', 'Disaster Type', 'QPR_Date'])
qpr_std['Quarter_Index'] = qpr_std.groupby(['Grantee', 'Disaster Type']).cumcount()
print(f"  Quarter index range: {qpr_std['Quarter_Index'].min()} to {qpr_std['Quarter_Index'].max()}")

# Limit to first 80 quarters (20 years) to focus on meaningful recovery period
max_quarters = 80
print(f"  Limiting to first {max_quarters} quarters per program...")
qpr_std_limited = qpr_std[qpr_std['Quarter_Index'] < max_quarters].copy()
print(f"  Retained {len(qpr_std_limited):,} / {len(qpr_std):,} observations ({len(qpr_std_limited)/len(qpr_std):.1%})")
print()

# Pivot to wide format: rows = grantee-disaster, columns = quarter
print("Pivoting velocity data to wide format...")
velocity_matrix_raw = qpr_std_limited.pivot_table(
    index=['Grantee', 'Disaster Type'],
    columns='Quarter_Index',
    values=velocity_col,
    aggfunc='mean'  # Use mean if duplicate quarters
)
print(f"  Raw velocity matrix shape: {velocity_matrix_raw.shape[0]} programs × {velocity_matrix_raw.shape[1]} quarters")

# Filter to programs with at least 12 quarters of data
min_quarters = 12
obs_per_program = velocity_matrix_raw.notna().sum(axis=1)
valid_programs = obs_per_program >= min_quarters
print(f"  Filtering to programs with ≥{min_quarters} quarters: {valid_programs.sum()} / {len(valid_programs)} retained")
velocity_matrix_filtered = velocity_matrix_raw[valid_programs].copy()

# Forward-fill missing values (carry last observation forward)
velocity_matrix = velocity_matrix_filtered.ffill(axis=1)
# Back-fill remaining NaNs at the start
velocity_matrix = velocity_matrix.bfill(axis=1)
# Fill any remaining NaNs with 0
velocity_matrix = velocity_matrix.fillna(0)
print(f"  Final velocity matrix shape: {velocity_matrix.shape[0]} programs × {velocity_matrix.shape[1]} quarters")
print()

# Standardize for clustering
print("Standardizing velocity time series...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(velocity_matrix)
print(f"  Scaled matrix shape: {X_scaled.shape}")
print()

# K-means clustering (try k=3 for more balanced clusters)
n_clusters = 3
print(f"Running K-means clustering (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
clusters = kmeans.fit_predict(X_scaled)
print(f"  Cluster sizes: {np.bincount(clusters)}")
print()

# Create trajectory labels dataframe
trajectory_labels = pd.DataFrame({
    'Grantee': velocity_matrix.index.get_level_values(0),
    'Disaster Type': velocity_matrix.index.get_level_values(1),
    'Velocity_Trajectory_Cluster': clusters,
})

# Label clusters by interpretation
print("Labeling clusters based on velocity profiles...")
cluster_means = pd.DataFrame(X_scaled, columns=velocity_matrix.columns).groupby(clusters).mean()

# Heuristic: early velocity (first 4 quarters) vs late velocity (last 4 quarters)
n_quarters = velocity_matrix.shape[1]
early_cols = velocity_matrix.columns[:min(4, n_quarters // 3)]
late_cols = velocity_matrix.columns[max(-4, -(n_quarters // 3)):]

cluster_profiles = pd.DataFrame({
    'Cluster': range(n_clusters),
    'Early_Mean': cluster_means[early_cols].mean(axis=1).values,
    'Late_Mean': cluster_means[late_cols].mean(axis=1).values,
    'Overall_Mean': cluster_means.mean(axis=1).values,
})

# Label based on early/late velocity patterns
cluster_profiles['Label'] = 'Unknown'

for idx, row in cluster_profiles.iterrows():
    early = row['Early_Mean']
    late = row['Late_Mean']
    overall = row['Overall_Mean']

    # Determine trajectory pattern
    if early > 0.2 and late > 0.2:
        cluster_profiles.loc[idx, 'Label'] = 'Fast-Consistent'
    elif early < -0.1 and late > 0.2:
        cluster_profiles.loc[idx, 'Label'] = 'Slow-Ramp'
    elif early > 0.2 and late < -0.1:
        cluster_profiles.loc[idx, 'Label'] = 'Early-Surge'
    elif overall < -0.3:
        cluster_profiles.loc[idx, 'Label'] = 'Stalled'
    elif late - early > 0.3:
        cluster_profiles.loc[idx, 'Label'] = 'Accelerating'
    elif early - late > 0.3:
        cluster_profiles.loc[idx, 'Label'] = 'Decelerating'
    else:
        cluster_profiles.loc[idx, 'Label'] = 'Moderate'

print(cluster_profiles.to_string(index=False))
print()

# Map labels to trajectory_labels
trajectory_labels = trajectory_labels.merge(
    cluster_profiles[['Cluster', 'Label']],
    left_on='Velocity_Trajectory_Cluster',
    right_on='Cluster',
    how='left'
).drop(columns=['Cluster'])
trajectory_labels.rename(columns={'Label': 'Trajectory_Label'}, inplace=True)

print("Cluster distribution:")
print(trajectory_labels['Trajectory_Label'].value_counts())
print()

# Merge with panel for survival analysis
panel_with_clusters = panel.merge(
    trajectory_labels,
    on=['Grantee', 'Disaster Type'],
    how='left'
)

# Prepare survival data
panel_with_clusters['Duration_Surv'] = panel_with_clusters['Duration'].fillna(
    panel_with_clusters['N_Quarters']
)
panel_with_clusters['Event'] = (
    panel_with_clusters['Duration'].notna() & (panel_with_clusters['Duration'] > 0)
).astype(int)

# Cox PH analysis stratified by cluster
print("=" * 80)
print("Cox PH Analysis by Trajectory Cluster")
print("=" * 80)
print()

results = []

for label in trajectory_labels['Trajectory_Label'].unique():
    if pd.isna(label):
        continue

    print(f"\nCluster: {label}")
    print("-" * 80)

    subset = panel_with_clusters[panel_with_clusters['Trajectory_Label'] == label].copy()

    # Drop missing
    subset_clean = subset[['Duration_Surv', 'Event', 'Government_Type_State']].dropna()

    n_events = subset_clean['Event'].sum()
    print(f"  Sample: N={len(subset_clean)}, Events={n_events}")

    if n_events < 5:
        print(f"  ⚠ Too few events ({n_events}), skipping Cox PH")
        results.append({
            'Trajectory_Label': label,
            'N': len(subset_clean),
            'N_Events': int(n_events),
            'Median_Duration': subset_clean['Duration_Surv'].median(),
            'Completion_Rate': subset_clean['Event'].mean(),
        })
        continue

    # Kaplan-Meier estimate
    kmf = KaplanMeierFitter()
    kmf.fit(subset_clean['Duration_Surv'], subset_clean['Event'], label=label)
    median_survival = kmf.median_survival_time_

    results.append({
        'Trajectory_Label': label,
        'N': len(subset_clean),
        'N_Events': int(n_events),
        'Median_Duration': subset_clean['Duration_Surv'].median(),
        'Median_Survival_Time': median_survival,
        'Completion_Rate': subset_clean['Event'].mean(),
    })

    print(f"  Median duration: {subset_clean['Duration_Surv'].median():.1f} quarters")
    print(f"  Median survival time: {median_survival:.1f} quarters")
    print(f"  Completion rate: {subset_clean['Event'].mean():.1%}")

# Save cluster results
results_df = pd.DataFrame(results)
output_path = Path("../data_work/diagnostics/temporal_dynamics_trajectory_clusters.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved cluster results: {output_path}")

# Save trajectory assignments
trajectory_output_path = Path("../data_work/diagnostics/trajectory_cluster_assignments.csv")
trajectory_labels.to_csv(trajectory_output_path, index=False)
print(f"✓ Saved trajectory assignments: {trajectory_output_path}")

print("\n" + "=" * 80)
print("CLUSTER SUMMARY")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

# Visualization 1: Cluster velocity profiles
print("Creating cluster velocity profile visualization...")
n_cols = min(3, n_clusters)
n_rows = (n_clusters + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows), dpi=300)
if n_clusters == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_clusters > 3 else axes

for cluster_id in range(n_clusters):
    ax = axes[cluster_id]

    # Get cluster members
    cluster_mask = clusters == cluster_id
    cluster_velocities = X_scaled[cluster_mask]

    # Plot all trajectories in cluster (thin lines)
    for trajectory in cluster_velocities[:20]:  # Limit to 20 for visibility
        ax.plot(velocity_matrix.columns, trajectory, alpha=0.2, color='gray', linewidth=0.5)

    # Plot cluster mean (thick line)
    cluster_mean = cluster_velocities.mean(axis=0)
    ax.plot(velocity_matrix.columns, cluster_mean, color='red', linewidth=2.5, label='Cluster Mean')

    # Label
    label = cluster_profiles.loc[cluster_profiles['Cluster'] == cluster_id, 'Label'].values[0]
    count = cluster_mask.sum()
    ax.set_title(f"Cluster {cluster_id}: {label} (N={count})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Quarter Index', fontsize=9)
    ax.set_ylabel('Standardized Velocity', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(fontsize=8)

plt.tight_layout()
cluster_viz_path = Path("../figures/velocity_trajectories_kmeans.png")
cluster_viz_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(cluster_viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved cluster visualization: {cluster_viz_path}")
plt.close()

# Visualization 2: Kaplan-Meier survival curves by cluster
print("Creating Kaplan-Meier survival curves...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

colors = {
    'Fast-Consistent': '#2ecc71',
    'Slow-Ramp': '#3498db',
    'Early-Surge': '#9b59b6',
    'Stalled': '#e74c3c',
    'Accelerating': '#1abc9c',
    'Decelerating': '#e67e22',
    'Moderate': '#95a5a6',
    'Unknown': '#34495e'
}

for label in trajectory_labels['Trajectory_Label'].unique():
    if pd.isna(label):
        continue

    subset = panel_with_clusters[panel_with_clusters['Trajectory_Label'] == label].copy()
    subset_clean = subset[['Duration_Surv', 'Event']].dropna()

    if len(subset_clean) < 5 or subset_clean['Event'].sum() < 3:
        continue

    kmf = KaplanMeierFitter()
    kmf.fit(subset_clean['Duration_Surv'], subset_clean['Event'], label=label)

    n = len(subset_clean)
    events = subset_clean['Event'].sum()
    kmf.plot_survival_function(
        ax=ax,
        color=colors.get(label, '#95a5a6'),
        linewidth=2.5,
        label=f"{label} (N={n}, Events={events})"
    )

ax.set_xlabel('Time to Completion (Quarters)', fontsize=11)
ax.set_ylabel('Survival Probability (Not Yet Completed)', fontsize=11)
ax.set_title('Kaplan-Meier Survival Curves by Velocity Trajectory Cluster',
             fontsize=13, fontweight='bold', pad=20)
ax.grid(alpha=0.3, linestyle=':')
ax.legend(loc='best', fontsize=9, framealpha=0.9)

km_path = Path("../figures/kaplan_meier_by_trajectory.png")
plt.savefig(km_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Kaplan-Meier curves: {km_path}")
plt.close()

print("\n" + "=" * 80)
print("Trajectory Clustering Complete")
print("=" * 80)
