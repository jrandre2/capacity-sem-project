#!/usr/bin/env python3
"""Generate forest plot for Phase 2 Week 4."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Load results
results_path = Path("../data_work/diagnostics/multistage_efficiency.csv")
results_df = pd.read_csv(results_path)

print("Results loaded:")
print(results_df)
print()

# Filter to rows with valid results
plot_df = results_df.dropna(subset=['Velocity_HR', 'Velocity_CI_lower', 'Velocity_CI_upper']).copy()

if len(plot_df) == 0:
    print("No valid results to plot")
    exit(1)

# Sort by HR
plot_df = plot_df.sort_values('Velocity_HR', ascending=False)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Y positions
y_positions = np.arange(len(plot_df))

# Colors
colors = {
    'Completed': '#2ecc71',
    'Stalled_Stage1': '#e74c3c',
    'Stalled_Stage2': '#f39c12',
}

# Plot each HR
for idx, (i, row) in enumerate(plot_df.iterrows()):
    event_type = row['Event_Type']
    hr = row['Velocity_HR']
    ci_lower = row['Velocity_CI_lower']
    ci_upper = row['Velocity_CI_upper']
    p_value = row['Velocity_p']
    n = int(row['N'])
    n_events = int(row['N_Events'])

    color = colors.get(event_type, '#95a5a6')

    # Point estimate
    ax.plot(hr, y_positions[idx], 'o', markersize=10, color=color, zorder=3)

    # CI
    ax.plot([ci_lower, ci_upper], [y_positions[idx], y_positions[idx]],
            '-', linewidth=2, color=color, zorder=2)

    # Significance
    if p_value < 0.001:
        sig_marker = '***'
    elif p_value < 0.01:
        sig_marker = '**'
    elif p_value < 0.05:
        sig_marker = '*'
    else:
        sig_marker = ''

    # Label
    label_text = f"HR={hr:.2f} {sig_marker}\n(N={n}, Events={n_events})"
    ax.text(ci_upper * 1.15, y_positions[idx], label_text,
            va='center', ha='left', fontsize=9)

# Null line
ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

# X-axis (log scale)
ax.set_xscale('log')
ax.set_xlabel('Hazard Ratio (95% CI)\nper 1 pp/quarter increase in expenditure velocity', fontsize=11)

# Set x-axis limits to handle extreme HRs
ax.set_xlim(0.5, max(plot_df['Velocity_CI_upper']) * 2)

# Y-axis
ax.set_yticks(y_positions)
event_labels = [row['Event_Type'].replace('_', ' ').title() for _, row in plot_df.iterrows()]
ax.set_yticklabels(event_labels, fontsize=10)
ax.set_ylim(-0.5, len(plot_df) - 0.5)

# Title
ax.set_title('Velocity Effects by Event Type: Competing Risks Analysis',
             fontsize=13, fontweight='bold', pad=20)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)

# Legend
legend_elements = [
    mpatches.Patch(color=colors['Completed'], label='Completed (reached 95%)'),
    mpatches.Patch(color=colors['Stalled_Stage1'], label='Stalled Stage 1 (obligate→disburse bottleneck)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

# Notes
note_text = "*** p<0.001, ** p<0.01, * p<0.05\nHR>1: Higher velocity → higher hazard (faster outcome)\nVertical dashed line: Null effect (HR=1)"
ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
        fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save
output_path = Path("../figures/multistage_bottleneck_hazards.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved forest plot: {output_path}")

plt.close()
print("\nForest plot created successfully!")
