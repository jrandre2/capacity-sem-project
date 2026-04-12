"""
Forest plot visualization for multi-stage efficiency analysis.

Creates publication-quality forest plots showing hazard ratios with confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def create_forest_plot(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Velocity Effects by Event Type: Competing Risks Analysis",
    figsize: tuple = (10, 6),
    dpi: int = 300
):
    """
    Create forest plot showing hazard ratios with 95% confidence intervals.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: Event_Type, Velocity_HR, Velocity_CI_lower, Velocity_CI_upper, Velocity_p, N, N_Events
    output_path : Path
        Where to save the figure
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    """
    # Filter to rows with valid results
    plot_df = results_df.dropna(subset=['Velocity_HR', 'Velocity_CI_lower', 'Velocity_CI_upper']).copy()

    if len(plot_df) == 0:
        print("No valid results to plot")
        return

    # Sort by HR for better visualization
    plot_df = plot_df.sort_values('Velocity_HR', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Y positions for each event type
    y_positions = np.arange(len(plot_df))

    # Color palette
    colors = {
        'Completed': '#2ecc71',  # Green
        'Stalled_Stage1': '#e74c3c',  # Red
        'Stalled_Stage2': '#f39c12',  # Orange
    }

    # Plot each hazard ratio with confidence interval
    for idx, (i, row) in enumerate(plot_df.iterrows()):
        event_type = row['Event_Type']
        hr = row['Velocity_HR']
        ci_lower = row['Velocity_CI_lower']
        ci_upper = row['Velocity_CI_upper']
        p_value = row['Velocity_p']
        n = row['N']
        n_events = row['N_Events']

        color = colors.get(event_type, '#95a5a6')

        # Plot point estimate
        ax.plot(hr, y_positions[idx], 'o', markersize=10, color=color, zorder=3)

        # Plot confidence interval
        ax.plot([ci_lower, ci_upper], [y_positions[idx], y_positions[idx]],
                '-', linewidth=2, color=color, zorder=2)

        # Add significance marker
        if p_value < 0.001:
            sig_marker = '***'
        elif p_value < 0.01:
            sig_marker = '**'
        elif p_value < 0.05:
            sig_marker = '*'
        else:
            sig_marker = ''

        # Label with HR and sample size
        label_text = f"HR={hr:.2f} {sig_marker}\n(N={n}, Events={n_events})"
        ax.text(ci_upper * 1.1, y_positions[idx], label_text,
                va='center', ha='left', fontsize=9)

    # Add null effect line (HR=1)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    # Format x-axis (log scale)
    ax.set_xscale('log')
    ax.set_xlabel('Hazard Ratio (95% CI)\nper 1 pp/quarter increase in expenditure velocity', fontsize=11)

    # Format y-axis
    ax.set_yticks(y_positions)
    event_labels = [
        row['Event_Type'].replace('_', ' ').title()
        for _, row in plot_df.iterrows()
    ]
    ax.set_yticklabels(event_labels, fontsize=10)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)

    # Title
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['Completed'], label='Completed (reached 95%)'),
        mpatches.Patch(color=colors['Stalled_Stage1'], label='Stalled Stage 1 (obligate→disburse bottleneck)'),
    ]
    if 'Stalled_Stage2' in plot_df['Event_Type'].values:
        legend_elements.append(
            mpatches.Patch(color=colors['Stalled_Stage2'], label='Stalled Stage 2 (disburse→expend bottleneck)')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # Add notes
    note_text = "*** p<0.001, ** p<0.01, * p<0.05\nHR>1: Higher velocity → higher hazard (faster outcome)\nVertical dashed line: Null effect (HR=1)"
    ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ Saved forest plot: {output_path}")

    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from config import DATA_WORK_DIR

    # Load results
    results_path = DATA_WORK_DIR / "diagnostics" / "multistage_efficiency.csv"
    results_df = pd.read_csv(results_path)

    # Create forest plot
    output_path = Path("figures") / "multistage_bottleneck_hazards.png"
    create_forest_plot(results_df, output_path)

    print("\nForest plot created successfully!")
