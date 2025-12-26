"""
Stage 05: Figure Generation

Generate publication-ready figures for the manuscript.

Commands:
    python src/pipeline.py make_figures [--style STYLE]

Outputs:
    figures/*.png  - Publication figures
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from config import DATA_WORK_DIR, FIGURES_DIR

# Configure matplotlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Figure style settings
FIGURE_STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6.5, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def set_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(FIGURE_STYLE)


def load_panel_features() -> pd.DataFrame:
    """Load panel with features."""
    path = DATA_WORK_DIR / "panel_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_estimation_results() -> Dict[str, pd.DataFrame]:
    """Load estimation results from diagnostics directory."""
    diag_dir = DATA_WORK_DIR / "diagnostics"
    results = {}

    if not diag_dir.exists():
        return results

    for f in diag_dir.glob("*.csv"):
        results[f.stem] = pd.read_csv(f)

    return results


def figure_descriptive_stats(data: pd.DataFrame, output_path: Path) -> None:
    """
    Create descriptive statistics figure.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    output_path : Path
        Output file path.
    """
    set_style()

    # Key variables to plot
    vars_to_plot = [
        ('Duration_of_completion', 'Duration (months)'),
        ('Ratio_expended_to_obligated', 'Expenditure Ratio'),
        ('Experience_Index', 'Experience Index'),
        ('Progress_Rate', 'Progress Rate'),
    ]

    available_vars = [(v, l) for v, l in vars_to_plot if v in data.columns]

    if not available_vars:
        print("  No variables available for descriptive figure")
        return

    n_vars = len(available_vars)
    fig, axes = plt.subplots(1, n_vars, figsize=(3 * n_vars, 3))

    if n_vars == 1:
        axes = [axes]

    for ax, (var, label) in zip(axes, available_vars):
        values = data[var].dropna()
        ax.hist(values, bins=30, edgecolor='white', alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.axvline(values.mean(), color='red', linestyle='--', alpha=0.7,
                   label=f'Mean: {values.mean():.2f}')
        ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Created: {output_path.name}")


def figure_model_comparison(results: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """
    Create model comparison figure.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Estimation results.
    output_path : Path
        Output file path.
    """
    set_style()

    # Look for robustness specifications results
    if 'robustness_specifications' not in results:
        print("  No model comparison data available")
        return

    comparison = results['robustness_specifications']

    if 'Model' not in comparison.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # CFI comparison
    if 'CFI' in comparison.columns:
        ax = axes[0]
        models = comparison['Model'].values
        cfi = comparison['CFI'].values
        x = np.arange(len(models))
        bars = ax.barh(x, cfi, color='steelblue', alpha=0.8)
        ax.axvline(0.95, color='green', linestyle='--', alpha=0.7, label='Good (0.95)')
        ax.axvline(0.90, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.90)')
        ax.set_yticks(x)
        ax.set_yticklabels(models)
        ax.set_xlabel('CFI')
        ax.set_title('Comparative Fit Index')
        ax.legend(loc='lower right', frameon=False)
        ax.set_xlim(0.5, 1.0)

    # RMSEA comparison
    if 'RMSEA' in comparison.columns:
        ax = axes[1]
        models = comparison['Model'].values
        rmsea = comparison['RMSEA'].values
        x = np.arange(len(models))
        bars = ax.barh(x, rmsea, color='coral', alpha=0.8)
        ax.axvline(0.05, color='green', linestyle='--', alpha=0.7, label='Good (0.05)')
        ax.axvline(0.08, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.08)')
        ax.set_yticks(x)
        ax.set_yticklabels(models)
        ax.set_xlabel('RMSEA')
        ax.set_title('Root Mean Square Error of Approximation')
        ax.legend(loc='upper right', frameon=False)
        ax.set_xlim(0, 0.15)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Created: {output_path.name}")


def figure_subset_comparison(results: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """
    Create government subset comparison figure.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Estimation results.
    output_path : Path
        Output file path.
    """
    set_style()

    if 'robustness_subsets' not in results:
        print("  No subset comparison data available")
        return

    comparison = results['robustness_subsets']

    if 'Model' not in comparison.columns or 'N' not in comparison.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Sample sizes by subset
    models = comparison['Model'].values
    n_values = comparison['N'].values
    x = np.arange(len(models))

    bars = ax.bar(x, n_values, color=['#2c7bb6', '#abd9e9', '#fdae61'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Sample Size (N)')
    ax.set_title('Sample Sizes by Government Type')

    # Add value labels
    for bar, n in zip(bars, n_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(n):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Created: {output_path.name}")


def figure_sensitivity(results: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """
    Create sample sensitivity figure.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Estimation results.
    output_path : Path
        Output file path.
    """
    set_style()

    if 'robustness_sample_sensitivity' not in results:
        print("  No sensitivity data available")
        return

    sensitivity = results['robustness_sample_sensitivity']

    if 'Min_Quarters' not in sensitivity.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Sample size by min quarters
    ax = axes[0]
    ax.plot(sensitivity['Min_Quarters'], sensitivity['N'], 'o-', color='steelblue')
    ax.set_xlabel('Minimum Quarters')
    ax.set_ylabel('Sample Size (N)')
    ax.set_title('Sample Size Sensitivity')

    # Capacity effect by min quarters
    ax = axes[1]
    if 'Capacity_Effect' in sensitivity.columns:
        ax.errorbar(
            sensitivity['Min_Quarters'],
            sensitivity['Capacity_Effect'],
            yerr=sensitivity.get('Capacity_SE', 0),
            fmt='o-',
            color='coral',
            capsize=3
        )
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Minimum Quarters')
        ax.set_ylabel('Capacity Effect (β)')
        ax.set_title('Effect Stability')

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Created: {output_path.name}")


def figure_path_diagram(output_path: Path) -> None:
    """
    Create simplified path diagram.

    Parameters
    ----------
    output_path : Path
        Output file path.
    """
    set_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw simplified SEM path diagram
    # Latent variables as ellipses
    from matplotlib.patches import Ellipse, FancyArrowPatch

    # Government Capacity
    cap_ellipse = Ellipse((0.2, 0.5), 0.25, 0.15, fill=False, edgecolor='steelblue', linewidth=2)
    ax.add_patch(cap_ellipse)
    ax.text(0.2, 0.5, 'Government\nCapacity', ha='center', va='center', fontsize=10)

    # Recovery Outcome
    out_ellipse = Ellipse((0.8, 0.5), 0.25, 0.15, fill=False, edgecolor='coral', linewidth=2)
    ax.add_patch(out_ellipse)
    ax.text(0.8, 0.5, 'Recovery\nOutcome', ha='center', va='center', fontsize=10)

    # Structural path
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.35, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 0.55, 'β', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Capacity indicators
    cap_indicators = ['Disbursement\nRatio', 'Expenditure\nRatio']
    for i, ind in enumerate(cap_indicators):
        y = 0.8 - i * 0.3
        ax.add_patch(plt.Rectangle((0.02, y - 0.05), 0.12, 0.1, fill=False, edgecolor='gray'))
        ax.text(0.08, y, ind, ha='center', va='center', fontsize=8)
        ax.annotate('', xy=(0.14, y), xytext=(0.2 - 0.12, 0.5 + (0.1 if i == 0 else -0.1)),
                    arrowprops=dict(arrowstyle='<-', color='steelblue', lw=1))

    # Outcome indicators
    out_indicators = ['Duration\n(log)', 'Spending\nCV']
    for i, ind in enumerate(out_indicators):
        y = 0.8 - i * 0.3
        ax.add_patch(plt.Rectangle((0.86, y - 0.05), 0.12, 0.1, fill=False, edgecolor='gray'))
        ax.text(0.92, y, ind, ha='center', va='center', fontsize=8)
        ax.annotate('', xy=(0.86, y), xytext=(0.8 + 0.12, 0.5 + (0.1 if i == 0 else -0.1)),
                    arrowprops=dict(arrowstyle='<-', color='coral', lw=1))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.2, 0.95)
    ax.axis('off')
    ax.set_title('Structural Equation Model Path Diagram', fontsize=12, pad=20)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Created: {output_path.name}")


def main(style: str = 'publication'):
    """
    Main entry point for figure generation.

    Parameters
    ----------
    style : str
        Figure style: 'publication' or 'presentation'.
    """
    print("=" * 60)
    print("Stage 05: Figure Generation")
    print("=" * 60)

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = load_panel_features()
    results = load_estimation_results()
    print(f"  Panel: {len(data):,} observations")
    print(f"  Results files: {len(results)}")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Descriptive statistics
    figure_descriptive_stats(data, FIGURES_DIR / "fig_descriptive.png")

    # 2. Model comparison
    figure_model_comparison(results, FIGURES_DIR / "fig_model_comparison.png")

    # 3. Subset comparison
    figure_subset_comparison(results, FIGURES_DIR / "fig_subset_comparison.png")

    # 4. Sensitivity analysis
    figure_sensitivity(results, FIGURES_DIR / "fig_sensitivity.png")

    # 5. Path diagram
    figure_path_diagram(FIGURES_DIR / "fig_path_diagram.png")

    print(f"\n✓ Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--style", "-s", default="publication",
                        choices=["publication", "presentation"],
                        help="Figure style")

    args = parser.parse_args()
    main(style=args.style)
