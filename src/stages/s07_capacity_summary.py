"""
Stage 07: Capacity Summary Reporting

Combines static survival capacity-set results with time-varying Cox outputs,
applies multiple-testing corrections, and produces a compact table + figure.

Outputs:
    data_work/diagnostics/multiple_testing_capacity_sets_time_varying.csv
    data_work/diagnostics/capacity_corrected_table.csv
    figures/fig_capacity_corrected.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_WORK_DIR, FIGURES_DIR


def load_capacity_sets() -> pd.DataFrame:
    path = DATA_WORK_DIR / "diagnostics" / "alternatives_survival_capacity_sets.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_alternatives first.")

    df = pd.read_csv(path)
    df = df[df['p_value'].notna()].copy()
    df['source'] = 'capacity_sets'
    df['model'] = df['Model']
    df['capacity_set'] = df['Capacity_Set']
    df['variable'] = df['Variable']
    df['effect_ratio'] = df['Effect_Ratio']
    df['effect_lower'] = df['Effect_Lower']
    df['effect_upper'] = df['Effect_Upper']
    df['n'] = df.get('N')
    df['events'] = df.get('Events')
    df['interpretation'] = df.get('Interpretation')
    return df[[
        'source', 'model', 'capacity_set', 'variable', 'p_value',
        'effect_ratio', 'effect_lower', 'effect_upper', 'n', 'events', 'interpretation'
    ]]


def load_time_varying() -> pd.DataFrame:
    path = DATA_WORK_DIR / "diagnostics" / "survival_hazard_ratios.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_survival first.")

    df = pd.read_csv(path)
    df = df[df['p_value'].notna()].copy()
    df['source'] = 'time_varying'
    df['model'] = df['model']
    df['capacity_set'] = None
    df['variable'] = df['Variable']
    df['effect_ratio'] = df['HR']
    df['effect_lower'] = df['HR_Lower']
    df['effect_upper'] = df['HR_Upper']
    df['n'] = None
    df['events'] = None
    df['interpretation'] = 'Hazard Ratio (>1 = faster completion)'
    return df[[
        'source', 'model', 'capacity_set', 'variable', 'p_value',
        'effect_ratio', 'effect_lower', 'effect_upper', 'n', 'events', 'interpretation'
    ]]


def apply_multiple_testing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('p_value').reset_index(drop=True)
    m = len(df)
    df['bonferroni'] = (df['p_value'] * m).clip(upper=1.0)
    df['rank'] = np.arange(1, m + 1)
    df['bh_raw'] = df['p_value'] * m / df['rank']
    df['bh_fdr'] = df['bh_raw'][::-1].cummin()[::-1].clip(upper=1.0)
    return df


def build_corrected_table(df: pd.DataFrame) -> pd.DataFrame:
    # Focus on Cox PH results and time-varying capacity terms
    cox = df[(df['model'] == 'Cox_PH') | (df['source'] == 'time_varying')].copy()
    cox = cox[cox['variable'].str.contains('Ratio_|Velocity|Capacity_', na=False)].copy()
    cox['sig_bonferroni'] = cox['bonferroni'] < 0.05
    cox['sig_bh_fdr'] = cox['bh_fdr'] < 0.05
    cox = cox.sort_values('p_value').reset_index(drop=True)
    return cox


def plot_capacity_effects(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    baseline_mask = df['variable'].str.contains('Ratio_disbursed_to_obligated|Ratio_expended_to_disbursed', na=False)
    plot_df = df[(df['bh_fdr'] < 0.05) | baseline_mask].copy()
    plot_df = plot_df.sort_values('p_value').reset_index(drop=True)

    def label_row(row):
        if row['source'] == 'time_varying':
            return f"tv:{row['model']}:{row['variable']}"
        return f"{row['capacity_set']}:{row['variable']}"

    plot_df['label'] = plot_df.apply(label_row, axis=1)

    fig_height = max(2.5, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    y = np.arange(len(plot_df))
    x = plot_df['effect_ratio'].values
    lower = plot_df['effect_lower'].values
    upper = plot_df['effect_upper'].values
    xerr = [x - lower, upper - x]

    colors = ['#d95f02' if sig else '#666666' for sig in plot_df['sig_bh_fdr']]
    for idx, (xi, err, color) in enumerate(zip(x, xerr, colors)):
        ax.errorbar(xi, idx, xerr=[[err[0]], [err[1]]], fmt='o', color=color, ecolor=color, capsize=3)

    ax.axvline(1.0, color='#333333', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['label'])
    ax.set_xscale('log')
    ax.set_xlabel('Hazard Ratio (log scale)')
    ax.set_title('Capacity Effects (Corrected p-values; BH-FDR < 0.05 in orange)')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main() -> None:
    diag_dir = DATA_WORK_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    capacity_sets = load_capacity_sets()
    time_varying = load_time_varying()
    combined = pd.concat([capacity_sets, time_varying], ignore_index=True)
    combined = apply_multiple_testing(combined)

    combined_path = diag_dir / "multiple_testing_capacity_sets_time_varying.csv"
    combined.to_csv(combined_path, index=False)

    corrected_table = build_corrected_table(combined)
    table_path = diag_dir / "capacity_corrected_table.csv"
    corrected_table.to_csv(table_path, index=False)

    fig_path = FIGURES_DIR / "fig_capacity_corrected.png"
    plot_capacity_effects(corrected_table, fig_path)

    print(f"Saved combined results → {combined_path}")
    print(f"Saved corrected table → {table_path}")
    print(f"Saved figure → {fig_path}")


if __name__ == "__main__":
    main()
