"""
Common utilities for manuscript table and figure rendering.
Loads diagnostic CSVs and provides helper functions for Capacity-SEM project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import Markdown, display


def show_table(df: pd.DataFrame) -> None:
    """Display a DataFrame as a properly rendered markdown table.

    This function works correctly in both HTML and PDF output formats.
    """
    display(Markdown(df.to_markdown(index=False)))


# Paths relative to manuscript_quarto/
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"


def load_diagnostic(name: str) -> pd.DataFrame:
    """Load a diagnostic CSV file by name (without .csv extension)."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic file not found: {path}")
    return pd.read_csv(path)


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if pd.isna(p):
        return "—"
    if p < threshold:
        return f"<{threshold}"
    return f"{p:.3f}"


def format_ci(lo: float, hi: float, decimals: int = 3) -> str:
    """Format confidence interval as [lo, hi]."""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def add_significance_stars(p: float) -> str:
    """Add significance stars based on p-value."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def format_coefficient(est: float, se: float, p: float, decimals: int = 3) -> str:
    """Format coefficient with standard error and significance stars."""
    stars = add_significance_stars(p)
    return f"{est:.{decimals}f}{stars}\n({se:.{decimals}f})"


def format_fit_index(value: float, threshold_good: float, threshold_acceptable: float,
                     lower_is_better: bool = False) -> str:
    """Format fit index with quality indicator."""
    if pd.isna(value):
        return "—"

    if lower_is_better:
        if value <= threshold_good:
            quality = "✓"
        elif value <= threshold_acceptable:
            quality = "~"
        else:
            quality = "✗"
    else:
        if value >= threshold_good:
            quality = "✓"
        elif value >= threshold_acceptable:
            quality = "~"
        else:
            quality = "✗"

    return f"{value:.3f} {quality}"


# Capacity-SEM specific loaders
def load_estimation_results() -> pd.DataFrame:
    """Load main SEM estimation results."""
    return load_diagnostic("estimates_exp_optimal_v1_all")


def load_fit_statistics() -> pd.DataFrame:
    """Load model fit statistics."""
    return load_diagnostic("fit_stats_exp_optimal_v1_all")


def load_robustness_specs() -> pd.DataFrame:
    """Load robustness specification comparison."""
    return load_diagnostic("robustness_specifications")


def load_robustness_subsets() -> pd.DataFrame:
    """Load robustness subset comparison."""
    return load_diagnostic("robustness_subsets")


def load_sample_sensitivity() -> pd.DataFrame:
    """Load sample sensitivity analysis."""
    return load_diagnostic("robustness_sample_sensitivity")


def create_fit_summary_table() -> pd.DataFrame:
    """Create formatted fit summary table."""
    try:
        fit = load_fit_statistics()

        # Format for display
        summary = pd.DataFrame({
            'Index': ['CFI', 'TLI', 'RMSEA', 'SRMR', 'Chi-Square', 'df'],
            'Value': [
                format_fit_index(fit.get('CFI', np.nan), 0.95, 0.90),
                format_fit_index(fit.get('TLI', np.nan), 0.95, 0.90),
                format_fit_index(fit.get('RMSEA', np.nan), 0.05, 0.08, lower_is_better=True),
                format_fit_index(fit.get('SRMR', np.nan), 0.05, 0.08, lower_is_better=True),
                f"{fit.get('chi2', np.nan):.2f}",
                f"{fit.get('dof', np.nan):.0f}"
            ],
            'Threshold': ['≥0.95', '≥0.95', '≤0.05', '≤0.05', '—', '—']
        })
        return summary
    except FileNotFoundError:
        return pd.DataFrame({'Note': ['Results not yet available']})
