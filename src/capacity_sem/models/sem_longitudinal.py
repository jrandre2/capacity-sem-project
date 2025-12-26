"""
Longitudinal SEM and cross-lagged panel models.

This module provides infrastructure for:
1. Restructuring quarterly data into wide panel format
2. Building cross-lagged panel model specifications
3. Testing temporal precedence of capacity vs. outcomes
4. Latent growth curve model preparation

Cross-Lagged Panel Models (CLPM) test temporal precedence:
- Does capacity at time t predict outcomes at time t+1?
- Or do outcomes at time t predict capacity at time t+1?

This helps address causal inference by examining temporal ordering.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

try:
    from semopy import Model
    from semopy import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

from .sem_fitting import fit_sem_model
from .sem_diagnostics import extract_fit_stat


def prepare_panel_data(
    qpr_quarterly: pd.DataFrame,
    n_waves: int = 4,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    quarter_col: str = 'QPR Actual Quarter',
    capacity_vars: Optional[List[str]] = None,
    outcome_vars: Optional[List[str]] = None,
    min_quarters: int = 4,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Restructure quarterly data into wide panel format for longitudinal SEM.

    Converts from long format (grantee × disaster × quarter) to wide format
    with T1, T2, T3, T4 measurements per grantee-disaster unit.

    Parameters
    ----------
    qpr_quarterly : pd.DataFrame
        Quarterly QPR data in long format.
    n_waves : int, default 4
        Number of time waves to extract.
    grantee_col : str, default 'Grantee'
        Column name for grantee identifier.
    disaster_col : str, default 'Disaster Type'
        Column name for disaster identifier.
    quarter_col : str, default 'QPR Actual Quarter'
        Column name for quarter identifier.
    capacity_vars : List[str], optional
        Variables to use as capacity indicators. If None, uses defaults.
    outcome_vars : List[str], optional
        Variables to use as outcome indicators. If None, uses defaults.
    min_quarters : int, default 4
        Minimum quarters required per grantee-disaster.
    verbose : bool, default True
        Whether to print summary statistics.

    Returns
    -------
    pd.DataFrame
        Wide-format data with variables suffixed by wave number.
        Example columns: Ratio_disbursed_T1, Ratio_disbursed_T2, etc.
    """
    if capacity_vars is None:
        capacity_vars = [
            'Ratio_disbursed_to_obligated',
            'Ratio_expended_to_disbursed'
        ]

    if outcome_vars is None:
        outcome_vars = [
            'Duration_log',
            'Spending_CV'
        ]

    all_vars = capacity_vars + outcome_vars

    # Filter to grantee-disasters with enough quarters
    counts = qpr_quarterly.groupby([grantee_col, disaster_col]).size()
    valid_pairs = counts[counts >= min_quarters].index

    if verbose:
        print(f"Total grantee-disaster pairs: {len(counts)}")
        print(f"Pairs with {min_quarters}+ quarters: {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        return pd.DataFrame()

    # Filter to valid pairs
    qpr = qpr_quarterly.set_index([grantee_col, disaster_col])
    qpr = qpr.loc[qpr.index.isin(valid_pairs)].reset_index()

    # Sort by grantee, disaster, quarter
    qpr = qpr.sort_values([grantee_col, disaster_col, quarter_col])

    # Create wave numbers within each grantee-disaster
    qpr['Wave'] = qpr.groupby([grantee_col, disaster_col]).cumcount() + 1

    # Filter to requested number of waves
    qpr = qpr[qpr['Wave'] <= n_waves]

    # Pivot to wide format
    wide_data = []

    for (grantee, disaster), group in qpr.groupby([grantee_col, disaster_col]):
        if len(group) < n_waves:
            continue  # Skip if missing waves

        row = {
            grantee_col: grantee,
            disaster_col: disaster
        }

        for wave in range(1, n_waves + 1):
            wave_data = group[group['Wave'] == wave]
            if wave_data.empty:
                continue

            for var in all_vars:
                if var in wave_data.columns:
                    row[f'{var}_T{wave}'] = wave_data[var].iloc[0]

        wide_data.append(row)

    wide_df = pd.DataFrame(wide_data)

    if verbose:
        print(f"Wide panel observations: {len(wide_df)}")
        print(f"Variables per wave: {len(all_vars)}")
        print(f"Total columns: {len(wide_df.columns)}")

    return wide_df


def compute_panel_variables(
    qpr_quarterly: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Compute capacity and outcome variables at each quarter.

    This function computes time-varying versions of capacity and outcome
    indicators for panel analysis.

    Parameters
    ----------
    qpr_quarterly : pd.DataFrame
        Quarterly QPR data.
    grantee_col : str, default 'Grantee'
        Column name for grantee identifier.
    disaster_col : str, default 'Disaster Type'
        Column name for disaster identifier.

    Returns
    -------
    pd.DataFrame
        Quarterly data with computed panel variables.
    """
    df = qpr_quarterly.copy()

    # Compute quarterly ratios
    if 'QPR Fund Obligated $' in df.columns and 'QPR Fund Disbursed $' in df.columns:
        df['Ratio_disbursed_to_obligated'] = np.where(
            df['QPR Fund Obligated $'] > 0,
            df['QPR Fund Disbursed $'] / df['QPR Fund Obligated $'],
            np.nan
        )

    if 'QPR Fund Disbursed $' in df.columns and 'QPR Fund Expended $' in df.columns:
        df['Ratio_expended_to_disbursed'] = np.where(
            df['QPR Fund Disbursed $'] > 0,
            df['QPR Fund Expended $'] / df['QPR Fund Disbursed $'],
            np.nan
        )

    # Compute cumulative completion percentage
    if 'QPR Fund Obligated $' in df.columns and 'QPR Fund Expended $' in df.columns:
        df['Completion_Pct'] = np.where(
            df['QPR Fund Obligated $'] > 0,
            df['QPR Fund Expended $'] / df['QPR Fund Obligated $'],
            np.nan
        )

    # Compute quarterly spending variability (rolling)
    if 'QPR Fund Expended Q $' in df.columns:
        df['Spending_CV'] = df.groupby([grantee_col, disaster_col])['QPR Fund Expended Q $'].transform(
            lambda x: x.expanding().std() / x.expanding().mean() if x.expanding().mean().iloc[-1] > 0 else np.nan
        )

    return df


def build_clpm_specification(
    n_waves: int = 4,
    capacity_var: str = 'Ratio_disbursed_to_obligated',
    outcome_var: str = 'Completion_Pct',
    include_cross_lagged: bool = True,
    include_autoregressive: bool = True,
    include_contemporaneous: bool = False
) -> str:
    """
    Generate cross-lagged panel model (CLPM) specification.

    The CLPM tests temporal precedence by examining:
    - Autoregressive paths (X_t → X_t+1): Stability of constructs
    - Cross-lagged paths (X_t → Y_t+1): Predictive relationships
    - Contemporaneous correlations (X_t ~~ Y_t): Same-time associations

    Parameters
    ----------
    n_waves : int, default 4
        Number of time waves.
    capacity_var : str, default 'Ratio_disbursed_to_obligated'
        Capacity indicator variable name (without time suffix).
    outcome_var : str, default 'Completion_Pct'
        Outcome indicator variable name (without time suffix).
    include_cross_lagged : bool, default True
        Whether to include cross-lagged paths.
    include_autoregressive : bool, default True
        Whether to include autoregressive paths.
    include_contemporaneous : bool, default False
        Whether to include contemporaneous correlations.

    Returns
    -------
    str
        Model specification in lavaan/semopy syntax.

    Examples
    --------
    >>> spec = build_clpm_specification(n_waves=3)
    >>> print(spec)
    # Autoregressive paths
    Cap_T2 ~ Cap_T1
    Cap_T3 ~ Cap_T2
    Out_T2 ~ Out_T1
    Out_T3 ~ Out_T2
    # Cross-lagged paths
    Out_T2 ~ Cap_T1
    Out_T3 ~ Cap_T2
    Cap_T2 ~ Out_T1
    Cap_T3 ~ Out_T2
    """
    lines = [
        "# Cross-Lagged Panel Model",
        f"# Waves: {n_waves}",
        f"# Capacity: {capacity_var}",
        f"# Outcome: {outcome_var}",
        ""
    ]

    cap_short = 'Cap'
    out_short = 'Out'

    # Autoregressive paths
    if include_autoregressive:
        lines.append("# Autoregressive paths (stability)")
        for t in range(1, n_waves):
            lines.append(f"{capacity_var}_T{t+1} ~ {capacity_var}_T{t}")
        for t in range(1, n_waves):
            lines.append(f"{outcome_var}_T{t+1} ~ {outcome_var}_T{t}")
        lines.append("")

    # Cross-lagged paths
    if include_cross_lagged:
        lines.append("# Cross-lagged paths (predictive)")
        lines.append("# Capacity → Outcome (hypothesis: capacity predicts outcomes)")
        for t in range(1, n_waves):
            lines.append(f"{outcome_var}_T{t+1} ~ {capacity_var}_T{t}")
        lines.append("# Outcome → Capacity (reverse causation)")
        for t in range(1, n_waves):
            lines.append(f"{capacity_var}_T{t+1} ~ {outcome_var}_T{t}")
        lines.append("")

    # Contemporaneous correlations
    if include_contemporaneous:
        lines.append("# Contemporaneous correlations")
        for t in range(1, n_waves + 1):
            lines.append(f"{capacity_var}_T{t} ~~ {outcome_var}_T{t}")
        lines.append("")

    return "\n".join(lines)


def build_riclpm_specification(
    n_waves: int = 4,
    capacity_var: str = 'Ratio_disbursed_to_obligated',
    outcome_var: str = 'Completion_Pct'
) -> str:
    """
    Generate Random Intercept Cross-Lagged Panel Model (RI-CLPM) specification.

    The RI-CLPM separates between-person and within-person variance by
    including random intercepts. This addresses the critique that standard
    CLPM conflates trait-like and state-like variation.

    Note: This model requires many observations and may not converge with
    small samples.

    Parameters
    ----------
    n_waves : int, default 4
        Number of time waves.
    capacity_var : str, default 'Ratio_disbursed_to_obligated'
        Capacity indicator variable name.
    outcome_var : str, default 'Completion_Pct'
        Outcome indicator variable name.

    Returns
    -------
    str
        RI-CLPM specification in lavaan/semopy syntax.
    """
    lines = [
        "# Random Intercept Cross-Lagged Panel Model (RI-CLPM)",
        f"# Waves: {n_waves}",
        ""
    ]

    # Random intercepts (between-person stable traits)
    lines.append("# Random intercepts (stable between-person differences)")
    cap_indicators = " + ".join([f"1*{capacity_var}_T{t}" for t in range(1, n_waves + 1)])
    out_indicators = " + ".join([f"1*{outcome_var}_T{t}" for t in range(1, n_waves + 1)])
    lines.append(f"RI_cap =~ {cap_indicators}")
    lines.append(f"RI_out =~ {out_indicators}")
    lines.append("")

    # Within-person centered variables
    lines.append("# Within-person latent variables")
    for t in range(1, n_waves + 1):
        lines.append(f"cap_w{t} =~ 1*{capacity_var}_T{t}")
        lines.append(f"out_w{t} =~ 1*{outcome_var}_T{t}")
    lines.append("")

    # Cross-lagged paths at within-person level
    lines.append("# Within-person cross-lagged paths")
    for t in range(1, n_waves):
        lines.append(f"cap_w{t+1} ~ cap_w{t} + out_w{t}")
        lines.append(f"out_w{t+1} ~ out_w{t} + cap_w{t}")
    lines.append("")

    # Correlation between random intercepts
    lines.append("# Between-person correlation")
    lines.append("RI_cap ~~ RI_out")

    return "\n".join(lines)


def build_latent_growth_specification(
    n_waves: int = 4,
    outcome_var: str = 'Completion_Pct',
    predictors: Optional[List[str]] = None
) -> str:
    """
    Generate Latent Growth Curve Model (LGCM) specification.

    LGCM models trajectories of change over time by estimating:
    - Intercept: Initial level (or average level)
    - Slope: Rate of change

    Predictors can explain individual differences in intercept and slope.

    Parameters
    ----------
    n_waves : int, default 4
        Number of time waves.
    outcome_var : str, default 'Completion_Pct'
        Outcome variable to model growth.
    predictors : List[str], optional
        Variables to predict intercept and slope (e.g., capacity indicators).

    Returns
    -------
    str
        LGCM specification in lavaan/semopy syntax.
    """
    lines = [
        "# Latent Growth Curve Model",
        f"# Waves: {n_waves}",
        f"# Outcome: {outcome_var}",
        ""
    ]

    # Intercept factor (all loadings = 1)
    intercept_indicators = " + ".join([f"1*{outcome_var}_T{t}" for t in range(1, n_waves + 1)])
    lines.append(f"Intercept =~ {intercept_indicators}")

    # Slope factor (loadings = 0, 1, 2, 3, ...)
    slope_indicators = " + ".join([f"{t-1}*{outcome_var}_T{t}" for t in range(1, n_waves + 1)])
    lines.append(f"Slope =~ {slope_indicators}")
    lines.append("")

    # Intercept and slope correlation
    lines.append("# Intercept-slope covariance")
    lines.append("Intercept ~~ Slope")
    lines.append("")

    # Predictors of growth
    if predictors:
        lines.append("# Predictors of growth parameters")
        for pred in predictors:
            lines.append(f"Intercept ~ {pred}")
            lines.append(f"Slope ~ {pred}")
        lines.append("")

    return "\n".join(lines)


def fit_clpm(
    model_spec: str,
    panel_data: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fit a cross-lagged panel model.

    Parameters
    ----------
    model_spec : str
        CLPM specification from build_clpm_specification().
    panel_data : pd.DataFrame
        Wide-format panel data from prepare_panel_data().
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    Dict[str, Any]
        Fitted model results including:
        - 'model': Fitted semopy Model
        - 'estimates': Parameter estimates
        - 'fit_stats': Fit statistics
        - 'cross_lagged': Extracted cross-lagged path coefficients
    """
    results = {
        'model': None,
        'estimates': pd.DataFrame(),
        'fit_stats': pd.DataFrame(),
        'cross_lagged': {}
    }

    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required for CLPM")

    try:
        model, fit_result = fit_sem_model(model_spec, panel_data)
        results['model'] = model

        if model is not None:
            results['estimates'] = model.inspect()
            results['fit_stats'] = calc_stats(model)

            # Extract cross-lagged paths
            estimates = results['estimates']
            if not estimates.empty:
                struct = estimates[estimates['op'] == '~']
                for _, row in struct.iterrows():
                    path = f"{row['rval']} → {row['lval']}"
                    results['cross_lagged'][path] = {
                        'estimate': row['Estimate'],
                        'se': row.get('Std. Err', np.nan),
                        'p': row.get('p-value', np.nan)
                    }

    except Exception as e:
        if verbose:
            print(f"Error fitting CLPM: {e}")

    if verbose and results['model'] is not None:
        print("\nCross-Lagged Panel Model Results")
        print("=" * 50)

        # Print fit statistics
        fit = results['fit_stats']
        if not fit.empty:
            cfi = extract_fit_stat(fit, 'CFI')
            rmsea = extract_fit_stat(fit, 'RMSEA')
            print(f"CFI: {cfi:.3f}" if pd.notna(cfi) else "CFI: NA")
            print(f"RMSEA: {rmsea:.3f}" if pd.notna(rmsea) else "RMSEA: NA")

        # Print cross-lagged paths
        print("\nCross-Lagged Paths:")
        for path, stats in results['cross_lagged'].items():
            sig = '***' if stats['p'] < 0.001 else '**' if stats['p'] < 0.01 else '*' if stats['p'] < 0.05 else ''
            print(f"  {path}: β = {stats['estimate']:.3f} (p = {stats['p']:.3f}) {sig}")

    return results


def interpret_clpm_results(
    clpm_results: Dict[str, Any],
    capacity_var: str,
    outcome_var: str
) -> str:
    """
    Interpret cross-lagged panel model results.

    Examines the pattern of cross-lagged paths to determine:
    1. Does capacity precede outcomes? (capacity → outcome paths significant)
    2. Do outcomes precede capacity? (outcome → capacity paths significant)
    3. Bidirectional causation? (both directions significant)

    Parameters
    ----------
    clpm_results : Dict[str, Any]
        Output from fit_clpm().
    capacity_var : str
        Name of capacity variable.
    outcome_var : str
        Name of outcome variable.

    Returns
    -------
    str
        Interpretation of causal precedence.
    """
    cross_lagged = clpm_results.get('cross_lagged', {})

    if not cross_lagged:
        return "Unable to interpret: No cross-lagged paths found."

    # Find paths in each direction
    cap_to_out_sig = []
    out_to_cap_sig = []

    for path, stats in cross_lagged.items():
        p = stats.get('p', 1.0)
        if pd.isna(p):
            continue

        # Check if capacity → outcome
        if capacity_var in path.split(' → ')[0] and outcome_var in path.split(' → ')[1]:
            if p < 0.05:
                cap_to_out_sig.append((path, stats['estimate'], p))

        # Check if outcome → capacity
        if outcome_var in path.split(' → ')[0] and capacity_var in path.split(' → ')[1]:
            if p < 0.05:
                out_to_cap_sig.append((path, stats['estimate'], p))

    # Interpret
    lines = ["CLPM Interpretation:", ""]

    if cap_to_out_sig and not out_to_cap_sig:
        lines.append("FINDING: Capacity temporally precedes outcomes")
        lines.append("- Capacity → Outcome paths are significant")
        lines.append("- Outcome → Capacity paths are not significant")
        lines.append("This supports the hypothesis that government capacity")
        lines.append("causally affects recovery outcomes.")

    elif out_to_cap_sig and not cap_to_out_sig:
        lines.append("FINDING: Outcomes temporally precede capacity")
        lines.append("- Outcome → Capacity paths are significant")
        lines.append("- Capacity → Outcome paths are not significant")
        lines.append("This suggests reverse causation: performance affects")
        lines.append("subsequent capacity ratings.")

    elif cap_to_out_sig and out_to_cap_sig:
        lines.append("FINDING: Bidirectional/reciprocal relationship")
        lines.append("- Both Capacity → Outcome and Outcome → Capacity are significant")
        lines.append("This suggests a feedback loop between capacity and outcomes.")

    else:
        lines.append("FINDING: No significant cross-lagged relationships")
        lines.append("- Neither direction shows temporal precedence")
        lines.append("Capacity and outcomes may be independent or")
        lines.append("related through unmeasured confounders.")

    # Add specific paths
    lines.append("")
    if cap_to_out_sig:
        lines.append("Significant Capacity → Outcome paths:")
        for path, est, p in cap_to_out_sig:
            lines.append(f"  {path}: β={est:.3f}, p={p:.3f}")

    if out_to_cap_sig:
        lines.append("Significant Outcome → Capacity paths:")
        for path, est, p in out_to_cap_sig:
            lines.append(f"  {path}: β={est:.3f}, p={p:.3f}")

    return "\n".join(lines)
