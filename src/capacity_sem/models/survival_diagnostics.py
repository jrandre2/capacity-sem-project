"""
Diagnostic tools for survival analysis models.

This module provides functions to assess model fit, check assumptions,
identify influential observations, and visualize predicted survival curves.

Key functions:
- compute_martingale_residuals(): Check linearity of covariates
- compute_cox_snell_residuals(): Assess overall model fit
- compute_influence_diagnostics(): Identify influential observations
- plot_predicted_survival_curves(): Visualize survival by covariate levels
- test_proportional_hazards(): Test PH assumption using Schoenfeld residuals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import chi2
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn(f"Missing dependencies: {e}. Install with: pip install lifelines matplotlib seaborn scipy")


def compute_martingale_residuals(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame,
    predictor_col: str,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, 'plt.Figure']:
    """
    Compute martingale residuals to check linearity assumption.

    Martingale residuals = observed events - expected events under the model.
    Plotting residuals vs a continuous predictor helps assess whether the
    functional form is correct (should show no pattern if linear).

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data used to fit the model
    predictor_col : str
        Column name of predictor to check (e.g., 'Ratio_disbursed_to_obligated_lag1')
    save_path : str, optional
        Path to save the diagnostic plot

    Returns
    -------
    residuals : np.ndarray
        Martingale residuals
    fig : matplotlib.Figure
        Diagnostic plot

    Notes
    -----
    - Residuals should scatter around zero with no systematic pattern
    - Systematic curvature suggests nonlinearity (try transformations or splines)
    - Outliers may indicate influential observations
    """

    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Required dependencies not available")

    # Compute martingale residuals
    # Martingale = E - cumulative hazard (predicted events)
    residuals = model.compute_residuals(tv_data, kind='martingale')

    # Get predictor values (aggregate to grantee-disaster level for plotting)
    # Use mean predictor value across intervals for each grantee-disaster
    if 'Grantee' in tv_data.columns and 'Disaster Type' in tv_data.columns:
        plot_data = tv_data.copy()
        plot_data['martingale'] = residuals.values if hasattr(residuals, 'values') else residuals

        # Aggregate to grantee-disaster level
        agg_data = plot_data.groupby(['Grantee', 'Disaster Type']).agg({
            predictor_col: 'mean',
            'martingale': 'sum'  # Sum residuals across intervals
        }).reset_index()

        x = agg_data[predictor_col]
        y = agg_data['martingale']
    else:
        x = tv_data[predictor_col]
        y = residuals

    # Create diagnostic plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=50)

    # Add lowess smooth
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='LOWESS smooth')
    except ImportError:
        warnings.warn("statsmodels not available for LOWESS smoothing")

    # Add reference line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel(predictor_col, fontsize=12)
    ax.set_ylabel('Martingale Residuals', fontsize=12)
    ax.set_title(f'Martingale Residuals vs {predictor_col}\n(Check for linearity)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved martingale residual plot → {save_path}")

    return residuals, fig


def compute_cox_snell_residuals(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, 'plt.Figure']:
    """
    Compute Cox-Snell residuals for overall model fit assessment.

    If the model fits well, Cox-Snell residuals should follow a unit
    exponential distribution (equivalently, the KM curve of residuals
    should follow the theoretical survival curve S(t) = exp(-t)).

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data
    save_path : str, optional
        Path to save diagnostic plot

    Returns
    -------
    residuals : np.ndarray
        Cox-Snell residuals
    fig : matplotlib.Figure
        Diagnostic plot comparing KM curve to exponential

    Notes
    -----
    - Cox-Snell residuals = cumulative hazard at observed time
    - If model fits well, these should follow unit exponential
    - Plot KM curve of residuals vs exp(-t) diagonal
    - Deviations indicate model misspecification
    """

    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Required dependencies not available")

    # Compute Cox-Snell residuals
    residuals = model.compute_residuals(tv_data, kind='cox-snell')

    # Fit Kaplan-Meier to residuals (treating them as "survival times")
    # For Cox-Snell, event indicator is always 1 (all "fail" at their residual value)
    kmf = KaplanMeierFitter()

    # Aggregate residuals to grantee-disaster level
    if 'Grantee' in tv_data.columns and 'Disaster Type' in tv_data.columns:
        plot_data = tv_data.copy()
        plot_data['cox_snell'] = residuals.values if hasattr(residuals, 'values') else residuals

        # Sum residuals across intervals for each grantee-disaster
        cs_by_grantee = plot_data.groupby(['Grantee', 'Disaster Type']).agg({
            'cox_snell': 'sum',
            'E': 'max'  # Event indicator (1 if completed)
        }).reset_index()

        cs_residuals = cs_by_grantee['cox_snell']
        events = cs_by_grantee['E']
    else:
        cs_residuals = residuals
        events = tv_data['E']

    # Fit KM to Cox-Snell residuals
    kmf.fit(cs_residuals, event_observed=events)

    # Create diagnostic plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot KM curve of residuals
    kmf.plot_survival_function(ax=ax, label='KM curve of Cox-Snell residuals')

    # Plot theoretical exponential survival curve S(t) = exp(-t)
    t_range = np.linspace(0, cs_residuals.max(), 100)
    theoretical = np.exp(-t_range)
    ax.plot(t_range, theoretical, 'r--', linewidth=2, label='Exponential S(t) = exp(-t)')

    ax.set_xlabel('Cox-Snell Residuals', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Cox-Snell Residuals: Model Fit Check\n(Should follow exponential)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved Cox-Snell residual plot → {save_path}")

    return residuals, fig


def compute_influence_diagnostics(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Identify influential observations using score residuals.

    Score residuals measure the influence of each observation on the
    regression coefficients. Large score residuals indicate observations
    that have disproportionate influence on the model fit.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data
    threshold : float, default=3.0
        Standard deviations from mean to flag as influential

    Returns
    -------
    pd.DataFrame
        Influence statistics by grantee-disaster with columns:
        - Grantee, Disaster Type
        - score_residual (max absolute score across intervals)
        - influential (flagged if > threshold SD)

    Notes
    -----
    - Score residuals are also called dfbeta residuals
    - They show how coefficients would change if observation removed
    - Flagged observations should be investigated for data quality
    """

    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Required dependencies not available")

    # Compute score residuals (dfbeta)
    # These show the influence on each coefficient
    try:
        score_residuals = model.compute_residuals(tv_data, kind='score')

        # Aggregate to grantee-disaster level
        if 'Grantee' in tv_data.columns and 'Disaster Type' in tv_data.columns:
            plot_data = tv_data.copy()

            # Add score residuals to data
            if isinstance(score_residuals, pd.DataFrame):
                for col in score_residuals.columns:
                    plot_data[f'score_{col}'] = score_residuals[col].values
            else:
                plot_data['score'] = score_residuals

            # Aggregate: sum of absolute scores across intervals
            score_cols = [col for col in plot_data.columns if col.startswith('score_')]
            if not score_cols:
                score_cols = ['score']

            agg_dict = {col: lambda x: np.abs(x).sum() for col in score_cols}
            influence_data = plot_data.groupby(['Grantee', 'Disaster Type']).agg(agg_dict).reset_index()

            # Compute overall influence score (max across coefficients)
            influence_data['score_residual'] = influence_data[score_cols].max(axis=1)

        else:
            # No grantee-disaster structure
            influence_data = pd.DataFrame({
                'score_residual': np.abs(score_residuals).max(axis=1) if hasattr(score_residuals, 'max') else np.abs(score_residuals)
            })

        # Flag influential observations
        mean_score = influence_data['score_residual'].mean()
        std_score = influence_data['score_residual'].std()
        influence_data['influential'] = influence_data['score_residual'] > (mean_score + threshold * std_score)

        n_influential = influence_data['influential'].sum()
        print(f"  Found {n_influential} influential observations (>{threshold} SD from mean)")

        return influence_data

    except Exception as e:
        warnings.warn(f"Could not compute influence diagnostics: {e}")
        return pd.DataFrame()


def plot_predicted_survival_curves(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame,
    predictor_col: str,
    stratify_by: str = 'quartile',
    percentiles: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> 'plt.Figure':
    """
    Plot predicted survival curves by covariate level.

    Shows how survival curves differ across levels of a key predictor
    (e.g., capacity ratios). Useful for visualizing effect sizes and
    interpreting hazard ratios in terms of survival probabilities.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data
    predictor_col : str
        Predictor to stratify by (e.g., 'Ratio_disbursed_to_obligated_lag1')
    stratify_by : str, default='quartile'
        How to stratify: 'quartile', 'tertile', 'median', or 'percentile'
    percentiles : list of float, optional
        Custom percentiles if stratify_by='percentile' (e.g., [10, 50, 90])
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.Figure
        Survival curve plot

    Notes
    -----
    - Uses mean predictor value across intervals for each grantee-disaster
    - Creates representative covariate profiles for each stratum
    - Shows 95% confidence intervals if available
    """

    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Required dependencies not available")

    # Aggregate data to grantee-disaster level
    if 'Grantee' in tv_data.columns and 'Disaster Type' in tv_data.columns:
        # Get mean predictor values
        agg_data = tv_data.groupby(['Grantee', 'Disaster Type']).agg({
            predictor_col: 'mean',
            'E': 'max'
        }).reset_index()

        predictor_values = agg_data[predictor_col].dropna()
    else:
        predictor_values = tv_data[predictor_col].dropna()

    # Determine stratification cutpoints
    if stratify_by == 'quartile':
        cutpoints = np.percentile(predictor_values, [25, 50, 75])
        labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    elif stratify_by == 'tertile':
        cutpoints = np.percentile(predictor_values, [33.3, 66.7])
        labels = ['T1 (Low)', 'T2', 'T3 (High)']
    elif stratify_by == 'median':
        cutpoints = [np.median(predictor_values)]
        labels = ['Below Median', 'Above Median']
    elif stratify_by == 'percentile' and percentiles:
        cutpoints = np.percentile(predictor_values, percentiles)
        labels = [f'P{int(p)}' for p in percentiles] + [f'P{int(percentiles[-1])}+']
    else:
        cutpoints = np.percentile(predictor_values, [25, 75])
        labels = ['Low', 'Medium', 'High']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define representative covariate values for each stratum
    # Use median within each stratum
    stratum_values = []
    stratum_values.append(predictor_values[predictor_values <= cutpoints[0]].median())
    for i in range(len(cutpoints) - 1):
        mask = (predictor_values > cutpoints[i]) & (predictor_values <= cutpoints[i+1])
        stratum_values.append(predictor_values[mask].median())
    stratum_values.append(predictor_values[predictor_values > cutpoints[-1]].median())

    # Get other covariates at their means
    covariate_cols = [col for col in model.params_.index if col != predictor_col]
    baseline_covariates = {}
    for col in covariate_cols:
        if col in tv_data.columns:
            baseline_covariates[col] = tv_data[col].mean()

    # Plot survival curve for each stratum
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))

    for i, (value, label) in enumerate(zip(stratum_values, labels)):
        # Create covariate profile
        profile = baseline_covariates.copy()
        profile[predictor_col] = value

        # Convert to DataFrame
        profile_df = pd.DataFrame([profile])

        # Predict survival curve
        try:
            survival_fn = model.predict_survival_function(profile_df)

            # Plot
            survival_fn.plot(ax=ax, color=colors[i], linewidth=2.5, label=f'{label} ({predictor_col}={value:.2f})')

        except Exception as e:
            warnings.warn(f"Could not predict survival for stratum {label}: {e}")

    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability (Not Yet Completed)', fontsize=12)
    ax.set_title(f'Predicted Survival Curves by {predictor_col}\n({stratify_by.title()} Stratification)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved predicted survival curves → {save_path}")

    return fig


def test_proportional_hazards(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Test proportional hazards assumption using Schoenfeld residuals.

    The proportional hazards assumption states that hazard ratios are
    constant over time. Schoenfeld residuals test whether the effect
    of each covariate changes over time.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data

    Returns
    -------
    pd.DataFrame
        Test statistics and p-values for each covariate with columns:
        - Variable
        - test_statistic (chi-square)
        - p_value
        - significant (True if p < 0.05)

    Notes
    -----
    - Uses weighted Schoenfeld residuals
    - p < 0.05 suggests PH assumption may be violated
    - If violated, consider:
      - Time interaction terms
      - Stratification
      - AFT models (don't assume PH)
    """

    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Required dependencies not available")

    try:
        # lifelines provides built-in PH test
        ph_test_results = model.check_assumptions(tv_data, show_plots=False)

        # Extract results
        if hasattr(ph_test_results, 'summary'):
            test_df = ph_test_results.summary
        elif isinstance(ph_test_results, pd.DataFrame):
            test_df = ph_test_results
        else:
            # Try to construct from model attributes
            test_df = pd.DataFrame({
                'Variable': model.params_.index,
                'test_statistic': np.nan,
                'p_value': np.nan
            })

        # Add significance flag
        if 'p' in test_df.columns:
            test_df['significant'] = test_df['p'] < 0.05
        elif 'p_value' in test_df.columns:
            test_df['significant'] = test_df['p_value'] < 0.05

        print("Proportional Hazards Test Results:")
        for _, row in test_df.iterrows():
            var = row.get('Variable', row.name)
            p = row.get('p', row.get('p_value', np.nan))
            sig = "VIOLATED" if p < 0.05 else "OK"
            print(f"  {var}: p={p:.3f} [{sig}]")

        return test_df

    except Exception as e:
        warnings.warn(f"Could not perform PH test: {e}")
        return pd.DataFrame()


def plot_residual_diagnostics(
    model: 'CoxPHFitter',
    tv_data: pd.DataFrame,
    capacity_cols: List[str],
    output_dir: str = 'figures'
) -> Dict[str, 'plt.Figure']:
    """
    Generate comprehensive residual diagnostic plots.

    Creates martingale residual plots for all capacity predictors
    and Cox-Snell residuals for overall fit assessment.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    tv_data : pd.DataFrame
        Time-varying survival data
    capacity_cols : list of str
        Capacity predictor columns to check
    output_dir : str, default='figures'
        Directory to save plots

    Returns
    -------
    dict
        Dictionary of {plot_name: Figure} for all generated plots
    """

    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Martingale residuals for each capacity predictor
    for col in capacity_cols:
        if col in tv_data.columns:
            try:
                _, fig = compute_martingale_residuals(
                    model, tv_data, col,
                    save_path=str(output_path / f'martingale_{col}.png')
                )
                figures[f'martingale_{col}'] = fig
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Could not create martingale plot for {col}: {e}")

    # Cox-Snell residuals
    try:
        _, fig = compute_cox_snell_residuals(
            model, tv_data,
            save_path=str(output_path / 'cox_snell_residuals.png')
        )
        figures['cox_snell'] = fig
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Could not create Cox-Snell plot: {e}")

    print(f"Generated {len(figures)} diagnostic plots in {output_dir}/")

    return figures
