"""
Stage 06: Alternative Modeling Approaches

Runs alternative analyses to address right-censoring in duration data:
1. Survival analysis (Cox PH, AFT)
2. Lower threshold SEM (50%, 70%, 90%)
3. Duration-free SEM
4. Milestone-based SEM

Commands:
    python src/pipeline.py run_alternatives [--methods METHODS]
    python src/pipeline.py run_alternatives --survival-only
    python src/pipeline.py run_alternatives --sem-only

Outputs:
    data_work/diagnostics/alternatives_survival.csv
    data_work/diagnostics/alternatives_survival_capacity_sets.csv
    data_work/diagnostics/alternatives_threshold_sensitivity.csv
    data_work/diagnostics/alternatives_duration_free.csv
    data_work/diagnostics/alternatives_milestone.csv
    data_work/diagnostics/alternatives_comparison.csv
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import warnings

from config import (
    DATA_WORK_DIR,
    STATE_GOVERNMENTS,
    LOCAL_GOVERNMENTS,
    SURVIVAL_CAPACITY_COLS,
    CAPACITY_ALTERNATIVE_SETS,
    AFT_DISTRIBUTIONS,
    COX_PENALIZER,
    COX_STRATIFIED_LOW_EVENT_THRESHOLD,
    COX_STRATIFIED_LOW_EVENT_PENALIZER,
)
from stages._io_utils import safe_read_parquet

from capacity_sem.models.sem_specifications import MODEL_REGISTRY, get_model_spec
from capacity_sem.models.sem_fitting import fit_sem_model, SEMOPY_AVAILABLE
from capacity_sem.models.sem_diagnostics import evaluate_model_fit, extract_fit_stat
from capacity_sem.models.sem_alternatives import (
    LIFELINES_AVAILABLE,
    check_lifelines,
    prepare_survival_data,
    fit_cox_model,
    fit_aft_model,
    compare_survival_models,
    extract_survival_coefficients,
    run_threshold_sensitivity_sem,
    compare_methods,
    summarize_alternatives_findings,
    get_available_duration_thresholds,
)

SCIPY_AVAILABLE = False
try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except ImportError:
    chi2 = None


def load_panel_features() -> pd.DataFrame:
    """Load panel with features."""
    path = DATA_WORK_DIR / "panel_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Panel features not found at {path}. Run compute_features first."
        )
    return safe_read_parquet(path)


def ensure_diagnostics_dir() -> Path:
    """Ensure diagnostics directory exists."""
    diag_dir = DATA_WORK_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    return diag_dir


def resolve_capacity_sets(capacity_sets: Optional[List[str]]) -> Dict[str, List[str]]:
    """Resolve capacity set names into column lists."""
    if capacity_sets is None:
        return {'ratios': SURVIVAL_CAPACITY_COLS}

    if 'all' in capacity_sets:
        return CAPACITY_ALTERNATIVE_SETS

    resolved = {}
    for name in capacity_sets:
        if name in CAPACITY_ALTERNATIVE_SETS:
            resolved[name] = CAPACITY_ALTERNATIVE_SETS[name]
        else:
            warnings.warn(f"Unknown capacity set '{name}'. Available: {list(CAPACITY_ALTERNATIVE_SETS)}")

    return resolved or {'ratios': SURVIVAL_CAPACITY_COLS}


# =============================================================================
# SURVIVAL ANALYSIS
# =============================================================================

def run_survival_analysis(
    data: pd.DataFrame,
    capacity_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive survival analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    capacity_cols : list, optional
        Capacity indicator columns. Defaults to SURVIVAL_CAPACITY_COLS.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        Results including Cox and AFT models.
    """
    if not LIFELINES_AVAILABLE:
        if verbose:
            print("  Skipping survival analysis: lifelines not installed")
            print("  Install with: pip install lifelines>=0.27.0")
        return {'error': 'lifelines not available'}

    if capacity_cols is None:
        capacity_cols = SURVIVAL_CAPACITY_COLS

    if verbose:
        print("\n  --- Survival Analysis ---")
        print(f"  Capacity predictors: {capacity_cols}")

    # Prepare survival data
    surv_data = prepare_survival_data(data, capacity_cols=capacity_cols)

    if verbose:
        n_obs = len(surv_data)
        n_events = surv_data['E'].sum()
        n_censored = n_obs - n_events
        print(f"  Total observations: {n_obs}")
        print(f"  Completed (events): {n_events} ({100*n_events/n_obs:.1f}%)")
        print(f"  Censored: {n_censored} ({100*n_censored/n_obs:.1f}%)")

    results = {
        'n_obs': len(surv_data),
        'n_events': int(surv_data['E'].sum()),
        'n_censored': int(len(surv_data) - surv_data['E'].sum()),
    }

    # Fit Cox model
    if verbose:
        print("\n  Fitting Cox Proportional Hazards...")

    cox_result = fit_cox_model(surv_data, capacity_cols, penalizer=COX_PENALIZER)
    results['cox'] = cox_result

    if 'error' not in cox_result:
        if verbose:
            print(f"    Concordance: {cox_result['concordance']:.3f}")
            if 'summary' in cox_result:
                for var in capacity_cols:
                    if var in cox_result['summary'].index:
                        hr = cox_result['summary'].loc[var, 'exp(coef)']
                        p = cox_result['summary'].loc[var, 'p']
                        sig = '*' if p < 0.05 else ''
                        print(f"    {var}: HR={hr:.3f}, p={p:.3f}{sig}")

    # Fit AFT models
    results['aft'] = {}
    for dist in AFT_DISTRIBUTIONS:
        if verbose:
            print(f"\n  Fitting AFT ({dist})...")

        try:
            aft_result = fit_aft_model(surv_data, capacity_cols, distribution=dist)
            results['aft'][dist] = aft_result

            if 'error' not in aft_result:
                if verbose:
                    print(f"    AIC: {aft_result['aic']:.1f}")
                    print(f"    Concordance: {aft_result['concordance']:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            results['aft'][dist] = {'error': str(e)}

    # Model comparison
    if verbose:
        print("\n  --- Model Comparison ---")

    comparison = compare_survival_models(surv_data, capacity_cols, AFT_DISTRIBUTIONS)
    results['comparison'] = comparison

    if verbose and not comparison.empty:
        print(comparison.to_string(index=False))

    # Extract unified coefficients
    aft_results_list = [r for r in results['aft'].values() if 'error' not in r]
    results['coefficients'] = extract_survival_coefficients(
        cox_result if 'error' not in cox_result else None,
        aft_results_list,
        capacity_cols
    )

    return results


def run_survival_capacity_sets(
    data: pd.DataFrame,
    capacity_sets: Dict[str, List[str]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run survival analysis across multiple capacity indicator sets.

    Returns a stacked coefficient table with Capacity_Set labels.
    """
    results = []

    for set_name, cols in capacity_sets.items():
        available_cols = [col for col in cols if col in data.columns]
        if not available_cols:
            if verbose:
                print(f"\n  Skipping capacity set '{set_name}': no columns found")
            continue

        if verbose:
            print("\n" + "-" * 60)
            print(f"  Capacity set: {set_name}")
            print(f"  Columns: {available_cols}")

        survival_results = run_survival_analysis(
            data,
            capacity_cols=available_cols,
            verbose=verbose
        )

        coeffs = survival_results.get('coefficients', pd.DataFrame())
        if coeffs.empty:
            continue

        coeffs = coeffs.copy()
        coeffs['Capacity_Set'] = set_name
        coeffs['N'] = survival_results.get('n_obs')
        coeffs['Events'] = survival_results.get('n_events')
        results.append(coeffs)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame()


# =============================================================================
# STRATIFIED VELOCITY ANALYSIS
# =============================================================================

def select_stratum_penalizer(scheme: str, label: str, n_events: int) -> tuple[float, str]:
    """Select penalizer for low-stratum or low-event Cox models."""
    penalizer = COX_PENALIZER
    reasons = []
    low_labels = {
        'median': ['low'],
        'tercile': ['low'],
        'quartile': ['q1'],
    }

    if label in low_labels.get(scheme, []):
        penalizer = max(penalizer, COX_STRATIFIED_LOW_EVENT_PENALIZER)
        reasons.append('low_stratum')

    if n_events < COX_STRATIFIED_LOW_EVENT_THRESHOLD:
        penalizer = max(penalizer, COX_STRATIFIED_LOW_EVENT_PENALIZER)
        reasons.append('low_events')

    return penalizer, '+'.join(reasons) if reasons else 'base'


def run_velocity_stratified_by_ratio(
    data: pd.DataFrame,
    ratio_col: str = 'Ratio_disbursed_to_obligated',
    velocity_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Re-estimate velocity effects within strata of baseline ratios.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with capacity measures.
    ratio_col : str
        Baseline ratio column used to define strata.
    velocity_cols : list of str, optional
        Velocity measures to re-estimate within strata.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Stratified Cox results for velocity measures.
    """
    if not LIFELINES_AVAILABLE:
        if verbose:
            print("  Skipping stratified velocity analysis: lifelines not installed")
        return pd.DataFrame()

    if velocity_cols is None:
        velocity_cols = ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp']

    if ratio_col not in data.columns:
        warnings.warn(f"Ratio column '{ratio_col}' not found; skipping stratified velocity analysis")
        return pd.DataFrame()

    ratio_series = data[ratio_col].dropna()
    if ratio_series.empty:
        warnings.warn("No valid ratio values for stratification; skipping")
        return pd.DataFrame()

    q25 = ratio_series.quantile(0.25)
    q33 = ratio_series.quantile(0.33)
    q50 = ratio_series.quantile(0.50)
    q67 = ratio_series.quantile(0.67)
    q75 = ratio_series.quantile(0.75)

    strat_specs = [
        {
            'scheme': 'median',
            'bins': [(-np.inf, q50, 'low'), (q50, np.inf, 'high')]
        },
        {
            'scheme': 'tercile',
            'bins': [(-np.inf, q33, 'low'), (q33, q67, 'mid'), (q67, np.inf, 'high')]
        },
        {
            'scheme': 'quartile',
            'bins': [(-np.inf, q25, 'q1'), (q25, q50, 'q2'), (q50, q75, 'q3'), (q75, np.inf, 'q4')]
        },
    ]

    results = []
    for spec in strat_specs:
        scheme = spec['scheme']
        for low, high, label in spec['bins']:
            subset = data[(data[ratio_col] > low) & (data[ratio_col] <= high)].copy()
            if subset.empty:
                continue

            for velocity_col in velocity_cols:
                if velocity_col not in subset.columns:
                    continue

                surv_data = prepare_survival_data(subset, capacity_cols=[velocity_col])
                if surv_data.empty:
                    continue

                try:
                    n_events = int(surv_data['E'].sum())
                    if n_events < 2:
                        results.append({
                            'Scheme': scheme,
                            'Stratum': label,
                            'Ratio_Low': low if np.isfinite(low) else np.nan,
                            'Ratio_High': high if np.isfinite(high) else np.nan,
                            'Velocity_Var': velocity_col,
                            'HR': np.nan,
                            'HR_Lower': np.nan,
                            'HR_Upper': np.nan,
                            'p_value': np.nan,
                            'N': len(surv_data),
                            'Events': n_events,
                            'Penalizer': np.nan,
                            'Penalty_Reason': 'insufficient_events',
                            'Status': 'skipped',
                        })
                        continue

                    penalizer, penalty_reason = select_stratum_penalizer(scheme, label, n_events)
                    cox_result = fit_cox_model(surv_data, [velocity_col], penalizer=penalizer)
                except Exception as e:
                    if verbose:
                        print(f"  Stratified Cox failed ({scheme}, {label}, {velocity_col}): {e}")
                    results.append({
                        'Scheme': scheme,
                        'Stratum': label,
                        'Ratio_Low': low if np.isfinite(low) else np.nan,
                        'Ratio_High': high if np.isfinite(high) else np.nan,
                        'Velocity_Var': velocity_col,
                        'HR': np.nan,
                        'HR_Lower': np.nan,
                        'HR_Upper': np.nan,
                        'p_value': np.nan,
                        'N': len(surv_data),
                        'Events': int(surv_data['E'].sum()) if 'E' in surv_data else 0,
                        'Penalizer': penalizer,
                        'Penalty_Reason': penalty_reason,
                        'Status': 'failed',
                    })
                    continue

                if 'summary' not in cox_result or velocity_col not in cox_result['summary'].index:
                    results.append({
                        'Scheme': scheme,
                        'Stratum': label,
                        'Ratio_Low': low if np.isfinite(low) else np.nan,
                        'Ratio_High': high if np.isfinite(high) else np.nan,
                        'Velocity_Var': velocity_col,
                        'HR': np.nan,
                        'HR_Lower': np.nan,
                        'HR_Upper': np.nan,
                        'p_value': np.nan,
                        'N': len(surv_data),
                        'Events': int(surv_data['E'].sum()),
                        'Penalizer': penalizer,
                        'Penalty_Reason': penalty_reason,
                        'Status': 'failed',
                    })
                    continue

                summary = cox_result['summary'].loc[velocity_col]
                hr = np.exp(summary['coef'])
                ci_lower = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
                ci_upper = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
                p_val = summary['p']

                results.append({
                    'Scheme': scheme,
                    'Stratum': label,
                    'Ratio_Low': low if np.isfinite(low) else np.nan,
                    'Ratio_High': high if np.isfinite(high) else np.nan,
                    'Velocity_Var': velocity_col,
                    'HR': hr,
                    'HR_Lower': ci_lower,
                    'HR_Upper': ci_upper,
                    'p_value': p_val,
                    'N': len(surv_data),
                    'Events': int(surv_data['E'].sum()),
                    'Penalizer': penalizer,
                    'Penalty_Reason': penalty_reason,
                    'Status': 'ok',
                })

    return pd.DataFrame(results)


# =============================================================================
# POOLED / STRATIFIED INTERACTION MODELS
# =============================================================================

def add_ratio_strata_dummies(
    data: pd.DataFrame,
    ratio_col: str = 'Ratio_disbursed_to_obligated',
    q: int = 4
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Add quartile-style ratio strata with dummy indicators."""
    if ratio_col not in data.columns:
        warnings.warn(f"Ratio column '{ratio_col}' not found; cannot build strata")
        return data, {}

    ratio_series = data[ratio_col].dropna()
    if ratio_series.empty or ratio_series.nunique() < 2:
        warnings.warn("Insufficient ratio variation for strata; skipping pooled models")
        return data, {}

    try:
        bins = pd.qcut(ratio_series, q=q, labels=False, duplicates='drop') + 1
    except ValueError as exc:
        warnings.warn(f"Unable to compute ratio strata: {exc}")
        return data, {}

    data = data.copy()
    data['Ratio_Stratum'] = np.nan
    data.loc[bins.index, 'Ratio_Stratum'] = bins.astype(int)

    max_bin = int(bins.max())
    dummy_cols = []
    for bin_id in range(2, max_bin + 1):
        dummy_col = f'Ratio_Stratum_Q{bin_id}'
        data[dummy_col] = (data['Ratio_Stratum'] == bin_id).astype(int)
        dummy_cols.append(dummy_col)

    quantiles = {}
    for quantile in np.linspace(1 / q, (q - 1) / q, q - 1):
        label = int(round(quantile * 100))
        quantiles[f'q{label}'] = ratio_series.quantile(quantile)

    return data, {
        'n_bins': max_bin,
        'dummy_cols': dummy_cols,
        'quantiles': quantiles,
        'stratum_col': 'Ratio_Stratum',
    }


def fit_velocity_strata_models(
    data: pd.DataFrame,
    velocity_col: str,
    base_covariates: List[str],
    interaction_covariates: List[str],
    strata_cols: Optional[List[str]] = None,
    penalizer: float = COX_PENALIZER
) -> Optional[Dict[str, Any]]:
    """Fit base and interaction Cox models for pooled/stratified comparisons."""
    full_cols = interaction_covariates + (strata_cols or [])
    surv_full = prepare_survival_data(data, capacity_cols=full_cols)
    if surv_full.empty:
        return None

    base_cols = ['T', 'E'] + base_covariates + (strata_cols or [])
    int_cols = ['T', 'E'] + interaction_covariates + (strata_cols or [])
    base_data = surv_full[base_cols].dropna()
    int_data = surv_full[int_cols].dropna()

    if base_data.empty or int_data.empty:
        return None

    base_fit = fit_cox_model(
        base_data,
        base_covariates,
        penalizer=penalizer,
        strata_cols=strata_cols
    )
    int_fit = fit_cox_model(
        int_data,
        interaction_covariates,
        penalizer=penalizer,
        strata_cols=strata_cols
    )

    if 'model' not in base_fit or 'model' not in int_fit:
        return None

    lrt_stat = 2 * (int_fit['model'].log_likelihood_ - base_fit['model'].log_likelihood_)
    lrt_df = max(len(interaction_covariates) - len(base_covariates), 1)
    lrt_p = chi2.sf(lrt_stat, lrt_df) if SCIPY_AVAILABLE else np.nan

    return {
        'base_fit': base_fit,
        'int_fit': int_fit,
        'n_obs': len(int_data),
        'n_events': int(int_data['E'].sum()),
        'lrt_stat': lrt_stat,
        'lrt_df': lrt_df,
        'lrt_p': lrt_p,
        'velocity_col': velocity_col,
        'penalizer': penalizer,
    }


def run_velocity_ratio_strata_models(
    data: pd.DataFrame,
    ratio_col: str = 'Ratio_disbursed_to_obligated',
    velocity_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fit pooled and stratified-baseline interaction models to test differential velocity effects.
    """
    if not LIFELINES_AVAILABLE:
        if verbose:
            print("  Skipping pooled/hierarchical models: lifelines not installed")
        return pd.DataFrame()

    if velocity_cols is None:
        velocity_cols = ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp']

    data, strata_meta = add_ratio_strata_dummies(data, ratio_col=ratio_col, q=4)
    if not strata_meta or strata_meta.get('n_bins', 0) < 2:
        return pd.DataFrame()

    dummy_cols = strata_meta['dummy_cols']
    stratum_col = strata_meta['stratum_col']
    quantiles = strata_meta.get('quantiles', {})

    results = []
    for velocity_col in velocity_cols:
        if velocity_col not in data.columns:
            continue

        model_data = data.copy()
        centered_col = f'{velocity_col}_c'
        model_data[centered_col] = model_data[velocity_col] - model_data[velocity_col].mean(skipna=True)

        interaction_cols = []
        for dummy_col in dummy_cols:
            inter_col = f'{centered_col}_x_{dummy_col}'
            model_data[inter_col] = model_data[centered_col] * model_data[dummy_col]
            interaction_cols.append(inter_col)

        pooled_base = [centered_col] + dummy_cols
        pooled_int = pooled_base + interaction_cols
        pooled_fit = fit_velocity_strata_models(
            model_data,
            velocity_col=velocity_col,
            base_covariates=pooled_base,
            interaction_covariates=pooled_int,
            strata_cols=None,
            penalizer=COX_PENALIZER
        )

        strat_base = [centered_col]
        strat_int = strat_base + interaction_cols
        strat_fit = fit_velocity_strata_models(
            model_data,
            velocity_col=velocity_col,
            base_covariates=strat_base,
            interaction_covariates=strat_int,
            strata_cols=[stratum_col],
            penalizer=COX_PENALIZER
        )

        for model_type, fit_result in [('pooled', pooled_fit), ('stratified_baseline', strat_fit)]:
            if fit_result is None:
                continue

            summary = fit_result['int_fit'].get('summary', pd.DataFrame())
            if summary.empty:
                continue

            for term, row in summary.iterrows():
                results.append({
                    'Model_Type': model_type,
                    'Velocity_Var': velocity_col,
                    'Term': term,
                    'coef': row.get('coef'),
                    'HR': row.get('exp(coef)'),
                    'p_value': row.get('p'),
                    'N': fit_result['n_obs'],
                    'Events': fit_result['n_events'],
                    'Penalizer': fit_result['penalizer'],
                    'LRT_stat': fit_result['lrt_stat'],
                    'LRT_df': fit_result['lrt_df'],
                    'LRT_p': fit_result['lrt_p'],
                    'Ratio_Q25': quantiles.get('q25'),
                    'Ratio_Q50': quantiles.get('q50'),
                    'Ratio_Q75': quantiles.get('q75'),
                })

    return pd.DataFrame(results)

# =============================================================================
# LOWER THRESHOLD ANALYSIS
# =============================================================================

def run_lower_threshold_analysis(
    data: pd.DataFrame,
    thresholds: List[str] = ['50pct', '70pct', '90pct'],
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run SEM at multiple duration thresholds.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    thresholds : list
        Threshold suffixes to test.
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Comparison table across thresholds.
    """
    if not SEMOPY_AVAILABLE:
        if verbose:
            print("  Skipping threshold analysis: semopy not installed")
        return pd.DataFrame()

    if verbose:
        print("\n  --- Lower Threshold Analysis ---")
        print(f"  Thresholds: {thresholds}")
        print(f"  Subset: {subset}")

        # Show available observations
        avail = get_available_duration_thresholds(data)
        print("\n  Duration availability:")
        for col, n in avail.items():
            pct = 100 * n / len(data)
            print(f"    {col}: {n} ({pct:.1f}%)")

    results = run_threshold_sensitivity_sem(data, thresholds, subset, verbose)

    return results


# =============================================================================
# DURATION-FREE ANALYSIS
# =============================================================================

def run_duration_free_analysis(
    data: pd.DataFrame,
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run duration-free SEM models.

    Tests capacity effects using only ratio-based outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Results for duration-free models.
    """
    if not SEMOPY_AVAILABLE:
        if verbose:
            print("  Skipping duration-free analysis: semopy not installed")
        return pd.DataFrame()

    if verbose:
        print("\n  --- Duration-Free Analysis ---")
        print(f"  Subset: {subset}")

    # Filter by subset
    if subset == 'state':
        data = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        data = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]

    if verbose:
        print(f"  Sample size: {len(data)}")

    # Models to test
    duration_free_models = [
        'duration_free_cv',
        'duration_free_single',
        'duration_free_multiple',
        'duration_free_3x2',
    ]

    results = []

    for model_name in duration_free_models:
        if model_name not in MODEL_REGISTRY:
            if verbose:
                print(f"  Skipping {model_name}: not in registry")
            continue

        if verbose:
            print(f"\n  Testing: {model_name}")

        try:
            model_spec = get_model_spec(model_name)

            from semopy import Model, calc_stats

            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Model(model_spec)
                model.fit(data)

            estimates = model.inspect()
            fit_stats = calc_stats(model)

            # Get sample size from model
            if hasattr(model, 'mx_data') and model.mx_data is not None:
                n_obs = model.mx_data.shape[0]
            else:
                n_obs = len(data)

            # Extract structural path(s)
            structural = estimates[estimates['op'] == '~']

            beta = np.nan
            se = np.nan
            pval = np.nan

            # Look for capacity effect
            cap_paths = structural[structural['rval'] == 'gov_cap']
            if not cap_paths.empty:
                row = cap_paths.iloc[0]
                beta = float(row['Estimate'])
                se = float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan
                pval = float(row['p-value']) if pd.notna(row['p-value']) else np.nan

            results.append({
                'Model': model_name,
                'N': n_obs,
                'CFI': extract_fit_stat(fit_stats, 'CFI'),
                'RMSEA': extract_fit_stat(fit_stats, 'RMSEA'),
                'Capacity_Beta': beta,
                'Capacity_SE': se,
                'Capacity_p': pval,
                'Significant': pval < 0.05 if pd.notna(pval) else False,
                'Subset': subset,
            })

            if verbose:
                sig = '*' if pval < 0.05 else ''
                print(f"    N={n_obs}, Beta={beta:.3f}, p={pval:.3f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# MILESTONE-BASED ANALYSIS
# =============================================================================

def run_milestone_analysis(
    data: pd.DataFrame,
    subset: str = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run milestone-based SEM models.

    Uses Time_to_50pct, Progress_Rate, etc. as outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subset : str
        Government type: 'all', 'state', or 'local'.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Results for milestone-based models.
    """
    if not SEMOPY_AVAILABLE:
        if verbose:
            print("  Skipping milestone analysis: semopy not installed")
        return pd.DataFrame()

    if verbose:
        print("\n  --- Milestone-Based Analysis ---")
        print(f"  Subset: {subset}")

    # Filter by subset
    if subset == 'state':
        data = data[data['Grantee'].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        data = data[data['Grantee'].isin(LOCAL_GOVERNMENTS)]

    if verbose:
        print(f"  Sample size: {len(data)}")

    # Models to test
    milestone_models = [
        'milestone_time_to_50',
        'milestone_progress_rate',
        'milestone_velocity',
        'milestone_direct',
        'exp_time_to_milestone',  # Also include existing milestone model
    ]

    results = []

    for model_name in milestone_models:
        if model_name not in MODEL_REGISTRY:
            if verbose:
                print(f"  Skipping {model_name}: not in registry")
            continue

        if verbose:
            print(f"\n  Testing: {model_name}")

        try:
            model_spec = get_model_spec(model_name)

            from semopy import Model, calc_stats

            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Model(model_spec)
                model.fit(data)

            estimates = model.inspect()
            fit_stats = calc_stats(model)

            # Get sample size from model
            if hasattr(model, 'mx_data') and model.mx_data is not None:
                n_obs = model.mx_data.shape[0]
            else:
                n_obs = len(data)

            # Extract structural path(s)
            structural = estimates[estimates['op'] == '~']

            beta = np.nan
            se = np.nan
            pval = np.nan

            # Look for capacity effect
            cap_paths = structural[structural['rval'] == 'gov_cap']
            if not cap_paths.empty:
                row = cap_paths.iloc[0]
                beta = float(row['Estimate'])
                se = float(row['Std. Err']) if pd.notna(row['Std. Err']) else np.nan
                pval = float(row['p-value']) if pd.notna(row['p-value']) else np.nan

            results.append({
                'Model': model_name,
                'N': n_obs,
                'CFI': extract_fit_stat(fit_stats, 'CFI'),
                'RMSEA': extract_fit_stat(fit_stats, 'RMSEA'),
                'Capacity_Beta': beta,
                'Capacity_SE': se,
                'Capacity_p': pval,
                'Significant': pval < 0.05 if pd.notna(pval) else False,
                'Subset': subset,
            })

            if verbose:
                sig = '*' if pval < 0.05 else ''
                print(f"    N={n_obs}, Beta={beta:.3f}, p={pval:.3f}{sig}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# COMBINED ANALYSIS
# =============================================================================

def run_all_alternatives(
    data: pd.DataFrame,
    subsets: List[str] = ['all', 'state', 'local'],
    methods: Optional[List[str]] = None,
    capacity_sets: Optional[Dict[str, List[str]]] = None,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete alternative analysis battery.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with features.
    subsets : list
        Government subsets to analyze.
    methods : list, optional
        Methods to run. If None, runs all.
        Options: 'survival', 'threshold', 'duration_free', 'milestone'
    save_results : bool
        Whether to save results to disk.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        All results organized by method.
    """
    if methods is None:
        methods = ['survival', 'threshold', 'duration_free', 'milestone']

    results = {}
    diag_dir = ensure_diagnostics_dir()

    # 1. Survival Analysis (uses full sample, ignores subsets)
    if 'survival' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("SURVIVAL ANALYSIS")
            print("=" * 60)

        resolved_sets = capacity_sets or {'ratios': SURVIVAL_CAPACITY_COLS}
        if len(resolved_sets) == 1 and 'ratios' in resolved_sets:
            survival_results = run_survival_analysis(
                data,
                capacity_cols=resolved_sets['ratios'],
                verbose=verbose
            )
            results['survival'] = survival_results

            if save_results and 'coefficients' in survival_results:
                path = diag_dir / "alternatives_survival.csv"
                survival_results['coefficients'].to_csv(path, index=False)
                if verbose:
                    print(f"\n  Saved to: {path}")
        else:
            survival_capacity_df = run_survival_capacity_sets(
                data,
                capacity_sets=resolved_sets,
                verbose=verbose
            )
            results['survival_capacity_sets'] = survival_capacity_df

            if save_results and not survival_capacity_df.empty:
                path = diag_dir / "alternatives_survival_capacity_sets.csv"
                survival_capacity_df.to_csv(path, index=False)
                if verbose:
                    print(f"\n  Saved to: {path}")

        stratified_df = run_velocity_stratified_by_ratio(data, verbose=verbose)
        results['survival_stratified_velocity'] = stratified_df

        if save_results and not stratified_df.empty:
            path = diag_dir / "alternatives_survival_stratified_velocity.csv"
            stratified_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

        pooled_df = run_velocity_ratio_strata_models(data, verbose=verbose)
        results['survival_velocity_strata_models'] = pooled_df

        if save_results and not pooled_df.empty:
            path = diag_dir / "alternatives_survival_velocity_strata_models.csv"
            pooled_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 2. Lower Threshold Analysis
    if 'threshold' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("LOWER THRESHOLD ANALYSIS")
            print("=" * 60)

        threshold_results = []
        for subset in subsets:
            if verbose:
                print(f"\n  Subset: {subset}")
            subset_results = run_lower_threshold_analysis(
                data, subset=subset, verbose=verbose
            )
            threshold_results.append(subset_results)

        threshold_df = pd.concat(threshold_results, ignore_index=True)
        results['threshold'] = threshold_df

        if save_results and not threshold_df.empty:
            path = diag_dir / "alternatives_threshold_sensitivity.csv"
            threshold_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 3. Duration-Free Analysis
    if 'duration_free' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("DURATION-FREE ANALYSIS")
            print("=" * 60)

        duration_free_results = []
        for subset in subsets:
            if verbose:
                print(f"\n  Subset: {subset}")
            subset_results = run_duration_free_analysis(
                data, subset=subset, verbose=verbose
            )
            duration_free_results.append(subset_results)

        duration_free_df = pd.concat(duration_free_results, ignore_index=True)
        results['duration_free'] = duration_free_df

        if save_results and not duration_free_df.empty:
            path = diag_dir / "alternatives_duration_free.csv"
            duration_free_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 4. Milestone-Based Analysis
    if 'milestone' in methods:
        if verbose:
            print("\n" + "=" * 60)
            print("MILESTONE-BASED ANALYSIS")
            print("=" * 60)

        milestone_results = []
        for subset in subsets:
            if verbose:
                print(f"\n  Subset: {subset}")
            subset_results = run_milestone_analysis(
                data, subset=subset, verbose=verbose
            )
            milestone_results.append(subset_results)

        milestone_df = pd.concat(milestone_results, ignore_index=True)
        results['milestone'] = milestone_df

        if save_results and not milestone_df.empty:
            path = diag_dir / "alternatives_milestone.csv"
            milestone_df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved to: {path}")

    # 5. Cross-Method Comparison
    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-METHOD COMPARISON")
        print("=" * 60)

    comparison = compare_methods(
        survival_results=results.get('survival', {}).get('coefficients'),
        threshold_results=results.get('threshold'),
        duration_free_results=results.get('duration_free'),
        milestone_results=results.get('milestone'),
    )
    results['comparison'] = comparison

    if save_results and not comparison.empty:
        path = diag_dir / "alternatives_comparison.csv"
        comparison.to_csv(path, index=False)
        if verbose:
            print(f"\n  Saved to: {path}")

    # Summary
    if verbose:
        summary = summarize_alternatives_findings(comparison)
        print("\n" + summary)

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(
    methods: Optional[List[str]] = None,
    subset: str = 'all',
    save_results: bool = True,
    capacity_sets: Optional[List[str]] = None
):
    """
    Main entry point for alternative analyses.

    Parameters
    ----------
    methods : list, optional
        Which methods to run: 'survival', 'threshold', 'duration_free', 'milestone'.
        If None, runs all.
    subset : str
        Government subset (used for selecting single subset).
    save_results : bool
        Whether to save to disk.
    capacity_sets : list, optional
        Capacity sets to run for survival analysis (names from config).
    """
    print("=" * 60)
    print("Stage 06: Alternative Modeling Approaches")
    print("=" * 60)

    # Load data
    print("\nLoading panel features...")
    try:
        data = load_panel_features()
        print(f"  Loaded {len(data)} observations")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Determine subsets
    if subset == 'all':
        subsets = ['all', 'state', 'local']
    else:
        subsets = [subset]

    # Resolve capacity sets for survival analysis
    resolved_sets = resolve_capacity_sets(capacity_sets)

    # Run analyses
    results = run_all_alternatives(
        data,
        subsets=subsets,
        methods=methods,
        capacity_sets=resolved_sets,
        save_results=save_results,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Alternative analyses complete!")
    print("=" * 60)

    return results


# =============================================================================
# Phase 1: Measurement Validation Functions
# =============================================================================


def run_qa_flag_sensitivity_analysis(
    panel: pd.DataFrame,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Analysis 1.1: QA Flag Sensitivity Analysis

    Test if velocity effects persist after excluding quality-flagged observations.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with QA flag columns
    save_results : bool
        Save results to diagnostics directory

    Returns
    -------
    pd.DataFrame
        Comparison of velocity HR with/without flagged observations
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines not available")

    from lifelines import CoxPHFitter

    print("\n" + "=" * 80)
    print("Analysis 1.1: QA Flag Sensitivity Analysis")
    print("=" * 80)

    # Prepare survival data
    # Event = 1 if Duration notna (reached 95% threshold), 0 if Duration is NA (censored)
    panel_surv = panel.copy()
    panel_surv['Event'] = panel_surv['Duration'].notna() & (panel_surv['Duration'] > 0)

    # For censored observations, use last observed quarter as duration
    if 'N_Quarters' in panel_surv.columns:
        panel_surv['Duration_Surv'] = panel_surv['Duration'].fillna(panel_surv['N_Quarters'])
    else:
        # If N_Quarters not available, drop censored observations
        panel_surv = panel_surv[panel_surv['Event']].copy()
        panel_surv['Duration_Surv'] = panel_surv['Duration']

    # Scale velocity by 100 to get true percentage points and avoid convergence issues
    for vel_col in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
        if vel_col in panel_surv.columns:
            panel_surv[f'{vel_col}_scaled'] = panel_surv[vel_col] * 100

    results = []

    # Baseline model (all observations)
    print("\n1. Baseline model (all observations, N={}, Events={})".format(len(panel_surv), panel_surv['Event'].sum()))

    for vel_var in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
        vel_var_scaled = f'{vel_var}_scaled'
        if vel_var_scaled not in panel_surv.columns:
            continue

        subset = panel_surv[['Duration_Surv', 'Event', vel_var_scaled, 'Government_Type_State']].dropna()

        if len(subset) < 20:
            print(f"  Skipping {vel_var}: insufficient sample (N={len(subset)})")
            continue

        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(subset, duration_col='Duration_Surv', event_col='Event')

            results.append({
                'Model': 'Baseline',
                'Velocity_Measure': vel_var.replace('_pp', ''),
                'Sample': 'All',
                'N': len(subset),
                'Events': subset['Event'].sum(),
                'Velocity_HR': np.exp(cph.params_[vel_var_scaled]),
                'Velocity_CI_lower': np.exp(cph.confidence_intervals_[vel_var_scaled][0]),
                'Velocity_CI_upper': np.exp(cph.confidence_intervals_[vel_var_scaled][1]),
                'Velocity_p': cph.summary.loc[vel_var_scaled, 'p'],
                'Velocity_HR_Interpretation': f'Per 1 pp/quarter increase',
            })
            print(f"  {vel_var}: HR = {np.exp(cph.params_[vel_var_scaled]):.3f} per 1 pp/quarter, p = {cph.summary.loc[vel_var_scaled, 'p']:.4f}")
        except Exception as e:
            print(f"  {vel_var} failed: {e}")

    # Exclude high-flag programs
    print("\n2. Excluding high-flag programs (>2 extreme velocity or obligated jump flags)")
    panel_clean = panel_surv[panel_surv['QA_High_Flag_Program'] == False].copy()
    print(f"   Clean sample: N={len(panel_clean)} ({len(panel_clean)/len(panel_surv)*100:.1f}% of total)")

    for vel_var in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
        vel_var_scaled = f'{vel_var}_scaled'
        if vel_var_scaled not in panel_clean.columns:
            continue

        subset = panel_clean[['Duration_Surv', 'Event', vel_var_scaled, 'Government_Type_State']].dropna()

        if len(subset) < 10:
            print(f"  Skipping {vel_var}: insufficient sample (N={len(subset)})")
            continue

        cph = CoxPHFitter(penalizer=0.05)  # Higher penalization for small sample
        try:
            cph.fit(subset, duration_col='Duration_Surv', event_col='Event')

            results.append({
                'Model': 'Exclude_High_Flag',
                'Velocity_Measure': vel_var.replace('_pp', ''),
                'Sample': 'QA_High_Flag_Program=False',
                'N': len(subset),
                'Events': subset['Event'].sum(),
                'Velocity_HR': np.exp(cph.params_[vel_var_scaled]),
                'Velocity_CI_lower': np.exp(cph.confidence_intervals_[vel_var_scaled][0]),
                'Velocity_CI_upper': np.exp(cph.confidence_intervals_[vel_var_scaled][1]),
                'Velocity_p': cph.summary.loc[vel_var_scaled, 'p'],
            })
            print(f"  {vel_var}: HR = {np.exp(cph.params_[vel_var_scaled]):.3f}, p = {cph.summary.loc[vel_var_scaled, 'p']:.4f}")
        except Exception as e:
            print(f"  {vel_var} failed: {e}")

    # Exclude any extreme velocity flags
    print("\n3. Excluding programs with ANY extreme velocity flags")
    panel_no_extreme = panel_surv[panel_surv['Flag_Count_Extreme_Velocity'] == 0].copy()
    print(f"   No-extreme sample: N={len(panel_no_extreme)} ({len(panel_no_extreme)/len(panel_surv)*100:.1f}% of total)")

    for vel_var in ['Expenditure_Velocity_pp', 'Capacity_Velocity_Index_pp', 'Disbursement_Velocity_pp']:
        vel_var_scaled = f'{vel_var}_scaled'
        if vel_var_scaled not in panel_no_extreme.columns:
            continue

        subset = panel_no_extreme[['Duration_Surv', 'Event', vel_var_scaled, 'Government_Type_State']].dropna()

        if len(subset) < 10:
            print(f"  Skipping {vel_var}: insufficient sample (N={len(subset)})")
            continue

        cph = CoxPHFitter(penalizer=0.05)
        try:
            cph.fit(subset, duration_col='Duration_Surv', event_col='Event')

            results.append({
                'Model': 'Exclude_Any_Extreme',
                'Velocity_Measure': vel_var.replace('_pp', ''),
                'Sample': 'Flag_Count_Extreme_Velocity=0',
                'N': len(subset),
                'Events': subset['Event'].sum(),
                'Velocity_HR': np.exp(cph.params_[vel_var_scaled]),
                'Velocity_CI_lower': np.exp(cph.confidence_intervals_[vel_var_scaled][0]),
                'Velocity_CI_upper': np.exp(cph.confidence_intervals_[vel_var_scaled][1]),
                'Velocity_p': cph.summary.loc[vel_var_scaled, 'p'],
            })
            print(f"  {vel_var}: HR = {np.exp(cph.params_[vel_var_scaled]):.3f}, p = {cph.summary.loc[vel_var_scaled, 'p']:.4f}")
        except Exception as e:
            print(f"  {vel_var} failed: {e}")

    results_df = pd.DataFrame(results)

    if save_results:
        diag_dir = ensure_diagnostics_dir()
        output_path = diag_dir / "measurement_validation_qa_flags.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved results to {output_path}")

    return results_df


def run_velocity_operationalization_comparison(
    panel: pd.DataFrame,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Analysis 1.2: Alternative Velocity Operationalizations

    Test velocity effect robustness across different measurement approaches.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with multiple velocity operationalizations
    save_results : bool
        Save results to diagnostics directory

    Returns
    -------
    pd.DataFrame
        Meta-analysis of velocity HRs across operationalizations
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines not available")

    from lifelines import CoxPHFitter

    print("\n" + "=" * 80)
    print("Analysis 1.2: Velocity Operationalization Comparison")
    print("=" * 80)

    # Prepare survival data
    panel_surv = panel[panel['Duration'].notna() & (panel['Duration'] > 0)].copy()
    panel_surv['Event'] = panel_surv['Completion_Pct'] >= 0.95

    # Define velocity variants to test
    velocity_variants = {
        'Static_Mean_pp': 'Expenditure_Velocity_pp',
        'Static_Median_pp': 'Expenditure_Velocity_median',
        'Early_2q': 'Expenditure_Velocity_early_2q_pp',
        'Early_3q': 'Expenditure_Velocity_early_3q_pp',
        'Early_4q': 'Expenditure_Velocity_early_4q_pp',
        'Early_6q': 'Expenditure_Velocity_early_6q_pp',
        'Fixed_12m': 'Expenditure_Velocity_fixed_12m_pp',
        'Fixed_18m': 'Expenditure_Velocity_fixed_18m_pp',
        'Index_Mean_pp': 'Capacity_Velocity_Index_pp',
        'Index_Median_pp': 'Capacity_Velocity_Index_median',
        'Disbursement_Mean_pp': 'Disbursement_Velocity_pp',
        'Disbursement_Median_pp': 'Disbursement_Velocity_median',
    }

    results = []

    for variant_name, vel_col in velocity_variants.items():
        if vel_col not in panel_surv.columns:
            print(f"Skipping {variant_name}: column {vel_col} not found")
            continue

        subset = panel_surv[['Duration', 'Event', vel_col, 'Government_Type_State']].dropna()

        if len(subset) < 20:
            print(f"Skipping {variant_name}: insufficient sample (N={len(subset)})")
            continue

        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(subset, duration_col='Duration_Surv', event_col='Event')

            results.append({
                'Operationalization': variant_name,
                'Column': vel_col,
                'N': len(subset),
                'Events': subset['Event'].sum(),
                'Velocity_HR': np.exp(cph.params_[vel_col]),
                'Velocity_log_HR': cph.params_[vel_col],
                'Velocity_SE': cph.standard_errors_[vel_col],
                'Velocity_CI_lower': np.exp(cph.confidence_intervals_[vel_col][0]),
                'Velocity_CI_upper': np.exp(cph.confidence_intervals_[vel_col][1]),
                'Velocity_p': cph.summary.loc[vel_col, 'p'],
            })
            print(f"{variant_name:25s}: HR = {np.exp(cph.params_[vel_col]):.3f} (95% CI: {np.exp(cph.confidence_intervals_[vel_col][0]):.3f}-{np.exp(cph.confidence_intervals_[vel_col][1]):.3f}), p = {cph.summary.loc[vel_col, 'p']:.4f}, N={len(subset)}")
        except Exception as e:
            print(f"{variant_name:25s}: FAILED - {e}")

    results_df = pd.DataFrame(results)

    # Meta-analysis: average log HR with inverse-variance weighting
    if len(results_df) > 0:
        results_df['Weight'] = 1 / (results_df['Velocity_SE'] ** 2)
        meta_log_HR = (results_df['Velocity_log_HR'] * results_df['Weight']).sum() / results_df['Weight'].sum()
        meta_SE = np.sqrt(1 / results_df['Weight'].sum())
        meta_HR = np.exp(meta_log_HR)
        meta_CI_lower = np.exp(meta_log_HR - 1.96 * meta_SE)
        meta_CI_upper = np.exp(meta_log_HR + 1.96 * meta_SE)

        print("\n" + "-" * 80)
        print("META-ANALYSIS (inverse-variance weighted):")
        print(f"  Average HR = {meta_HR:.3f} (95% CI: {meta_CI_lower:.3f}-{meta_CI_upper:.3f})")
        print(f"  Range: {results_df['Velocity_HR'].min():.3f} - {results_df['Velocity_HR'].max():.3f}")
        print(f"  Std dev: {results_df['Velocity_HR'].std():.3f}")
        print("-" * 80)

        # Add meta-analysis row
        results.append({
            'Operationalization': 'META_ANALYSIS',
            'Column': 'Weighted_Average',
            'N': results_df['N'].max(),
            'Events': results_df['Events'].max(),
            'Velocity_HR': meta_HR,
            'Velocity_log_HR': meta_log_HR,
            'Velocity_SE': meta_SE,
            'Velocity_CI_lower': meta_CI_lower,
            'Velocity_CI_upper': meta_CI_upper,
            'Velocity_p': np.nan,
        })
        results_df = pd.DataFrame(results)

    if save_results:
        diag_dir = ensure_diagnostics_dir()
        output_path = diag_dir / "measurement_validation_velocity_variants.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved results to {output_path}")

    return results_df


# =============================================================================
# Phase 2: Mechanistic Deep Dive - Multi-Stage Efficiency Analysis
# =============================================================================

def run_multistage_efficiency_analysis(
    panel: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Competing risks survival analysis by pipeline stage.

    Tests if velocity effects differ by bottleneck location:
    - Event type 1: "Completed" (reached 95% threshold)
    - Event type 2: "Stalled_Stage1" (low Stage1 efficiency < 0.5)
    - Event type 3: "Stalled_Stage2" (low Stage2 efficiency < 0.5)

    Parameters
    ----------
    panel : pd.DataFrame, optional
        Panel with stage lag features. If None, loads from panel_features_std.parquet
    output_path : Path, optional
        Where to save results. Defaults to data_work/diagnostics/multistage_efficiency.csv

    Returns
    -------
    pd.DataFrame
        Results with HR and p-values for each event type
    """
    if not LIFELINES_AVAILABLE:
        warnings.warn("lifelines not available, skipping multi-stage analysis")
        return pd.DataFrame()

    from lifelines import CoxPHFitter

    print("=" * 80)
    print("Phase 2 Analysis 2.1: Multi-Stage Efficiency & Bottleneck Identification")
    print("=" * 80)
    print()

    # Load panel if not provided
    if panel is None:
        panel_path = DATA_WORK_DIR / "panel_features_std.parquet"
        print(f"Loading panel: {panel_path}")
        panel = safe_read_parquet(panel_path)
        print(f"  Loaded {len(panel)} grantee-disaster pairs")
        print()

    # Check required columns
    required_cols = ['Duration', 'Expenditure_Velocity_pp', 'Stage1_Efficiency',
                     'Stage2_Efficiency', 'Lag_Total_Pipeline', 'Government_Type_State']
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        warnings.warn(f"Missing required columns: {missing_cols}. Run build_features_std first.")
        return pd.DataFrame()

    # Define competing events
    panel = panel.copy()
    panel['Event_Type'] = 'Censored'

    # Event 1: Completed (reached 95% threshold)
    panel.loc[(panel['Duration'].notna()) & (panel['Duration'] > 0), 'Event_Type'] = 'Completed'

    # Event 2: Stalled at Stage 1 (low Stage1 efficiency, not completed)
    panel.loc[
        (panel['Event_Type'] == 'Censored') &
        (panel['Stage1_Efficiency'].notna()) &
        (panel['Stage1_Efficiency'] < 0.5),
        'Event_Type'
    ] = 'Stalled_Stage1'

    # Event 3: Stalled at Stage 2 (low Stage2 efficiency, not completed)
    panel.loc[
        (panel['Event_Type'] == 'Censored') &
        (panel['Stage2_Efficiency'].notna()) &
        (panel['Stage2_Efficiency'] < 0.5),
        'Event_Type'
    ] = 'Stalled_Stage2'

    print("Event type distribution:")
    print(panel['Event_Type'].value_counts())
    print()

    # Prepare survival duration
    panel['Duration_Surv'] = panel['Duration'].fillna(panel['N_Quarters'])

    # Scale velocity by 100 for proper interpretation (pp/quarter)
    panel['Velocity_scaled'] = panel['Expenditure_Velocity_pp'] * 100

    results = []

    # Fit Cox PH for each event type
    for event_type in ['Completed', 'Stalled_Stage1', 'Stalled_Stage2']:
        print(f"\nFitting Cox PH for event: {event_type}")

        # Binary event indicator
        panel[f'Event_{event_type}'] = (panel['Event_Type'] == event_type).astype(int)
        n_events = panel[f'Event_{event_type}'].sum()

        print(f"  Events: {n_events}")

        if n_events < 5:
            print(f"  ⚠ Too few events ({n_events}), skipping")
            continue

        # Subset to complete cases
        subset = panel[
            ['Duration_Surv', f'Event_{event_type}', 'Velocity_scaled',
             'Lag_Total_Pipeline', 'Government_Type_State']
        ].dropna()

        print(f"  Sample: N={len(subset)}, Events={subset[f'Event_{event_type}'].sum()}")

        # Fit Cox PH
        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(
                subset,
                duration_col='Duration_Surv',
                event_col=f'Event_{event_type}'
            )

            # Debug: print model params
            print(f"  Model params: {cph.params_.index.tolist()}")

            # Extract results
            results.append({
                'Event_Type': event_type,
                'N': len(subset),
                'N_Events': int(subset[f'Event_{event_type}'].sum()),
                'Velocity_HR': np.exp(cph.params_['Velocity_scaled']),
                'Velocity_CI_lower': np.exp(cph.confidence_intervals_.loc['Velocity_scaled', '95% lower-bound']),
                'Velocity_CI_upper': np.exp(cph.confidence_intervals_.loc['Velocity_scaled', '95% upper-bound']),
                'Velocity_p': cph.summary.loc['Velocity_scaled', 'p'],
                'Lag_HR': np.exp(cph.params_['Lag_Total_Pipeline']),
                'Lag_p': cph.summary.loc['Lag_Total_Pipeline', 'p'],
                'Government_Type_State_HR': np.exp(cph.params_['Government_Type_State']),
                'Government_Type_State_p': cph.summary.loc['Government_Type_State', 'p'],
            })

            print(f"  Velocity HR: {np.exp(cph.params_['Velocity_scaled']):.3f} (p={cph.summary.loc['Velocity_scaled', 'p']:.4f})")
            print(f"  Lag HR: {np.exp(cph.params_['Lag_Total_Pipeline']):.3f} (p={cph.summary.loc['Lag_Total_Pipeline', 'p']:.4f})")

        except Exception as e:
            import traceback
            print(f"  ✗ Model failed: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()
            results.append({
                'Event_Type': event_type,
                'N': len(subset),
                'N_Events': int(subset[f'Event_{event_type}'].sum()),
                'Velocity_HR': np.nan,
                'Velocity_p': np.nan,
                'Lag_HR': np.nan,
                'Lag_p': np.nan,
                'Error': str(e),
            })

    results_df = pd.DataFrame(results)

    # Save results
    if output_path is None:
        output_path = DATA_WORK_DIR / "diagnostics" / "multistage_efficiency.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to {output_path}")

    print("\n" + "=" * 80)
    print("Multi-Stage Efficiency Analysis Complete")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run alternative modeling approaches")
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        choices=['survival', 'threshold', 'duration_free', 'milestone', 'all'],
        help="Methods to run (default: all)"
    )
    parser.add_argument(
        "--subset", "-s",
        default="all",
        choices=["all", "state", "local"],
        help="Government subset"
    )
    parser.add_argument(
        "--survival-only",
        action="store_true",
        help="Run only survival analysis"
    )
    parser.add_argument(
        "--sem-only",
        action="store_true",
        help="Run only SEM alternatives (no survival)"
    )
    parser.add_argument(
        "--capacity-sets",
        nargs="+",
        default=None,
        help="Capacity sets for survival analysis (names from config, or 'all')"
    )

    args = parser.parse_args()

    # Handle convenience flags
    if args.survival_only:
        methods = ['survival']
    elif args.sem_only:
        methods = ['threshold', 'duration_free', 'milestone']
    elif args.methods and 'all' in args.methods:
        methods = None
    else:
        methods = args.methods

    main(methods=methods, subset=args.subset, capacity_sets=args.capacity_sets)
