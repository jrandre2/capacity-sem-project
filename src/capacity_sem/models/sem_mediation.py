"""
Mediation analysis for SEM models.

This module provides functions for:
1. Computing indirect effects (a × b paths)
2. Decomposing total effects into direct and indirect components
3. Bootstrap confidence intervals for indirect effects
4. Testing mediation significance (Sobel test, bootstrap)

Mediation occurs when a third variable (mediator) intervenes in the
relationship between an independent and dependent variable:

    IV → Mediator → DV  (indirect effect = a × b)
    IV → DV            (direct effect = c')
    Total effect = c' + a × b

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


def extract_mediation_paths(
    estimates: pd.DataFrame,
    iv: str,
    mediator: str,
    dv: str
) -> Dict[str, float]:
    """
    Extract mediation path coefficients from model estimates.

    Parameters
    ----------
    estimates : pd.DataFrame
        Parameter estimates from model.inspect().
    iv : str
        Independent variable (predictor) name.
    mediator : str
        Mediator variable name.
    dv : str
        Dependent variable (outcome) name.

    Returns
    -------
    Dict[str, float]
        Dictionary with path coefficients:
        - 'a': IV → Mediator path
        - 'b': Mediator → DV path (controlling for IV)
        - 'c_prime': IV → DV direct path (controlling for mediator)
        - 'indirect': a × b (indirect effect)
        - 'total': c' + a×b (total effect)
    """
    paths = {
        'a': np.nan,
        'b': np.nan,
        'c_prime': np.nan,
        'indirect': np.nan,
        'total': np.nan,
        'a_se': np.nan,
        'b_se': np.nan,
        'c_prime_se': np.nan,
        'a_p': np.nan,
        'b_p': np.nan,
        'c_prime_p': np.nan
    }

    if estimates.empty:
        return paths

    # Get structural paths (~ operator)
    struct = estimates[estimates['op'] == '~']

    # Path a: IV → Mediator
    a_row = struct[(struct['lval'] == mediator) & (struct['rval'] == iv)]
    if not a_row.empty:
        paths['a'] = a_row['Estimate'].iloc[0]
        if 'Std. Err' in a_row.columns:
            paths['a_se'] = a_row['Std. Err'].iloc[0]
        if 'p-value' in a_row.columns:
            paths['a_p'] = a_row['p-value'].iloc[0]

    # Path b: Mediator → DV
    b_row = struct[(struct['lval'] == dv) & (struct['rval'] == mediator)]
    if not b_row.empty:
        paths['b'] = b_row['Estimate'].iloc[0]
        if 'Std. Err' in b_row.columns:
            paths['b_se'] = b_row['Std. Err'].iloc[0]
        if 'p-value' in b_row.columns:
            paths['b_p'] = b_row['p-value'].iloc[0]

    # Path c': IV → DV (direct)
    c_row = struct[(struct['lval'] == dv) & (struct['rval'] == iv)]
    if not c_row.empty:
        paths['c_prime'] = c_row['Estimate'].iloc[0]
        if 'Std. Err' in c_row.columns:
            paths['c_prime_se'] = c_row['Std. Err'].iloc[0]
        if 'p-value' in c_row.columns:
            paths['c_prime_p'] = c_row['p-value'].iloc[0]

    # Compute indirect effect
    if pd.notna(paths['a']) and pd.notna(paths['b']):
        paths['indirect'] = paths['a'] * paths['b']

    # Compute total effect
    if pd.notna(paths['indirect']) and pd.notna(paths['c_prime']):
        paths['total'] = paths['c_prime'] + paths['indirect']

    return paths


def sobel_test(a: float, b: float, se_a: float, se_b: float) -> Tuple[float, float]:
    """
    Perform Sobel test for indirect effect significance.

    The Sobel test uses the formula:
    SE_indirect = sqrt(b² × SE_a² + a² × SE_b²)
    z = (a × b) / SE_indirect

    Note: This test assumes normal distribution of the indirect effect,
    which is often violated. Bootstrap methods are preferred.

    Parameters
    ----------
    a : float
        Path a coefficient (IV → Mediator).
    b : float
        Path b coefficient (Mediator → DV).
    se_a : float
        Standard error of path a.
    se_b : float
        Standard error of path b.

    Returns
    -------
    Tuple[float, float]
        (z-statistic, two-tailed p-value)
    """
    from scipy import stats

    if any(pd.isna([a, b, se_a, se_b])):
        return np.nan, np.nan

    if se_a <= 0 or se_b <= 0:
        return np.nan, np.nan

    # Sobel standard error
    se_indirect = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)

    if se_indirect <= 0:
        return np.nan, np.nan

    # Z statistic
    indirect = a * b
    z = indirect / se_indirect

    # Two-tailed p-value
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p


def bootstrap_indirect_effect(
    model_spec: str,
    data: pd.DataFrame,
    iv: str,
    mediator: str,
    dv: str,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval for indirect effect.

    Bootstrap is the preferred method for testing indirect effects
    because it doesn't assume normal distribution of the indirect effect.

    Parameters
    ----------
    model_spec : str
        SEM model specification.
    data : pd.DataFrame
        Data for fitting.
    iv : str
        Independent variable name.
    mediator : str
        Mediator variable name.
    dv : str
        Dependent variable name.
    n_boot : int, default 1000
        Number of bootstrap samples.
    ci_level : float, default 0.95
        Confidence interval level.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default True
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        Bootstrap results including:
        - 'indirect': Point estimate of indirect effect
        - 'ci_lower': Lower CI bound
        - 'ci_upper': Upper CI bound
        - 'boot_samples': Array of bootstrapped indirect effects
        - 'sig': Whether CI excludes zero
    """
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required for bootstrap mediation analysis")

    if seed is not None:
        np.random.seed(seed)

    results = {
        'indirect': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'boot_samples': [],
        'sig': False,
        'n_boot': n_boot,
        'ci_level': ci_level
    }

    # Fit original model
    try:
        model, _ = fit_sem_model(model_spec, data)
        if model is None:
            return results

        estimates = model.inspect()
        original_paths = extract_mediation_paths(estimates, iv, mediator, dv)
        results['indirect'] = original_paths['indirect']
        results['direct'] = original_paths['c_prime']
        results['total'] = original_paths['total']

    except Exception as e:
        if verbose:
            print(f"Error fitting original model: {e}")
        return results

    # Bootstrap
    boot_indirect = []
    n_failed = 0

    if verbose:
        print(f"Bootstrapping {n_boot} samples...")

    for i in range(n_boot):
        # Resample with replacement
        boot_data = data.sample(n=len(data), replace=True)

        try:
            boot_model, _ = fit_sem_model(model_spec, boot_data)
            if boot_model is not None:
                boot_estimates = boot_model.inspect()
                boot_paths = extract_mediation_paths(boot_estimates, iv, mediator, dv)
                if pd.notna(boot_paths['indirect']):
                    boot_indirect.append(boot_paths['indirect'])
                else:
                    n_failed += 1
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_boot} samples ({n_failed} failed)")

    if len(boot_indirect) < 10:
        if verbose:
            print(f"Warning: Only {len(boot_indirect)} successful bootstrap samples")
        return results

    boot_indirect = np.array(boot_indirect)
    results['boot_samples'] = boot_indirect

    # Compute percentile CI
    alpha = 1 - ci_level
    results['ci_lower'] = np.percentile(boot_indirect, 100 * alpha / 2)
    results['ci_upper'] = np.percentile(boot_indirect, 100 * (1 - alpha / 2))

    # Significant if CI doesn't include zero
    results['sig'] = not (results['ci_lower'] <= 0 <= results['ci_upper'])

    if verbose:
        print(f"\nBootstrap Results ({len(boot_indirect)} successful samples):")
        print(f"  Indirect effect: {results['indirect']:.4f}")
        print(f"  {int(ci_level*100)}% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        print(f"  Significant: {'Yes' if results['sig'] else 'No'}")

    return results


def compute_mediation_effects(
    model_spec: str,
    data: pd.DataFrame,
    iv: str,
    mediator: str,
    dv: str,
    bootstrap: bool = True,
    n_boot: int = 500,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute complete mediation analysis.

    Includes:
    - Point estimates for all paths
    - Sobel test for indirect effect
    - Bootstrap CI for indirect effect (optional)
    - Effect decomposition (direct, indirect, total)

    Parameters
    ----------
    model_spec : str
        SEM model specification.
    data : pd.DataFrame
        Data for fitting.
    iv : str
        Independent variable name.
    mediator : str
        Mediator variable name.
    dv : str
        Dependent variable name.
    bootstrap : bool, default True
        Whether to compute bootstrap CI.
    n_boot : int, default 500
        Number of bootstrap samples.
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    Dict[str, Any]
        Complete mediation analysis results.
    """
    results = {
        'paths': None,
        'sobel': None,
        'bootstrap': None,
        'interpretation': None
    }

    # Fit model and extract paths
    try:
        model, _ = fit_sem_model(model_spec, data)
        if model is None:
            return results

        estimates = model.inspect()
        paths = extract_mediation_paths(estimates, iv, mediator, dv)
        results['paths'] = paths

    except Exception as e:
        if verbose:
            print(f"Error fitting model: {e}")
        return results

    # Sobel test
    sobel_z, sobel_p = sobel_test(
        paths['a'], paths['b'],
        paths['a_se'], paths['b_se']
    )
    results['sobel'] = {
        'z': sobel_z,
        'p': sobel_p,
        'sig': sobel_p < 0.05 if pd.notna(sobel_p) else False
    }

    # Bootstrap (if requested)
    if bootstrap:
        boot_results = bootstrap_indirect_effect(
            model_spec, data, iv, mediator, dv,
            n_boot=n_boot, verbose=verbose
        )
        results['bootstrap'] = boot_results

    # Interpretation
    results['interpretation'] = _interpret_mediation(
        paths, results['sobel'], results.get('bootstrap')
    )

    if verbose:
        print("\n" + "=" * 60)
        print("MEDIATION ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nPath Coefficients:")
        print(f"  a (IV → Med): {paths['a']:.4f}" + (f" (p={paths['a_p']:.3f})" if pd.notna(paths['a_p']) else ""))
        print(f"  b (Med → DV): {paths['b']:.4f}" + (f" (p={paths['b_p']:.3f})" if pd.notna(paths['b_p']) else ""))
        print(f"  c' (IV → DV): {paths['c_prime']:.4f}" + (f" (p={paths['c_prime_p']:.3f})" if pd.notna(paths['c_prime_p']) else ""))
        print(f"\nEffect Decomposition:")
        print(f"  Indirect (a×b): {paths['indirect']:.4f}")
        print(f"  Direct (c'): {paths['c_prime']:.4f}")
        print(f"  Total: {paths['total']:.4f}")
        print(f"\nSobel Test:")
        print(f"  z = {sobel_z:.3f}, p = {sobel_p:.4f}")
        print(f"\nInterpretation: {results['interpretation']}")

    return results


def _interpret_mediation(
    paths: Dict[str, float],
    sobel: Dict[str, Any],
    bootstrap: Optional[Dict[str, Any]] = None
) -> str:
    """Generate interpretation of mediation results."""
    indirect_sig = False
    direct_sig = False

    # Check indirect significance
    if bootstrap and bootstrap.get('sig'):
        indirect_sig = True
    elif sobel and sobel.get('sig'):
        indirect_sig = True

    # Check direct significance
    if pd.notna(paths.get('c_prime_p')) and paths['c_prime_p'] < 0.05:
        direct_sig = True

    # Interpret
    if indirect_sig and direct_sig:
        return "Partial mediation: Both indirect and direct effects are significant"
    elif indirect_sig and not direct_sig:
        return "Full mediation: Indirect effect significant, direct effect not significant"
    elif not indirect_sig and direct_sig:
        return "No mediation: Only direct effect is significant"
    else:
        return "No effect: Neither indirect nor direct effects are significant"


def analyze_parallel_mediation(
    model_spec: str,
    data: pd.DataFrame,
    iv: str,
    mediators: List[str],
    dv: str,
    bootstrap: bool = True,
    n_boot: int = 500,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze parallel mediation with multiple mediators.

    In parallel mediation, multiple mediators simultaneously mediate
    the IV → DV relationship. This allows comparing which mediator
    has a stronger indirect effect.

    Parameters
    ----------
    model_spec : str
        SEM model specification with all mediator paths.
    data : pd.DataFrame
        Data for fitting.
    iv : str
        Independent variable name.
    mediators : List[str]
        List of mediator variable names.
    dv : str
        Dependent variable name.
    bootstrap : bool, default True
        Whether to compute bootstrap CIs.
    n_boot : int, default 500
        Number of bootstrap samples.
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    Dict[str, Any]
        Parallel mediation results for all mediators.
    """
    results = {
        'mediators': {},
        'total_indirect': np.nan,
        'direct': np.nan,
        'total': np.nan
    }

    # Fit model
    try:
        model, _ = fit_sem_model(model_spec, data)
        if model is None:
            return results

        estimates = model.inspect()

    except Exception as e:
        if verbose:
            print(f"Error fitting model: {e}")
        return results

    total_indirect = 0.0

    for mediator in mediators:
        paths = extract_mediation_paths(estimates, iv, mediator, dv)
        results['mediators'][mediator] = paths

        if pd.notna(paths['indirect']):
            total_indirect += paths['indirect']

    results['total_indirect'] = total_indirect

    # Get direct effect (should be same across mediators)
    if mediators:
        first_paths = results['mediators'][mediators[0]]
        results['direct'] = first_paths['c_prime']
        results['total'] = results['direct'] + total_indirect

    if verbose:
        print("\n" + "=" * 60)
        print("PARALLEL MEDIATION ANALYSIS")
        print("=" * 60)
        print(f"\nSpecific Indirect Effects:")
        for med, paths in results['mediators'].items():
            print(f"  Through {med}: {paths['indirect']:.4f}")
        print(f"\nTotal indirect: {results['total_indirect']:.4f}")
        print(f"Direct effect: {results['direct']:.4f}")
        print(f"Total effect: {results['total']:.4f}")

    return results


def analyze_serial_mediation(
    model_spec: str,
    data: pd.DataFrame,
    iv: str,
    mediator1: str,
    mediator2: str,
    dv: str,
    bootstrap: bool = True,
    n_boot: int = 500,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze serial mediation with two sequential mediators.

    In serial mediation:
    IV → M1 → M2 → DV (serial indirect: a1 × d × b2)
    IV → M1 → DV (specific indirect 1: a1 × b1)
    IV → M2 → DV (specific indirect 2: a2 × b2)
    IV → DV (direct: c')

    Parameters
    ----------
    model_spec : str
        SEM model specification with serial mediation paths.
    data : pd.DataFrame
        Data for fitting.
    iv : str
        Independent variable name.
    mediator1 : str
        First mediator in sequence.
    mediator2 : str
        Second mediator in sequence.
    dv : str
        Dependent variable name.
    bootstrap : bool, default True
        Whether to compute bootstrap CIs.
    n_boot : int, default 500
        Number of bootstrap samples.
    verbose : bool, default True
        Whether to print results.

    Returns
    -------
    Dict[str, Any]
        Serial mediation results including:
        - Path coefficients
        - Serial indirect effect (a1 × d × b2)
        - Specific indirect effects
        - Total indirect and direct effects
    """
    results = {
        'paths': {},
        'serial_indirect': np.nan,
        'specific_indirect_m1': np.nan,
        'specific_indirect_m2': np.nan,
        'total_indirect': np.nan,
        'direct': np.nan,
        'total': np.nan
    }

    # Fit model
    try:
        model, _ = fit_sem_model(model_spec, data)
        if model is None:
            return results

        estimates = model.inspect()
        struct = estimates[estimates['op'] == '~']

    except Exception as e:
        if verbose:
            print(f"Error fitting model: {e}")
        return results

    # Extract all paths
    def get_path(lval, rval):
        row = struct[(struct['lval'] == lval) & (struct['rval'] == rval)]
        return row['Estimate'].iloc[0] if not row.empty else np.nan

    # a1: IV → M1
    a1 = get_path(mediator1, iv)
    # d: M1 → M2
    d = get_path(mediator2, mediator1)
    # b1: M1 → DV
    b1 = get_path(dv, mediator1)
    # a2: IV → M2
    a2 = get_path(mediator2, iv)
    # b2: M2 → DV
    b2 = get_path(dv, mediator2)
    # c': IV → DV
    c_prime = get_path(dv, iv)

    results['paths'] = {
        'a1': a1, 'd': d, 'b1': b1,
        'a2': a2, 'b2': b2, 'c_prime': c_prime
    }

    # Compute indirect effects
    if pd.notna(a1) and pd.notna(d) and pd.notna(b2):
        results['serial_indirect'] = a1 * d * b2

    if pd.notna(a1) and pd.notna(b1):
        results['specific_indirect_m1'] = a1 * b1

    if pd.notna(a2) and pd.notna(b2):
        results['specific_indirect_m2'] = a2 * b2

    # Total indirect (sum of all indirect paths)
    total_indirect = 0.0
    for key in ['serial_indirect', 'specific_indirect_m1', 'specific_indirect_m2']:
        if pd.notna(results[key]):
            total_indirect += results[key]
    results['total_indirect'] = total_indirect

    results['direct'] = c_prime
    if pd.notna(c_prime):
        results['total'] = c_prime + total_indirect

    if verbose:
        print("\n" + "=" * 60)
        print("SERIAL MEDIATION ANALYSIS")
        print("=" * 60)
        print(f"\nPath Coefficients:")
        print(f"  a1 (IV → {mediator1}): {a1:.4f}")
        print(f"  d  ({mediator1} → {mediator2}): {d:.4f}")
        print(f"  b1 ({mediator1} → DV): {b1:.4f}")
        print(f"  a2 (IV → {mediator2}): {a2:.4f}")
        print(f"  b2 ({mediator2} → DV): {b2:.4f}")
        print(f"  c' (IV → DV): {c_prime:.4f}")
        print(f"\nIndirect Effects:")
        print(f"  Serial (a1×d×b2): {results['serial_indirect']:.4f}")
        print(f"  Through {mediator1} (a1×b1): {results['specific_indirect_m1']:.4f}")
        print(f"  Through {mediator2} (a2×b2): {results['specific_indirect_m2']:.4f}")
        print(f"  Total indirect: {results['total_indirect']:.4f}")
        print(f"\nDirect effect: {results['direct']:.4f}")
        print(f"Total effect: {results['total']:.4f}")

    return results
