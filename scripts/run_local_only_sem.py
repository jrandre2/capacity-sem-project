"""
Local-only SEM re-estimation for R8 M4b (equal-billing).

R8 argues the pooled model mixes 30 state agencies with 543 local jurisdictions
when state-vs-local measurement invariance is untestable at N=30. We report
the local-only coefficients alongside pooled to demonstrate the substantive
conclusions hold on both samples.

Outputs:
  data_work/diagnostics/sem_local_only_paths.csv     — four main path coefficients
  data_work/diagnostics/sem_local_only_fit.csv       — fit statistics
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from semopy import Model, calc_stats
except ImportError:
    raise SystemExit("semopy required: pip install semopy")

warnings.filterwarnings("ignore")

SEM_DATA_PATH = (
    PROJECT_ROOT / "data_work" / "diagnostics" / "kaifa_recovered_analysis"
    / "recovered_notebook_reference_573_sem_ready_dataset.csv"
)
OUT_PATHS = PROJECT_ROOT / "data_work" / "diagnostics" / "sem_local_only_paths.csv"
OUT_FIT = PROJECT_ROOT / "data_work" / "diagnostics" / "sem_local_only_fit.csv"

# Reference two-factor SEM spec (inlined to avoid kaifa module's `from config` import)
TWO_FACTOR_MODEL_SPEC = """
AdminResources =~ z_avg_employment + z_avg_payroll
AdminBurdenCapacity =~ z_rev_programs_per_staff + z_rev_disasters_per_staff
RecoveryPerformance =~ z_Ratio_disbursed_to_obligated + z_Ratio_expended_to_disbursed + z_Ratio_obligated_funds_fully_expended + z_Ratio_Program_Completed
RecoveryTimeliness =~ z_rev_Duration_of_completion + z_rev_Average_Duration_Program_Completion
RecoveryPerformance ~ AdminResources + AdminBurdenCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
RecoveryTimeliness ~ AdminResources + AdminBurdenCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
AdminResources ~~ AdminBurdenCapacity
RecoveryPerformance ~~ RecoveryTimeliness
"""

# Local-only spec (drops state_level predictor because it is collinear with
# constant-0 once state agencies are removed)
LOCAL_MODEL_SPEC = """
AdminResources =~ z_avg_employment + z_avg_payroll
AdminBurdenCapacity =~ z_rev_programs_per_staff + z_rev_disasters_per_staff
RecoveryPerformance =~ z_Ratio_disbursed_to_obligated + z_Ratio_expended_to_disbursed + z_Ratio_obligated_funds_fully_expended + z_Ratio_Program_Completed
RecoveryTimeliness =~ z_rev_Duration_of_completion + z_rev_Average_Duration_Program_Completion
RecoveryPerformance ~ AdminResources + AdminBurdenCapacity + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
RecoveryTimeliness ~ AdminResources + AdminBurdenCapacity + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
AdminResources ~~ AdminBurdenCapacity
RecoveryPerformance ~~ RecoveryTimeliness
"""

PATHS_OF_INTEREST = [
    ("RecoveryPerformance", "AdminResources"),
    ("RecoveryPerformance", "AdminBurdenCapacity"),
    ("RecoveryTimeliness", "AdminResources"),
    ("RecoveryTimeliness", "AdminBurdenCapacity"),
    ("RecoveryPerformance", "z_SPL_THEME1"),
    ("RecoveryPerformance", "z_SPL_THEME2"),
    ("RecoveryPerformance", "z_SPL_THEME3"),
    ("RecoveryPerformance", "z_SPL_THEME4"),
    ("RecoveryTimeliness", "z_SPL_THEME1"),
    ("RecoveryTimeliness", "z_SPL_THEME2"),
    ("RecoveryTimeliness", "z_SPL_THEME3"),
    ("RecoveryTimeliness", "z_SPL_THEME4"),
]


def fit_and_extract(spec: str, data: pd.DataFrame, label: str) -> tuple[dict, dict]:
    m = Model(spec)
    m.fit(data)
    est = m.inspect(std_est=True)
    try:
        # Include p-value from the Std vs. unstd row set
        p_col = "p-value" if "p-value" in est.columns else "p-value"
    except Exception:
        p_col = None
    paths = {}
    for lval, rval in PATHS_OF_INTEREST:
        match = est[(est["lval"] == lval) & (est["op"] == "~") & (est["rval"] == rval)]
        if len(match) == 1:
            paths[(lval, rval)] = {
                "beta_std": float(match["Est. Std"].iloc[0]),
                "b_unstd": float(match["Estimate"].iloc[0]),
                "p": float(match["p-value"].iloc[0]) if "p-value" in match.columns else np.nan,
                "se": float(match["Std. Err"].iloc[0]) if "Std. Err" in match.columns else np.nan,
            }
        else:
            paths[(lval, rval)] = {"beta_std": np.nan, "b_unstd": np.nan, "p": np.nan, "se": np.nan}
    # Fit
    try:
        fit = calc_stats(m)
        fit_d = {
            "label": label,
            "N": len(data),
            "CFI": float(fit.loc["Value", "CFI"]) if "CFI" in fit.columns else np.nan,
            "TLI": float(fit.loc["Value", "TLI"]) if "TLI" in fit.columns else np.nan,
            "RMSEA": float(fit.loc["Value", "RMSEA"]) if "RMSEA" in fit.columns else np.nan,
            "AIC": float(fit.loc["Value", "AIC"]) if "AIC" in fit.columns else np.nan,
            "BIC": float(fit.loc["Value", "BIC"]) if "BIC" in fit.columns else np.nan,
        }
    except Exception as e:
        fit_d = {"label": label, "N": len(data), "CFI": np.nan, "TLI": np.nan, "RMSEA": np.nan, "AIC": np.nan, "BIC": np.nan}
    return paths, fit_d


def main():
    print(f"Loading SEM data: {SEM_DATA_PATH}")
    df = pd.read_csv(SEM_DATA_PATH)
    print(f"  Pooled N = {len(df)}; state_level=1 N = {(df['state_level']==1).sum()}; state_level=0 N = {(df['state_level']==0).sum()}")

    # Fit pooled and local-only
    print("\n[Pooled] Fitting reference two-factor SEM...")
    pooled_paths, pooled_fit = fit_and_extract(TWO_FACTOR_MODEL_SPEC, df, "pooled")

    local_df = df[df["state_level"] == 0].copy()
    print(f"\n[Local-only] Fitting two-factor SEM (drops state_level predictor), N = {len(local_df)}...")
    local_paths, local_fit = fit_and_extract(LOCAL_MODEL_SPEC, local_df, "local_only")

    # Build comparison table
    rows = []
    for (lval, rval) in PATHS_OF_INTEREST:
        p = pooled_paths[(lval, rval)]
        l = local_paths[(lval, rval)]
        rows.append({
            "path_lval": lval,
            "path_rval": rval,
            "pooled_beta_std": p["beta_std"],
            "pooled_b": p["b_unstd"],
            "pooled_p": p["p"],
            "local_only_beta_std": l["beta_std"],
            "local_only_b": l["b_unstd"],
            "local_only_p": l["p"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATHS, index=False)
    print(f"\nPaths saved: {OUT_PATHS}")
    print(out.to_string(index=False))

    # Fit comparison
    fit_out = pd.DataFrame([pooled_fit, local_fit])
    fit_out.to_csv(OUT_FIT, index=False)
    print(f"\nFit saved: {OUT_FIT}")
    print(fit_out.to_string(index=False))


if __name__ == "__main__":
    main()
