"""
QCEW imputation-bounds sensitivity (R9 M2 substitute for Tobit).

R9 asks to treat QCEW NAICS 925110 zero values as BLS-suppressed rather than
literal zero. A Tobit-style censored-data SEM is not supported by semopy
without extensive custom work. As a bounds analysis, this script re-fits
the reference two-factor SEM with the 462 zero-QCEW jurisdictions imputed
at each of several plausible values consistent with BLS's suppression-floor
conventions (typically 1--5 employees per establishment; cell suppression
when the establishment count is below the disclosure threshold of 3 for
NAICS-industry-by-county). The coefficient range across imputations bounds
the reference estimate under different assumptions about what the zeros
really mean.

Imputation schema:
  0.00 — literal zero (status quo, baseline ε-offset specification)
  0.25 — very low: assume zeros = 0.25 employees per thousand population on average
  0.50 — low: 0.5
  1.00 — moderate
  2.50 — suppression floor: BLS suppresses when <3 establishments or <=10 jobs
  5.00 — state-level median suppression floor
  10.00 — aggressive

For each imputation, the SEM is refit on the full N=573 sample, and the four
main structural paths are extracted. The resulting bounds on the
capacity-timeliness coefficient indicate how much of the pooled ε-coded
β = 0.266 is attributable to the suppression handling.

Output: data_work/diagnostics/sem_qcew_bounds_sensitivity.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from semopy import Model
except ImportError:
    raise SystemExit("semopy required: pip install semopy")

warnings.filterwarnings("ignore")

# Raw (unstandardized) SEM-input dataset — we need to re-standardize after imputation
RAW_CSV_PATH = (
    PROJECT_ROOT / "manuscript_kaifa_archive" / "data" / "recovered_code_bundle"
    / "extracted" / "all_state_local_fund_latent_var_4_v2.csv"
)
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "sem_qcew_bounds_sensitivity.csv"

# Imputation values to scan (employment per thousand population)
IMPUTATION_VALUES = [0.00, 0.25, 0.50, 1.00, 2.50, 5.00, 10.00]

TWO_FACTOR_SPEC = """
AdminResources =~ z_avg_employment + z_avg_payroll
AdminBurdenCapacity =~ z_rev_programs_per_staff + z_rev_disasters_per_staff
RecoveryPerformance =~ z_Ratio_disbursed_to_obligated + z_Ratio_expended_to_disbursed + z_Ratio_obligated_funds_fully_expended + z_Ratio_Program_Completed
RecoveryTimeliness =~ z_rev_Duration_of_completion + z_rev_Average_Duration_Program_Completion
RecoveryPerformance ~ AdminResources + AdminBurdenCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
RecoveryTimeliness ~ AdminResources + AdminBurdenCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
AdminResources ~~ AdminBurdenCapacity
RecoveryPerformance ~~ RecoveryTimeliness
"""

PATHS_OF_INTEREST = [
    ("RecoveryPerformance", "AdminResources"),
    ("RecoveryPerformance", "AdminBurdenCapacity"),
    ("RecoveryTimeliness", "AdminResources"),
    ("RecoveryTimeliness", "AdminBurdenCapacity"),
]


def impute_and_prepare(raw_df: pd.DataFrame, qcew_imputation: float) -> pd.DataFrame:
    """Replace zero QCEW values with the imputation value, then standardize."""
    df = raw_df.copy()

    # Coerce Grantee-state marker
    import re
    df["state_level"] = df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").astype(int)

    # Impute zero QCEW with the specified value (both employment and payroll)
    zero_mask = df["avg_employment"] == 0
    n_zero = zero_mask.sum()
    if qcew_imputation > 0:
        df.loc[zero_mask, "avg_employment"] = qcew_imputation
        # Payroll imputation: proportional — use the observed payroll/employment ratio
        # among nonzero rows, multiplied by the imputed employment
        nonzero = df[~zero_mask].copy()
        if len(nonzero) > 0:
            pay_emp_ratio = (nonzero["avg_payroll"] / nonzero["avg_employment"]).median()
        else:
            pay_emp_ratio = 1000.0  # fallback
        df.loc[zero_mask, "avg_payroll"] = qcew_imputation * pay_emp_ratio

    # Rebuild workload ratios with the imputed denominators
    eps = 1e-6
    denominator = df["avg_employment"] * df["E_TOTPOP"] + eps
    df["programs_per_staff"] = df["Num_Program"] / denominator
    df["disasters_per_staff"] = df["Num_Disaster"] / denominator
    df["rev_programs_per_staff"] = -df["programs_per_staff"]
    df["rev_disasters_per_staff"] = -df["disasters_per_staff"]
    df["rev_Duration_of_completion"] = -df["Duration_of_completion"]
    df["rev_Average_Duration_Program_Completion"] = -df["Average_Duration_Program_Completion"]

    sem_vars = [
        "avg_employment", "avg_payroll",
        "rev_programs_per_staff", "rev_disasters_per_staff",
        "Ratio_disbursed_to_obligated", "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended", "Ratio_Program_Completed",
        "rev_Duration_of_completion", "rev_Average_Duration_Program_Completion",
        "state_level", "E_TOTPOP",
        "SPL_THEME1", "SPL_THEME2", "SPL_THEME3", "SPL_THEME4",
    ]
    data = df[["Grantee"] + sem_vars].dropna().copy()

    # Z-score all continuous variables
    continuous_vars = [
        "avg_employment", "avg_payroll",
        "rev_programs_per_staff", "rev_disasters_per_staff",
        "Ratio_disbursed_to_obligated", "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended", "Ratio_Program_Completed",
        "rev_Duration_of_completion", "rev_Average_Duration_Program_Completion",
        "E_TOTPOP", "SPL_THEME1", "SPL_THEME2", "SPL_THEME3", "SPL_THEME4",
    ]
    for v in continuous_vars:
        m = data[v].mean()
        s = data[v].std(ddof=0)
        if s > 0:
            data[f"z_{v}"] = (data[v] - m) / s
        else:
            data[f"z_{v}"] = 0.0

    return data, n_zero


def fit_and_extract(data: pd.DataFrame) -> dict:
    """Fit the two-factor SEM and return the four path coefficients."""
    model = Model(TWO_FACTOR_SPEC)
    model.fit(data)
    est = model.inspect(std_est=True)
    out = {}
    for lval, rval in PATHS_OF_INTEREST:
        match = est[(est["lval"] == lval) & (est["op"] == "~") & (est["rval"] == rval)]
        out[(lval, rval)] = float(match["Est. Std"].iloc[0]) if len(match) == 1 else np.nan
    return out


def main():
    print(f"Loading raw SEM input: {RAW_CSV_PATH}")
    raw = pd.read_csv(RAW_CSV_PATH)
    print(f"  {len(raw)} raw records")
    print(f"  avg_employment zero-count: {(raw['avg_employment'] == 0).sum()}")

    rows = []
    for val in IMPUTATION_VALUES:
        print(f"\n[Imputation = {val:.2f}] Rebuilding SEM input + refitting...")
        data, n_zero = impute_and_prepare(raw, val)
        print(f"  N jurisdictions: {len(data)}; imputed: {n_zero}")
        try:
            paths = fit_and_extract(data)
            for (lval, rval), v in paths.items():
                rows.append({
                    "qcew_imputation": val,
                    "n_imputed": n_zero,
                    "n_total": len(data),
                    "path_lval": lval,
                    "path_rval": rval,
                    "beta_std": v,
                })
            print(
                f"  RP←AR={paths[('RecoveryPerformance', 'AdminResources')]:+.3f}  "
                f"RT←ABC={paths[('RecoveryTimeliness', 'AdminBurdenCapacity')]:+.3f}"
            )
        except Exception as e:
            print(f"  Failed: {e}")
            for (lval, rval) in PATHS_OF_INTEREST:
                rows.append({
                    "qcew_imputation": val,
                    "n_imputed": n_zero,
                    "n_total": len(data),
                    "path_lval": lval,
                    "path_rval": rval,
                    "beta_std": np.nan,
                })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print()

    # Pivot for a cleaner display: rows = imputation, cols = path
    piv = out.pivot_table(
        index="qcew_imputation", columns=["path_lval", "path_rval"], values="beta_std"
    )
    print("Bounds summary:")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))
    print()

    # Print focused range for the two key paths
    burden_timeliness = out[
        (out["path_lval"] == "RecoveryTimeliness")
        & (out["path_rval"] == "AdminBurdenCapacity")
    ]
    res_perf = out[
        (out["path_lval"] == "RecoveryPerformance")
        & (out["path_rval"] == "AdminResources")
    ]
    print(f"Burden → Timeliness β range across QCEW imputations:")
    print(f"  min = {burden_timeliness['beta_std'].min():+.3f}")
    print(f"  max = {burden_timeliness['beta_std'].max():+.3f}")
    print(f"Resources → Performance β range across QCEW imputations:")
    print(f"  min = {res_perf['beta_std'].min():+.3f}")
    print(f"  max = {res_perf['beta_std'].max():+.3f}")


if __name__ == "__main__":
    main()
