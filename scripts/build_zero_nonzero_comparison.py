"""
Zero-QCEW vs non-suppressed-QCEW descriptive comparison (R10 M3 transportability).

R10 asks: if substantive claims rest on the non-suppressed-QCEW subsample
(N=111), we need to show whether that subsample is transportable to the
broader target population of CDBG-DR administering jurisdictions. If the
two groups differ systematically on observable characteristics, then
non-suppressed-subsample findings cannot be straightforwardly generalized.

This script compares, on the 573-jurisdiction SEM sample, the 462 zero-QCEW
and 111 non-suppressed-QCEW jurisdictions on:
  - Population (E_TOTPOP, mean and median)
  - Government type (% state, % local)
  - Portfolio size (Num_Program, Num_Disaster)
  - Maturity (Duration_of_completion; Average_Duration_Program_Completion)
  - Financial outcomes (Ratio_disbursed_to_obligated, etc.)
  - SVI themes (means across the 4 themes)

Output: data_work/diagnostics/zero_nonzero_qcew_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = (
    PROJECT_ROOT / "manuscript_kaifa_archive" / "data" / "recovered_code_bundle"
    / "extracted" / "all_state_local_fund_latent_var_4_v2.csv"
)
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "zero_nonzero_qcew_comparison.csv"


def main():
    print(f"Loading: {RAW_CSV}")
    raw = pd.read_csv(RAW_CSV)
    import re
    raw["state_level"] = raw["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").astype(int)

    # Identify complete-case SEM sample (same columns as the 573-row file)
    needed_cols = [
        "Grantee", "avg_employment", "avg_payroll",
        "Num_Program", "Num_Disaster",
        "Duration_of_completion", "Average_Duration_Program_Completion",
        "Ratio_disbursed_to_obligated", "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended", "Ratio_Program_Completed",
        "E_TOTPOP", "SPL_THEME1", "SPL_THEME2", "SPL_THEME3", "SPL_THEME4",
    ]
    df = raw[["state_level"] + needed_cols].dropna().copy()
    print(f"  N (complete-case) = {len(df)}")

    # Identify non-suppressed QCEW subsample
    df["qcew_nonzero"] = (df["avg_employment"] > 0).astype(int)
    print(f"  N non-suppressed QCEW (avg_employment > 0): {df['qcew_nonzero'].sum()}")
    print(f"  N zero-QCEW: {(df['qcew_nonzero'] == 0).sum()}")

    # Variables to compare
    comparison_vars = {
        "E_TOTPOP": "Population (thousands)",
        "state_level": "State-agency fraction",
        "Num_Program": "Number of CDBG-DR programs",
        "Num_Disaster": "Number of disaster contexts",
        "Duration_of_completion": "Duration of completion (months)",
        "Average_Duration_Program_Completion": "Avg program-completion duration (months)",
        "Ratio_disbursed_to_obligated": "Disbursed/obligated ratio",
        "Ratio_expended_to_disbursed": "Expended/disbursed ratio",
        "Ratio_obligated_funds_fully_expended": "Fully-expended share",
        "Ratio_Program_Completed": "Program-completion share",
        "SPL_THEME1": "SVI Theme 1 (socioeconomic)",
        "SPL_THEME2": "SVI Theme 2 (household comp.)",
        "SPL_THEME3": "SVI Theme 3 (minority/language)",
        "SPL_THEME4": "SVI Theme 4 (housing/transport)",
    }

    rows = []
    zero_df = df[df["qcew_nonzero"] == 0]
    nonzero_df = df[df["qcew_nonzero"] == 1]

    for var, label in comparison_vars.items():
        z_mean = zero_df[var].mean()
        z_med = zero_df[var].median()
        n_mean = nonzero_df[var].mean()
        n_med = nonzero_df[var].median()
        # Welch's t-test
        try:
            t, p = stats.ttest_ind(nonzero_df[var], zero_df[var], equal_var=False, nan_policy="omit")
        except Exception:
            t, p = np.nan, np.nan
        # Standardized mean difference (Cohen's d with pooled SD)
        z_sd = zero_df[var].std(ddof=1)
        n_sd = nonzero_df[var].std(ddof=1)
        pooled_sd = np.sqrt((z_sd**2 + n_sd**2) / 2) if np.isfinite(z_sd) and np.isfinite(n_sd) else np.nan
        d = (n_mean - z_mean) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan
        rows.append({
            "variable": var,
            "label": label,
            "zero_mean": z_mean, "zero_median": z_med, "zero_n": len(zero_df),
            "nonzero_mean": n_mean, "nonzero_median": n_med, "nonzero_n": len(nonzero_df),
            "diff_mean": n_mean - z_mean,
            "cohens_d": d,
            "welch_t": t, "welch_p": p,
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print()

    # Print a readable table
    display_cols = ["label", "zero_mean", "nonzero_mean", "cohens_d", "welch_p"]
    display = out[display_cols].copy()
    display.columns = ["Variable", "Zero-QCEW mean", "Non-suppressed mean", "Cohen's d", "Welch p"]
    print(display.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else str(v)))
    print()
    print("Key differences (|d| > 0.3):")
    big = out[out["cohens_d"].abs() > 0.3][display_cols]
    if len(big):
        big.columns = ["Variable", "Zero-QCEW mean", "Non-suppressed mean", "Cohen's d", "Welch p"]
        print(big.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else str(v)))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
