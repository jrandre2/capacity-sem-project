"""
ε-sensitivity analysis for the workload-denominator offset (R10 M3).

R10 asks for explicit ε-sensitivity: the reference specification uses
ε = 10⁻⁶ in `avg_employment × E_TOTPOP + ε` to avoid division by zero
when QCEW employment is suppressed. ε is analytically consequential
because for 80.6% of the sample `avg_employment = 0`, so the denominator
collapses to ε × E_TOTPOP + ε ≈ ε, and the workload ratio becomes roughly
`Num_Program / ε`.

This script refits the reference two-factor SEM at several ε values
{10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²} and reports the four main structural paths.
The expected pattern: the coefficient amplifies as ε shrinks (because
the z-scored ratio becomes increasingly dominated by the suppressed-mass
mode, and Num_Program differences among suppressed jurisdictions become
the only variation).

Output: data_work/diagnostics/sem_epsilon_sensitivity.csv
"""

from __future__ import annotations

import re
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

RAW_CSV_PATH = (
    PROJECT_ROOT / "manuscript_kaifa_archive" / "data" / "recovered_code_bundle"
    / "extracted" / "all_state_local_fund_latent_var_4_v2.csv"
)
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "sem_epsilon_sensitivity.csv"

EPSILON_VALUES = [1e-3, 1e-6, 1e-9, 1e-12]

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


def prepare_data_with_epsilon(raw_df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    df = raw_df.copy()
    df["state_level"] = df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").astype(int)

    denominator = df["avg_employment"] * df["E_TOTPOP"] + epsilon
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
        data[f"z_{v}"] = (data[v] - m) / s if s > 0 else 0.0

    return data


def fit_extract(data: pd.DataFrame) -> dict:
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

    rows = []
    for eps in EPSILON_VALUES:
        print(f"\n[ε = {eps:.0e}] Refitting...")
        data = prepare_data_with_epsilon(raw, eps)
        paths = fit_extract(data)
        for (lval, rval), v in paths.items():
            rows.append({
                "epsilon": eps, "path_lval": lval, "path_rval": rval, "beta_std": v,
            })
        print(
            f"  RP←AR={paths[('RecoveryPerformance', 'AdminResources')]:+.3f}  "
            f"RT←ABC={paths[('RecoveryTimeliness', 'AdminBurdenCapacity')]:+.3f}"
        )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    piv = out.pivot_table(
        index="epsilon", columns=["path_lval", "path_rval"], values="beta_std"
    )
    print("\nε-sensitivity summary:")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
