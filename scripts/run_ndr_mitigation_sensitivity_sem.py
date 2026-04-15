"""
NDR/Mitigation exclusion sensitivity for the reference two-factor SEM.

R7 minor #6: "Report a sensitivity excluding the 2013 National Disaster
Resilience and 2015–2018 Mitigation program allocations, since those are
not disaster-event allocations in the same sense as the others."

Approach (partial sensitivity due to upstream-aggregation limitation):
- The SEM-ready dataset is already aggregated to the administering-jurisdiction
  level (573 profiles). Dropping NDR/MIT *rows* and re-aggregating from the
  raw F31 file would require rebuilding the upstream geography-matching
  pipeline. As a tractable substitute, this script identifies SEM
  jurisdictions located in states with substantial (>25%) NDR/MIT exposure
  in the F31 data, drops them, and re-fits the SEM. It also runs a stricter
  variant dropping any state with ANY NDR/MIT exposure.

This is a partial sensitivity. A complete check would re-run upstream
aggregation excluding NDR/MIT rows; that is documented as future work.

Output: data_work/diagnostics/sem_ndr_mitigation_sensitivity.csv
"""

from __future__ import annotations

import re
import sys
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

SEM_DATA_PATH = (
    PROJECT_ROOT
    / "data_work"
    / "diagnostics"
    / "kaifa_recovered_analysis"
    / "recovered_notebook_reference_573_sem_ready_dataset.csv"
)
F31_PATH = PROJECT_ROOT / "data_raw" / "F31 - QPR - Fin Data by Project, Activity and Quarter_20250214.csv"
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "sem_ndr_mitigation_sensitivity.csv"

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

PATHS_OF_INTEREST = [
    ("RecoveryPerformance", "AdminResources"),
    ("RecoveryPerformance", "AdminBurdenCapacity"),
    ("RecoveryTimeliness", "AdminResources"),
    ("RecoveryTimeliness", "AdminBurdenCapacity"),
]

# Map full state name (used in F31 'Grantee State') to 2-letter abbreviation
STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR", "Virgin Islands": "VI",
    "American Samoa": "AS", "Northern Mariana Islands": "MP", "Hawaii": "HI",
}


def state_abbr(grantee: str) -> str:
    g = str(grantee)
    if re.fullmatch(r"[A-Z]{2}", g):
        return g
    parts = g.rsplit(",", 1)
    return parts[1].strip() if len(parts) == 2 else "UNKNOWN"


def fit_sem_paths(data: pd.DataFrame) -> tuple[dict, dict]:
    """Return (path coefficients, fit stats)."""
    model = Model(TWO_FACTOR_MODEL_SPEC)
    model.fit(data)
    estimates = model.inspect(std_est=True)
    paths = {}
    for lval, rval in PATHS_OF_INTEREST:
        match = estimates[
            (estimates["lval"] == lval) & (estimates["op"] == "~") & (estimates["rval"] == rval)
        ]
        paths[(lval, rval)] = float(match["Est. Std"].iloc[0]) if len(match) == 1 else np.nan

    try:
        from semopy import calc_stats
        fit = calc_stats(model)
        fit_d = {
            "CFI": float(fit.loc["Value", "CFI"]) if "CFI" in fit.columns else np.nan,
            "TLI": float(fit.loc["Value", "TLI"]) if "TLI" in fit.columns else np.nan,
            "RMSEA": float(fit.loc["Value", "RMSEA"]) if "RMSEA" in fit.columns else np.nan,
        }
    except Exception:
        fit_d = {"CFI": np.nan, "TLI": np.nan, "RMSEA": np.nan}

    return paths, fit_d


def main():
    print(f"Loading SEM data: {SEM_DATA_PATH}")
    sem = pd.read_csv(SEM_DATA_PATH)
    sem["state_abbr"] = sem["Grantee"].map(state_abbr)
    print(f"  N = {len(sem)} jurisdictions; {sem['state_abbr'].nunique()} states")

    print(f"\nLoading F31: {F31_PATH}")
    f31 = pd.read_csv(F31_PATH, low_memory=False)
    f31["is_non_event"] = f31["Appropriation"].isin(["2013 NDR", "MIT"])

    # Per-state NDR/MIT exposure
    state_summary = f31.groupby("Grantee State").agg(
        n_total=("is_non_event", "size"),
        n_non_event=("is_non_event", "sum"),
    ).reset_index()
    state_summary["share"] = state_summary["n_non_event"] / state_summary["n_total"]
    state_summary["state_abbr"] = state_summary["Grantee State"].map(STATE_NAME_TO_ABBR)
    state_summary = state_summary.dropna(subset=["state_abbr"])

    # Defining exposure tiers
    states_with_any_exposure = set(
        state_summary[state_summary["n_non_event"] > 0]["state_abbr"]
    )
    states_with_substantial_exposure = set(
        state_summary[state_summary["share"] > 0.25]["state_abbr"]
    )
    print(f"  States with ANY NDR/MIT exposure: {len(states_with_any_exposure)}")
    print(f"  States with >25% NDR/MIT row share: {len(states_with_substantial_exposure)}")
    print(f"    > Substantial exposure: {sorted(states_with_substantial_exposure)}")

    # Reference fit
    print("\n[Reference] Full sample (N=573):")
    ref_paths, ref_fit = fit_sem_paths(sem)
    for k, v in ref_paths.items():
        print(f"  {k[0]} ~ {k[1]}: {v:+.4f}")
    print(f"  Fit: CFI={ref_fit['CFI']:.3f} TLI={ref_fit['TLI']:.3f} RMSEA={ref_fit['RMSEA']:.3f}")

    # Sensitivity 1: drop states with >25% NDR/MIT exposure
    keep_substantial = ~sem["state_abbr"].isin(states_with_substantial_exposure)
    sub_data = sem[keep_substantial].copy()
    print(f"\n[Sensitivity 1] Drop states >25% NDR/MIT exposure (N={len(sub_data)}):")
    sub_paths, sub_fit = fit_sem_paths(sub_data)
    for k, v in sub_paths.items():
        print(f"  {k[0]} ~ {k[1]}: {v:+.4f}")
    print(f"  Fit: CFI={sub_fit['CFI']:.3f} TLI={sub_fit['TLI']:.3f} RMSEA={sub_fit['RMSEA']:.3f}")

    # Sensitivity 2: drop states with ANY NDR/MIT exposure
    keep_any = ~sem["state_abbr"].isin(states_with_any_exposure)
    any_data = sem[keep_any].copy()
    print(f"\n[Sensitivity 2] Drop states with ANY NDR/MIT exposure (N={len(any_data)}):")
    any_paths, any_fit = fit_sem_paths(any_data)
    for k, v in any_paths.items():
        print(f"  {k[0]} ~ {k[1]}: {v:+.4f}")
    print(f"  Fit: CFI={any_fit['CFI']:.3f} TLI={any_fit['TLI']:.3f} RMSEA={any_fit['RMSEA']:.3f}")

    # Sensitivity 3: drop only state-agency rows with substantial exposure (locals retained)
    keep_partial = ~((sem["state_level"] == 1) & sem["state_abbr"].isin(states_with_substantial_exposure))
    partial_data = sem[keep_partial].copy()
    print(f"\n[Sensitivity 3] Drop only state-agency profiles with >25% NDR/MIT exposure (N={len(partial_data)}):")
    partial_paths, partial_fit = fit_sem_paths(partial_data)
    for k, v in partial_paths.items():
        print(f"  {k[0]} ~ {k[1]}: {v:+.4f}")
    print(f"  Fit: CFI={partial_fit['CFI']:.3f} TLI={partial_fit['TLI']:.3f} RMSEA={partial_fit['RMSEA']:.3f}")

    rows = []
    for label, paths, fit, n in [
        ("reference_full_sample", ref_paths, ref_fit, len(sem)),
        ("drop_states_substantial_ndrmit_exposure", sub_paths, sub_fit, len(sub_data)),
        ("drop_states_any_ndrmit_exposure", any_paths, any_fit, len(any_data)),
        ("drop_state_agencies_only_substantial_exposure", partial_paths, partial_fit, len(partial_data)),
    ]:
        for (lval, rval), v in paths.items():
            rows.append(
                {
                    "specification": label,
                    "n": n,
                    "path_lval": lval,
                    "path_rval": rval,
                    "beta_std": v,
                    "CFI": fit["CFI"],
                    "TLI": fit["TLI"],
                    "RMSEA": fit["RMSEA"],
                }
            )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
