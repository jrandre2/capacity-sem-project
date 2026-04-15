"""
State-clustered bootstrap inference for the reference two-factor SEM.

R7 critique: 200-iteration bootstrap is too few; expand to 1,000+ replications
and (where feasible) report alternative cluster structures.

Strategy:
- State-clustered bootstrap at n=1000: resample 35 unique state clusters with
  replacement, refit the reference SEM, record the four main structural paths.
- "Alternative cluster structure" for SEM is naturally limited because the
  cross-sectional unit is the administering jurisdiction (one row per
  jurisdiction). Disaster-context clustering would require collapsing
  jurisdictions onto a single dominant disaster, which conflicts with the
  multi-disaster portfolio aggregation already baked into the SEM input.
  The survival framework already reports grantee-clustered bootstrap. We
  therefore report the state-clustered SEM bootstrap at n=1000 and document
  the multiway / disaster-clustering limitation explicitly.

Output: data_work/diagnostics/sem_state_clustered_bootstrap_n1000.csv
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reproduce the reference two-factor SEM specification inline to avoid
# the kaifa_recovered_analysis.py module's `from config import` (which only
# resolves when src/ is on sys.path before stdlib).
TWO_FACTOR_MODEL_SPEC = """
# Reference two-factor SEM (matches kaifa_recovered_analysis._build_two_factor_model_spec)
AdminResources =~ z_avg_employment + z_avg_payroll
AdminBurdenCapacity =~ z_rev_programs_per_staff + z_rev_disasters_per_staff
RecoveryPerformance =~ z_Ratio_disbursed_to_obligated + z_Ratio_expended_to_disbursed + z_Ratio_obligated_funds_fully_expended + z_Ratio_Program_Completed
RecoveryTimeliness =~ z_rev_Duration_of_completion + z_rev_Average_Duration_Program_Completion
RecoveryPerformance ~ AdminResources + AdminBurdenCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
RecoveryTimeliness ~ AdminResources + AdminBurdenCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
AdminResources ~~ AdminBurdenCapacity
RecoveryPerformance ~~ RecoveryTimeliness
"""

try:
    from semopy import Model
except ImportError as exc:
    raise SystemExit("semopy required: pip install semopy") from exc

warnings.filterwarnings("ignore")

DATA_PATH = (
    PROJECT_ROOT
    / "data_work"
    / "diagnostics"
    / "kaifa_recovered_analysis"
    / "recovered_notebook_reference_573_sem_ready_dataset.csv"
)
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "sem_state_clustered_bootstrap_n1000.csv"

PATHS_OF_INTEREST = [
    ("RecoveryPerformance", "AdminResources"),
    ("RecoveryPerformance", "AdminBurdenCapacity"),
    ("RecoveryTimeliness", "AdminResources"),
    ("RecoveryTimeliness", "AdminBurdenCapacity"),
]

N_BOOTSTRAP = 1000
SEED = 42


def state_abbr(grantee: str) -> str:
    g = str(grantee)
    if re.fullmatch(r"[A-Z]{2}", g):
        return g
    parts = g.rsplit(",", 1)
    return parts[1].strip() if len(parts) == 2 else "UNKNOWN"


def fit_sem_paths(data: pd.DataFrame) -> dict:
    """Fit reference SEM on data, return standardized path coefficients."""
    model = Model(TWO_FACTOR_MODEL_SPEC)
    model.fit(data)
    estimates = model.inspect(std_est=True)
    out = {}
    for lval, rval in PATHS_OF_INTEREST:
        match = estimates[
            (estimates["lval"] == lval)
            & (estimates["op"] == "~")
            & (estimates["rval"] == rval)
        ]
        if len(match) == 1:
            out[(lval, rval)] = float(match["Est. Std"].iloc[0])
        else:
            out[(lval, rval)] = np.nan
    return out


def main():
    print(f"Loading SEM data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  N = {len(df)} jurisdictions")

    df["state_abbr"] = df["Grantee"].map(state_abbr)
    states = df["state_abbr"].unique()
    n_states = len(states)
    print(f"  N states (clusters) = {n_states}")

    # Original point estimates
    print("Fitting reference SEM on full sample for point estimates...")
    point = fit_sem_paths(df)
    for (lval, rval), v in point.items():
        print(f"  {lval} ~ {rval}: {v:+.4f}")

    # Bootstrap
    print(f"\nRunning state-clustered bootstrap (n={N_BOOTSTRAP}, seed={SEED})...")
    rng = np.random.default_rng(SEED)
    boot_records = {(lval, rval): [] for lval, rval in PATHS_OF_INTEREST}

    state_to_rows = {s: df[df["state_abbr"] == s].copy() for s in states}

    t0 = time.time()
    n_success = 0
    n_failed = 0
    for i in range(N_BOOTSTRAP):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_BOOTSTRAP - i - 1) / rate
            print(
                f"  Iter {i+1}/{N_BOOTSTRAP} | success={n_success} failed={n_failed} "
                f"| {rate:.1f} it/s | ETA {eta/60:.1f} min"
            )

        sampled_states = rng.choice(states, size=n_states, replace=True)
        boot_df = pd.concat([state_to_rows[s] for s in sampled_states], ignore_index=True)

        try:
            paths = fit_sem_paths(boot_df)
            for k, v in paths.items():
                boot_records[k].append(v)
            n_success += 1
        except Exception:
            n_failed += 1
            continue

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. success={n_success} failed={n_failed}")

    # Compute summary statistics
    rows = []
    for (lval, rval), values in boot_records.items():
        arr = np.array([v for v in values if not np.isnan(v)])
        rows.append(
            {
                "path_lval": lval,
                "path_op": "~",
                "path_rval": rval,
                "point_beta_std": point[(lval, rval)],
                "boot_mean": float(np.mean(arr)) if len(arr) else np.nan,
                "boot_se": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "boot_ci_lo_2_5": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                "boot_ci_hi_97_5": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                "n_success": len(arr),
                "n_attempted": N_BOOTSTRAP,
                "n_clusters": n_states,
                "cluster_var": "state_abbr",
            }
        )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
