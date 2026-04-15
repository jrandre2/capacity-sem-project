"""
Included-vs-excluded sample comparison (R10 M5).

R10 asks for a main-text table comparing the 573 SEM-retained jurisdictions
to the upstream-excluded ones. Full characterization of the 1,708 unresolved
ARO labels is not possible (that's the Reproducibility Boundary), but
characterizable subsets exist:

  - **Stage 4 city-of-type labels** (N=677): we know their state and the
    raw label; we can compute aggregate row counts and disaster contexts.
  - **Stage 5 institutional labels** (N=444): same.
  - **Stage 6 unresolved** (N=1,708): partial — we know F31-row
    counts and aggregate flow values per ARO.

This gives us a defensible "what was excluded?" picture even though the
full ARO→jurisdiction assignment is not recoverable.

Output: data_work/diagnostics/included_vs_excluded_comparison.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
F31_PATH = PROJECT_ROOT / "data_raw" / "F31 - QPR - Fin Data by Project, Activity and Quarter_20250214.csv"
LINKER_DETAIL = PROJECT_ROOT / "data_work" / "diagnostics" / "upstream_sample_flow_detail.csv"
SEM_INPUT = (
    PROJECT_ROOT / "manuscript_kaifa_archive" / "data" / "recovered_code_bundle"
    / "extracted" / "all_state_local_fund_latent_var_4_v2.csv"
)
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "included_vs_excluded_comparison.csv"


def main():
    print(f"Loading F31: {F31_PATH}")
    f31 = pd.read_csv(F31_PATH, low_memory=False)
    print(f"  {len(f31):,} rows; {f31['Activity Responsible Org'].nunique():,} distinct AROs")

    print(f"Loading linker detail: {LINKER_DETAIL}")
    linker = pd.read_csv(LINKER_DETAIL)

    print(f"Loading SEM input: {SEM_INPUT}")
    sem = pd.read_csv(SEM_INPUT)
    print(f"  {len(sem):,} SEM-included jurisdictions")

    # Merge linker stage info onto F31 rows
    merged = f31.merge(
        linker[["Activity Responsible Org", "stage", "state_abbr"]],
        on="Activity Responsible Org",
        how="left",
    )
    merged["stage"] = merged["stage"].fillna("unmatched")

    # Per-ARO summary: for each ARO, total F31 rows, total flow values, distinct disasters
    aro_summary = merged.groupby(["Activity Responsible Org", "stage"]).agg(
        n_rows=("Activity Responsible Org", "size"),
        n_disaster_contexts=("Disaster Type", "nunique"),
        n_distinct_states=("Grantee State", "nunique"),
        total_obligated_flow=("QPR Funds Obligated $", "sum"),
        total_expended_flow=("QPR Fund Expended $", "sum"),
    ).reset_index()

    # Stage-level aggregates
    stage_agg = aro_summary.groupby("stage").agg(
        n_aros=("Activity Responsible Org", "size"),
        median_rows_per_aro=("n_rows", "median"),
        mean_rows_per_aro=("n_rows", "mean"),
        median_obligated_per_aro=("total_obligated_flow", "median"),
        mean_obligated_per_aro=("total_obligated_flow", "mean"),
        median_disaster_contexts=("n_disaster_contexts", "median"),
        total_obligated_aggregate=("total_obligated_flow", "sum"),
    ).reset_index()
    print("\nStage-level summary:")
    print(stage_agg.to_string(index=False))

    # Now construct the included-vs-excluded comparison
    # Included = SEM jurisdictions (N=573 in complete-case; 577 raw)
    # Excluded = unresolved (Stage 6) + institutional (Stage 5) + city-of-type that didn't resolve to a SEM jurisdiction
    # For simplicity, treat:
    #   - Included F31-rows: rows where sem_grantee was assigned by build_full_fixed_horizon_panel.py linker
    #   - Excluded F31-rows: stage 5 and 6, plus stage 4 that didn't get city-to-county-resolved

    included_aros = set(linker[linker["stage"].isin(["3_direct_county_match", "4_city_needs_lookup"])]["Activity Responsible Org"])
    excluded_aros_5 = set(linker[linker["stage"] == "5_institutional"]["Activity Responsible Org"])
    excluded_aros_6 = set(linker[linker["stage"] == "6_unresolved"]["Activity Responsible Org"])

    summaries = {}
    for label, aros in [
        ("included_county_or_city", included_aros),
        ("excluded_institutional", excluded_aros_5),
        ("excluded_unresolved", excluded_aros_6),
    ]:
        sub = aro_summary[aro_summary["Activity Responsible Org"].isin(aros)]
        summaries[label] = {
            "n_aros": len(sub),
            "median_rows_per_aro": sub["n_rows"].median(),
            "mean_rows_per_aro": sub["n_rows"].mean(),
            "median_disaster_contexts": sub["n_disaster_contexts"].median(),
            "median_obligated_per_aro_$": sub["total_obligated_flow"].median(),
            "mean_obligated_per_aro_$": sub["total_obligated_flow"].mean(),
            "total_obligated_aggregate_$": sub["total_obligated_flow"].sum(),
        }

    out = pd.DataFrame(summaries).T.reset_index().rename(columns={"index": "category"})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print()
    print("Included vs Excluded comparison:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
