"""
Export the 573-jurisdiction crosswalk as a replication artifact (R9 M7).

R9 reviewer: "Ideally the final jurisdiction crosswalk used for the 573-case
SEM sample should be deposited directly."

This script reads the SEM-ready dataset and the upstream linker output,
assembles the minimum crosswalk needed to reproduce the downstream SEM /
survival analyses, and writes it to:
  data_work/replication/jurisdiction_crosswalk.csv

Crosswalk columns:
  Grantee: SEM jurisdiction label (e.g., "AK", "Cole County, MO")
  state_level: 1 if state agency, 0 if local jurisdiction
  state_abbr: 2-letter state code
  inferred_geography: 'state' | 'county' | 'other'
  inferred_county_name: county/parish/borough label where applicable (may be NA)
  n_f31_aro_matched: approximate number of F31 Activity Responsible Org labels
                     that plausibly aggregated to this jurisdiction (from the
                     linker's decidable matches; does NOT include the manual
                     adjudication inherited from the upstream Kaifa pipeline,
                     which is not recoverable)
  notes: descriptive flag (e.g., "2-letter state code match", "county-name match")
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEM_PATH = (
    PROJECT_ROOT / "data_work" / "diagnostics"
    / "kaifa_recovered_analysis"
    / "recovered_notebook_reference_573_sem_ready_dataset.csv"
)
LINKER_DETAIL = PROJECT_ROOT / "data_work" / "diagnostics" / "upstream_sample_flow_detail.csv"
OUT_PATH = PROJECT_ROOT / "data_work" / "replication" / "jurisdiction_crosswalk.csv"


def parse_grantee(label: str) -> tuple[str, str, str]:
    """Parse a SEM Grantee label into (state_abbr, geography_type, county_name)."""
    s = str(label).strip()
    if re.fullmatch(r"[A-Z]{2}", s):
        return s, "state", ""
    parts = s.rsplit(",", 1)
    if len(parts) == 2:
        name = parts[0].strip()
        state = parts[1].strip()
        if re.fullmatch(r"[A-Z]{2}", state):
            return state, "county", name
    return "UNKNOWN", "other", s


def main():
    print(f"Loading SEM-ready dataset: {SEM_PATH}")
    sem = pd.read_csv(SEM_PATH)
    print(f"  {len(sem)} jurisdictions")

    records = []
    for _, row in sem.iterrows():
        grantee = row["Grantee"]
        state_abbr, geo_type, county = parse_grantee(grantee)
        records.append({
            "Grantee": grantee,
            "state_level": int(row["state_level"]),
            "state_abbr": state_abbr,
            "inferred_geography": geo_type,
            "inferred_county_name": county,
            "notes": (
                "2-letter state code" if geo_type == "state"
                else "county/parish/borough name, state" if geo_type == "county"
                else "non-standard label"
            ),
        })

    crosswalk = pd.DataFrame(records)

    # Optional: count F31 ARO labels that plausibly map to this jurisdiction
    # (decidable matches only — not including upstream manual adjudication)
    if LINKER_DETAIL.exists():
        print(f"Loading linker detail: {LINKER_DETAIL}")
        linker = pd.read_csv(LINKER_DETAIL)
        # For state-agency rows, count ARO labels assigned to that state
        state_counts = (
            linker[linker["state_abbr"].notna()]
            .groupby("state_abbr")
            .size()
            .rename("aro_label_count_state")
            .to_frame()
            .reset_index()
        )
        crosswalk = crosswalk.merge(state_counts, on="state_abbr", how="left")
        crosswalk["aro_label_count_state"] = (
            crosswalk["aro_label_count_state"].fillna(0).astype(int)
        )
        print(f"  Mean ARO label count per state: {crosswalk['aro_label_count_state'].mean():.1f}")
    else:
        print(f"  Linker detail not found at {LINKER_DETAIL}; skipping ARO counts")
        crosswalk["aro_label_count_state"] = -1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(crosswalk)} rows to {OUT_PATH}")
    print()
    print("Sample rows:")
    print(crosswalk.head(10).to_string(index=False))
    print()
    print(f"State agencies: {(crosswalk['state_level'] == 1).sum()}")
    print(f"Local jurisdictions: {(crosswalk['state_level'] == 0).sum()}")
    print(f"Distinct states covered: {crosswalk['state_abbr'].nunique()}")


if __name__ == "__main__":
    main()
