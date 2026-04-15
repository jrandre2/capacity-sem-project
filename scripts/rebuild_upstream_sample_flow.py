"""
Reconstruct per-stage upstream sample-flow counts for Appendix A.5.

R8 M4a: the prior A.5 table used "~2,400 / ~500 / ~100 / ~150" approximate
values because the original Kaifa pipeline's per-stage row counts were not
preserved. This script reconstructs the linking pipeline using the same raw
F31 file and the 2022 CDC/ATSDR SVI county reference as the authoritative
county list (3,143 US counties plus DC and territories).

Reconstruction is approximate because the original pipeline's manual
adjudication and city-to-county lookup tables are not recoverable. However,
this script logs EXACT counts at each decidable stage:

  Stage 1 — Raw F31: total distinct Activity Responsible Org labels.
  Stage 2 — State-agency profiles: grantees whose Grantee field is a
    two-letter state code OR matches a US state name. Each becomes one profile.
  Stage 3 — Direct Census county match: local Activity Responsible Org labels
    whose normalized name matches a Census county (with suffix handling for
    Parish/Borough/Census Area/Municipio/Municipality).
  Stage 4 — City-label requiring lookup: Activity Responsible Org labels of
    the form "City of X" or "X, ST" that do not match a county name directly.
  Stage 5 — Institutional / non-geographic: universities, non-profits,
    authorities, consortia, housing agencies.
  Stage 6 — Unresolved: everything else.

Output: data_work/diagnostics/upstream_sample_flow.csv

This reconstruction is documented in the manuscript as an approximation of
the original pipeline, not a perfect replication.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
F31_PATH = PROJECT_ROOT / "data_raw" / "F31 - QPR - Fin Data by Project, Activity and Quarter_20250214.csv"
COUNTY_REF_PATH = PROJECT_ROOT / "data_raw" / "svi_historical" / "SVI2022_US_COUNTY.csv"
OUT_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "upstream_sample_flow.csv"
DETAIL_PATH = PROJECT_ROOT / "data_work" / "diagnostics" / "upstream_sample_flow_detail.csv"

STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
    # Territories
    "American Samoa": "AS", "Puerto Rico": "PR", "Virgin Islands": "VI",
    "Northern Mariana Islands": "MP", "Guam": "GU",
}

COUNTY_SUFFIXES = [
    "county", "parish", "borough", "census area", "municipality",
    "municipio", "city and borough",
]

LOCAL_GOVT_SUFFIXES = [
    "police jury", "government", "council", "board of supervisors",
    "commission", "consolidated government", "metropolitan government",
    "city-county", "unified government",
]

INSTITUTIONAL_KEYWORDS = [
    "university", "college", "institute", "research",
    "authority", "agency", "department of", "division of",
    "non-profit", "habitat for humanity", "salvation army", "red cross",
    "llc", "inc.", "corp.", "corporation", "consortium", "coalition",
    "archives", "mental health", "development finance",
    "housing authority", "housing agency", "redevelopment", "community action",
    "transit", "public works",
]


def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_county_suffix(name: str) -> str:
    """Return the bare county name with the geographic suffix removed."""
    norm = normalize(name)
    for suffix in sorted(COUNTY_SUFFIXES, key=len, reverse=True):
        if norm.endswith(" " + suffix):
            return norm[: -len(suffix)].strip()
    return norm


def load_county_reference() -> pd.DataFrame:
    ref = pd.read_csv(COUNTY_REF_PATH, low_memory=False)
    ref = ref[["ST_ABBR", "STATE", "COUNTY", "FIPS"]].copy()
    ref["county_norm"] = ref["COUNTY"].map(strip_county_suffix)
    ref["state_norm"] = ref["ST_ABBR"].str.upper()
    return ref


def classify_label(label: str, state: str, county_ref: pd.DataFrame) -> str:
    """Classify a single Activity Responsible Org label into a stage."""
    if pd.isna(label) or not str(label).strip():
        return "6_unresolved"

    norm = normalize(label)

    # Stage 2: State-agency profile
    # (Grantees matching two-letter code or full state name are handled at grantee level, not ARO level)

    # Stage 5: Institutional / non-geographic (check before county match to avoid
    # false positives like "Washington University")
    for kw in INSTITUTIONAL_KEYWORDS:
        if kw in norm:
            return "5_institutional"

    # Stage 3: Direct county name match within the grantee state
    if state and state in county_ref["state_norm"].values:
        state_counties = county_ref[county_ref["state_norm"] == state]
        candidate = strip_county_suffix(label)
        # Also try dropping local-govt suffixes
        for suffix in sorted(LOCAL_GOVT_SUFFIXES, key=len, reverse=True):
            if candidate.endswith(" " + suffix):
                candidate = candidate[: -len(suffix)].strip()
                break
        # Drop "city of" / "town of" / "village of" prefixes for city→county lookup later
        is_city = bool(re.match(r"^(city|town|village) of ", candidate))
        if is_city:
            return "4_city_needs_lookup"
        # Direct county match (with or without suffix)
        if candidate in state_counties["county_norm"].values:
            return "3_direct_county_match"
        # Try without common prefix "county of"
        if candidate.startswith("county of "):
            bare = candidate.replace("county of ", "").strip()
            if bare in state_counties["county_norm"].values:
                return "3_direct_county_match"
        # Check for ", ST" suffix pattern (e.g., "Houston, TX" → city)
        if re.search(r",\s*[a-z]{2}$", norm):
            return "4_city_needs_lookup"

    # Fall-through: unresolved
    return "6_unresolved"


def main():
    print(f"Loading F31: {F31_PATH}")
    f31 = pd.read_csv(F31_PATH, low_memory=False)
    print(f"  {len(f31):,} rows")

    # Unique Activity Responsible Org × Grantee State pairs
    aro_state = (
        f31[["Activity Responsible Org", "Grantee State", "Grantee"]]
        .dropna(subset=["Activity Responsible Org"])
        .drop_duplicates()
        .copy()
    )
    print(f"  {len(aro_state):,} distinct ARO×state pairs")

    # Normalize
    aro_state["state_abbr"] = aro_state["Grantee State"].map(STATE_NAME_TO_ABBR)
    aro_state["aro_norm"] = aro_state["Activity Responsible Org"].map(normalize)

    # Identify state-grantee profiles: Grantee is a 2-letter code OR a state name
    state_grantees = set()
    for g in f31["Grantee"].dropna().unique():
        g_str = str(g)
        if re.fullmatch(r"[A-Z]{2}", g_str):
            state_grantees.add(g_str)
        elif g_str in STATE_NAME_TO_ABBR:
            state_grantees.add(STATE_NAME_TO_ABBR[g_str])

    print(f"\n[Stage 2] State-agency grantees: {len(state_grantees)}")
    print(f"  States: {sorted(state_grantees)}")

    # Load county reference
    county_ref = load_county_reference()
    print(f"\n  County reference: {len(county_ref):,} US counties")

    # Classify ARO-level labels
    # For ARO-level classification, use the grantee state
    distinct_aro = (
        aro_state[["Activity Responsible Org", "state_abbr", "aro_norm"]]
        .drop_duplicates(subset=["Activity Responsible Org"])
        .reset_index(drop=True)
    )
    print(f"\nClassifying {len(distinct_aro):,} distinct Activity Responsible Org labels...")
    distinct_aro["stage"] = distinct_aro.apply(
        lambda row: classify_label(row["Activity Responsible Org"], row["state_abbr"], county_ref),
        axis=1,
    )

    stage_counts = distinct_aro["stage"].value_counts().sort_index()
    print("\nARO-level stage breakdown:")
    for stage, n in stage_counts.items():
        print(f"  {stage}: {n:,}")
    print(f"  Total distinct ARO labels: {len(distinct_aro):,}")

    # Write detail file
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    distinct_aro.to_csv(DETAIL_PATH, index=False)
    print(f"\nDetail written: {DETAIL_PATH}")

    # Summary file
    summary_rows = [
        {
            "stage": "1_raw_aro",
            "count": len(distinct_aro),
            "description": "Distinct Activity Responsible Org labels in raw F31",
        },
        {
            "stage": "2_state_agency_grantees",
            "count": len(state_grantees),
            "description": "Distinct state-agency profiles (Grantee = 2-letter code or state name)",
        },
        {
            "stage": "3_direct_county_match",
            "count": int(stage_counts.get("3_direct_county_match", 0)),
            "description": "ARO labels resolving directly to a Census county/parish/borough (normalized)",
        },
        {
            "stage": "4_city_needs_lookup",
            "count": int(stage_counts.get("4_city_needs_lookup", 0)),
            "description": "ARO labels of the form 'City/Town/Village of X' or 'X, ST' requiring city-to-county crosswalk",
        },
        {
            "stage": "5_institutional",
            "count": int(stage_counts.get("5_institutional", 0)),
            "description": "Institutional labels (universities, authorities, non-profits, LLC/Inc., consortia)",
        },
        {
            "stage": "6_unresolved",
            "count": int(stage_counts.get("6_unresolved", 0)),
            "description": "ARO labels not resolved by automated rules; would require manual adjudication",
        },
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_PATH, index=False)
    print(f"Summary written: {OUT_PATH}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Note: The final SEM sample (N=577 recovered, N=573 after complete-case) is smaller")
    print("than Stage 3 + Stage 2 because (a) city-to-county lookup and manual adjudication")
    print("aggregate multiple AROs to the same county-profile, and (b) complete-case deletion")
    print("on SEM indicators drops 4 additional profiles. This reconstruction does not claim")
    print("to perfectly reproduce the original Kaifa pipeline's aggregation logic.")


if __name__ == "__main__":
    main()
