"""
Full-sample fixed-horizon panel for the 573 SEM jurisdictions (R9 M3 strong form).

The previous fixed-horizon analysis (Appendix C.8) was N=32--36 because it
merged jurisdiction-quarter data (~45 grantees) with the SEM input (N=573).
R9 asks for a full-sample censoring-aware design. The irrecoverable step is
the *original* Kaifa ARO-to-jurisdiction adjudication, but we now have the
final 573-jurisdiction crosswalk. Using the crosswalk as the target, we can
re-match F31 rows to SEM jurisdictions directly: a state-level jurisdiction
picks up F31 rows whose Grantee equals the 2-letter state code or state
name; a county-level local jurisdiction picks up F31 rows whose Grantee
State matches and whose normalized Activity Responsible Org contains the
county name (with parish/borough variants).

This is an approximation of the original pipeline — it will miss F31 rows
that the original pipeline assigned via manual adjudication that our
normalized-name matching cannot recover — but it produces a defensibly
larger fixed-horizon sample than N=32--36.

Output:
  data_work/diagnostics/fixed_horizon_full_sample_panel.parquet
  data_work/diagnostics/fixed_horizon_full_sample_outcomes.csv
  data_work/diagnostics/fixed_horizon_full_sample_regression.csv
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
F31_PATH = PROJECT_ROOT / "data_raw" / "F31 - QPR - Fin Data by Project, Activity and Quarter_20250214.csv"
CROSSWALK_PATH = PROJECT_ROOT / "data_work" / "replication" / "jurisdiction_crosswalk.csv"
SEM_INPUT_PATH = (
    PROJECT_ROOT / "data_work" / "diagnostics"
    / "kaifa_recovered_analysis"
    / "recovered_notebook_reference_573_sem_ready_dataset.csv"
)
OUT_PANEL = PROJECT_ROOT / "data_work" / "diagnostics" / "fixed_horizon_full_sample_panel.csv"
OUT_OUTCOMES = PROJECT_ROOT / "data_work" / "diagnostics" / "fixed_horizon_full_sample_outcomes.csv"
OUT_REG = PROJECT_ROOT / "data_work" / "diagnostics" / "fixed_horizon_full_sample_regression.csv"

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
    "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR",
    "Virgin Islands": "VI", "American Samoa": "AS",
}

warnings.filterwarnings("ignore")


def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_county_suffix(name: str) -> str:
    norm = normalize(name)
    for suffix in ("county", "parish", "borough", "census area", "municipality", "municipio"):
        if norm.endswith(" " + suffix):
            return norm[: -len(suffix)].strip()
    return norm


def parse_quarter_string(qs: str) -> tuple[int, int] | None:
    """Parse '2011 Q1' -> (2011, 1)."""
    if pd.isna(qs):
        return None
    m = re.match(r"(\d{4})\s*Q([1-4])", str(qs).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def quarter_to_index(year: int, q: int) -> int:
    """Convert (year, quarter) to continuous integer index."""
    return year * 4 + (q - 1)


def match_f31_to_jurisdiction(f31: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Assign each F31 row to a SEM jurisdiction where possible.

    Returns the F31 rows augmented with a `sem_grantee` column (the SEM
    Grantee label), plus `match_method` for transparency. Rows that cannot
    be matched are dropped.
    """
    f31 = f31.copy()
    f31["aro_norm"] = f31["Activity Responsible Org"].map(normalize)
    f31["aro_county_candidate"] = f31["Activity Responsible Org"].map(strip_county_suffix)
    f31["state_abbr"] = f31["Grantee State"].map(STATE_NAME_TO_ABBR)

    # State-agency matches: F31 rows where Grantee matches the SEM state-agency label
    state_grantees = crosswalk[crosswalk["state_level"] == 1]["Grantee"].tolist()
    state_to_grantee = {g: g for g in state_grantees}  # Grantee = 2-letter code

    # County-agency matches: (state_abbr, normalized county name) -> SEM grantee
    local_cw = crosswalk[crosswalk["state_level"] == 0].copy()
    local_cw["county_norm"] = local_cw["inferred_county_name"].map(strip_county_suffix)
    county_lookup = {
        (row["state_abbr"], row["county_norm"]): row["Grantee"]
        for _, row in local_cw.iterrows()
        if row["county_norm"]
    }

    # Match each F31 row
    sem_grantees = []
    methods = []
    for _, row in f31.iterrows():
        grantee = str(row.get("Grantee", ""))
        state = row.get("state_abbr")
        aro_cand = row.get("aro_county_candidate", "")

        # Try state-agency match (Grantee is 2-letter code that maps to a SEM state agency)
        if grantee in state_to_grantee:
            sem_grantees.append(state_to_grantee[grantee])
            methods.append("state_grantee")
            continue
        # Try state-name match (Grantee is a full state name like "Louisiana")
        if grantee in STATE_NAME_TO_ABBR and STATE_NAME_TO_ABBR[grantee] in state_to_grantee:
            sem_grantees.append(STATE_NAME_TO_ABBR[grantee])
            methods.append("state_grantee_fullname")
            continue
        # Try county match via ARO + state_abbr
        key = (state, aro_cand)
        if key in county_lookup:
            sem_grantees.append(county_lookup[key])
            methods.append("county_aro_match")
            continue
        # Try partial: aro contains the county name
        # (e.g., "St. Landry Parish Police Jury" -> "St. Landry Parish")
        match_found = False
        if state and aro_cand:
            for (cstate, cname), sem_g in county_lookup.items():
                if cstate == state and cname in aro_cand:
                    sem_grantees.append(sem_g)
                    methods.append("county_partial_match")
                    match_found = True
                    break
        if not match_found:
            sem_grantees.append(None)
            methods.append("unmatched")

    f31["sem_grantee"] = sem_grantees
    f31["match_method"] = methods
    matched = f31.dropna(subset=["sem_grantee"]).copy()
    print(f"  F31 rows matched: {len(matched):,} / {len(f31):,}")
    print(f"  Match method breakdown:")
    print(matched["match_method"].value_counts().to_string())
    print(f"  Unique SEM jurisdictions matched: {matched['sem_grantee'].nunique()}")
    return matched


def build_jurisdiction_quarter_panel(matched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate F31 rows to sem_grantee × quarter, using flows → cumsum.

    F31 "QPR Funds Obligated $", "QPR Fund Expended $", "QPR Grant Disbursed $"
    are QUARTERLY FLOWS (not cumulative). For each jurisdiction, we sum flows
    across activities within each quarter, then cumsum across quarters to
    obtain jurisdiction-level cumulative values at each quarter.
    """
    matched = matched.copy()
    matched[["year", "q"]] = matched["QPR Actual Quarter"].apply(
        lambda s: pd.Series(parse_quarter_string(s) or (None, None))
    )
    matched = matched.dropna(subset=["year", "q"])
    matched["year"] = matched["year"].astype(int)
    matched["q"] = matched["q"].astype(int)
    matched["q_index"] = matched.apply(lambda r: quarter_to_index(r["year"], r["q"]), axis=1)

    # Step 1: sum flows across activities within each jurisdiction-quarter
    flow = matched.groupby(["sem_grantee", "year", "q", "q_index"]).agg(
        obligated_flow=("QPR Funds Obligated $", "sum"),
        disbursed_flow=("QPR Grant Disbursed $", "sum"),
        expended_flow=("QPR Fund Expended $", "sum"),
        n_rows=("QPR Fund Expended $", "size"),
    ).reset_index().sort_values(["sem_grantee", "q_index"])

    # Step 2: cumsum within each jurisdiction across the chronologically ordered quarters
    flow["obligated"] = flow.groupby("sem_grantee")["obligated_flow"].cumsum()
    flow["disbursed"] = flow.groupby("sem_grantee")["disbursed_flow"].cumsum()
    flow["expended"] = flow.groupby("sem_grantee")["expended_flow"].cumsum()
    return flow


def compute_fixed_horizon_outcomes(panel: pd.DataFrame, horizons=(8, 12, 16)) -> pd.DataFrame:
    """For each jurisdiction, compute cumulative expended/obligated at q_0 + h quarters."""
    records = []
    for g, grp in panel.groupby("sem_grantee"):
        grp = grp.sort_values("q_index")
        # First quarter with positive obligations
        obl_pos = grp[grp["obligated"] > 0]
        if len(obl_pos) == 0:
            continue
        q0 = obl_pos["q_index"].min()
        final_obl = grp[grp["q_index"] == grp["q_index"].max()]["obligated"].iloc[0]
        rec = {"sem_grantee": g, "q0_index": q0, "n_quarters_observed": len(grp),
               "final_obligated": final_obl}
        for h in horizons:
            target = q0 + h
            # Find the row at or closest to (but not after) target
            avail = grp[grp["q_index"] <= target]
            if len(avail) == 0:
                rec[f"exp_share_q{h}"] = np.nan
                continue
            # Pick the row at the largest q_index ≤ target
            row = avail.loc[avail["q_index"].idxmax()]
            if row["obligated"] > 0:
                rec[f"exp_share_q{h}"] = row["expended"] / row["obligated"]
                # Observed at the target or before?
                rec[f"observed_at_q{h}"] = int(row["q_index"] >= target)
            else:
                rec[f"exp_share_q{h}"] = np.nan
                rec[f"observed_at_q{h}"] = 0
        records.append(rec)
    return pd.DataFrame(records)


def run_regression(outcomes: pd.DataFrame, sem: pd.DataFrame, horizons=(8, 12, 16)) -> pd.DataFrame:
    """Regress fixed-horizon outcome on SEM capacity indicators + covariates."""
    import statsmodels.api as sm

    merged = outcomes.merge(sem, left_on="sem_grantee", right_on="Grantee", how="inner")
    print(f"  Merged outcome + SEM rows: {len(merged)}")

    predictors = [
        "z_rev_programs_per_staff", "z_rev_disasters_per_staff",
        "z_avg_employment", "z_avg_payroll",
        "state_level", "z_E_TOTPOP",
        "z_SPL_THEME1", "z_SPL_THEME2", "z_SPL_THEME3", "z_SPL_THEME4",
    ]

    rows = []
    for h in horizons:
        outcome_col = f"exp_share_q{h}"
        # Only jurisdictions observed at the horizon
        obs_col = f"observed_at_q{h}"
        df = merged.dropna(subset=[outcome_col] + predictors).copy()
        if obs_col in df.columns:
            df = df[df[obs_col] == 1]
        n = len(df)
        if n < 20:
            print(f"  q={h}: N={n}, skipped (insufficient)")
            continue

        X = sm.add_constant(df[predictors])
        y = df[outcome_col]
        model = sm.OLS(y, X).fit()
        for var in predictors:
            rows.append({
                "horizon": h, "N": n, "r2": model.rsquared,
                "predictor": var,
                "b": model.params[var],
                "se": model.bse[var],
                "p": model.pvalues[var],
            })
        print(f"  q={h}: N={n}, R²={model.rsquared:.3f}, b(rev_programs_per_staff)={model.params['z_rev_programs_per_staff']:+.3f} (p={model.pvalues['z_rev_programs_per_staff']:.3f})")
    return pd.DataFrame(rows)


def main():
    print(f"Loading F31: {F31_PATH}")
    f31 = pd.read_csv(F31_PATH, low_memory=False)
    print(f"  {len(f31):,} rows")

    print(f"\nLoading crosswalk: {CROSSWALK_PATH}")
    crosswalk = pd.read_csv(CROSSWALK_PATH)
    print(f"  {len(crosswalk)} jurisdictions")

    print(f"\nMatching F31 rows to SEM jurisdictions...")
    matched = match_f31_to_jurisdiction(f31, crosswalk)

    print(f"\nBuilding jurisdiction-quarter panel...")
    panel = build_jurisdiction_quarter_panel(matched)
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PANEL, index=False)
    print(f"  Saved: {OUT_PANEL}")
    print(f"  Panel rows: {len(panel):,}; jurisdictions: {panel['sem_grantee'].nunique()}")

    print(f"\nComputing fixed-horizon outcomes (q=8, 12, 16)...")
    outcomes = compute_fixed_horizon_outcomes(panel)
    outcomes.to_csv(OUT_OUTCOMES, index=False)
    print(f"  Saved: {OUT_OUTCOMES}")
    print(f"  Jurisdictions with outcomes: {len(outcomes)}")
    for h in (8, 12, 16):
        n_obs = outcomes[f"observed_at_q{h}"].sum()
        mean = outcomes[outcomes[f"observed_at_q{h}"] == 1][f"exp_share_q{h}"].mean()
        print(f"    q={h}: N observed = {n_obs}, mean expenditure share = {mean:.3f}")

    print(f"\nLoading SEM input for regression covariates: {SEM_INPUT_PATH}")
    sem = pd.read_csv(SEM_INPUT_PATH)
    print(f"  {len(sem)} SEM-sample jurisdictions")

    print(f"\nRunning fixed-horizon regressions...")
    reg = run_regression(outcomes, sem)
    reg.to_csv(OUT_REG, index=False)
    print(f"  Saved: {OUT_REG}")

    print("\nKey coefficient: rev_programs_per_staff (capacity-timeliness path):")
    print(reg[reg["predictor"] == "z_rev_programs_per_staff"].to_string(index=False))


if __name__ == "__main__":
    main()
