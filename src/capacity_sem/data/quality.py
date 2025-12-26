"""
Data quality utilities for QPR datasets.

These helpers produce a cleaned dataset alongside row-level quality flags
without modifying the original raw export.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.date_utils import quarter_to_date


FIPS_STATE_CODES = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "60": "American Samoa",
    "66": "Guam",
    "69": "Northern Mariana Islands",
    "72": "Puerto Rico",
    "78": "Virgin Islands",
}

STATE_NAMES = set(FIPS_STATE_CODES.values())

GRANT_CODE_RE = re.compile(r"^[A-Z]-\d{2}-[A-Z]{2}-(\d{2})-\d{4}$")


def parse_grant_state_code(grant: Optional[str]) -> Optional[str]:
    """Extract the state/territory code from a HUD grant identifier."""
    if not isinstance(grant, str):
        return None
    match = GRANT_CODE_RE.match(grant.strip())
    if not match:
        return None
    return match.group(1)


def clean_qpr_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a cleaned QPR dataset with row-level quality flags.

    Returns a tuple of (clean_df, quality_report_df).
    """
    df = df.copy()

    # Preserve original state values before imputation.
    if "Grantee State" in df.columns:
        df["Grantee State Raw"] = df["Grantee State"]

    if "Grant" in df.columns:
        df["Grant_State_Code"] = df["Grant"].apply(parse_grant_state_code)
        df["Grant_State_Name"] = df["Grant_State_Code"].map(FIPS_STATE_CODES)

    if "Grantee State" in df.columns:
        if "Grant_State_Name" in df.columns:
            df["Grantee State"] = df["Grantee State"].fillna(df["Grant_State_Name"])
        source = pd.Series(index=df.index, dtype="object")
        source = source.mask(df["Grantee State Raw"].notna(), "raw")
        source = source.mask(source.isna() & df["Grantee State"].notna(), "grant_code")
        df["Grantee State Source"] = source

    # Quarter parsing and flags.
    if "QPR Actual Quarter" in df.columns:
        df["QPR_Date"] = df["QPR Actual Quarter"].apply(quarter_to_date)
        df["QA_missing_qpr_actual_quarter"] = df["QPR Actual Quarter"].isna()
        df["QA_invalid_qpr_actual_quarter"] = (
            df["QPR Actual Quarter"].notna() & df["QPR_Date"].isna()
        )
    else:
        df["QA_missing_qpr_actual_quarter"] = True
        df["QA_invalid_qpr_actual_quarter"] = False

    # Core missingness flags.
    for col, flag in [
        ("Grantee", "QA_missing_grantee"),
        ("Grant", "QA_missing_grant"),
        ("Disaster Type", "QA_missing_disaster_type"),
    ]:
        if col in df.columns:
            df[flag] = df[col].isna()
        else:
            df[flag] = True

    # Grantee state validity checks.
    if "Grantee State" in df.columns:
        df["QA_missing_grantee_state"] = df["Grantee State"].isna()
        df["QA_unknown_grantee_state"] = (
            df["Grantee State"].notna()
            & ~df["Grantee State"].isin(STATE_NAMES)
        )
    else:
        df["QA_missing_grantee_state"] = True
        df["QA_unknown_grantee_state"] = False

    if "Grant_State_Name" in df.columns and "Grantee State Raw" in df.columns:
        df["QA_grant_state_mismatch"] = (
            df["Grantee State Raw"].notna()
            & df["Grant_State_Name"].notna()
            & (df["Grantee State Raw"] != df["Grant_State_Name"])
        )
    else:
        df["QA_grant_state_mismatch"] = False

    # Negative dollar values (may indicate adjustments).
    negative_cols = {
        "QPR Fund Obligated $": "QA_negative_obligated",
        "QPR Fund Disbursed $": "QA_negative_disbursed",
        "QPR Fund Expended $": "QA_negative_expended",
    }
    for col, flag in negative_cols.items():
        if col in df.columns:
            df[flag] = df[col] < 0
        else:
            df[flag] = False

    # Duplicate row indicator.
    df["QA_duplicate_row"] = df.duplicated(keep=False)

    report = build_quality_report(df)
    return df, report


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize quality flags and key counts."""
    rows = len(df)
    metrics = [
        {"metric": "rows", "value": rows, "percent": 1.0},
        {
            "metric": "unique_grantees",
            "value": df["Grantee"].nunique(dropna=True) if "Grantee" in df.columns else 0,
            "percent": np.nan,
        },
        {
            "metric": "unique_disasters",
            "value": df["Disaster Type"].nunique(dropna=True) if "Disaster Type" in df.columns else 0,
            "percent": np.nan,
        },
        {
            "metric": "unique_grants",
            "value": df["Grant"].nunique(dropna=True) if "Grant" in df.columns else 0,
            "percent": np.nan,
        },
    ]

    if "Grantee State Source" in df.columns:
        imputed = (df["Grantee State Source"] == "grant_code").sum()
        metrics.append({
            "metric": "grantee_state_imputed",
            "value": int(imputed),
            "percent": (imputed / rows) if rows else 0.0,
        })

    qa_cols = [col for col in df.columns if col.startswith("QA_")]
    for col in sorted(qa_cols):
        flagged = int(df[col].sum()) if rows else 0
        metrics.append({
            "metric": col,
            "value": flagged,
            "percent": (flagged / rows) if rows else 0.0,
        })

    return pd.DataFrame(metrics)


def build_quarterly_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize quality checks for the quarterly rollup."""
    rows = len(df)
    n_pairs = (
        df[["Grantee", "Disaster Type"]].drop_duplicates().shape[0]
        if {"Grantee", "Disaster Type"}.issubset(df.columns)
        else 0
    )

    metrics = [
        {"metric": "rows", "value": rows, "percent": 1.0},
        {"metric": "unique_grantee_disaster_pairs", "value": n_pairs, "percent": np.nan},
    ]

    if {"Grantee", "Disaster Type"}.issubset(df.columns):
        counts = df.groupby(["Grantee", "Disaster Type"]).size()
        metrics.extend([
            {"metric": "quarters_per_pair_min", "value": int(counts.min()), "percent": np.nan},
            {"metric": "quarters_per_pair_median", "value": float(counts.median()), "percent": np.nan},
            {"metric": "quarters_per_pair_mean", "value": float(counts.mean()), "percent": np.nan},
            {"metric": "quarters_per_pair_max", "value": int(counts.max()), "percent": np.nan},
        ])

    flow_cols = [
        "QPR Fund Obligated Q $",
        "QPR Fund Disbursed Q $",
        "QPR Fund Expended Q $",
    ]
    for col in flow_cols:
        if col in df.columns:
            flagged = int((df[col] < 0).sum())
            metrics.append({
                "metric": f"negative_{col}",
                "value": flagged,
                "percent": (flagged / rows) if rows else 0.0,
            })

    cumulative_cols = [
        "QPR Fund Obligated $",
        "QPR Fund Disbursed $",
        "QPR Fund Expended $",
    ]
    if {"Grantee", "Disaster Type"}.issubset(df.columns):
        for col in cumulative_cols:
            if col not in df.columns:
                continue
            decreases = 0
            for (_, _), grp in df.groupby(["Grantee", "Disaster Type"]):
                series = grp[col].dropna().values
                if series.size < 2:
                    continue
                if (np.diff(series) < 0).any():
                    decreases += 1
            metrics.append({
                "metric": f"groups_with_cumulative_decrease_{col}",
                "value": decreases,
                "percent": (decreases / max(1, n_pairs)),
            })

    # Add adjustment tracking statistics if columns are present
    adjustment_cols = [
        "QPR Fund Obligated Adjustment $",
        "QPR Fund Disbursed Adjustment $",
        "QPR Fund Expended Adjustment $",
    ]
    for col in adjustment_cols:
        if col in df.columns:
            # Count quarters with adjustments
            has_adjustment = (df[col] > 0).sum()
            metrics.append({
                "metric": f"quarters_with_{col.replace(' $', '')}",
                "value": int(has_adjustment),
                "percent": (has_adjustment / rows) if rows else 0.0,
            })
            # Total adjustment amount
            total_adj = df[col].sum()
            metrics.append({
                "metric": f"total_{col}",
                "value": float(total_adj),
                "percent": np.nan,
            })

    # Add groups with any adjustments
    if {"Grantee", "Disaster Type"}.issubset(df.columns):
        for col in adjustment_cols:
            if col not in df.columns:
                continue
            groups_with_adj = 0
            for (_, _), grp in df.groupby(["Grantee", "Disaster Type"]):
                if (grp[col] > 0).any():
                    groups_with_adj += 1
            metrics.append({
                "metric": f"groups_with_{col.replace(' $', '')}",
                "value": groups_with_adj,
                "percent": (groups_with_adj / max(1, n_pairs)),
            })

    return pd.DataFrame(metrics)
