"""Helpers for collapsing mixed activity-row QPR data to quarter-level panels."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from config import VELOCITY_ROLLING_WINDOWS, VELOCITY_WINSOR_PERCENTILES


DEFAULT_GROUP_COLS = ("Grantee", "Disaster Type")

_SUM_CUMULATIVE_COLS = {
    # Dollar columns: SUM across activities to get grant-level totals.
    "QPR Fund Obligated $",
    "QPR Fund Disbursed $",
    "QPR Fund Expended $",
    "Obligated_Final",
    "Obligated_Clean",
    "Disbursed_Clean",
    "Expended_Clean",
}

_MAX_COLS = {
    # Ratio columns use max; they're recomputed from summed dollars after collapse.
    "Ratio_Disbursed_Std",
    "Ratio_Expended_Std",
}

_SUM_FLOW_COLS = {
    "QPR Fund Obligated Q $",
    "QPR Fund Disbursed Q $",
    "QPR Fund Expended Q $",
}


def detect_quarter_col(df: pd.DataFrame) -> str:
    """Return the canonical quarter column available in a panel."""
    for col in ("QPR Actual Quarter", "Quarter_Index"):
        if col in df.columns:
            return col
    raise ValueError("Expected 'QPR Actual Quarter' or 'Quarter_Index' in dataframe")


def collapse_to_quarterly_panel(
    df: pd.DataFrame,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    quarter_col: str | None = None,
    rolling_windows: Iterable[int] | None = None,
    winsor_percentiles: tuple[float, float] | None = VELOCITY_WINSOR_PERCENTILES,
) -> pd.DataFrame:
    """
    Collapse mixed activity-row data to one row per grantee-disaster-quarter.

    The standardized QPR data can contain many activity rows inside a single
    quarter. This helper collapses those rows to quarter-end values and then
    recomputes standardized velocity from quarter-end ratios so downstream
    analyses are based on the correct unit of analysis.
    """
    if df.empty:
        return df.copy()

    quarter_col = quarter_col or detect_quarter_col(df)
    sort_cols = [col for col in (*group_cols, "QPR_Date", quarter_col) if col in df.columns]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    agg_map: dict[str, str] = {}
    for col in df_sorted.columns:
        if col in group_cols or col == quarter_col:
            continue
        if col == "QPR_Date":
            agg_map[col] = "max"
        elif col in _SUM_FLOW_COLS:
            agg_map[col] = "sum"
        elif col in _SUM_CUMULATIVE_COLS:
            # Dollar columns: SUM across activities to get grant-level totals
            agg_map[col] = "sum"
        elif col in _MAX_COLS or col.startswith("Ratio_") or col.endswith("_Clean"):
            agg_map[col] = "max"
        elif pd.api.types.is_bool_dtype(df_sorted[col]):
            agg_map[col] = "max"
        else:
            agg_map[col] = "last"

    quarterly = (
        df_sorted.groupby(list(group_cols) + [quarter_col], as_index=False)
        .agg(agg_map)
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    # Recompute ratios from summed grant-level dollar totals.
    # Per-activity ratios are stale after dollar columns were summed
    # across activities -- override them with grant-level ratios.
    import numpy as np

    _MIN_DENOMINATOR = 1000  # $1K minimum to avoid division-by-near-zero

    if "QPR Fund Obligated $" in quarterly.columns and "QPR Fund Disbursed $" in quarterly.columns:
        quarterly["Ratio_Disbursed_Std"] = np.where(
            quarterly["QPR Fund Obligated $"] > _MIN_DENOMINATOR,
            quarterly["QPR Fund Disbursed $"] / quarterly["QPR Fund Obligated $"],
            np.nan,
        )
        # Clip to reasonable range: negative values from data adjustments
        # and values > 2 from supplemental allocations are capped
        quarterly["Ratio_Disbursed_Std"] = quarterly["Ratio_Disbursed_Std"].clip(lower=0.0, upper=2.0)

    if "QPR Fund Disbursed $" in quarterly.columns and "QPR Fund Expended $" in quarterly.columns:
        quarterly["Ratio_Expended_Std"] = np.where(
            quarterly["QPR Fund Disbursed $"] > _MIN_DENOMINATOR,
            quarterly["QPR Fund Expended $"] / quarterly["QPR Fund Disbursed $"],
            np.nan,
        )
        quarterly["Ratio_Expended_Std"] = quarterly["Ratio_Expended_Std"].clip(lower=0.0, upper=2.0)

    if rolling_windows is None:
        rolling_windows = VELOCITY_ROLLING_WINDOWS

    if "Ratio_Disbursed_Std" in quarterly.columns:
        quarterly["Velocity_Disb_Std"] = quarterly.groupby(list(group_cols))["Ratio_Disbursed_Std"].diff()
        quarterly["Velocity_Disb_Std_pp"] = quarterly["Velocity_Disb_Std"] * 100

    if "Ratio_Expended_Std" in quarterly.columns:
        quarterly["Velocity_Exp_Std"] = quarterly.groupby(list(group_cols))["Ratio_Expended_Std"].diff()
        quarterly["Velocity_Exp_Std_pp"] = quarterly["Velocity_Exp_Std"] * 100

    if {"Velocity_Disb_Std_pp", "Velocity_Exp_Std_pp"}.issubset(quarterly.columns):
        quarterly["Velocity_Index_Std_pp"] = quarterly[
            ["Velocity_Disb_Std_pp", "Velocity_Exp_Std_pp"]
        ].mean(axis=1, skipna=True)

    for raw_col in ("Velocity_Disb_Std_pp", "Velocity_Exp_Std_pp", "Velocity_Index_Std_pp"):
        if raw_col not in quarterly.columns:
            continue
        winsor_col = f"{raw_col}_winsor"
        quarterly[winsor_col] = quarterly[raw_col]
        if winsor_percentiles is None:
            continue
        valid = quarterly[raw_col].dropna()
        if valid.empty:
            continue
        lower = valid.quantile(winsor_percentiles[0])
        upper = valid.quantile(winsor_percentiles[1])
        quarterly[winsor_col] = quarterly[raw_col].clip(lower=lower, upper=upper)

    for prefix in ("Disb", "Exp", "Index"):
        base_col = f"Velocity_{prefix}_Std_pp"
        if base_col not in quarterly.columns:
            continue
        for window in rolling_windows:
            quarterly[f"{base_col}_roll{window}"] = quarterly.groupby(list(group_cols))[base_col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        quarterly[f"{base_col}_cum"] = quarterly.groupby(list(group_cols))[base_col].transform(
            lambda s: s.expanding(min_periods=1).mean()
        )

    return quarterly
