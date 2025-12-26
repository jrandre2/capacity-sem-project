"""
Data loading functions for QPR financial data.

This module provides functions to load and parse the Quarterly Performance
Report (QPR) data from HUD DRGR system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional

from config import DATA_RAW_DIR, QPR_DATA_FILE, YEAR_MAPPINGS, COLUMN_MAPPING
from ..utils.date_utils import quarter_to_date


def load_qpr_data(
    filepath: Optional[Union[str, Path]] = None,
    parse_year: bool = True
) -> pd.DataFrame:
    """
    Load and parse QPR financial data from CSV.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the QPR CSV file. If not provided, uses default path.
    parse_year : bool, default True
        Whether to parse Year and Disaster Abbr from Appropriation column.

    Returns
    -------
    pd.DataFrame
        Loaded and initially processed DataFrame.

    Examples
    --------
    >>> df = load_qpr_data()
    >>> df.columns
    Index(['Grant', 'Appropriation', 'Disaster Type', ...])
    """
    if filepath is None:
        filepath = DATA_RAW_DIR / QPR_DATA_FILE

    df = pd.read_csv(filepath)

    # Rename columns for consistency using the mapping
    df = df.rename(columns=COLUMN_MAPPING)

    if parse_year:
        # Extract Year and Disaster Abbreviation from Appropriation
        df[['Year', 'Disaster Abbr']] = df['Appropriation'].str.split(
            ' ', n=1, expand=True
        )

        # Apply year mappings for special cases
        for old_val, new_val in YEAR_MAPPINGS.items():
            df.loc[df['Year'] == old_val, 'Year'] = new_val

    # Fill NaN values consistently
    df = df.fillna(np.nan)

    return df


def build_qpr_quarterly(
    df: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    quarter_col: str = 'QPR Actual Quarter',
    obligated_col: str = 'QPR Fund Obligated $',
    disbursed_col: str = 'QPR Fund Disbursed $',
    expended_col: str = 'QPR Fund Expended $',
    flows_are_net: bool = True
) -> pd.DataFrame:
    """
    Aggregate QPR data to grantee-disaster-quarter level and build cumulative series.

    Uses quarterly net changes as inputs and computes cumulative totals per group.
    """
    df = df.copy()

    rename_map = {}
    if obligated_col not in df.columns and 'QPR Funds Obligated $' in df.columns:
        rename_map['QPR Funds Obligated $'] = obligated_col
    if disbursed_col not in df.columns and 'QPR Grant Disbursed $' in df.columns:
        rename_map['QPR Grant Disbursed $'] = disbursed_col
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in [grantee_col, disaster_col, quarter_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for quarterly build: {missing}")

    value_cols = [c for c in [obligated_col, disbursed_col, expended_col] if c in df.columns]
    if not value_cols:
        return df[[grantee_col, disaster_col, quarter_col]].drop_duplicates()

    df = df.dropna(subset=[quarter_col])
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    grouped = (
        df.groupby([grantee_col, disaster_col, quarter_col], dropna=True)[value_cols]
        .sum()
        .reset_index()
    )
    grouped['QPR_Date'] = grouped[quarter_col].apply(quarter_to_date)
    grouped = grouped.sort_values([grantee_col, disaster_col, 'QPR_Date'])

    flow_map = {
        obligated_col: 'QPR Fund Obligated Q $',
        disbursed_col: 'QPR Fund Disbursed Q $',
        expended_col: 'QPR Fund Expended Q $',
    }

    if flows_are_net:
        grouped = grouped.rename(columns=flow_map)
        cumulative_map = {
            'QPR Fund Obligated Q $': obligated_col,
            'QPR Fund Disbursed Q $': disbursed_col,
            'QPR Fund Expended Q $': expended_col,
        }
        for flow_col, cum_col in cumulative_map.items():
            if flow_col in grouped.columns:
                grouped[cum_col] = grouped.groupby([grantee_col, disaster_col])[flow_col].cumsum()

        # Add adjustment tracking columns (separate negative values from positive flows)
        grouped = _add_adjustment_tracking(grouped, grantee_col, disaster_col)
    else:
        for base_col, flow_col in flow_map.items():
            if base_col in grouped.columns:
                grouped[flow_col] = grouped.groupby([grantee_col, disaster_col])[base_col].diff()
                grouped[flow_col] = grouped[flow_col].fillna(grouped[base_col])

    return grouped


def _add_adjustment_tracking(
    df: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Add adjustment tracking columns to quarterly data.

    For each financial flow column, creates:
    - Adjustment column: absolute value of negative flows (de-obligations, corrections)
    - Flow column: positive flows only (new obligations, disbursements, expenditures)
    - Clean cumulative column: monotonic cumulative from positive flows only

    Parameters
    ----------
    df : pd.DataFrame
        Quarterly QPR data with flow columns.
    grantee_col : str
        Name of grantee column for grouping.
    disaster_col : str
        Name of disaster column for grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional adjustment tracking columns.
    """
    df = df.copy()

    # Define flow columns and their corresponding new column names
    flow_columns = [
        ('QPR Fund Obligated Q $', 'QPR Fund Obligated'),
        ('QPR Fund Disbursed Q $', 'QPR Fund Disbursed'),
        ('QPR Fund Expended Q $', 'QPR Fund Expended'),
    ]

    for flow_col, base_name in flow_columns:
        if flow_col not in df.columns:
            continue

        # Track negative values as adjustments (absolute value)
        adj_col = f'{base_name} Adjustment $'
        df[adj_col] = df[flow_col].clip(upper=0).abs()

        # Track positive values as flows only
        positive_flow_col = f'{base_name} Flow $'
        df[positive_flow_col] = df[flow_col].clip(lower=0)

        # Compute "clean" cumulative from positive flows only (guaranteed monotonic)
        clean_col = f'{base_name} Clean $'
        df[clean_col] = df.groupby([grantee_col, disaster_col])[positive_flow_col].cumsum()

        # Also add cumulative adjustments for tracking total corrections
        cum_adj_col = f'{base_name} Cumulative Adjustment $'
        df[cum_adj_col] = df.groupby([grantee_col, disaster_col])[adj_col].cumsum()

    return df


def get_disaster_events(df: pd.DataFrame) -> List[str]:
    """
    Get sorted list of unique disaster events.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame with 'Disaster Type' column.

    Returns
    -------
    List[str]
        Sorted list of unique disaster event names.
    """
    return sorted(df['Disaster Type'].dropna().unique().tolist())


def get_grantees(
    df: pd.DataFrame,
    disaster_type: Optional[str] = None
) -> List[str]:
    """
    Get sorted list of grantees, optionally filtered by disaster type.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame with 'Grantee' column.
    disaster_type : str, optional
        Filter grantees by specific disaster type.

    Returns
    -------
    List[str]
        Sorted list of grantee names.
    """
    if disaster_type is not None:
        df = df[df['Disaster Type'] == disaster_type]

    return sorted(df['Grantee'].dropna().unique().tolist())


def get_years(df: pd.DataFrame) -> List[str]:
    """
    Get sorted list of unique years in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame with 'Year' column.

    Returns
    -------
    List[str]
        Sorted list of year values.
    """
    return sorted(df['Year'].dropna().unique().tolist())


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for the QPR dataset.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame.

    Returns
    -------
    dict
        Dictionary containing summary statistics.
    """
    return {
        'total_rows': len(df),
        'n_disasters': len(get_disaster_events(df)),
        'n_grantees': len(get_grantees(df)),
        'n_years': len(get_years(df)),
        'year_range': (min(get_years(df)), max(get_years(df))),
        'disaster_events': get_disaster_events(df),
        'columns': df.columns.tolist()
    }
