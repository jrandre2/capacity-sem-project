"""
Data loading functions for QPR financial data.

This module provides functions to load and parse the Quarterly Performance
Report (QPR) data from HUD DRGR system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional

from ..config import RAW_DATA_DIR, QPR_DATA_FILE, YEAR_MAPPINGS, COLUMN_MAPPING


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
        filepath = RAW_DATA_DIR / QPR_DATA_FILE

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
