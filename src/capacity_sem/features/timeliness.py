"""
Timeliness and duration metrics for recovery outcomes.

This module provides functions to calculate duration and timeliness metrics
that measure disaster recovery performance.
"""

import pandas as pd
import numpy as np
from typing import Optional

from ..config import COMPLETION_THRESHOLD
from ..utils.date_utils import quarter_to_date


def calculate_duration_of_completion(
    df: pd.DataFrame,
    completion_threshold: float = COMPLETION_THRESHOLD,
    quarter_col: str = 'QPR Actual Quarter',
    obligated_col: str = 'QPR Fund Obligated $',
    expended_col: str = 'QPR Fund Expended $',
    censor_incomplete: bool = True
) -> float:
    """
    Calculate duration in months to reach completion threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Grantee funding DataFrame with date and funding columns.
    completion_threshold : float, default 0.95
        Threshold for considering funds "completed" (e.g., 0.95 = 95%).
    quarter_col : str
        Column name for quarter identifier.
    obligated_col : str
        Column name for obligated funds.
    expended_col : str
        Column name for expended funds.

    Returns
    -------
    float
        Duration in months to reach completion threshold.
    """
    if df.empty or quarter_col not in df.columns:
        return np.nan

    df = df.copy()

    # Add date column if not present
    if 'QPR_Date' not in df.columns:
        df['QPR_Date'] = df[quarter_col].apply(quarter_to_date)

    # Sort by date
    df = df.sort_values('QPR_Date')

    # Calculate percentage of obligated funds expended
    final_obligated = df[obligated_col].iloc[-1]
    if final_obligated == 0:
        return 3.0  # Minimum duration

    df['pct_obligated'] = df[expended_col] / final_obligated

    # Find quarter when threshold is reached
    completion_row = df[df['pct_obligated'] >= completion_threshold].head(1)

    if not completion_row.empty:
        completion_date = completion_row['QPR_Date'].iloc[0]
    else:
        # If not yet completed, optionally censor rather than assume completion
        if censor_incomplete:
            return np.nan
        completion_date = df['QPR_Date'].iloc[-1]

    start_date = df['QPR_Date'].iloc[0]

    # Calculate duration in months (approximately)
    duration = (completion_date - start_date).days / 30.4

    # Add minimum quarter length
    duration += 3

    return duration


def calculate_timeliness(duration: float) -> float:
    """
    Calculate timeliness as inverse of duration.

    Higher timeliness values indicate faster completion.

    Parameters
    ----------
    duration : float
        Duration in months.

    Returns
    -------
    float
        Timeliness score (inverse of duration).
    """
    if np.isnan(duration):
        return np.nan
    if duration <= 0:
        return 0.0

    return 1.0 / duration


def calculate_quarter_variance(
    df: pd.DataFrame,
    expended_col: str = 'QPR Fund Expended Q $'
) -> float:
    """
    Calculate normalized standard deviation of quarterly expenditures.

    Lower variance indicates more consistent spending patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Grantee funding DataFrame with quarterly expenditure column.
    expended_col : str
        Column name for quarterly expenditure values.

    Returns
    -------
    float
        Normalized standard deviation of quarterly expenditures.
    """
    if df.empty or expended_col not in df.columns:
        return np.nan

    expended_arr = df[expended_col].values

    # Handle all-zero or single-value cases
    if len(expended_arr) < 2 or max(expended_arr) == 0:
        return 0.0

    # Normalize by maximum value
    normalized = expended_arr / max(expended_arr)

    return np.std(normalized)


def calculate_all_timeliness_metrics(
    df: pd.DataFrame,
    completion_threshold: float = COMPLETION_THRESHOLD
) -> dict:
    """
    Calculate all timeliness-related metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Grantee funding DataFrame.
    completion_threshold : float
        Threshold for completion calculations.

    Returns
    -------
    dict
        Dictionary containing duration, timeliness, and variance metrics.
    """
    duration = calculate_duration_of_completion(df, completion_threshold)
    timeliness = calculate_timeliness(duration)
    variance = calculate_quarter_variance(df)

    return {
        'Duration_of_completion': duration,
        'Timeliness': timeliness,
        'Quarter_by_quarter_variance_expended': variance
    }
