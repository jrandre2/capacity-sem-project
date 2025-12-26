"""
Date utility functions for QPR quarter handling.

This module provides functions to convert QPR quarter strings to datetime
objects and perform date-related calculations.
"""

import pandas as pd
from typing import Union


def quarter_to_date(quarter_string: str) -> pd.Timestamp:
    """
    Convert QPR quarter string to pandas Timestamp.

    The function converts a quarter string in the format "YYYY QN" to
    a Timestamp representing the last day of that quarter.

    Parameters
    ----------
    quarter_string : str
        Quarter string in format "YYYY QN" (e.g., "2017 Q1", "2020 Q4").

    Returns
    -------
    pd.Timestamp
        Timestamp at the end of the specified quarter.

    Examples
    --------
    >>> quarter_to_date("2017 Q1")
    Timestamp('2017-03-31 00:00:00')

    >>> quarter_to_date("2020 Q4")
    Timestamp('2020-12-31 00:00:00')
    """
    if pd.isna(quarter_string):
        return pd.NaT

    try:
        year_str, qtr_str = quarter_string.split()
        year = int(year_str)
        q = int(qtr_str[1])  # Turn "Q1" -> 1

        # Calculate the last month of the quarter
        month = q * 3  # Q1=3 (March), Q2=6 (June), etc.

        # Return the last day of that month
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

    except (ValueError, AttributeError, IndexError):
        return pd.NaT


def get_quarter_from_date(date: Union[pd.Timestamp, str]) -> str:
    """
    Convert a date to QPR quarter string format.

    Parameters
    ----------
    date : pd.Timestamp or str
        Date to convert.

    Returns
    -------
    str
        Quarter string in format "YYYY QN".

    Examples
    --------
    >>> get_quarter_from_date(pd.Timestamp("2017-03-15"))
    "2017 Q1"
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)

    if pd.isna(date):
        return None

    quarter = (date.month - 1) // 3 + 1
    return f"{date.year} Q{quarter}"


def get_quarters_between(
    start_quarter: str,
    end_quarter: str
) -> list:
    """
    Get list of all quarters between two quarter strings (inclusive).

    Parameters
    ----------
    start_quarter : str
        Starting quarter in format "YYYY QN".
    end_quarter : str
        Ending quarter in format "YYYY QN".

    Returns
    -------
    list
        List of quarter strings.
    """
    start_date = quarter_to_date(start_quarter)
    end_date = quarter_to_date(end_quarter)

    if pd.isna(start_date) or pd.isna(end_date):
        return []

    quarters = []
    current = start_date

    while current <= end_date:
        quarters.append(get_quarter_from_date(current))
        # Move to next quarter (add 3 months)
        current = current + pd.DateOffset(months=3)

    return quarters


def calculate_duration_months(
    start_quarter: str,
    end_quarter: str
) -> float:
    """
    Calculate duration in months between two quarters.

    Parameters
    ----------
    start_quarter : str
        Starting quarter.
    end_quarter : str
        Ending quarter.

    Returns
    -------
    float
        Duration in months (approximate).
    """
    start_date = quarter_to_date(start_quarter)
    end_date = quarter_to_date(end_quarter)

    if pd.isna(start_date) or pd.isna(end_date):
        return 0.0

    return (end_date - start_date).days / 30.4
