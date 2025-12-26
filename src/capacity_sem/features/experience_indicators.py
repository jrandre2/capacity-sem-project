"""
Experience/Learning proxy indicators derived from QPR data.

Computes organizational learning measures based on prior grant experience.
These indicators capture how much experience a grantee has with CDBG-DR
programs at the time of each disaster, which may influence their capacity
to manage recovery funds effectively.

Indicators:
- Years_Experience: Years since first CDBG-DR grant
- Prior_Grant_Count: Number of prior disaster grants managed
- Prior_Grant_Dollars: Cumulative prior obligated dollars
- Experience_Index: Composite 0-1 score (normalized)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def parse_quarter_to_date(quarter_str) -> Optional[datetime]:
    """
    Parse quarter string to datetime.

    Parameters
    ----------
    quarter_str : str or None
        Quarter string in format "YYYY Qq" (e.g., "2020 Q3").

    Returns
    -------
    datetime or None
        First day of the quarter, or None if invalid.
    """
    if quarter_str is None or (isinstance(quarter_str, float) and pd.isna(quarter_str)):
        return None

    try:
        quarter_str = str(quarter_str).strip()
        parts = quarter_str.split()
        year = int(parts[0])
        quarter = int(parts[1].replace("Q", "").replace("q", ""))

        # Convert quarter to month (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1)
    except (IndexError, ValueError, AttributeError):
        return None


def compute_first_grant_date(
    df: pd.DataFrame,
    grantee: str,
    grantee_col: str = 'Grantee',
    quarter_col: str = 'QPR Actual Quarter'
) -> Optional[datetime]:
    """
    Find the first CDBG-DR grant quarter for a grantee.

    Parameters
    ----------
    df : pd.DataFrame
        QPR data with grantee and quarter columns.
    grantee : str
        Grantee name.
    grantee_col : str
        Column name for grantee identifier.
    quarter_col : str
        Column name for quarter identifier.

    Returns
    -------
    datetime or None
        Date of first grant quarter, or None if not found.
    """
    grantee_data = df[df[grantee_col] == grantee]
    if grantee_data.empty:
        return None

    # Get all quarters and parse to dates
    quarters = grantee_data[quarter_col].dropna().unique()
    dates = [parse_quarter_to_date(q) for q in quarters]
    valid_dates = [d for d in dates if d is not None]

    if not valid_dates:
        return None

    return min(valid_dates)


def compute_years_of_experience(
    df: pd.DataFrame,
    grantee: str,
    disaster_type: str,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> float:
    """
    Compute years since first CDBG-DR grant at time of current disaster.

    Parameters
    ----------
    df : pd.DataFrame
        QPR data.
    grantee : str
        Grantee name.
    disaster_type : str
        DRGR disaster type string.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster type.

    Returns
    -------
    float
        Years of experience (0 if this is first disaster).
    """
    from ..data.external_data import DRGR_DISASTER_YEARS

    # Get disaster year
    disaster_year = DRGR_DISASTER_YEARS.get(disaster_type, 2020)

    # Get first grant date for this grantee
    first_grant_date = compute_first_grant_date(df, grantee, grantee_col)

    if first_grant_date is None:
        return 0.0

    first_year = first_grant_date.year
    years_exp = max(0, disaster_year - first_year)

    return float(years_exp)


def compute_prior_grant_count(
    df: pd.DataFrame,
    grantee: str,
    disaster_type: str,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> int:
    """
    Count number of prior disaster grants managed before current disaster.

    Parameters
    ----------
    df : pd.DataFrame
        QPR data.
    grantee : str
        Grantee name.
    disaster_type : str
        DRGR disaster type string.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster type.

    Returns
    -------
    int
        Number of prior disaster grants (0 if this is first).
    """
    from ..data.external_data import DRGR_DISASTER_YEARS

    # Get current disaster year
    current_year = DRGR_DISASTER_YEARS.get(disaster_type, 2020)

    # Get all disasters for this grantee
    grantee_data = df[df[grantee_col] == grantee]
    grantee_disasters = grantee_data[disaster_col].unique()

    # Count disasters that occurred before current one
    prior_count = 0
    for d in grantee_disasters:
        if d == disaster_type:
            continue
        d_year = DRGR_DISASTER_YEARS.get(d, 9999)
        if d_year < current_year:
            prior_count += 1

    return prior_count


def compute_cumulative_prior_dollars(
    df: pd.DataFrame,
    grantee: str,
    disaster_type: str,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    obligated_col: str = 'QPR Fund Obligated $',
    quarter_col: str = 'QPR Actual Quarter'
) -> float:
    """
    Compute total dollars from prior grants administered.

    Sum of QPR Fund Obligated $ from all prior disaster grants.
    Uses the sum of quarterly net changes per disaster (final obligated value
    when QPR fields are net changes), excluding total rows.

    Parameters
    ----------
    df : pd.DataFrame
        QPR data.
    grantee : str
        Grantee name.
    disaster_type : str
        DRGR disaster type string.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster type.
    obligated_col : str
        Column name for obligated funds.
    quarter_col : str
        Column name for quarter identifier (used to exclude total rows).

    Returns
    -------
    float
        Total prior obligated dollars (0 if none).
    """
    from ..data.external_data import DRGR_DISASTER_YEARS

    # Get current disaster year
    current_year = DRGR_DISASTER_YEARS.get(disaster_type, 2020)

    # Get grantee data
    grantee_data = df[df[grantee_col] == grantee]

    if obligated_col not in grantee_data.columns:
        logger.warning(f"Column {obligated_col} not found in data")
        return 0.0
    if quarter_col not in grantee_data.columns:
        logger.warning(f"Column {quarter_col} not found in data")
        return 0.0

    total_prior_dollars = 0.0

    # Get unique disasters for this grantee
    for d in grantee_data[disaster_col].unique():
        if d == disaster_type:
            continue

        d_year = DRGR_DISASTER_YEARS.get(d, 9999)
        if d_year < current_year:
            # Sum quarterly net changes for prior disaster (exclude total rows)
            disaster_data = grantee_data[
                (grantee_data[disaster_col] == d) &
                (grantee_data[quarter_col].notna())
            ]
            prior_total = disaster_data[obligated_col].sum()
            if pd.notna(prior_total) and prior_total > 0:
                total_prior_dollars += prior_total

    return total_prior_dollars


def compute_experience_index(
    years_exp: float,
    prior_count: int,
    prior_dollars: float,
    max_years: float = 20.0,
    max_grants: int = 10,
    max_log_dollars: float = 10.0
) -> float:
    """
    Compute composite experience index (0-1 scale).

    Formula:
        Experience_Index = (norm_years + norm_grants + norm_dollars) / 3

    Where each component is normalized to 0-1 range.

    Parameters
    ----------
    years_exp : float
        Years of experience.
    prior_count : int
        Number of prior grants.
    prior_dollars : float
        Cumulative prior dollars.
    max_years : float
        Maximum years for normalization (default 20).
    max_grants : int
        Maximum grants for normalization (default 10).
    max_log_dollars : float
        Maximum log10(dollars) for normalization (default 10 = $10B).

    Returns
    -------
    float
        Experience index between 0 and 1.
    """
    # Normalize years
    norm_years = min(years_exp / max_years, 1.0)

    # Normalize grant count
    norm_grants = min(prior_count / max_grants, 1.0)

    # Normalize dollars (log scale)
    if prior_dollars > 0:
        norm_dollars = min(np.log10(prior_dollars) / max_log_dollars, 1.0)
    else:
        norm_dollars = 0.0

    # Weighted average (equal weights)
    experience_index = (norm_years + norm_grants + norm_dollars) / 3.0

    return round(experience_index, 4)


def compute_grantee_experience(
    df: pd.DataFrame,
    grantee: str,
    disaster_type: str,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    obligated_col: str = 'QPR Fund Obligated $',
    quarter_col: str = 'QPR Actual Quarter'
) -> Dict[str, float]:
    """
    Compute all experience/learning indicators for a grantee-disaster pair.

    Parameters
    ----------
    df : pd.DataFrame
        QPR data.
    grantee : str
        Grantee name.
    disaster_type : str
        DRGR disaster type string.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster type.
    obligated_col : str
        Column name for obligated funds.

    Returns
    -------
    Dict[str, float]
        Dictionary with experience indicators:
        - Years_Experience
        - Prior_Grant_Count
        - Prior_Grant_Dollars
        - Experience_Index
    """
    years_exp = compute_years_of_experience(
        df, grantee, disaster_type, grantee_col, disaster_col
    )

    prior_count = compute_prior_grant_count(
        df, grantee, disaster_type, grantee_col, disaster_col
    )

    prior_dollars = compute_cumulative_prior_dollars(
        df,
        grantee,
        disaster_type,
        grantee_col,
        disaster_col,
        obligated_col,
        quarter_col
    )

    experience_index = compute_experience_index(years_exp, prior_count, prior_dollars)

    return {
        'Years_Experience': years_exp,
        'Prior_Grant_Count': prior_count,
        'Prior_Grant_Dollars': prior_dollars,
        'Experience_Index': experience_index
    }


def build_experience_dataset(
    df: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    obligated_col: str = 'QPR Fund Obligated $'
) -> pd.DataFrame:
    """
    Build experience indicators for all grantee-disaster combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Full QPR dataset.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster type.
    obligated_col : str
        Column name for obligated funds.

    Returns
    -------
    pd.DataFrame
        Experience indicators at grantee-disaster level with columns:
        - Grantee
        - Disaster_Type
        - Disaster_Year
        - Years_Experience
        - Prior_Grant_Count
        - Prior_Grant_Dollars
        - Experience_Index
    """
    from ..data.external_data import DRGR_DISASTER_YEARS

    records = []

    # Get unique grantee-disaster pairs
    pairs = df.groupby([grantee_col, disaster_col]).size().reset_index()[[grantee_col, disaster_col]]

    logger.info(f"Computing experience indicators for {len(pairs)} grantee-disaster pairs")

    for _, row in pairs.iterrows():
        grantee = row[grantee_col]
        disaster = row[disaster_col]

        try:
            exp_data = compute_grantee_experience(
                df, grantee, disaster, grantee_col, disaster_col, obligated_col
            )

            record = {
                'Grantee': grantee,
                'Disaster_Type': disaster,
                'Disaster_Year': DRGR_DISASTER_YEARS.get(disaster),
                **exp_data
            }
            records.append(record)

        except Exception as e:
            logger.warning(f"Error computing experience for {grantee}-{disaster}: {e}")
            continue

    df_exp = pd.DataFrame(records)

    # Reorder columns
    col_order = [
        'Grantee', 'Disaster_Type', 'Disaster_Year',
        'Years_Experience', 'Prior_Grant_Count',
        'Prior_Grant_Dollars', 'Experience_Index'
    ]

    # Only include columns that exist
    col_order = [c for c in col_order if c in df_exp.columns]

    logger.info(f"Built experience dataset with {len(df_exp)} rows")

    return df_exp[col_order]


def get_experience_summary(df_experience: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for experience indicators.

    Parameters
    ----------
    df_experience : pd.DataFrame
        Experience indicator dataset from build_experience_dataset().

    Returns
    -------
    pd.DataFrame
        Summary statistics for each indicator.
    """
    numeric_cols = [
        'Years_Experience', 'Prior_Grant_Count',
        'Prior_Grant_Dollars', 'Experience_Index'
    ]

    summary = df_experience[numeric_cols].describe()

    return summary
