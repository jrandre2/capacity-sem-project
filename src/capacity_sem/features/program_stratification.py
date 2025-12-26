"""
Program-type stratification for activity-level analysis.

This module maps the 51 Activity Types in the QPR data to 6 major program
categories and provides functions for stratified SEM analysis by program type.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Mapping of Activity Types to major program categories
PROGRAM_TYPE_MAPPING: Dict[str, List[str]] = {
    'Housing': [
        'Affordable Rental Housing',
        'Construction of new housing',
        'Construction of new replacement housing',
        'Homeownership Assistance (with waiver only)',
        'Homeownership Assistance to low- and moderate-income',
        'Housing incentives to encourage resettlement',
        'Rehabilitation/reconstruction of residential structures',
        'Relocation payments and assistance',
        'Rental Assistance (waiver only)',
        'MIT - Residential New Construction',
        'MIT - Rehabilitation/reconstruction of residential structures',
    ],
    'Infrastructure': [
        'Acquisition, construction,reconstruction of public facilities',
        'Construction/reconstruction of streets',
        'Construction/reconstruction of water lift stations',
        'Construction/reconstruction of water/sewer lines or systems',
        'Dike/dam/stream-river bank repairs',
        'Electrical power system improvements',
        'Privately owned utilities',
        'Rehabilitation/reconstruction of a public improvement',
        'Rehabilitation/reconstruction of public facilities',
        'MIT - Public Facilities and Improvements-Covered Projects Only',
        'MIT - Public Facilities and Improvements-Non Covered Projects',
    ],
    'Administration': [
        'Administration',
        'Planning',
        'Capacity building for nonprofit or public entities',
        'MIT - Planning and Capacity Building',
    ],
    'Economic Development': [
        'Econ. development or recovery activity that creates/retains jobs',
        'Economic Development Center (Virginia waiver only)',
        'MIT - Economic Development',
        'Tourism (Waiver Only)',
        'Travel and Tourism per 107-117 - (WTC only)',
    ],
    'Acquisition': [
        'Acquisition - buyout of residential properties',
        'Acquisition - general',
        'Acquisition of property for replacement housing',
        'Acquisition of relocation properties',
        'MIT - Buyout of Properties',
        'Clearance and Demolition',
        'Disposition',
    ],
    'Other': [
        'Code enforcement',
        'Compensation for disaster-related losses (Louisiana and Texas)',
        'Debris removal',
        'NDR - Environmental Value',
        'Payment for compensation and incentives (Louisiana only)',
        'Payment for compensation for economic losses (WTC-only)',
        'Payment for homeowner compensation (Mississippi only)',
        'Public services',
        'MIT - Public Services and Information',
        'Residential Location Incentive Grants - (Waiver only)',
        'Windpool Mitigation (Mississippi only)',
        'Construction of buildings for the general conduct of government',
        'Rehabilitation/reconstruction of other non-residential structures',
    ],
}

# Create reverse mapping for quick lookup
ACTIVITY_TO_PROGRAM: Dict[str, str] = {}
for program, activities in PROGRAM_TYPE_MAPPING.items():
    for activity in activities:
        ACTIVITY_TO_PROGRAM[activity] = program


def map_activity_to_program_type(activity_type) -> str:
    """
    Map an Activity Type to its major program category.

    Parameters
    ----------
    activity_type : str or None
        Activity Type from QPR data.

    Returns
    -------
    str
        Program category: Housing, Infrastructure, Administration,
        Economic Development, Acquisition, or Other.
    """
    # Handle missing values
    if activity_type is None or (isinstance(activity_type, float) and pd.isna(activity_type)):
        return 'Other'

    # Convert to string if needed
    activity_type = str(activity_type)

    # Direct match
    if activity_type in ACTIVITY_TO_PROGRAM:
        return ACTIVITY_TO_PROGRAM[activity_type]

    # Fuzzy match for similar names
    activity_lower = activity_type.lower()

    if 'housing' in activity_lower or 'residential' in activity_lower:
        return 'Housing'
    elif 'infrastructure' in activity_lower or 'public facilit' in activity_lower:
        return 'Infrastructure'
    elif 'admin' in activity_lower or 'planning' in activity_lower:
        return 'Administration'
    elif 'economic' in activity_lower or 'job' in activity_lower:
        return 'Economic Development'
    elif 'acquisition' in activity_lower or 'buyout' in activity_lower:
        return 'Acquisition'
    else:
        return 'Other'


def add_program_type_column(
    df: pd.DataFrame,
    activity_col: str = 'Activity Type'
) -> pd.DataFrame:
    """
    Add a Program_Type column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame with Activity Type column.
    activity_col : str
        Name of the Activity Type column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new Program_Type column.
    """
    df = df.copy()
    df['Program_Type'] = df[activity_col].apply(map_activity_to_program_type)
    return df


def get_program_type_distribution(
    df: pd.DataFrame,
    activity_col: str = 'Activity Type'
) -> pd.DataFrame:
    """
    Get distribution of activities across program types.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame.
    activity_col : str
        Activity Type column name.

    Returns
    -------
    pd.DataFrame
        Summary of activities per program type.
    """
    df_typed = add_program_type_column(df, activity_col)

    summary = df_typed.groupby('Program_Type').agg({
        activity_col: 'count',
        'QPR Fund Obligated $': 'sum',
        'QPR Fund Expended $': 'sum'
    }).reset_index()

    summary = summary.rename(columns={
        activity_col: 'N_Activities',
        'QPR Fund Obligated $': 'Total_Obligated',
        'QPR Fund Expended $': 'Total_Expended'
    })

    # Add percentages
    summary['Pct_Activities'] = summary['N_Activities'] / summary['N_Activities'].sum() * 100
    summary['Pct_Obligated'] = summary['Total_Obligated'] / summary['Total_Obligated'].sum() * 100

    return summary.sort_values('Total_Obligated', ascending=False)


def filter_by_program_type(
    df: pd.DataFrame,
    program_type: str,
    activity_col: str = 'Activity Type'
) -> pd.DataFrame:
    """
    Filter DataFrame to a specific program type.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame.
    program_type : str
        Program type to filter to.
    activity_col : str
        Activity Type column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    df_typed = add_program_type_column(df, activity_col)
    return df_typed[df_typed['Program_Type'] == program_type].copy()


def compute_indicators_by_program_type(
    df: pd.DataFrame,
    grantee: str,
    disaster: str,
    program_type: str,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    activity_col: str = 'Activity Type'
) -> Dict[str, float]:
    """
    Compute capacity/outcome indicators for a specific program type.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame.
    grantee : str
        Grantee name.
    disaster : str
        Disaster type.
    program_type : str
        Program type to compute indicators for.
    grantee_col, disaster_col, activity_col : str
        Column names.

    Returns
    -------
    Dict[str, float]
        Indicator values for this grantee-disaster-program combination.
    """
    # Filter to grantee-disaster
    mask = (df[grantee_col] == grantee) & (df[disaster_col] == disaster)
    df_grantee = df[mask].copy()

    if df_grantee.empty:
        return {}

    # Add program type and filter
    df_grantee = add_program_type_column(df_grantee, activity_col)
    df_program = df_grantee[df_grantee['Program_Type'] == program_type]

    if df_program.empty:
        return {}

    # Compute totals
    obligated = df_program['QPR Fund Obligated $'].sum()
    disbursed = df_program['QPR Fund Disbursed $'].sum() if 'QPR Fund Disbursed $' in df_program.columns else 0
    expended = df_program['QPR Fund Expended $'].sum()

    # Handle alternative column name
    if disbursed == 0 and 'QPR Grant Disbursed $' in df_program.columns:
        disbursed = df_program['QPR Grant Disbursed $'].sum()

    # Count quarters
    n_quarters = df_program['QPR Actual Quarter'].nunique() if 'QPR Actual Quarter' in df_program.columns else 0

    indicators = {
        'Grantee': grantee,
        'Disaster_Type': disaster,
        'Program_Type': program_type,
        'N_Quarters': n_quarters,
        'Total_Obligated': obligated,
        'Total_Disbursed': disbursed,
        'Total_Expended': expended
    }

    # Compute ratios
    if obligated > 0:
        indicators['Ratio_disbursed_to_obligated'] = disbursed / obligated
        indicators['Ratio_expended_to_obligated'] = expended / obligated

    if disbursed > 0:
        indicators['Ratio_expended_to_disbursed'] = expended / disbursed

    return indicators


def build_program_stratified_dataset(
    df: pd.DataFrame,
    program_types: Optional[List[str]] = None,
    min_quarters: int = 3,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type',
    activity_col: str = 'Activity Type'
) -> Dict[str, pd.DataFrame]:
    """
    Build separate indicator datasets for each program type.

    Parameters
    ----------
    df : pd.DataFrame
        QPR DataFrame.
    program_types : List[str], optional
        Program types to include. Defaults to Housing, Infrastructure, Administration.
    min_quarters : int, default 3
        Minimum quarters required for inclusion.
    grantee_col, disaster_col, activity_col : str
        Column names.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping program type to indicator DataFrame.
    """
    if program_types is None:
        program_types = ['Housing', 'Infrastructure', 'Administration']

    # Get unique grantee-disaster combinations
    grantee_disasters = df.groupby([grantee_col, disaster_col]).size().reset_index()[[grantee_col, disaster_col]]

    datasets = {}

    for program_type in program_types:
        logger.info(f"Building dataset for {program_type}...")

        results = []
        for _, row in grantee_disasters.iterrows():
            indicators = compute_indicators_by_program_type(
                df,
                row[grantee_col],
                row[disaster_col],
                program_type,
                grantee_col,
                disaster_col,
                activity_col
            )

            if indicators and indicators.get('N_Quarters', 0) >= min_quarters:
                results.append(indicators)

        if results:
            datasets[program_type] = pd.DataFrame(results)
            logger.info(f"  {program_type}: {len(results)} grantee-disaster combinations")
        else:
            logger.warning(f"  {program_type}: No data meeting criteria")

    return datasets


def run_stratified_sem_analysis(
    datasets: Dict[str, pd.DataFrame],
    model_spec: str,
    min_sample: int = 20
) -> Dict[str, Dict[str, Any]]:
    """
    Run SEM analysis for each program type.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Output from build_program_stratified_dataset().
    model_spec : str
        SEM model specification (semopy syntax).
    min_sample : int, default 20
        Minimum sample size to attempt SEM fitting.

    Returns
    -------
    Dict[str, Dict]
        Results for each program type.
    """
    try:
        from semopy import Model
        from semopy.stats import calc_stats
    except ImportError:
        logger.error("semopy is required for SEM analysis")
        return {}

    results = {}

    for program_type, df in datasets.items():
        logger.info(f"Fitting SEM for {program_type} (n={len(df)})...")

        if len(df) < min_sample:
            logger.warning(f"  Skipping {program_type}: sample size {len(df)} < {min_sample}")
            results[program_type] = {
                'status': 'skipped',
                'reason': f'sample_size_{len(df)}_less_than_{min_sample}',
                'n': len(df)
            }
            continue

        try:
            model = Model(model_spec)
            model.fit(df)

            fit_stats = calc_stats(model)
            estimates = model.inspect()

            results[program_type] = {
                'status': 'fitted',
                'n': len(df),
                'model': model,
                'fit_statistics': fit_stats,
                'estimates': estimates
            }

            logger.info(f"  CFI: {fit_stats.get('CFI', 'N/A'):.3f}, "
                       f"RMSEA: {fit_stats.get('RMSEA', 'N/A'):.3f}")

        except Exception as e:
            logger.error(f"  Failed to fit {program_type}: {e}")
            results[program_type] = {
                'status': 'failed',
                'error': str(e),
                'n': len(df)
            }

    return results


def compare_program_type_results(
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create comparison table across program types.

    Parameters
    ----------
    results : Dict
        Output from run_stratified_sem_analysis().

    Returns
    -------
    pd.DataFrame
        Comparison table with fit statistics.
    """
    rows = []

    for program_type, result in results.items():
        row = {
            'Program_Type': program_type,
            'Status': result.get('status', 'unknown'),
            'N': result.get('n', 0)
        }

        if result.get('status') == 'fitted':
            fit_stats = result.get('fit_statistics', {})
            row.update({
                'Chi_Square': fit_stats.get('chi2', np.nan),
                'df': fit_stats.get('dof', np.nan),
                'CFI': fit_stats.get('CFI', np.nan),
                'TLI': fit_stats.get('TLI', np.nan),
                'RMSEA': fit_stats.get('RMSEA', np.nan),
                'SRMR': fit_stats.get('SRMR', np.nan)
            })

        rows.append(row)

    return pd.DataFrame(rows)


def get_program_type_summary_statistics(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Get descriptive statistics for each program type dataset.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Output from build_program_stratified_dataset().

    Returns
    -------
    pd.DataFrame
        Summary statistics for each program type.
    """
    rows = []

    for program_type, df in datasets.items():
        row = {
            'Program_Type': program_type,
            'N_Observations': len(df),
            'Mean_Quarters': df['N_Quarters'].mean() if 'N_Quarters' in df.columns else np.nan,
            'Mean_Obligated': df['Total_Obligated'].mean() if 'Total_Obligated' in df.columns else np.nan,
            'Mean_Ratio_Disbursed': df['Ratio_disbursed_to_obligated'].mean() if 'Ratio_disbursed_to_obligated' in df.columns else np.nan,
            'Mean_Ratio_Expended': df['Ratio_expended_to_obligated'].mean() if 'Ratio_expended_to_obligated' in df.columns else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)
