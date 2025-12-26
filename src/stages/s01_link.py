"""
Stage 01: Record Linkage and Panel Construction

Link QPR data with external covariates and construct analysis panel.

Commands:
    python src/pipeline.py build_panel

Outputs:
    data_work/panel.parquet  - Analysis panel with all indicators
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from config import DATA_WORK_DIR, RATIO_DEFINITION, QPR_DOLLAR_FIELDS_ARE_FLOW
from stages._io_utils import safe_to_parquet, safe_read_parquet
from capacity_sem.data.loader import build_qpr_quarterly


def load_quarterly_data() -> pd.DataFrame:
    """Load or build grantee-disaster quarterly QPR data."""
    quarterly_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
    if quarterly_path.exists():
        return safe_read_parquet(quarterly_path)

    qpr_path = DATA_WORK_DIR / "qpr_raw.parquet"
    if not qpr_path.exists():
        raise FileNotFoundError(
            f"QPR data not found at {qpr_path}. Run ingest_data first."
        )
    qpr_raw = safe_read_parquet(qpr_path)
    return build_qpr_quarterly(qpr_raw, flows_are_net=QPR_DOLLAR_FIELDS_ARE_FLOW)


def load_covariates() -> dict:
    """Load all covariate files."""
    covariates = {}

    for name in ['population', 'grantee_disaster_population', 'severity', 'employment']:
        path = DATA_WORK_DIR / f"{name}.parquet"
        if path.exists():
            covariates[name] = safe_read_parquet(path)

    return covariates


def merge_population(
    df: pd.DataFrame,
    pop_df: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Merge population data with main DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Main data with grantee column.
    pop_df : pd.DataFrame
        Population data with grantee and population columns.
    grantee_col : str
        Column name for grantee identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with population merged.
    """
    if pop_df is None or pop_df.empty:
        return df

    # Ensure proper column names
    if 'grantee' in pop_df.columns and grantee_col not in pop_df.columns:
        pop_df = pop_df.rename(columns={'grantee': grantee_col})

    if disaster_col not in pop_df.columns and 'Disaster_Type' in pop_df.columns:
        pop_df = pop_df.rename(columns={'Disaster_Type': disaster_col})

    # Merge by grantee only or grantee + disaster if available
    merge_cols = [grantee_col]
    if disaster_col in pop_df.columns:
        merge_cols.append(disaster_col)

    df = df.merge(pop_df, on=merge_cols, how='left')

    return df


def merge_severity(
    df: pd.DataFrame,
    sev_df: pd.DataFrame,
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Merge severity data with main DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Main data with disaster column.
    sev_df : pd.DataFrame
        Severity data with disaster and severity columns.
    disaster_col : str
        Column name for disaster identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with severity merged.
    """
    if sev_df is None or sev_df.empty:
        return df

    # Ensure proper column names
    if disaster_col not in sev_df.columns:
        if 'disaster' in sev_df.columns:
            sev_df = sev_df.rename(columns={'disaster': disaster_col})
        elif 'Disaster_Type' in sev_df.columns:
            sev_df = sev_df.rename(columns={'Disaster_Type': disaster_col})

    # Merge
    df = df.merge(sev_df, on=disaster_col, how='left')

    return df


def create_grantee_disaster_panel(
    df: pd.DataFrame,
    grantee_col: str = 'Grantee',
    disaster_col: str = 'Disaster Type'
) -> pd.DataFrame:
    """
    Create grantee-disaster level panel from quarterly data.

    Aggregates quarterly observations to grantee-disaster level,
    computing summary statistics for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Quarterly QPR data.
    grantee_col : str
        Column name for grantee identifier.
    disaster_col : str
        Column name for disaster identifier.

    Returns
    -------
    pd.DataFrame
        Panel at grantee-disaster level.
    """
    grouped = df.groupby([grantee_col, disaster_col])

    agg = {}
    def add_agg(col: str, func: str) -> None:
        if col not in agg:
            agg[col] = func
            return
        if isinstance(agg[col], list):
            agg[col].append(func)
        else:
            agg[col] = [agg[col], func]

    if 'QPR Fund Obligated Q $' in df.columns:
        add_agg('QPR Fund Obligated Q $', 'sum')
    elif QPR_DOLLAR_FIELDS_ARE_FLOW and 'QPR Fund Obligated $' in df.columns:
        add_agg('QPR Fund Obligated $', 'sum')
    if 'QPR Fund Disbursed Q $' in df.columns:
        add_agg('QPR Fund Disbursed Q $', 'sum')
    elif QPR_DOLLAR_FIELDS_ARE_FLOW and 'QPR Fund Disbursed $' in df.columns:
        add_agg('QPR Fund Disbursed $', 'sum')
    if 'QPR Fund Expended Q $' in df.columns:
        add_agg('QPR Fund Expended Q $', 'sum')
    elif QPR_DOLLAR_FIELDS_ARE_FLOW and 'QPR Fund Expended $' in df.columns:
        add_agg('QPR Fund Expended $', 'sum')

    if 'QPR Fund Obligated $' in df.columns:
        add_agg('QPR Fund Obligated $', 'max')
    if 'QPR Fund Disbursed $' in df.columns:
        add_agg('QPR Fund Disbursed $', 'max')
    if 'QPR Fund Expended $' in df.columns:
        add_agg('QPR Fund Expended $', 'max')
    if 'QPR Actual Quarter' in df.columns:
        agg['QPR Actual Quarter'] = 'nunique'

    panel = grouped.agg(agg).reset_index()

    # Flatten column names
    panel.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in panel.columns
    ]

    # Rename for clarity
    panel = panel.rename(columns={
        'QPR Fund Obligated Q $_sum': 'Total_Obligated',
        'QPR Fund Disbursed Q $_sum': 'Total_Disbursed',
        'QPR Fund Expended Q $_sum': 'Total_Expended',
        'QPR Fund Obligated $_sum': 'Total_Obligated',
        'QPR Fund Disbursed $_sum': 'Total_Disbursed',
        'QPR Fund Expended $_sum': 'Total_Expended',
        'QPR Fund Obligated $_max': 'Max_Obligated',
        'QPR Fund Disbursed $_max': 'Max_Disbursed',
        'QPR Fund Expended $_max': 'Max_Expended',
        'QPR Actual Quarter_nunique': 'N_Quarters'
    })

    return panel


def compute_ratios(
    df: pd.DataFrame,
    qpr_quarterly: Optional[pd.DataFrame] = None,
    ratio_definition: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute key financial ratios.

    Computes both standard ratios (from original cumulative, may exceed 1.0)
    and clean ratios (from monotonic cumulative, guaranteed [0,1]).

    Parameters
    ----------
    df : pd.DataFrame
        Panel with funding columns.
    qpr_quarterly : pd.DataFrame, optional
        Quarterly data for mean_cumulative calculation.
    ratio_definition : str, optional
        How to compute ratios: "mean_cumulative" or "final_cumulative".

    Returns
    -------
    pd.DataFrame
        Panel with ratio columns and quality flags added.
    """
    df = df.copy()
    ratio_definition = ratio_definition or RATIO_DEFINITION
    if ratio_definition == "mean_cumulative" and qpr_quarterly is None:
        ratio_definition = "final_cumulative"

    def pick_total(col: str) -> Optional[str]:
        if QPR_DOLLAR_FIELDS_ARE_FLOW and f"Total_{col}" in df.columns:
            return f"Total_{col}"
        if f"Max_{col}" in df.columns:
            return f"Max_{col}"
        if f"Total_{col}" in df.columns:
            return f"Total_{col}"
        return None

    if ratio_definition == "mean_cumulative" and qpr_quarterly is not None:
        if {'QPR Fund Obligated $', 'QPR Fund Disbursed $'}.issubset(qpr_quarterly.columns):
            ratios = qpr_quarterly.copy()

            # Standard ratios from original cumulative (may exceed 1.0 due to adjustments)
            ratios['Ratio_disbursed_to_obligated'] = np.where(
                ratios['QPR Fund Obligated $'] > 0,
                ratios['QPR Fund Disbursed $'] / ratios['QPR Fund Obligated $'],
                np.nan
            )
            if 'QPR Fund Expended $' in ratios.columns:
                ratios['Ratio_expended_to_disbursed'] = np.where(
                    ratios['QPR Fund Disbursed $'] > 0,
                    ratios['QPR Fund Expended $'] / ratios['QPR Fund Disbursed $'],
                    np.nan
                )

            # Clean ratios from monotonic cumulative (guaranteed [0,1])
            if 'QPR Fund Obligated Clean $' in ratios.columns and 'QPR Fund Disbursed Clean $' in ratios.columns:
                ratios['Ratio_disbursed_to_obligated_clean'] = np.where(
                    ratios['QPR Fund Obligated Clean $'] > 0,
                    ratios['QPR Fund Disbursed Clean $'] / ratios['QPR Fund Obligated Clean $'],
                    np.nan
                )
                if 'QPR Fund Expended Clean $' in ratios.columns:
                    ratios['Ratio_expended_to_disbursed_clean'] = np.where(
                        ratios['QPR Fund Disbursed Clean $'] > 0,
                        ratios['QPR Fund Expended Clean $'] / ratios['QPR Fund Disbursed Clean $'],
                        np.nan
                    )

            # Aggregate to grantee-disaster level
            ratio_cols = ['Ratio_disbursed_to_obligated']
            if 'Ratio_expended_to_disbursed' in ratios.columns:
                ratio_cols.append('Ratio_expended_to_disbursed')
            if 'Ratio_disbursed_to_obligated_clean' in ratios.columns:
                ratio_cols.append('Ratio_disbursed_to_obligated_clean')
            if 'Ratio_expended_to_disbursed_clean' in ratios.columns:
                ratio_cols.append('Ratio_expended_to_disbursed_clean')

            ratios = ratios.groupby(['Grantee', 'Disaster Type'])[ratio_cols].mean().reset_index()
            df = df.merge(ratios, on=['Grantee', 'Disaster Type'], how='left')

    if ratio_definition != "mean_cumulative":
        obligated_col = pick_total("Obligated")
        disbursed_col = pick_total("Disbursed")
        expended_col = pick_total("Expended")

        if disbursed_col and obligated_col:
            df['Ratio_disbursed_to_obligated'] = np.where(
                df[obligated_col] > 0,
                df[disbursed_col] / df[obligated_col],
                np.nan
            )
        if expended_col and disbursed_col:
            df['Ratio_expended_to_disbursed'] = np.where(
                df[disbursed_col] > 0,
                df[expended_col] / df[disbursed_col],
                np.nan
            )

    obligated_col = pick_total("Obligated")
    expended_col = pick_total("Expended")
    if expended_col and obligated_col:
        df['Ratio_expended_to_obligated'] = np.where(
            df[obligated_col] > 0,
            df[expended_col] / df[obligated_col],
            np.nan
        )

    if 'Ratio_expended_to_obligated' in df.columns:
        df['Ratio_obligated_funds_fully_expended'] = df['Ratio_expended_to_obligated']

    # Add quality flags for anomalous ratios (preserve raw values for investigation)
    df = _add_ratio_quality_flags(df)

    return df


def _add_ratio_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quality flags for anomalous ratio values.

    Flags ratios that exceed 1.0 (logically impossible in accounting terms)
    and zero-denominator cases. Raw values are preserved for investigation.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ratio columns.

    Returns
    -------
    pd.DataFrame
        Panel with QA flags added.
    """
    df = df.copy()

    # Flag ratios > 1.0 (impossible values indicating data quality issues)
    if 'Ratio_disbursed_to_obligated' in df.columns:
        df['QA_ratio_disbursed_exceeds_one'] = df['Ratio_disbursed_to_obligated'] > 1.0

    if 'Ratio_expended_to_disbursed' in df.columns:
        df['QA_ratio_expended_exceeds_one'] = df['Ratio_expended_to_disbursed'] > 1.0

    if 'Ratio_expended_to_obligated' in df.columns:
        df['QA_ratio_expended_obligated_exceeds_one'] = df['Ratio_expended_to_obligated'] > 1.0

    # Flag zero denominators (NaN ratios due to division by zero)
    if 'Ratio_disbursed_to_obligated' in df.columns:
        df['QA_zero_obligated'] = df['Ratio_disbursed_to_obligated'].isna()

    if 'Ratio_expended_to_disbursed' in df.columns:
        df['QA_zero_disbursed'] = df['Ratio_expended_to_disbursed'].isna()

    return df


def scale_covariates(
    df: pd.DataFrame,
    covariate_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    Scale covariates for SEM analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with covariate columns.
    covariate_cols : list, optional
        Columns to scale. If None, scales common covariates.

    Returns
    -------
    pd.DataFrame
        Panel with scaled covariate columns added.
    """
    df = df.copy()

    if covariate_cols is None:
        covariate_cols = [
            'Population', 'Severity_Index', 'Experience_Index', 'Employment'
        ]

    for col in covariate_cols:
        if col in df.columns:
            # Z-score scaling
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}_scaled'] = (df[col] - mean) / std
            else:
                df[f'{col}_scaled'] = 0

            # Log-scaled version for Population
            if col == 'Population':
                df['Population_log'] = np.log1p(df[col])
                df['Population_log_scaled'] = (
                    df['Population_log'] - df['Population_log'].mean()
                ) / df['Population_log'].std()

    return df


def main():
    """Main entry point for panel construction stage."""
    print("=" * 60)
    print("Stage 01: Panel Construction")
    print("=" * 60)

    DATA_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading ingested data...")
    try:
        qpr = load_quarterly_data()
        print(f"  QPR data: {len(qpr):,} rows")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Run 'python src/pipeline.py ingest_data' first.")
        return

    # Load covariates
    covariates = load_covariates()
    for name, df in covariates.items():
        print(f"  {name}: {len(df):,} rows")

    # Create panel
    print("\nCreating grantee-disaster panel...")
    panel = create_grantee_disaster_panel(qpr)
    print(f"  Panel: {len(panel):,} observations")

    # Merge covariates
    print("\nMerging covariates...")
    if 'grantee_disaster_population' in covariates:
        panel = merge_population(panel, covariates['grantee_disaster_population'])
    elif 'population' in covariates:
        panel = merge_population(panel, covariates['population'])
    if 'severity' in covariates:
        panel = merge_severity(panel, covariates['severity'])

    # Compute ratios
    print("\nComputing financial ratios...")
    panel = compute_ratios(panel, qpr, RATIO_DEFINITION)

    # Scale covariates
    print("Scaling covariates...")
    panel = scale_covariates(panel)

    # Save
    output_path = DATA_WORK_DIR / "panel.parquet"
    safe_to_parquet(panel, output_path)
    print(f"\n  Saved panel → {output_path}")
    print(f"  Columns: {list(panel.columns)}")

    print("\n✓ Panel construction complete")


if __name__ == "__main__":
    main()
