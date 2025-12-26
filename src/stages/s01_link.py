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

from config import DATA_WORK_DIR


def load_ingested_data() -> pd.DataFrame:
    """Load ingested QPR data."""
    qpr_path = DATA_WORK_DIR / "qpr_raw.parquet"
    if not qpr_path.exists():
        raise FileNotFoundError(
            f"QPR data not found at {qpr_path}. Run ingest_data first."
        )
    return pd.read_parquet(qpr_path)


def load_covariates() -> dict:
    """Load all covariate files."""
    covariates = {}

    for name in ['population', 'severity', 'employment']:
        path = DATA_WORK_DIR / f"{name}.parquet"
        if path.exists():
            covariates[name] = pd.read_parquet(path)

    return covariates


def merge_population(
    df: pd.DataFrame,
    pop_df: pd.DataFrame,
    grantee_col: str = 'Grantee'
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

    # Merge
    df = df.merge(pop_df, on=grantee_col, how='left')

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
    if 'disaster' in sev_df.columns and disaster_col not in sev_df.columns:
        sev_df = sev_df.rename(columns={'disaster': disaster_col})

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
    # Group by grantee-disaster
    grouped = df.groupby([grantee_col, disaster_col])

    # Compute aggregates
    panel = grouped.agg({
        'QPR Fund Obligated $': ['sum', 'max'],
        'QPR Fund Expended $': ['sum', 'max'],
        'QPR Actual Quarter': 'nunique'
    }).reset_index()

    # Flatten column names
    panel.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in panel.columns
    ]

    # Rename for clarity
    panel = panel.rename(columns={
        'QPR Fund Obligated $_sum': 'Total_Obligated',
        'QPR Fund Obligated $_max': 'Max_Obligated',
        'QPR Fund Expended $_sum': 'Total_Expended',
        'QPR Fund Expended $_max': 'Max_Expended',
        'QPR Actual Quarter_nunique': 'N_Quarters'
    })

    return panel


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key financial ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with funding columns.

    Returns
    -------
    pd.DataFrame
        Panel with ratio columns added.
    """
    df = df.copy()

    # Disbursement ratio
    if 'Total_Disbursed' in df.columns and 'Total_Obligated' in df.columns:
        df['Ratio_disbursed_to_obligated'] = np.where(
            df['Total_Obligated'] > 0,
            df['Total_Disbursed'] / df['Total_Obligated'],
            np.nan
        )

    # Expenditure to disbursement ratio
    if 'Total_Expended' in df.columns and 'Total_Disbursed' in df.columns:
        df['Ratio_expended_to_disbursed'] = np.where(
            df['Total_Disbursed'] > 0,
            df['Total_Expended'] / df['Total_Disbursed'],
            np.nan
        )

    # Expenditure to obligated ratio
    if 'Total_Expended' in df.columns and 'Total_Obligated' in df.columns:
        df['Ratio_expended_to_obligated'] = np.where(
            df['Total_Obligated'] > 0,
            df['Total_Expended'] / df['Total_Obligated'],
            np.nan
        )

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
        qpr = load_ingested_data()
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
    if 'population' in covariates:
        panel = merge_population(panel, covariates['population'])
    if 'severity' in covariates:
        panel = merge_severity(panel, covariates['severity'])

    # Compute ratios
    print("\nComputing financial ratios...")
    panel = compute_ratios(panel)

    # Scale covariates
    print("Scaling covariates...")
    panel = scale_covariates(panel)

    # Save
    output_path = DATA_WORK_DIR / "panel.parquet"
    panel.to_parquet(output_path, index=False)
    print(f"\n  Saved panel → {output_path}")
    print(f"  Columns: {list(panel.columns)}")

    print("\n✓ Panel construction complete")


if __name__ == "__main__":
    main()
