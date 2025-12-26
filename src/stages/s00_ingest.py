"""
Stage 00: Data Ingestion

Load QPR data and external covariates (population, severity, employment).

Commands:
    python src/pipeline.py ingest_data [--demo]

Outputs:
    data_work/qpr_raw.parquet     - Raw QPR data
    data_work/covariates.parquet  - External covariates
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from config import DATA_RAW_DIR, DATA_WORK_DIR, QPR_DATA_FILE

# Import from existing modules
from capacity_sem.data.loader import (
    load_qpr_data,
    get_disaster_events,
    get_grantees,
    get_years,
    get_data_summary,
)

from capacity_sem.data.external_data import (
    # Embedded data dictionaries
    GRANTEE_POPULATION_BY_DECADE,
    GRANTEE_POPULATION_DATA,
    DISASTER_SEVERITY_INDEX,
    GRANTEE_EMPLOYMENT_BY_YEAR,
    DRGR_TO_FEMA_MAPPING,
    DRGR_DISASTER_YEARS,
    # Population functions
    get_embedded_population,
    get_population_for_disaster,
    get_population_by_decade,
    # Severity functions
    get_disaster_severity,
    get_disaster_severity_components,
    get_severity_for_all_disasters,
    # Employment functions
    get_employment_for_year,
    get_employment_for_all_grantees,
    compute_employment_ratio,
    # FEMA/disaster functions
    map_drgr_disaster_to_fema,
    create_grantee_to_fips_mapping,
    get_all_external_data,
    get_covariates_simple,
)


def ingest_qpr_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate QPR data.

    Parameters
    ----------
    filepath : Path, optional
        Path to QPR CSV file. Uses default if not provided.

    Returns
    -------
    pd.DataFrame
        Validated QPR data.
    """
    df = load_qpr_data(filepath)

    # Basic validation
    required_cols = ['Grantee', 'Grant', 'Appropriation', 'Disaster Type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def ingest_covariates(
    grantees: Optional[list] = None,
    disasters: Optional[list] = None
) -> dict:
    """
    Load all external covariates.

    Parameters
    ----------
    grantees : list, optional
        List of grantee names.
    disasters : list, optional
        List of disaster types.

    Returns
    -------
    dict
        Dictionary with population, severity, and employment DataFrames.
    """
    result = {}

    # Population data
    result['population'] = get_embedded_population(grantees)

    # Severity data
    result['severity'] = get_severity_for_all_disasters()

    # Employment data
    result['employment'] = get_employment_for_all_grantees()

    # Grantee-disaster level data (if both provided)
    if grantees and disasters:
        all_data = get_all_external_data(grantees, disasters)
        result.update(all_data)

    return result


def main(demo: bool = False):
    """
    Main entry point for data ingestion stage.

    Parameters
    ----------
    demo : bool
        If True, use demo/synthetic data.
    """
    print("=" * 60)
    print("Stage 00: Data Ingestion")
    print("=" * 60)

    DATA_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Check for QPR data
    qpr_path = DATA_RAW_DIR / QPR_DATA_FILE

    if demo or not qpr_path.exists():
        print("\nNo QPR data found. Ingesting covariates only...")
        covariates = ingest_covariates()

        # Save covariates
        for name, df in covariates.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                output_path = DATA_WORK_DIR / f"{name}.parquet"
                df.to_parquet(output_path, index=False)
                print(f"  Saved {name}: {len(df)} rows → {output_path}")
    else:
        print(f"\nLoading QPR data from: {qpr_path}")
        df = ingest_qpr_data(qpr_path)

        # Get summary
        summary = get_data_summary(df)
        print(f"\n  Rows: {summary['total_rows']:,}")
        print(f"  Disasters: {summary['n_disasters']}")
        print(f"  Grantees: {summary['n_grantees']}")
        print(f"  Years: {summary['year_range']}")

        # Save raw data
        output_path = DATA_WORK_DIR / "qpr_raw.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\n  Saved QPR data → {output_path}")

        # Ingest covariates
        grantees = get_grantees(df)
        disasters = get_disaster_events(df)
        covariates = ingest_covariates(grantees, disasters)

        for name, df in covariates.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                output_path = DATA_WORK_DIR / f"{name}.parquet"
                df.to_parquet(output_path, index=False)
                print(f"  Saved {name}: {len(df)} rows → {output_path}")

    print("\n✓ Data ingestion complete")


if __name__ == "__main__":
    main()
