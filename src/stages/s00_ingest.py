"""
Stage 00: Data Ingestion

Load QPR data and external covariates (population, severity, employment).

Commands:
    python src/pipeline.py ingest_data [--demo]

Outputs:
    data_work/qpr_raw.parquet                   - Raw QPR data
    data_work/qpr_clean.parquet                 - Cleaned QPR data with QA flags
    data_work/qpr_quarterly.parquet             - Quarterly rollup with cumulative totals
    data_work/quality/qpr_quality_report.csv    - QPR data quality summary
    data_work/quality/qpr_quarterly_quality_report.csv - Quarterly rollup quality summary
    data_work/population.parquet                - Grantee population covariates
    data_work/grantee_disaster_population.parquet - Grantee-disaster population covariates
    data_work/severity.parquet                  - Disaster severity covariates
    data_work/employment.parquet                - Employment covariates
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from config import (
    DATA_RAW_DIR,
    DATA_WORK_DIR,
    QPR_DATA_FILE,
    QPR_CLEAN_FILE,
    QPR_QUALITY_REPORT_FILE,
    QPR_QUARTERLY_QUALITY_REPORT_FILE,
    STATE_GOVERNMENTS,
    LOCAL_GOVERNMENTS,
    QPR_DOLLAR_FIELDS_ARE_FLOW,
)
from stages._io_utils import safe_to_parquet

# Import from existing modules
from capacity_sem.data.loader import (
    load_qpr_data,
    build_qpr_quarterly,
    get_disaster_events,
    get_grantees,
    get_years,
    get_data_summary,
)
from capacity_sem.data.quality import clean_qpr_data, build_quarterly_quality_report

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


def generate_demo_qpr_data(
    n_quarters: int = 8,
    seed: int = 42
) -> pd.DataFrame:
    """Generate a small synthetic QPR dataset for pipeline testing."""
    rng = np.random.default_rng(seed)

    n_states = min(20, len(STATE_GOVERNMENTS))
    n_locals = min(20, len(LOCAL_GOVERNMENTS))
    grantees = (
        rng.choice(STATE_GOVERNMENTS, size=n_states, replace=False).tolist()
        + rng.choice(LOCAL_GOVERNMENTS, size=n_locals, replace=False).tolist()
    )
    disaster_pool = list(DISASTER_SEVERITY_INDEX.keys())
    disasters = rng.choice(disaster_pool, size=len(grantees), replace=True).tolist()
    activity_types = [
        'Affordable Rental Housing',
        'Construction/reconstruction of streets',
        'Administration',
        'Econ. development or recovery activity that creates/retains jobs',
    ]

    rows = []
    def make_curve(start: float, end: float, length: int, noise: float = 0.02) -> np.ndarray:
        base = np.linspace(start, end, length)
        jitter = rng.normal(0, noise, length)
        curve = np.clip(base + jitter, 0, 1)
        curve = np.maximum.accumulate(curve)
        curve[-1] = end
        return curve

    for idx, (grantee, disaster) in enumerate(zip(grantees, disasters)):
        final_obligated = rng.uniform(50_000_000, 300_000_000)
        n_q = rng.integers(6, 13)
        obligated_share = make_curve(0.2, 1.0, n_q)
        disbursed_end = rng.uniform(0.95, 0.99)
        disbursed_share = make_curve(0.05, disbursed_end, n_q)
        disbursed_share = np.minimum(disbursed_share, obligated_share)
        expended_end = rng.uniform(0.95, min(0.99, disbursed_end))
        expended_share = make_curve(0.02, expended_end, n_q)
        expended_share = np.minimum(expended_share, disbursed_share)

        obligated_cum = final_obligated * obligated_share
        disbursed_cum = final_obligated * disbursed_share
        expended_cum = final_obligated * expended_share
        obligated_q = np.diff(obligated_cum, prepend=0)
        disbursed_q = np.diff(disbursed_cum, prepend=0)
        expended_q = np.diff(expended_cum, prepend=0)

        start_year = 2016 + (idx % 5)
        start_quarter = (idx % 4) + 1

        for q in range(n_q):
            year = start_year + ((start_quarter + q - 1) // 4)
            quarter = ((start_quarter + q - 1) % 4) + 1
            rows.append({
                'Grantee': grantee,
                'Grant': f"DEMO-{idx:03d}",
                'Appropriation': f"FY{year} DEMO",
                'Disaster Type': disaster,
                'QPR Fund Obligated $': obligated_q[q],
                'QPR Fund Disbursed $': disbursed_q[q],
                'QPR Fund Expended $': expended_q[q],
                'QPR Fund Expended Q $': expended_q[q],
                'QPR Actual Quarter': f"{year} Q{quarter}",
                'Activity Type': rng.choice(activity_types),
            })

    return pd.DataFrame(rows)


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
        if demo:
            print("\nUsing demo QPR data...")
            df = generate_demo_qpr_data()
            output_path = DATA_WORK_DIR / "qpr_raw.parquet"
            safe_to_parquet(df, output_path)
            print(f"  Saved demo QPR data → {output_path}")
            clean_df, report = clean_qpr_data(df)
            clean_path = DATA_WORK_DIR / QPR_CLEAN_FILE
            safe_to_parquet(clean_df, clean_path)
            print(f"  Saved cleaned QPR data → {clean_path}")
            quality_dir = DATA_WORK_DIR / "quality"
            quality_dir.mkdir(parents=True, exist_ok=True)
            report_path = quality_dir / QPR_QUALITY_REPORT_FILE
            report.to_csv(report_path, index=False)
            print(f"  Saved QPR quality report → {report_path}")
            quarterly = build_qpr_quarterly(clean_df, flows_are_net=QPR_DOLLAR_FIELDS_ARE_FLOW)
            quarterly_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
            safe_to_parquet(quarterly, quarterly_path)
            print(f"  Saved quarterly QPR data → {quarterly_path}")
            quarterly_report = build_quarterly_quality_report(quarterly)
            quarterly_report_path = quality_dir / QPR_QUARTERLY_QUALITY_REPORT_FILE
            quarterly_report.to_csv(quarterly_report_path, index=False)
            print(f"  Saved quarterly QPR quality report → {quarterly_report_path}")
            grantees = get_grantees(clean_df)
            disasters = get_disaster_events(clean_df)
            covariates = ingest_covariates(grantees, disasters)
        else:
            print("\nNo QPR data found. Ingesting covariates only...")
            covariates = ingest_covariates()

        # Save covariates
        for name, df in covariates.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                output_path = DATA_WORK_DIR / f"{name}.parquet"
                safe_to_parquet(df, output_path)
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
        safe_to_parquet(df, output_path)
        print(f"\n  Saved QPR data → {output_path}")
        clean_df, report = clean_qpr_data(df)
        clean_path = DATA_WORK_DIR / QPR_CLEAN_FILE
        safe_to_parquet(clean_df, clean_path)
        print(f"  Saved cleaned QPR data → {clean_path}")
        quality_dir = DATA_WORK_DIR / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        report_path = quality_dir / QPR_QUALITY_REPORT_FILE
        report.to_csv(report_path, index=False)
        print(f"  Saved QPR quality report → {report_path}")
        quarterly = build_qpr_quarterly(clean_df, flows_are_net=QPR_DOLLAR_FIELDS_ARE_FLOW)
        quarterly_path = DATA_WORK_DIR / "qpr_quarterly.parquet"
        safe_to_parquet(quarterly, quarterly_path)
        print(f"  Saved quarterly QPR data → {quarterly_path}")
        quarterly_report = build_quarterly_quality_report(quarterly)
        quarterly_report_path = quality_dir / QPR_QUARTERLY_QUALITY_REPORT_FILE
        quarterly_report.to_csv(quarterly_report_path, index=False)
        print(f"  Saved quarterly QPR quality report → {quarterly_report_path}")

        # Ingest covariates
        grantees = get_grantees(clean_df)
        disasters = get_disaster_events(clean_df)
        covariates = ingest_covariates(grantees, disasters)

        for name, df in covariates.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                output_path = DATA_WORK_DIR / f"{name}.parquet"
                safe_to_parquet(df, output_path)
                print(f"  Saved {name}: {len(df)} rows → {output_path}")

    print("\n✓ Data ingestion complete")


if __name__ == "__main__":
    main()
