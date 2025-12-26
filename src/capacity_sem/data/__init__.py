"""Data loading, cleaning, and transformation modules."""

from .loader import load_qpr_data, get_disaster_events, get_grantees, get_years, get_data_summary
from .cleaner import clean_qpr_data, validate_data, validate_quarter_sums_vs_totals
from .transformer import filter_by_criteria, aggregate_by_quarter, compute_cumulative_funding
from .external_data import (
    # API-based methods (require Census API key)
    fetch_census_population,
    fetch_fema_disaster_declarations,
    get_disaster_summary,
    create_grantee_to_fips_mapping,
    map_drgr_disaster_to_fema,
    fetch_population_for_grantees,
    build_covariate_dataset,
    DRGR_TO_FEMA_MAPPING,
    DRGR_DISASTER_YEARS,
    # Simple methods (NO API key required - recommended)
    get_embedded_population,
    get_population_for_disaster,
    get_census_decade,
    get_population_by_decade,
    download_fema_declarations_csv,
    get_covariates_simple,
    get_all_external_data,
    GRANTEE_POPULATION_DATA,
    GRANTEE_POPULATION_BY_DECADE,
    # Disaster severity data and functions
    DISASTER_SEVERITY_INDEX,
    get_disaster_severity,
    get_disaster_severity_components,
    get_severity_for_all_disasters,
    # Employment data and functions
    GRANTEE_EMPLOYMENT_BY_YEAR,
    get_employment_for_year,
    compute_employment_ratio,
    get_employment_for_all_grantees,
)
from .panel_data import (
    create_panel_structure,
    standardize_time_points,
    create_lagged_variables,
    build_ratio_panel,
    build_clpm_dataset,
    compute_growth_parameters,
    compute_ratios_at_timepoint,
)
