"""
Configuration constants for the Capacity-SEM project.

This module defines paths, column mappings, and classification constants
used throughout the analysis pipeline.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_WORK_DIR = PROJECT_ROOT / "data_work"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# =============================================================================
# QPR Data Configuration
# =============================================================================

QPR_DATA_FILE = "qpr_data.csv"  # Update with actual filename when available
QPR_QUARTERLY_FILE = "qpr_quarterly.parquet"
QPR_CLEAN_FILE = "qpr_clean.parquet"
QPR_QUALITY_REPORT_FILE = "qpr_quality_report.csv"
QPR_QUARTERLY_QUALITY_REPORT_FILE = "qpr_quarterly_quality_report.csv"

# QPR fields in the raw export represent quarterly net changes (not cumulative)
QPR_DOLLAR_FIELDS_ARE_FLOW = True

# Ratio definition: "mean_cumulative" (mean of cumulative ratios) or "final_cumulative"
RATIO_DEFINITION = "mean_cumulative"

# Column mapping from raw CSV to standardized names
COLUMN_MAPPING = {
    # Map raw column names to standardized names
    # Add mappings as needed based on actual CSV structure
    'QPR Funds Obligated $': 'QPR Fund Obligated $',
    'QPR Grant Disbursed $': 'QPR Fund Disbursed $',
}

# Year mappings for special cases in Appropriation parsing
YEAR_MAPPINGS = {
    # Map special year values to standardized format
    # Example:
    # 'FY2019': '2019',
    'MIT': '2015-2018',
}

# =============================================================================
# ETL Standardization Configuration
# =============================================================================

# Ratio denominator choice
RATIO_DENOMINATOR_METHOD = "final"
# Options:
#   - "final": Use last reported obligated amount (default)
#   - "max": Use maximum obligated amount across all quarters
#   - "quarter": Use quarter-specific obligated (NOT RECOMMENDED - creates artifacts)

# Velocity winsorization
VELOCITY_WINSOR_PERCENTILES = (0.01, 0.99)  # (lower, upper)
# Set to None to disable winsorization
# Recommended: (0.01, 0.99) for 1%/99% or (0.05, 0.95) for 5%/95%

# Extreme change flagging threshold
VELOCITY_EXTREME_THRESHOLD = 100
# Flag velocity changes >100 pp/quarter as extreme (physically implausible)
# These observations get QA flag but are not removed

# Rolling window sizes for velocity
VELOCITY_ROLLING_WINDOWS = [2, 4]
# Quarter windows for rolling average velocity

# Data quality thresholds
ETL_QA_THRESHOLDS = {
    'max_negative_pct': 0.05,  # Warn if >5% of observations have negative values
    'max_cumulative_decrease_pct': 0.05,  # Warn if >5% of groups have cumulative decreases
    'max_extreme_velocity_pct': 0.01,  # Warn if >1% have extreme velocity after winsorization
}

# Adjustment tracking
TRACK_ADJUSTMENTS = True  # Track retroactive adjustments to financial data
ADJUSTMENT_THRESHOLD = 1000.0  # Minimum absolute dollar amount to track as adjustment

# =============================================================================
# Analysis Thresholds
# =============================================================================

COMPLETION_THRESHOLD = 0.95  # 95% completion threshold for timeliness metrics

# Multi-threshold duration analysis: compute duration at each threshold
# from 20% to 100% in 5% increments (17 thresholds total)
DURATION_THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 1.00
]

# =============================================================================
# Grantee Classifications
# =============================================================================

# State governments (from embedded population data)
STATE_GOVERNMENTS = [
    'Alabama',
    'Alaska',
    'American Samoa',
    'Arkansas',
    'California',
    'Colorado',
    'Connecticut - DOH',
    'Florida',
    'Georgia',
    'Illinois',
    'Indiana - OCRA',
    'Iowa',
    'Kentucky',
    'Louisiana',
    'Maryland',
    'Michigan',
    'Mississippi',
    'Missouri',
    'Nebraska',
    'New Jersey',
    'New York',
    'North Carolina',
    'Northern Mariana Islands',
    'rogco',  # Abbreviation for Northern Mariana Islands
    'Ohio',
    'Oklahoma',
    'Oregon',
    'Pennsylvania',
    'Puerto Rico',
    'Rhode Island',
    'South Carolina',
    'Tennessee',
    'Texas - GLO',
    'Virgin Islands',
    'Virginia',
    'Washington',
    'West Virginia',
    'Wisconsin',
]

# Local governments (from embedded population data)
LOCAL_GOVERNMENTS = [
    'Baton Rouge, LA',
    'Chicago, IL',
    'City of Birmingham',
    'Columbia, SC',
    'Cook County, IL',
    'County Of Orange',
    'Dallas, TX',
    'Dauphin County, PA',
    'Dearborn, MI',
    'Detroit, MI',
    'Empire State Development Corporation (NYS)',
    'Fort Worth, TX',
    'Hawaii County, HI',
    'Houston, TX',
    'Jefferson County, AL',
    'Jefferson Parish, LA',
    'Joplin, MO',
    'Kauai County, HI',
    'Lake Charles, LA',
    'Lee County, FL',
    'Lexington County, SC',
    'Luzerne County, PA',
    'Minot, ND',
    'Moore, OK',
    'Nashville-Davidson, TN',
    'New Orleans, LA',
    'New York City, NY',
    'North Carolina-NCORR',
    'Orange County, FL',
    'Philadelphia, PA',
    'Richland County, SC',
    'San Marcos, TX',
    'Sarasota County, FL',
    'Shelby County, TN',
    'Springfield, MA',
    'St. Clair County, IL',
    'St. Tammany Parish',
    'Town of Union, NY',
    'Tuscaloosa, AL',
    'Volusia County, FL',
]

# Combined list of all grantees
ALL_GRANTEES = STATE_GOVERNMENTS + LOCAL_GOVERNMENTS

# =============================================================================
# SEM Model Configuration
# =============================================================================

# Default model specification to use
DEFAULT_MODEL = 'full'

# Fit index thresholds for model evaluation
FIT_THRESHOLDS = {
    'cfi_good': 0.95,
    'cfi_acceptable': 0.90,
    'tli_good': 0.95,
    'tli_acceptable': 0.90,
    'rmsea_good': 0.05,
    'rmsea_acceptable': 0.08,
    'srmr_good': 0.05,
    'srmr_acceptable': 0.08,
}

# =============================================================================
# Survival Analysis Configuration
# =============================================================================

# Default capacity columns for survival analysis
SURVIVAL_CAPACITY_COLS = [
    'Ratio_disbursed_to_obligated',
    'Ratio_expended_to_disbursed',
]

# Alternative capacity indicator sets for exploratory analyses
CAPACITY_ALTERNATIVE_SETS = {
    'ratios': [
        'Ratio_disbursed_to_obligated',
        'Ratio_expended_to_disbursed',
    ],
    'absolute_log': [
        'Disbursed_log',
        'Expended_log',
    ],
    'velocity': [
        'Disbursement_Velocity',
        'Expenditure_Velocity',
    ],
    'velocity_pp': [
        'Disbursement_Velocity_pp',
        'Expenditure_Velocity_pp',
    ],
    'velocity_winsor': [
        'Disbursement_Velocity_winsor',
        'Expenditure_Velocity_winsor',
    ],
    'velocity_scaled': [
        'Disbursement_Velocity_scaled',
        'Expenditure_Velocity_scaled',
    ],
    'absolute_index': [
        'Capacity_Absolute_Index',
    ],
    'velocity_index': [
        'Capacity_Velocity_Index',
    ],
    'velocity_index_pp': [
        'Capacity_Velocity_Index_pp',
    ],
    'velocity_index_winsor': [
        'Capacity_Velocity_Index_winsor',
    ],
    'velocity_index_scaled': [
        'Capacity_Velocity_Index_scaled',
    ],
    'ratio_velocity_pp_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_index_pp_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_pp_threshold_interaction': [
        'Ratio_disbursed_to_obligated_high',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_high_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_index_pp_threshold_interaction': [
        'Ratio_disbursed_to_obligated_high',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_high_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_pp_threshold_q25_interaction': [
        'Ratio_disbursed_to_obligated_high_q25',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_high_q25_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_pp_threshold_q33_interaction': [
        'Ratio_disbursed_to_obligated_high_q33',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_high_q33_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_pp_threshold_q67_interaction': [
        'Ratio_disbursed_to_obligated_high_q67',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_high_q67_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_pp_threshold_q75_interaction': [
        'Ratio_disbursed_to_obligated_high_q75',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_high_q75_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_index_pp_threshold_q25_interaction': [
        'Ratio_disbursed_to_obligated_high_q25',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_high_q25_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_index_pp_threshold_q33_interaction': [
        'Ratio_disbursed_to_obligated_high_q33',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_high_q33_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_index_pp_threshold_q67_interaction': [
        'Ratio_disbursed_to_obligated_high_q67',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_high_q67_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_index_pp_threshold_q75_interaction': [
        'Ratio_disbursed_to_obligated_high_q75',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_high_q75_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_pp_spline_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_above_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_index_pp_spline_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_above_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_pp_spline_q25_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q25',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_above_q25_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_pp_spline_q33_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q33',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_above_q33_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_pp_spline_q67_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q67',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_above_q67_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_pp_spline_q75_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q75',
        'Disbursement_Velocity_pp_c',
        'Ratio_disbursed_to_obligated_above_q75_x_Disbursement_Velocity_pp',
    ],
    'ratio_velocity_index_pp_spline_q25_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q25',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_above_q25_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_index_pp_spline_q33_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q33',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_above_q33_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_index_pp_spline_q67_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q67',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_above_q67_x_Capacity_Velocity_Index_pp',
    ],
    'ratio_velocity_index_pp_spline_q75_interaction': [
        'Ratio_disbursed_to_obligated_c',
        'Ratio_disbursed_to_obligated_above_q75',
        'Capacity_Velocity_Index_pp_c',
        'Ratio_disbursed_to_obligated_above_q75_x_Capacity_Velocity_Index_pp',
    ],
    'velocity_early_2q_pp': [
        'Disbursement_Velocity_early_2q_pp',
        'Expenditure_Velocity_early_2q_pp',
    ],
    'velocity_early_3q_pp': [
        'Disbursement_Velocity_early_3q_pp',
        'Expenditure_Velocity_early_3q_pp',
    ],
    'velocity_early_4q_pp': [
        'Disbursement_Velocity_early_4q_pp',
        'Expenditure_Velocity_early_4q_pp',
    ],
    'velocity_early_6q_pp': [
        'Disbursement_Velocity_early_6q_pp',
        'Expenditure_Velocity_early_6q_pp',
    ],
    'velocity_index_early_2q_pp': [
        'Capacity_Velocity_Index_early_2q_pp',
    ],
    'velocity_index_early_3q_pp': [
        'Capacity_Velocity_Index_early_3q_pp',
    ],
    'velocity_index_early_4q_pp': [
        'Capacity_Velocity_Index_early_4q_pp',
    ],
    'velocity_index_early_6q_pp': [
        'Capacity_Velocity_Index_early_6q_pp',
    ],
    'velocity_index_early_4q_scaled': [
        'Capacity_Velocity_Index_early_4q_scaled',
    ],
    'velocity_index_early_6q_scaled': [
        'Capacity_Velocity_Index_early_6q_scaled',
    ],
    'velocity_fixed_12m_pp': [
        'Disbursement_Velocity_fixed_12m_pp',
        'Expenditure_Velocity_fixed_12m_pp',
    ],
    'velocity_fixed_18m_pp': [
        'Disbursement_Velocity_fixed_18m_pp',
        'Expenditure_Velocity_fixed_18m_pp',
    ],
    'velocity_index_fixed_12m_pp': [
        'Capacity_Velocity_Index_fixed_12m_pp',
    ],
    'velocity_index_fixed_18m_pp': [
        'Capacity_Velocity_Index_fixed_18m_pp',
    ],
    'velocity_index_fixed_12m_scaled': [
        'Capacity_Velocity_Index_fixed_12m_scaled',
    ],
    'velocity_index_fixed_18m_scaled': [
        'Capacity_Velocity_Index_fixed_18m_scaled',
    ],
    'quartiles_ratio': [
        'Capacity_Index_Q2',
        'Capacity_Index_Q3',
        'Capacity_Index_Q4',
    ],
    'quartiles_absolute': [
        'Capacity_Absolute_Index_Q2',
        'Capacity_Absolute_Index_Q3',
        'Capacity_Absolute_Index_Q4',
    ],
    'quartiles_velocity': [
        'Capacity_Velocity_Index_Q2',
        'Capacity_Velocity_Index_Q3',
        'Capacity_Velocity_Index_Q4',
    ],
    'quartiles_velocity_pp': [
        'Capacity_Velocity_Index_pp_Q2',
        'Capacity_Velocity_Index_pp_Q3',
        'Capacity_Velocity_Index_pp_Q4',
    ],
    'quartiles_velocity_winsor': [
        'Capacity_Velocity_Index_winsor_Q2',
        'Capacity_Velocity_Index_winsor_Q3',
        'Capacity_Velocity_Index_winsor_Q4',
    ],
    'quartiles_velocity_scaled': [
        'Capacity_Velocity_Index_scaled_Q2',
        'Capacity_Velocity_Index_scaled_Q3',
        'Capacity_Velocity_Index_scaled_Q4',
    ],
    'pca': [
        'Capacity_PCA1',
    ],
}

# AFT distributions to compare
AFT_DISTRIBUTIONS = ['weibull', 'lognormal', 'loglogistic']

# Cox regularization penalty (for small samples)
COX_PENALIZER = 0.1
COX_STRATIFIED_LOW_EVENT_THRESHOLD = 10
COX_STRATIFIED_LOW_EVENT_PENALIZER = 1.0

# ==============================================================================
# Time-Varying Survival Analysis Configuration
# ==============================================================================

# Number of quarters to lag capacity ratios (avoid contemporaneous correlation)
TV_LAG_QUARTERS = 1

# Additional lag options for robustness (lag0, lag2)
TV_LAG_OPTIONS = [0, 1, 2]

# Bootstrap iterations for clustered standard errors
BOOTSTRAP_ITERATIONS = 1000

# Early-window velocity definitions (quarters)
EARLY_VELOCITY_WINDOWS = [2, 3, 4, 6]
FIXED_VELOCITY_WINDOWS_MONTHS = [12, 18]
RATIO_INTERACTION_QUANTILES = [0.25, 0.33, 0.67, 0.75]
TV_VELOCITY_ROLLING_WINDOWS = [2, 4]

# Cluster variable for bootstrap resampling
BOOTSTRAP_CLUSTER_COL = 'Grantee'

# Static covariates for survival models
SURVIVAL_COVARIATE_COLS = [
    'Government_Type_State',
    'Log_Obligated',
    'Prior_Grant_Count',
    'Prior_Grant_Dollars_log',
    'Disaster_Year',
    'Population_log',
]

# =============================================================================
# Program Type Classification (Phase 3 Week 7)
# =============================================================================

# Map 51 activity types to 6 major categories for heterogeneity analysis
PROGRAM_TYPE_MAPPING = {
    'Housing': [
        'Rehabilitation/reconstruction of residential structures',
        'Affordable Rental Housing',
        'Construction of new housing',
        'Homeownership Assistance to low- and moderate-income',
        'Acquisition - buyout of residential properties',
        'Construction of new replacement housing',
        'Homeownership Assistance (with waiver only)',
        'Rental Assistance (waiver only)',
        'Housing incentives to encourage resettlement',
        'Payment for homeowner compensation (Mississippi only)',
        'Residential Location Incentive Grants - (Waiver only)',
        'Acquisition of property for replacement housing',
        'MIT - Residential New Construction',
        'MIT - Rehabilitation/reconstruction of residential structures',
        'MIT - Buyout of Properties',
    ],
    'Infrastructure': [
        'Construction/reconstruction of water/sewer lines or systems',
        'Construction/reconstruction of streets',
        'Rehabilitation/reconstruction of public facilities',
        'Rehabilitation/reconstruction of a public improvement',
        'Acquisition, construction,reconstruction of public facilities',
        'MIT - Public Facilities and Improvements-Non Covered Projects',
        'Dike/dam/stream-river bank repairs',
        'Construction/reconstruction of water lift stations',
        'Construction of buildings for the general conduct of government',
        'MIT - Public Facilities and Improvements-Covered Projects Only',
        'Privately owned utilities',
        'Electrical power system improvements',
        'Debris removal',
        'Public services',
        'MIT - Public Services and Information',
    ],
    'Economic Development': [
        'Econ. development or recovery activity that creates/retains jobs',
        'Tourism (Waiver Only)',
        'Economic Development Center (Virginia waiver only)',
        'Travel and Tourism per 107-117 - (WTC only)',
        'MIT - Economic Development',
        'Payment for compensation for economic losses (WTC-only)',
        'Compensation for disaster-related losses (Louisiana and Texas)',
    ],
    'Acquisition': [
        'Clearance and Demolition',
        'Relocation payments and assistance',
        'Acquisition - general',
        'Disposition',
        'Acquisition of relocation properties',
    ],
    'Administration': [
        'Planning',
        'Administration',
        'MIT - Planning and Capacity Building',
        'Code enforcement',
        'Capacity building for nonprofit or public entities',
    ],
    # Note: 'Other' category for unmapped activities is implicit
    # Includes: Rehabilitation/reconstruction of other non-residential structures,
    #           NDR - Environmental Value, Windpool Mitigation, etc.
}


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_etl_config():
    """Validate ETL standardization configuration."""
    import warnings as warn_module

    errors = []
    warnings = []

    # Check ratio denominator method
    valid_methods = ['final', 'max', 'quarter']
    if RATIO_DENOMINATOR_METHOD not in valid_methods:
        errors.append(f"RATIO_DENOMINATOR_METHOD must be one of {valid_methods}")

    if RATIO_DENOMINATOR_METHOD == 'quarter':
        warnings.append("RATIO_DENOMINATOR_METHOD='quarter' may create computational artifacts")

    # Check winsorization percentiles
    if VELOCITY_WINSOR_PERCENTILES is not None:
        lower, upper = VELOCITY_WINSOR_PERCENTILES
        if not (0 <= lower < upper <= 1):
            errors.append("VELOCITY_WINSOR_PERCENTILES must be (lower, upper) with 0 <= lower < upper <= 1")
        if lower > 0.1 or upper < 0.9:
            warnings.append("Aggressive winsorization (>10% trimmed) may remove legitimate variation")

    # Check extreme threshold
    if VELOCITY_EXTREME_THRESHOLD <= 0:
        errors.append("VELOCITY_EXTREME_THRESHOLD must be positive")

    # Report errors and warnings
    if errors:
        raise ValueError(f"ETL configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    if warnings:
        for w in warnings:
            warn_module.warn(f"ETL configuration warning: {w}", UserWarning)

    return True
