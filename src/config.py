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

# AFT distributions to compare
AFT_DISTRIBUTIONS = ['weibull', 'lognormal', 'loglogistic']

# Cox regularization penalty (for small samples)
COX_PENALIZER = 0.1

# ==============================================================================
# Time-Varying Survival Analysis Configuration
# ==============================================================================

# Number of quarters to lag capacity ratios (avoid contemporaneous correlation)
TV_LAG_QUARTERS = 1

# Bootstrap iterations for clustered standard errors
BOOTSTRAP_ITERATIONS = 1000

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
