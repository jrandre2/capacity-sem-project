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

# Column mapping from raw CSV to standardized names
COLUMN_MAPPING = {
    # Map raw column names to standardized names
    # Add mappings as needed based on actual CSV structure
    # Example:
    # 'Raw Column Name': 'Standardized Name',
}

# Year mappings for special cases in Appropriation parsing
YEAR_MAPPINGS = {
    # Map special year values to standardized format
    # Example:
    # 'FY2019': '2019',
}

# =============================================================================
# Analysis Thresholds
# =============================================================================

COMPLETION_THRESHOLD = 0.95  # 95% completion threshold for timeliness metrics

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
