"""
External data acquisition functions for Census and FEMA data.

This module provides functions to fetch population data from the U.S. Census Bureau
and disaster declaration data from FEMA. Supports both API and direct CSV download methods.
"""

import os
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# API endpoints
FEMA_API_BASE = "https://www.fema.gov/api/open/v2"
FEMA_DISASTER_ENDPOINT = f"{FEMA_API_BASE}/DisasterDeclarationsSummaries"
CENSUS_ACS_BASE = "https://api.census.gov/data"

# Direct download URLs (no API key required)
FEMA_DECLARATIONS_CSV_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries.csv"

# Environment variable for Census API key
CENSUS_API_KEY_ENV = "CENSUS_API_KEY"

# Embedded population data by decade (2000, 2010, 2020 Census)
# This allows matching population to disaster year
GRANTEE_POPULATION_BY_DECADE: Dict[str, Dict[int, int]] = {
    # State populations (2000, 2010, 2020 Census)
    'Alabama': {2000: 4447100, 2010: 4779736, 2020: 5024279},
    'Alaska': {2000: 626932, 2010: 710231, 2020: 733391},
    'American Samoa': {2000: 57291, 2010: 55519, 2020: 55197},
    'Arkansas': {2000: 2673400, 2010: 2915918, 2020: 3011524},
    'California': {2000: 33871648, 2010: 37253956, 2020: 39538223},
    'Colorado': {2000: 4301261, 2010: 5029196, 2020: 5773714},
    'Connecticut - DOH': {2000: 3405565, 2010: 3574097, 2020: 3605944},
    'Florida': {2000: 15982378, 2010: 18801310, 2020: 21538187},
    'Georgia': {2000: 8186453, 2010: 9687653, 2020: 10711908},
    'Illinois': {2000: 12419293, 2010: 12830632, 2020: 12812508},
    'Indiana - OCRA': {2000: 6080485, 2010: 6483802, 2020: 6785528},
    'Iowa': {2000: 2926324, 2010: 3046355, 2020: 3190369},
    'Kentucky': {2000: 4041769, 2010: 4339367, 2020: 4505836},
    'Louisiana': {2000: 4468976, 2010: 4533372, 2020: 4657757},
    'Maryland': {2000: 5296486, 2010: 5773552, 2020: 6177224},
    'Michigan': {2000: 9938444, 2010: 9883640, 2020: 10077331},
    'Mississippi': {2000: 2844658, 2010: 2967297, 2020: 2961279},
    'Missouri': {2000: 5595211, 2010: 5988927, 2020: 6154913},
    'Nebraska': {2000: 1711263, 2010: 1826341, 2020: 1961504},
    'New Jersey': {2000: 8414350, 2010: 8791894, 2020: 9288994},
    'New York': {2000: 18976457, 2010: 19378102, 2020: 20201249},
    'North Carolina': {2000: 8049313, 2010: 9535483, 2020: 10439388},
    'Northern Mariana Islands': {2000: 69221, 2010: 53883, 2020: 47329},
    'Ohio': {2000: 11353140, 2010: 11536504, 2020: 11799448},
    'Oklahoma': {2000: 3450654, 2010: 3751351, 2020: 3959353},
    'Oregon': {2000: 3421399, 2010: 3831074, 2020: 4237256},
    'Pennsylvania': {2000: 12281054, 2010: 12702379, 2020: 13002700},
    'Puerto Rico': {2000: 3808610, 2010: 3725789, 2020: 3285874},
    'Rhode Island': {2000: 1048319, 2010: 1052567, 2020: 1097379},
    'South Carolina': {2000: 4012012, 2010: 4625364, 2020: 5118425},
    'Tennessee': {2000: 5689283, 2010: 6346105, 2020: 6910840},
    'Texas - GLO': {2000: 20851820, 2010: 25145561, 2020: 29145505},
    'Virgin Islands': {2000: 108612, 2010: 106405, 2020: 87146},
    'Virginia': {2000: 7078515, 2010: 8001024, 2020: 8631393},
    'Washington': {2000: 5894121, 2010: 6724540, 2020: 7705281},
    'West Virginia': {2000: 1808344, 2010: 1852994, 2020: 1793716},
    'Wisconsin': {2000: 5363675, 2010: 5686986, 2020: 5893718},
    # Local government populations (county/city - 2000, 2010, 2020 Census)
    'Baton Rouge, LA': {2000: 412852, 2010: 440171, 2020: 456781},
    'Chicago, IL': {2000: 2896016, 2010: 2695598, 2020: 2746388},
    'City of Birmingham': {2000: 242820, 2010: 212237, 2020: 200733},
    'Columbia, SC': {2000: 116278, 2010: 129272, 2020: 136632},
    'Cook County, IL': {2000: 5376741, 2010: 5194675, 2020: 5275541},
    'County Of Orange': {2000: 2846289, 2010: 3010232, 2020: 3186989},
    'Dallas, TX': {2000: 1188580, 2010: 1197816, 2020: 1304379},
    'Dauphin County, PA': {2000: 251798, 2010: 268100, 2020: 286401},
    'Dearborn, MI': {2000: 97775, 2010: 98153, 2020: 109976},
    'Detroit, MI': {2000: 951270, 2010: 713777, 2020: 639111},
    'Empire State Development Corporation (NYS)': {2000: 18976457, 2010: 19378102, 2020: 20201249},
    'Fort Worth, TX': {2000: 534694, 2010: 741206, 2020: 918915},
    'Hawaii County, HI': {2000: 148677, 2010: 185079, 2020: 200629},
    'Houston, TX': {2000: 1953631, 2010: 2099451, 2020: 2304580},
    'Jefferson County, AL': {2000: 662047, 2010: 658466, 2020: 674721},
    'Jefferson Parish, LA': {2000: 455466, 2010: 432552, 2020: 440781},
    'Joplin, MO': {2000: 45504, 2010: 50150, 2020: 51762},
    'Kauai County, HI': {2000: 58463, 2010: 67091, 2020: 73298},
    'Lake Charles, LA': {2000: 71757, 2010: 71993, 2020: 84872},
    'Lee County, FL': {2000: 440888, 2010: 618754, 2020: 760822},
    'Lexington County, SC': {2000: 216014, 2010: 262391, 2020: 293991},
    'Luzerne County, PA': {2000: 319250, 2010: 320918, 2020: 325594},
    'Minot, ND': {2000: 36567, 2010: 40888, 2020: 48377},
    'Moore, OK': {2000: 41138, 2010: 55081, 2020: 62793},
    'Nashville-Davidson, TN': {2000: 569891, 2010: 626681, 2020: 689447},
    'New Orleans, LA': {2000: 484674, 2010: 343829, 2020: 383997},
    'New York City, NY': {2000: 8008278, 2010: 8175133, 2020: 8336817},
    'North Carolina-NCORR': {2000: 8049313, 2010: 9535483, 2020: 10439388},
    'Orange County, FL': {2000: 896344, 2010: 1145956, 2020: 1393452},
    'Philadelphia, PA': {2000: 1517550, 2010: 1526006, 2020: 1584064},
    'Richland County, SC': {2000: 320677, 2010: 384504, 2020: 415759},
    'San Marcos, TX': {2000: 34733, 2010: 44894, 2020: 67553},
    'Sarasota County, FL': {2000: 325957, 2010: 379448, 2020: 434006},
    'Shelby County, TN': {2000: 897472, 2010: 927644, 2020: 929744},
    'Springfield, MA': {2000: 152082, 2010: 153060, 2020: 155929},
    'St. Clair County, IL': {2000: 256082, 2010: 270056, 2020: 257400},
    'St. Tammany Parish': {2000: 191268, 2010: 233740, 2020: 264570},
    'Town of Union, NY': {2000: 56298, 2010: 56346, 2020: 56298},
    'Tuscaloosa, AL': {2000: 77906, 2010: 90468, 2020: 99600},
    'Volusia County, FL': {2000: 443343, 2010: 494593, 2020: 553284},
}

# Legacy single-year format (for backward compatibility) - uses 2020 data
GRANTEE_POPULATION_DATA = {
    grantee: pops[2020] for grantee, pops in GRANTEE_POPULATION_BY_DECADE.items()
}


def get_census_api_key() -> Optional[str]:
    """
    Get Census API key from environment variable.

    Returns
    -------
    str or None
        API key if set, None otherwise.
    """
    return os.environ.get(CENSUS_API_KEY_ENV)


def fetch_census_population(
    state_fips: Optional[List[str]] = None,
    county_fips: Optional[List[str]] = None,
    year: int = 2020,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch population data from Census Bureau ACS 5-year estimates.

    Parameters
    ----------
    state_fips : List[str], optional
        List of 2-digit state FIPS codes to fetch (e.g., ['01', '06']).
        If None, fetches all states.
    county_fips : List[str], optional
        List of 5-digit county FIPS codes (state+county) to fetch.
    year : int, default 2020
        ACS survey year (5-year estimates available 2010-2023).
    api_key : str, optional
        Census API key. If not provided, uses CENSUS_API_KEY env var.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [FIPS, Name, Population, Year]

    Notes
    -----
    Uses ACS 5-year estimates variable B01003_001E (Total Population).
    API documentation: https://www.census.gov/data/developers/data-sets/acs-5year.html
    """
    api_key = api_key or get_census_api_key()

    if api_key is None:
        logger.warning(
            f"No Census API key found. Set {CENSUS_API_KEY_ENV} environment variable. "
            "Get a key at: https://api.census.gov/data/key_signup.html"
        )
        return pd.DataFrame()

    # Build API URL for ACS 5-year estimates
    base_url = f"{CENSUS_ACS_BASE}/{year}/acs/acs5"

    results = []

    if county_fips:
        # Fetch county-level data
        for fips in county_fips:
            state = fips[:2]
            county = fips[2:]
            params = {
                "get": "NAME,B01003_001E",
                "for": f"county:{county}",
                "in": f"state:{state}",
                "key": api_key
            }
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if len(data) > 1:
                    # First row is headers
                    for row in data[1:]:
                        results.append({
                            "FIPS": f"{row[2]}{row[3]}",
                            "Name": row[0],
                            "Population": int(row[1]) if row[1] else None,
                            "Year": year
                        })
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch county {fips}: {e}")

    else:
        # Fetch state-level data
        params = {
            "get": "NAME,B01003_001E",
            "for": "state:*",
            "key": api_key
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for row in data[1:]:  # Skip header
                fips = row[2]
                if state_fips is None or fips in state_fips:
                    results.append({
                        "FIPS": fips,
                        "Name": row[0],
                        "Population": int(row[1]) if row[1] else None,
                        "Year": year
                    })
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Census data: {e}")
            return pd.DataFrame()

    return pd.DataFrame(results)


def fetch_fema_disaster_declarations(
    disaster_numbers: Optional[List[int]] = None,
    state_codes: Optional[List[str]] = None,
    incident_types: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch disaster declaration data from FEMA OpenFEMA API.

    Parameters
    ----------
    disaster_numbers : List[int], optional
        Specific FEMA disaster numbers to fetch (e.g., [4332, 4336]).
    state_codes : List[str], optional
        Two-letter state codes to filter by (e.g., ['TX', 'FL']).
    incident_types : List[str], optional
        Incident types to filter (e.g., ['Hurricane', 'Flood']).
    start_year : int, optional
        Start year for declaration date filter.
    end_year : int, optional
        End year for declaration date filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with disaster declaration details.

    Notes
    -----
    API documentation: https://www.fema.gov/about/openfema/api
    No API key required.
    """
    params = {
        "$format": "json",
        "$top": 1000,
        "$orderby": "declarationDate desc"
    }

    # Build filter conditions
    filters = []

    if disaster_numbers:
        disaster_filter = " or ".join([f"disasterNumber eq {dn}" for dn in disaster_numbers])
        filters.append(f"({disaster_filter})")

    if state_codes:
        state_filter = " or ".join([f"stateCode eq '{sc}'" for sc in state_codes])
        filters.append(f"({state_filter})")

    if incident_types:
        type_filter = " or ".join([f"incidentType eq '{it}'" for it in incident_types])
        filters.append(f"({type_filter})")

    if start_year:
        filters.append(f"year(declarationDate) ge {start_year}")

    if end_year:
        filters.append(f"year(declarationDate) le {end_year}")

    if filters:
        params["$filter"] = " and ".join(filters)

    all_results = []
    skip = 0

    while True:
        params["$skip"] = skip

        try:
            response = requests.get(FEMA_DISASTER_ENDPOINT, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            records = data.get("DisasterDeclarationsSummaries", [])
            if not records:
                break

            all_results.extend(records)

            if len(records) < 1000:
                break

            skip += 1000

        except requests.RequestException as e:
            logger.error(f"Failed to fetch FEMA data: {e}")
            break

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Select and rename relevant columns
    columns_of_interest = [
        "disasterNumber",
        "declarationDate",
        "fyDeclared",
        "incidentType",
        "declarationTitle",
        "state",
        "stateCode",
        "designatedArea",
        "incidentBeginDate",
        "incidentEndDate",
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared"
    ]

    available_cols = [c for c in columns_of_interest if c in df.columns]
    return df[available_cols].copy()


def get_disaster_summary(df_fema: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize disaster declarations to one row per disaster.

    Parameters
    ----------
    df_fema : pd.DataFrame
        Output from fetch_fema_disaster_declarations().

    Returns
    -------
    pd.DataFrame
        One row per disaster with aggregated metrics.
    """
    if df_fema.empty:
        return pd.DataFrame()

    summary = df_fema.groupby("disasterNumber").agg({
        "declarationDate": "first",
        "fyDeclared": "first",
        "incidentType": "first",
        "declarationTitle": "first",
        "incidentBeginDate": "first",
        "incidentEndDate": "first",
        "state": lambda x: ", ".join(sorted(x.unique())),
        "designatedArea": "count"
    }).reset_index()

    summary = summary.rename(columns={
        "designatedArea": "CountiesAffected",
        "state": "StatesAffected"
    })

    return summary


# DRGR to FEMA disaster number mapping
# Maps the 18 DRGR disaster types to corresponding FEMA disaster numbers
DRGR_TO_FEMA_MAPPING: Dict[str, List[int]] = {
    "2001 World Trade Center": [1391],
    "2005 Hurricanes Katrina, Rita, Wilma": [1602, 1603, 1604, 1606, 1607],
    "2008 Hurricane Ike and Other Events": [1780, 1791, 1794],
    "2008 Midwest Floods": [1763, 1771, 1773],
    "2010 Severe Storms and Flooding": [1935, 1924, 1927],
    "2011 Multiple Disasters": [1991, 1994, 4012, 4024],
    "2011-2013 Hurricane Sandy and Other Events": [4085, 4086, 4087],
    "2013 National Disaster Resilience": [],  # NDR program, not tied to specific disaster
    "2015 Hurricane Joaquin and Patricia and Other Events": [4241, 4245],
    "2015-2018 Mitigation": [],  # Mitigation program
    "2016 Louisiana Floods and Other Events": [4277, 4280],
    "2017 Hurricanes Harvey, Irma and Maria": [4332, 4336, 4339],
    "2018 Disasters": [4393, 4396, 4399, 4407],
    "2019 Disasters": [4420, 4466, 4485, 4473],
    "2020 Hurricanes Laura, Delta and Zeta (LDZ)/2021 Hurricane Ida": [4559, 4570, 4577, 4611],
    "2022 Disasters including Hurricanes Fiona and Ian": [4671, 4673, 4675],
    "Electrical Power Systems in Puerto Rico and the U.S. Virgin Islands": [4339],
    "Storms, Flooding, and Other Disasters in California, Alabama, and Georgia": [4699, 4683],
}

# Disaster year mapping (approximate primary year)
DRGR_DISASTER_YEARS: Dict[str, int] = {
    "2001 World Trade Center": 2001,
    "2005 Hurricanes Katrina, Rita, Wilma": 2005,
    "2008 Hurricane Ike and Other Events": 2008,
    "2008 Midwest Floods": 2008,
    "2010 Severe Storms and Flooding": 2010,
    "2011 Multiple Disasters": 2011,
    "2011-2013 Hurricane Sandy and Other Events": 2012,  # Sandy was Oct 2012
    "2013 National Disaster Resilience": 2013,
    "2015 Hurricane Joaquin and Patricia and Other Events": 2015,
    "2015-2018 Mitigation": 2016,  # Midpoint
    "2016 Louisiana Floods and Other Events": 2016,
    "2017 Hurricanes Harvey, Irma and Maria": 2017,
    "2018 Disasters": 2018,
    "2019 Disasters": 2019,
    "2020 Hurricanes Laura, Delta and Zeta (LDZ)/2021 Hurricane Ida": 2020,
    "2022 Disasters including Hurricanes Fiona and Ian": 2022,
    "Electrical Power Systems in Puerto Rico and the U.S. Virgin Islands": 2017,
    "Storms, Flooding, and Other Disasters in California, Alabama, and Georgia": 2023,
}


def map_drgr_disaster_to_fema(disaster_type: str) -> Dict[str, Any]:
    """
    Map a DRGR disaster type to FEMA disaster information.

    Parameters
    ----------
    disaster_type : str
        DRGR disaster type string from the dataset.

    Returns
    -------
    Dict
        Contains: fema_disaster_numbers, disaster_year, and is_program flag
    """
    fema_numbers = DRGR_TO_FEMA_MAPPING.get(disaster_type, [])
    disaster_year = DRGR_DISASTER_YEARS.get(disaster_type)

    # Flag if this is a program (not tied to specific disaster)
    is_program = len(fema_numbers) == 0 and disaster_type in DRGR_TO_FEMA_MAPPING

    return {
        "drgr_disaster_type": disaster_type,
        "fema_disaster_numbers": fema_numbers,
        "disaster_year": disaster_year,
        "is_program": is_program
    }


def fetch_disaster_magnitude_for_drgr(disaster_type: str) -> pd.DataFrame:
    """
    Fetch FEMA disaster data for a specific DRGR disaster type.

    Parameters
    ----------
    disaster_type : str
        DRGR disaster type string.

    Returns
    -------
    pd.DataFrame
        FEMA disaster declaration data for matched disasters.
    """
    mapping = map_drgr_disaster_to_fema(disaster_type)

    if not mapping["fema_disaster_numbers"]:
        logger.info(f"No FEMA disaster numbers for: {disaster_type}")
        return pd.DataFrame()

    return fetch_fema_disaster_declarations(
        disaster_numbers=mapping["fema_disaster_numbers"]
    )


def create_grantee_to_fips_mapping() -> Dict[str, str]:
    """
    Create mapping from grantee names to FIPS codes.

    This maps the grantee names used in the QPR data to their corresponding
    state or county FIPS codes for Census data linkage.

    Returns
    -------
    Dict[str, str]
        Mapping of grantee name to FIPS code.
        - State grantees: 2-digit state FIPS
        - Local grantees: 5-digit county FIPS (state+county)
    """
    # State FIPS codes (2-digit)
    state_fips = {
        "Alabama": "01",
        "Alaska": "02",
        "American Samoa": "60",
        "Arkansas": "05",
        "California": "06",
        "Colorado": "08",
        "Connecticut - DOH": "09",
        "Florida": "12",
        "Georgia": "13",
        "Illinois": "17",
        "Indiana - OCRA": "18",
        "Iowa": "19",
        "Kentucky": "21",
        "Louisiana": "22",
        "Maryland": "24",
        "Michigan": "26",
        "Mississippi": "28",
        "Missouri": "29",
        "Nebraska": "31",
        "New Jersey": "34",
        "New York": "36",
        "North Carolina": "37",
        "Northern Mariana Islands": "69",
        "Ohio": "39",
        "Oklahoma": "40",
        "Oregon": "41",
        "Pennsylvania": "42",
        "Puerto Rico": "72",
        "Rhode Island": "44",
        "South Carolina": "45",
        "Tennessee": "47",
        "Texas - GLO": "48",
        "Virgin Islands": "78",
        "Virginia": "51",
        "Washington": "53",
        "West Virginia": "54",
        "Wisconsin": "55",
    }

    # County FIPS codes (5-digit: state + county)
    # Format: "City, ST" -> "SSCCC" where SS=state, CCC=county
    local_fips = {
        "Baton Rouge, LA": "22033",  # East Baton Rouge Parish
        "Chicago, IL": "17031",       # Cook County
        "City of Birmingham": "01073",  # Jefferson County, AL
        "Columbia, SC": "45079",      # Richland County
        "Cook County, IL": "17031",
        "County Of Orange": "06059",  # Orange County, CA
        "Dallas, TX": "48113",        # Dallas County
        "Dauphin County, PA": "42043",
        "Dearborn, MI": "26163",      # Wayne County
        "Detroit, MI": "26163",       # Wayne County
        "Empire State Development Corporation (NYS)": "36",  # State-level entity
        "Fort Worth, TX": "48439",    # Tarrant County
        "Hawaii County, HI": "15001",
        "Houston, TX": "48201",       # Harris County
        "Jefferson County, AL": "01073",
        "Jefferson Parish, LA": "22051",
        "Joplin, MO": "29097",        # Jasper County
        "Kauai County, HI": "15007",
        "Lake Charles, LA": "22019",  # Calcasieu Parish
        "Lee County, FL": "12071",
        "Lexington County, SC": "45063",
        "Luzerne County, PA": "42079",
        "Minot, ND": "38101",         # Ward County
        "Moore, OK": "40027",         # Cleveland County
        "Nashville-Davidson, TN": "47037",  # Davidson County
        "New Orleans, LA": "22071",   # Orleans Parish
        "New York City, NY": "36061", # New York County (Manhattan, representative)
        "North Carolina-NCORR": "37",  # State-level entity
        "Orange County, FL": "12095",
        "Philadelphia, PA": "42101",  # Philadelphia County
        "Richland County, SC": "45079",
        "San Marcos, TX": "48209",    # Hays County
        "Sarasota County, FL": "12115",
        "Shelby County, TN": "47157",
        "Springfield, MA": "25013",   # Hampden County
        "St. Clair County, IL": "17163",
        "St. Tammany Parish": "22103",
        "Town of Union, NY": "36007", # Broome County
        "Tuscaloosa, AL": "01125",    # Tuscaloosa County
        "Volusia County, FL": "12127",
    }

    return {**state_fips, **local_fips}


def fetch_population_for_grantees(
    grantees: List[str],
    year: int = 2020,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch population data for a list of grantees.

    Parameters
    ----------
    grantees : List[str]
        List of grantee names from the QPR data.
    year : int, default 2020
        Census year to use.
    api_key : str, optional
        Census API key.

    Returns
    -------
    pd.DataFrame
        Population data with columns: [Grantee, FIPS, Population, Year]
    """
    fips_mapping = create_grantee_to_fips_mapping()

    results = []

    # Separate state and county FIPS
    state_fips_list = []
    county_fips_list = []
    grantee_to_fips = {}

    for grantee in grantees:
        fips = fips_mapping.get(grantee)
        if fips:
            grantee_to_fips[grantee] = fips
            if len(fips) == 2:
                state_fips_list.append(fips)
            else:
                county_fips_list.append(fips)
        else:
            logger.warning(f"No FIPS mapping for grantee: {grantee}")

    # Fetch state-level populations
    if state_fips_list:
        df_states = fetch_census_population(
            state_fips=state_fips_list,
            year=year,
            api_key=api_key
        )

        for grantee, fips in grantee_to_fips.items():
            if len(fips) == 2:
                row = df_states[df_states["FIPS"] == fips]
                if not row.empty:
                    results.append({
                        "Grantee": grantee,
                        "FIPS": fips,
                        "Population": row.iloc[0]["Population"],
                        "Year": year
                    })

    # Fetch county-level populations
    if county_fips_list:
        df_counties = fetch_census_population(
            county_fips=county_fips_list,
            year=year,
            api_key=api_key
        )

        for grantee, fips in grantee_to_fips.items():
            if len(fips) == 5:
                row = df_counties[df_counties["FIPS"] == fips]
                if not row.empty:
                    results.append({
                        "Grantee": grantee,
                        "FIPS": fips,
                        "Population": row.iloc[0]["Population"],
                        "Year": year
                    })

    return pd.DataFrame(results)


def build_covariate_dataset(
    grantees: List[str],
    disaster_types: List[str],
    census_year: int = 2020,
    census_api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Build a complete covariate dataset for analysis.

    Combines population data with disaster information for each grantee-disaster
    combination.

    Parameters
    ----------
    grantees : List[str]
        List of grantee names.
    disaster_types : List[str]
        List of DRGR disaster types.
    census_year : int, default 2020
        Census year for population data.
    census_api_key : str, optional
        Census API key.

    Returns
    -------
    pd.DataFrame
        Covariate dataset with population and disaster information.
    """
    # Get population data
    df_pop = fetch_population_for_grantees(grantees, census_year, census_api_key)

    # Get disaster information
    disaster_info = []
    for disaster in disaster_types:
        info = map_drgr_disaster_to_fema(disaster)
        disaster_info.append({
            "Disaster Type": disaster,
            "Disaster_Year": info["disaster_year"],
            "Is_Program": info["is_program"],
            "FEMA_Numbers": ",".join(map(str, info["fema_disaster_numbers"]))
        })

    df_disaster = pd.DataFrame(disaster_info)

    # Cross-join grantees with disasters (each grantee-disaster is an observation)
    df_pop["_key"] = 1
    df_disaster["_key"] = 1
    df_combined = df_pop.merge(df_disaster, on="_key").drop("_key", axis=1)

    return df_combined


# =============================================================================
# ALTERNATIVE METHODS - No API key required
# =============================================================================

def get_census_decade(year: int) -> int:
    """
    Get the appropriate census decade for a given year.

    Uses the most recent decennial census available before the year.

    Parameters
    ----------
    year : int
        Year to find census data for.

    Returns
    -------
    int
        Census decade year (2000, 2010, or 2020).
    """
    if year >= 2020:
        return 2020
    elif year >= 2010:
        return 2010
    else:
        return 2000


def get_embedded_population(
    grantees: Optional[List[str]] = None,
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Get population data from embedded dataset (no API key required).

    Uses decennial Census data (2000, 2010, 2020) matched to the requested year.

    Parameters
    ----------
    grantees : List[str], optional
        List of grantee names. If None, returns all grantees.
    year : int, optional
        Year to match population to. Uses nearest prior census.
        If None, returns 2020 data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [Grantee, Population, Census_Year, FIPS]
    """
    fips_mapping = create_grantee_to_fips_mapping()
    census_year = get_census_decade(year) if year else 2020

    results = []
    for grantee, pop_by_decade in GRANTEE_POPULATION_BY_DECADE.items():
        if grantees is None or grantee in grantees:
            results.append({
                "Grantee": grantee,
                "Population": pop_by_decade.get(census_year, pop_by_decade[2020]),
                "Census_Year": census_year,
                "FIPS": fips_mapping.get(grantee, ""),
            })

    return pd.DataFrame(results)


def get_population_for_disaster(
    grantee: str,
    disaster_type: str
) -> Optional[int]:
    """
    Get population for a grantee matched to disaster year.

    Parameters
    ----------
    grantee : str
        Grantee name.
    disaster_type : str
        DRGR disaster type string.

    Returns
    -------
    int or None
        Population at time of disaster, or None if not found.
    """
    if grantee not in GRANTEE_POPULATION_BY_DECADE:
        return None

    disaster_year = DRGR_DISASTER_YEARS.get(disaster_type)
    if disaster_year is None:
        census_year = 2020
    else:
        census_year = get_census_decade(disaster_year)

    return GRANTEE_POPULATION_BY_DECADE[grantee].get(census_year)


def download_fema_declarations_csv(
    save_path: Optional[Path] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Download FEMA disaster declarations as CSV (no API pagination needed).

    This downloads the full disaster declarations dataset directly as CSV,
    which is simpler than the paginated API.

    Parameters
    ----------
    save_path : Path, optional
        Path to save the CSV file. If provided and file exists, loads from disk
        unless force_download is True.
    force_download : bool, default False
        Force re-download even if file exists.

    Returns
    -------
    pd.DataFrame
        FEMA disaster declarations data.
    """
    # Check for cached file
    if save_path and save_path.exists() and not force_download:
        logger.info(f"Loading cached FEMA data from {save_path}")
        return pd.read_csv(save_path)

    logger.info("Downloading FEMA disaster declarations CSV...")

    try:
        # Download the full CSV
        response = requests.get(FEMA_DECLARATIONS_CSV_URL, timeout=120)
        response.raise_for_status()

        # Parse CSV from response text
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Saved FEMA data to {save_path}")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to download FEMA CSV: {e}")
        return pd.DataFrame()


def get_covariates_simple(
    grantees: Optional[List[str]] = None,
    use_cached_fema: bool = True,
    fema_cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Get covariate data using embedded population and simple FEMA download.

    This is the recommended method as it requires no API keys.

    Parameters
    ----------
    grantees : List[str], optional
        List of grantee names. If None, uses all grantees.
    use_cached_fema : bool, default True
        Whether to use cached FEMA data if available.
    fema_cache_path : Path, optional
        Path to cache FEMA data.

    Returns
    -------
    pd.DataFrame
        Covariate dataset with population for all grantees.
    """
    # Get population from embedded data
    df_pop = get_embedded_population(grantees)

    if df_pop.empty:
        logger.warning("No population data available")
        return pd.DataFrame()

    logger.info(f"Retrieved population data for {len(df_pop)} grantees")

    return df_pop


def get_all_external_data(
    grantees: Optional[List[str]] = None,
    disasters: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Get all external data using simple methods (no API keys required).

    Population data is matched to disaster year using decennial census:
    - 2001-2009: Uses 2000 Census
    - 2010-2019: Uses 2010 Census
    - 2020+: Uses 2020 Census

    Parameters
    ----------
    grantees : List[str], optional
        List of grantee names. If None, uses all grantees.
    disasters : List[str], optional
        List of DRGR disaster types.
    output_dir : Path, optional
        Directory to save output files.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with 'population' and 'disaster_mapping' DataFrames.
        Population DataFrame includes disaster-year-matched data.
    """
    results = {}
    fips_mapping = create_grantee_to_fips_mapping()

    # Build disaster mapping first (needed for year-matching)
    disaster_info = []
    if disasters:
        for disaster in disasters:
            info = map_drgr_disaster_to_fema(disaster)
            disaster_info.append({
                "Disaster_Type": disaster,
                "Disaster_Year": info["disaster_year"],
                "Census_Year": get_census_decade(info["disaster_year"]) if info["disaster_year"] else 2020,
                "Is_Program": info["is_program"],
                "FEMA_Numbers": ",".join(map(str, info["fema_disaster_numbers"]))
            })
        df_disasters = pd.DataFrame(disaster_info)
        results['disaster_mapping'] = df_disasters

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            df_disasters.to_csv(output_dir / "disaster_mapping.csv", index=False)
            logger.info(f"Saved disaster mapping to {output_dir / 'disaster_mapping.csv'}")

    # Build grantee population data - one row per grantee per census decade used
    pop_results = []
    grantee_list = grantees if grantees else list(GRANTEE_POPULATION_BY_DECADE.keys())

    # Get unique census years needed
    if disaster_info:
        census_years_needed = set(d["Census_Year"] for d in disaster_info)
    else:
        census_years_needed = {2020}

    for grantee in grantee_list:
        if grantee not in GRANTEE_POPULATION_BY_DECADE:
            continue

        for census_year in census_years_needed:
            pop = GRANTEE_POPULATION_BY_DECADE[grantee].get(census_year)
            if pop:
                pop_results.append({
                    "Grantee": grantee,
                    "Population": pop,
                    "Census_Year": census_year,
                    "FIPS": fips_mapping.get(grantee, ""),
                })

    df_pop = pd.DataFrame(pop_results)
    results['population'] = df_pop

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_pop.to_csv(output_dir / "population_data.csv", index=False)
        logger.info(f"Saved population data to {output_dir / 'population_data.csv'}")

    # Build grantee-disaster level population (matched to disaster year)
    if disasters and grantee_list:
        grantee_disaster_pop = []
        for grantee in grantee_list:
            if grantee not in GRANTEE_POPULATION_BY_DECADE:
                continue
            for disaster in disasters:
                pop = get_population_for_disaster(grantee, disaster)
                disaster_year = DRGR_DISASTER_YEARS.get(disaster)
                census_year = get_census_decade(disaster_year) if disaster_year else 2020
                if pop:
                    grantee_disaster_pop.append({
                        "Grantee": grantee,
                        "Disaster_Type": disaster,
                        "Disaster_Year": disaster_year,
                        "Population": pop,
                        "Census_Year": census_year,
                    })

        df_grantee_disaster = pd.DataFrame(grantee_disaster_pop)
        results['grantee_disaster_population'] = df_grantee_disaster

        if output_dir:
            df_grantee_disaster.to_csv(
                output_dir / "grantee_disaster_population.csv", index=False
            )
            logger.info(f"Saved grantee-disaster population to {output_dir}")

    return results


# =============================================================================
# DISASTER SEVERITY INDEX DATA
# =============================================================================
# Composite severity index (0-10 scale) from FEMA data
# Components: program scope (25%), damage magnitude (35%), geographic scope (20%), duration (20%)
# Acquired via scripts/acquire_disaster_severity.py from FEMA OpenFEMA API

DISASTER_SEVERITY_INDEX: Dict[str, Dict[str, Any]] = {
    "2001 World Trade Center": {
        "severity_index": 3.96,
        "program_scope": 3,
        "counties_affected": 62,
        "duration_days": 0,
        "total_damage": 0,
        "valid_registrations": 0,
        "data_quality": "partial",
    },
    "2005 Hurricanes Katrina, Rita, Wilma": {
        "severity_index": 8.31,
        "program_scope": 3,
        "counties_affected": 392,
        "duration_days": 69,
        "total_damage": 6537244996,
        "valid_registrations": 1018542,
        "data_quality": "complete",
    },
    "2008 Hurricane Ike and Other Events": {
        "severity_index": 6.58,
        "program_scope": 3,
        "counties_affected": 81,
        "duration_days": 72,
        "total_damage": 714960608,
        "valid_registrations": 400070,
        "data_quality": "complete",
    },
    "2008 Midwest Floods": {
        "severity_index": 6.91,
        "program_scope": 3,
        "counties_affected": 142,
        "duration_days": 80,
        "total_damage": 207188615,
        "valid_registrations": 32010,
        "data_quality": "complete",
    },
    "2010 Severe Storms and Flooding (TN, RI)": {
        "severity_index": 5.36,
        "program_scope": 2,
        "counties_affected": 54,
        "duration_days": 67,
        "total_damage": 233683845,
        "valid_registrations": 63937,
        "data_quality": "complete",
    },
    "2011 National Disaster (Joplin, etc.)": {
        "severity_index": 6.15,
        "program_scope": 2,
        "counties_affected": 186,
        "duration_days": 54,
        "total_damage": 346784722,
        "valid_registrations": 87698,
        "data_quality": "complete",
    },
    "2011-2013 Hurricane Sandy and Other Events": {
        "severity_index": 5.01,
        "program_scope": 2,
        "counties_affected": 47,
        "duration_days": 13,
        "total_damage": 2125293230,
        "valid_registrations": 283694,
        "data_quality": "complete",
    },
    "2013 National Disaster Resilience": {
        "severity_index": 5.0,
        "program_scope": 2,
        "counties_affected": 0,
        "duration_days": 0,
        "total_damage": 0,
        "valid_registrations": 0,
        "data_quality": "program_default",
    },
    "2015 Hurricane Joaquin and Patricia and Other Events": {
        "severity_index": 5.26,
        "program_scope": 2,
        "counties_affected": 60,
        "duration_days": 52,
        "total_damage": 280654433,
        "valid_registrations": 85572,
        "data_quality": "complete",
    },
    "2015-2018 Mitigation": {
        "severity_index": 5.0,
        "program_scope": 2,
        "counties_affected": 0,
        "duration_days": 0,
        "total_damage": 0,
        "valid_registrations": 0,
        "data_quality": "program_default",
    },
    "2016 Louisiana Floods and Other Events": {
        "severity_index": 7.12,
        "program_scope": 2,
        "counties_affected": 102,
        "duration_days": 176,
        "total_damage": 1261431047,
        "valid_registrations": 131788,
        "data_quality": "complete",
    },
    "2017 Hurricanes Harvey, Irma and Maria": {
        "severity_index": 6.36,
        "program_scope": 2,
        "counties_affected": 131,
        "duration_days": 84,
        "total_damage": 2058494493,
        "valid_registrations": 467028,
        "data_quality": "complete",
    },
    "2018 Disasters": {
        "severity_index": 6.63,
        "program_scope": 2,
        "counties_affected": 82,
        "duration_days": 157,
        "total_damage": 412958464,
        "valid_registrations": 153712,
        "data_quality": "complete",
    },
    "2019 Disasters": {
        "severity_index": 6.9,
        "program_scope": 2,
        "counties_affected": 204,
        "duration_days": 127,
        "total_damage": 86779233,
        "valid_registrations": 11648,
        "data_quality": "complete",
    },
    "2020/21 Hurricanes Laura, Delta, Zeta / Hurricane Ida": {
        "severity_index": 6.88,
        "program_scope": 2,
        "counties_affected": 64,
        "duration_days": 377,
        "total_damage": 1019872984,
        "valid_registrations": 568145,
        "data_quality": "complete",
    },
    "2022 Hurricanes Fiona and Ian": {
        "severity_index": 6.01,
        "program_scope": 2,
        "counties_affected": 147,
        "duration_days": 48,
        "total_damage": 1374562379,
        "valid_registrations": 428224,
        "data_quality": "complete",
    },
    "Electrical Power Systems (PR & USVI)": {
        "severity_index": 5.0,
        "program_scope": 2,
        "counties_affected": 0,
        "duration_days": 0,
        "total_damage": 0,
        "valid_registrations": 0,
        "data_quality": "program_default",
    },
    "CA, AL, GA Storms and Flooding": {
        "severity_index": 7.97,
        "program_scope": 2,
        "counties_affected": 273,
        "duration_days": 191,
        "total_damage": 102858308,
        "valid_registrations": 80602,
        "data_quality": "complete",
    },
}


def get_disaster_severity(disaster_type: str) -> Optional[float]:
    """
    Get the composite severity index for a disaster type.

    Parameters
    ----------
    disaster_type : str
        DRGR disaster type string.

    Returns
    -------
    float or None
        Severity index (0-10 scale) or None if not found.
    """
    data = DISASTER_SEVERITY_INDEX.get(disaster_type)
    if data:
        return data["severity_index"]
    return None


def get_disaster_severity_components(disaster_type: str) -> Optional[Dict[str, Any]]:
    """
    Get all severity components for a disaster type.

    Parameters
    ----------
    disaster_type : str
        DRGR disaster type string.

    Returns
    -------
    Dict or None
        Dictionary with severity_index, program_scope, counties_affected,
        duration_days, total_damage, valid_registrations, data_quality.
    """
    return DISASTER_SEVERITY_INDEX.get(disaster_type)


def get_severity_for_all_disasters() -> pd.DataFrame:
    """
    Get severity data for all disaster types as a DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Disaster_Type, Severity_Index, and component fields.
    """
    records = []
    for disaster_type, data in DISASTER_SEVERITY_INDEX.items():
        records.append({
            "Disaster_Type": disaster_type,
            "Severity_Index": data["severity_index"],
            "Program_Scope": data["program_scope"],
            "Counties_Affected": data["counties_affected"],
            "Duration_Days": data["duration_days"],
            "Total_Damage": data["total_damage"],
            "Valid_Registrations": data["valid_registrations"],
            "Data_Quality": data["data_quality"],
        })
    return pd.DataFrame(records)


# =============================================================================
# BLS GOVERNMENT EMPLOYMENT DATA
# =============================================================================
# Total employment by grantee and year
# Acquired via scripts/acquire_bls_employment.py from BLS QCEW API
# Years: 2015-2023 (earlier years not available via API)

GRANTEE_EMPLOYMENT_BY_YEAR: Dict[str, Dict[int, Dict[str, int]]] = {
    "Alabama": {
        2015: {'total_gov': 1532167, 'state_gov': 0, 'local_gov': 358173},
        2016: {'total_gov': 1555542, 'state_gov': 0, 'local_gov': 359764},
        2017: {'total_gov': 1574772, 'state_gov': 0, 'local_gov': 362047},
        2018: {'total_gov': 1598129, 'state_gov': 0, 'local_gov': 363496},
        2019: {'total_gov': 1622460, 'state_gov': 0, 'local_gov': 367095},
        2020: {'total_gov': 1547125, 'state_gov': 0, 'local_gov': 362020},
        2021: {'total_gov': 1602985, 'state_gov': 0, 'local_gov': 362205},
        2022: {'total_gov': 1660156, 'state_gov': 0, 'local_gov': 365946},
        2023: {'total_gov': 1700160, 'state_gov': 0, 'local_gov': 375625},
    },
    "Alaska": {
        2015: {'total_gov': 255274, 'state_gov': 0, 'local_gov': 76407},
        2016: {'total_gov': 250255, 'state_gov': 0, 'local_gov': 76040},
        2017: {'total_gov': 246852, 'state_gov': 0, 'local_gov': 75284},
        2018: {'total_gov': 246319, 'state_gov': 0, 'local_gov': 74759},
        2019: {'total_gov': 249300, 'state_gov': 0, 'local_gov': 74396},
        2020: {'total_gov': 225077, 'state_gov': 0, 'local_gov': 72353},
        2021: {'total_gov': 232510, 'state_gov': 0, 'local_gov': 72635},
        2022: {'total_gov': 241031, 'state_gov': 0, 'local_gov': 72929},
        2023: {'total_gov': 248671, 'state_gov': 0, 'local_gov': 74129},
    },
    "Arkansas": {
        2015: {'total_gov': 978732, 'state_gov': 0, 'local_gov': 199152},
        2017: {'total_gov': 1004075, 'state_gov': 0, 'local_gov': 196467},
        2019: {'total_gov': 1022053, 'state_gov': 0, 'local_gov': 196053},
        2020: {'total_gov': 982047, 'state_gov': 0, 'local_gov': 192585},
        2022: {'total_gov': 1056644, 'state_gov': 0, 'local_gov': 192329},
    },
    "California": {
        2015: {'total_gov': 13916502, 'state_gov': 0, 'local_gov': 2378702},
        2017: {'total_gov': 14558652, 'state_gov': 0, 'local_gov': 2461050},
        2018: {'total_gov': 14876010, 'state_gov': 0, 'local_gov': 2479845},
        2019: {'total_gov': 15127578, 'state_gov': 0, 'local_gov': 2503911},
        2020: {'total_gov': 13966715, 'state_gov': 0, 'local_gov': 2411345},
        2022: {'total_gov': 15438555, 'state_gov': 0, 'local_gov': 2464984},
    },
    "Colorado": {
        2015: {'total_gov': 2097611, 'state_gov': 0, 'local_gov': 396840},
        2017: {'total_gov': 2197742, 'state_gov': 0, 'local_gov': 412028},
        2019: {'total_gov': 2308134, 'state_gov': 0, 'local_gov': 427971},
        2020: {'total_gov': 2182001, 'state_gov': 0, 'local_gov': 420370},
        2022: {'total_gov': 2384337, 'state_gov': 0, 'local_gov': 430638},
    },
    "Connecticut - DOH": {
        2015: {'total_gov': 1428395, 'state_gov': 0, 'local_gov': 234430},
        2017: {'total_gov': 1442388, 'state_gov': 0, 'local_gov': 227228},
        2019: {'total_gov': 1445817, 'state_gov': 0, 'local_gov': 224887},
        2020: {'total_gov': 1332441, 'state_gov': 0, 'local_gov': 212898},
        2022: {'total_gov': 1426728, 'state_gov': 0, 'local_gov': 215808},
    },
    "Florida": {
        2015: {'total_gov': 7005903, 'state_gov': 0, 'local_gov': 1033732},
        2017: {'total_gov': 7437388, 'state_gov': 0, 'local_gov': 1057235},
        2018: {'total_gov': 7635037, 'state_gov': 0, 'local_gov': 1065617},
        2019: {'total_gov': 7808474, 'state_gov': 0, 'local_gov': 1075592},
        2020: {'total_gov': 7389890, 'state_gov': 0, 'local_gov': 1058067},
        2022: {'total_gov': 8308654, 'state_gov': 0, 'local_gov': 1049575},
    },
    "Georgia": {
        2015: {'total_gov': 3513397, 'state_gov': 0, 'local_gov': 637614},
        2017: {'total_gov': 3699460, 'state_gov': 0, 'local_gov': 646994},
        2019: {'total_gov': 3859728, 'state_gov': 0, 'local_gov': 653300},
        2020: {'total_gov': 3663623, 'state_gov': 0, 'local_gov': 643516},
        2022: {'total_gov': 4062791, 'state_gov': 0, 'local_gov': 642546},
    },
    "Illinois": {
        2015: {'total_gov': 5060798, 'state_gov': 0, 'local_gov': 787652},
        2017: {'total_gov': 5153355, 'state_gov': 0, 'local_gov': 781194},
        2019: {'total_gov': 5211856, 'state_gov': 0, 'local_gov': 784049},
        2020: {'total_gov': 4822542, 'state_gov': 0, 'local_gov': 747657},
        2022: {'total_gov': 5161095, 'state_gov': 0, 'local_gov': 757737},
    },
    "Indiana - OCRA": {
        2015: {'total_gov': 2551010, 'state_gov': 0, 'local_gov': 390981},
        2017: {'total_gov': 2627391, 'state_gov': 0, 'local_gov': 390786},
        2019: {'total_gov': 2685388, 'state_gov': 0, 'local_gov': 392379},
        2020: {'total_gov': 2535558, 'state_gov': 0, 'local_gov': 383234},
        2022: {'total_gov': 2733492, 'state_gov': 0, 'local_gov': 379927},
    },
    "Iowa": {
        2015: {'total_gov': 1294792, 'state_gov': 0, 'local_gov': 235442},
        2017: {'total_gov': 1301192, 'state_gov': 0, 'local_gov': 239243},
        2019: {'total_gov': 1312038, 'state_gov': 0, 'local_gov': 241312},
        2020: {'total_gov': 1242572, 'state_gov': 0, 'local_gov': 233132},
        2022: {'total_gov': 1297333, 'state_gov': 0, 'local_gov': 239664},
    },
    "Kentucky": {
        2015: {'total_gov': 1539903, 'state_gov': 0, 'local_gov': 295647},
        2017: {'total_gov': 1580021, 'state_gov': 0, 'local_gov': 294434},
        2019: {'total_gov': 1606510, 'state_gov': 0, 'local_gov': 291386},
        2020: {'total_gov': 1511919, 'state_gov': 0, 'local_gov': 280677},
        2022: {'total_gov': 1640191, 'state_gov': 0, 'local_gov': 280266},
    },
    "Louisiana": {
        2015: {'total_gov': 1620672, 'state_gov': 0, 'local_gov': 310017},
        2016: {'total_gov': 1599363, 'state_gov': 0, 'local_gov': 309034},
        2017: {'total_gov': 1598324, 'state_gov': 0, 'local_gov': 309397},
        2019: {'total_gov': 1612743, 'state_gov': 0, 'local_gov': 311083},
        2020: {'total_gov': 1479871, 'state_gov': 0, 'local_gov': 300703},
        2022: {'total_gov': 1571606, 'state_gov': 0, 'local_gov': 293986},
    },
    "Maryland": {
        2015: {'total_gov': 2105913, 'state_gov': 0, 'local_gov': 485276},
        2017: {'total_gov': 2167072, 'state_gov': 0, 'local_gov': 486497},
        2019: {'total_gov': 2207512, 'state_gov': 0, 'local_gov': 490602},
        2020: {'total_gov': 2031687, 'state_gov': 0, 'local_gov': 480937},
        2022: {'total_gov': 2156462, 'state_gov': 0, 'local_gov': 484559},
    },
    "Michigan": {
        2015: {'total_gov': 3610636, 'state_gov': 0, 'local_gov': 551005},
        2017: {'total_gov': 3734432, 'state_gov': 0, 'local_gov': 560279},
        2019: {'total_gov': 3790764, 'state_gov': 0, 'local_gov': 567403},
        2020: {'total_gov': 3426882, 'state_gov': 0, 'local_gov': 541348},
        2022: {'total_gov': 3749727, 'state_gov': 0, 'local_gov': 551217},
    },
    "Mississippi": {
        2015: {'total_gov': 878389, 'state_gov': 0, 'local_gov': 235990},
        2017: {'total_gov': 894142, 'state_gov': 0, 'local_gov': 234357},
        2019: {'total_gov': 901829, 'state_gov': 0, 'local_gov': 233768},
        2020: {'total_gov': 860833, 'state_gov': 0, 'local_gov': 228988},
        2022: {'total_gov': 919857, 'state_gov': 0, 'local_gov': 225596},
    },
    "Missouri": {
        2015: {'total_gov': 2300643, 'state_gov': 0, 'local_gov': 414937},
        2017: {'total_gov': 2366305, 'state_gov': 0, 'local_gov': 414937},
        2019: {'total_gov': 2399573, 'state_gov': 0, 'local_gov': 413315},
        2020: {'total_gov': 2274758, 'state_gov': 0, 'local_gov': 400358},
        2022: {'total_gov': 2424908, 'state_gov': 0, 'local_gov': 396286},
    },
    "Nebraska": {
        2015: {'total_gov': 799388, 'state_gov': 0, 'local_gov': 159788},
        2017: {'total_gov': 811496, 'state_gov': 0, 'local_gov': 161268},
        2019: {'total_gov': 821384, 'state_gov': 0, 'local_gov': 161120},
        2020: {'total_gov': 790415, 'state_gov': 0, 'local_gov': 157722},
        2022: {'total_gov': 824945, 'state_gov': 0, 'local_gov': 159023},
    },
    "New Jersey": {
        2015: {'total_gov': 3317218, 'state_gov': 0, 'local_gov': 572757},
        2017: {'total_gov': 3435139, 'state_gov': 0, 'local_gov': 571660},
        2019: {'total_gov': 3510701, 'state_gov': 0, 'local_gov': 572313},
        2020: {'total_gov': 3205532, 'state_gov': 0, 'local_gov': 545823},
        2022: {'total_gov': 3581671, 'state_gov': 0, 'local_gov': 552949},
    },
    "New York": {
        2015: {'total_gov': 7648846, 'state_gov': 0, 'local_gov': 1365539},
        2017: {'total_gov': 7899798, 'state_gov': 0, 'local_gov': 1377070},
        2019: {'total_gov': 8121568, 'state_gov': 0, 'local_gov': 1421332},
        2020: {'total_gov': 7207032, 'state_gov': 0, 'local_gov': 1378149},
        2022: {'total_gov': 7908130, 'state_gov': 0, 'local_gov': 1356481},
    },
    "North Carolina": {
        2015: {'total_gov': 3474750, 'state_gov': 0, 'local_gov': 686904},
        2017: {'total_gov': 3633435, 'state_gov': 0, 'local_gov': 697171},
        2018: {'total_gov': 3710090, 'state_gov': 0, 'local_gov': 700702},
        2019: {'total_gov': 3793234, 'state_gov': 0, 'local_gov': 705339},
        2020: {'total_gov': 3632182, 'state_gov': 0, 'local_gov': 691143},
        2022: {'total_gov': 4018296, 'state_gov': 0, 'local_gov': 681259},
    },
    "Ohio": {
        2015: {'total_gov': 4552282, 'state_gov': 0, 'local_gov': 705689},
        2017: {'total_gov': 4646773, 'state_gov': 0, 'local_gov': 717852},
        2019: {'total_gov': 4711316, 'state_gov': 0, 'local_gov': 728035},
        2020: {'total_gov': 4418269, 'state_gov': 0, 'local_gov': 705498},
        2022: {'total_gov': 4684897, 'state_gov': 0, 'local_gov': 707715},
    },
    "Oklahoma": {
        2015: {'total_gov': 1270090, 'state_gov': 0, 'local_gov': 323921},
        2017: {'total_gov': 1260205, 'state_gov': 0, 'local_gov': 320993},
        2019: {'total_gov': 1296418, 'state_gov': 0, 'local_gov': 325640},
        2020: {'total_gov': 1230661, 'state_gov': 0, 'local_gov': 319333},
        2022: {'total_gov': 1304605, 'state_gov': 0, 'local_gov': 320300},
    },
    "Oregon": {
        2015: {'total_gov': 1509012, 'state_gov': 0, 'local_gov': 278385},
        2017: {'total_gov': 1596853, 'state_gov': 0, 'local_gov': 286554},
        2019: {'total_gov': 1677647, 'state_gov': 0, 'local_gov': 275820},
        2020: {'total_gov': 1570539, 'state_gov': 0, 'local_gov': 265668},
        2022: {'total_gov': 1675632, 'state_gov': 0, 'local_gov': 275143},
    },
    "Pennsylvania": {
        2015: {'total_gov': 5015915, 'state_gov': 0, 'local_gov': 675699},
        2017: {'total_gov': 5124456, 'state_gov': 0, 'local_gov': 674667},
        2019: {'total_gov': 5249732, 'state_gov': 0, 'local_gov': 675856},
        2020: {'total_gov': 4829417, 'state_gov': 0, 'local_gov': 659173},
        2022: {'total_gov': 5210459, 'state_gov': 0, 'local_gov': 652774},
    },
    "Puerto Rico": {
        2015: {'total_gov': 671398, 'state_gov': 0, 'local_gov': 225866},
        2017: {'total_gov': 656474, 'state_gov': 0, 'local_gov': 213454},
        2019: {'total_gov': 678407, 'state_gov': 0, 'local_gov': 197803},
        2020: {'total_gov': 637061, 'state_gov': 0, 'local_gov': 195693},
        2022: {'total_gov': 729767, 'state_gov': 0, 'local_gov': 192333},
    },
    "Rhode Island": {
        2015: {'total_gov': 411058, 'state_gov': 0, 'local_gov': 58923},
        2017: {'total_gov': 417997, 'state_gov': 0, 'local_gov': 59364},
        2019: {'total_gov': 424477, 'state_gov': 0, 'local_gov': 61161},
        2020: {'total_gov': 384927, 'state_gov': 0, 'local_gov': 59505},
        2022: {'total_gov': 420969, 'state_gov': 0, 'local_gov': 60373},
    },
    "South Carolina": {
        2015: {'total_gov': 1607203, 'state_gov': 0, 'local_gov': 342678},
        2017: {'total_gov': 1688066, 'state_gov': 0, 'local_gov': 347275},
        2019: {'total_gov': 1774305, 'state_gov': 0, 'local_gov': 354967},
        2020: {'total_gov': 1677040, 'state_gov': 0, 'local_gov': 351001},
        2022: {'total_gov': 1833574, 'state_gov': 0, 'local_gov': 348019},
    },
    "Tennessee": {
        2015: {'total_gov': 2414236, 'state_gov': 0, 'local_gov': 405961},
        2017: {'total_gov': 2519935, 'state_gov': 0, 'local_gov': 410997},
        2019: {'total_gov': 2614349, 'state_gov': 0, 'local_gov': 418544},
        2020: {'total_gov': 2502249, 'state_gov': 0, 'local_gov': 413932},
        2022: {'total_gov': 2741025, 'state_gov': 0, 'local_gov': 416862},
    },
    "Texas - GLO": {
        2015: {'total_gov': 9845769, 'state_gov': 0, 'local_gov': 1810150},
        2017: {'total_gov': 10151376, 'state_gov': 0, 'local_gov': 1863426},
        2018: {'total_gov': 10429485, 'state_gov': 0, 'local_gov': 1872873},
        2019: {'total_gov': 10694544, 'state_gov': 0, 'local_gov': 1895862},
        2020: {'total_gov': 10184330, 'state_gov': 0, 'local_gov': 1885881},
        2022: {'total_gov': 11337527, 'state_gov': 0, 'local_gov': 1911593},
    },
    "Virgin Islands": {
        2015: {'total_gov': 27298, 'state_gov': 0, 'local_gov': 10595},
        2017: {'total_gov': 26309, 'state_gov': 0, 'local_gov': 11142},
        2019: {'total_gov': 27961, 'state_gov': 0, 'local_gov': 10461},
        2020: {'total_gov': 25612, 'state_gov': 0, 'local_gov': 10768},
        2022: {'total_gov': 24037, 'state_gov': 0, 'local_gov': 10743},
    },
    "Virginia": {
        2015: {'total_gov': 3043782, 'state_gov': 0, 'local_gov': 691931},
        2017: {'total_gov': 3140917, 'state_gov': 0, 'local_gov': 697451},
        2019: {'total_gov': 3227013, 'state_gov': 0, 'local_gov': 711828},
        2020: {'total_gov': 3050133, 'state_gov': 0, 'local_gov': 693580},
        2022: {'total_gov': 3258793, 'state_gov': 0, 'local_gov': 699835},
    },
    "Washington": {
        2015: {'total_gov': 2589020, 'state_gov': 0, 'local_gov': 533729},
        2017: {'total_gov': 2735142, 'state_gov': 0, 'local_gov': 555067},
        2019: {'total_gov': 2871805, 'state_gov': 0, 'local_gov': 567353},
        2020: {'total_gov': 2709844, 'state_gov': 0, 'local_gov': 548319},
        2022: {'total_gov': 2955687, 'state_gov': 0, 'local_gov': 556581},
    },
    "West Virginia": {
        2015: {'total_gov': 559172, 'state_gov': 0, 'local_gov': 137022},
        2016: {'total_gov': 546617, 'state_gov': 0, 'local_gov': 137704},
        2017: {'total_gov': 547549, 'state_gov': 0, 'local_gov': 136258},
        2019: {'total_gov': 553809, 'state_gov': 0, 'local_gov': 134951},
        2020: {'total_gov': 509884, 'state_gov': 0, 'local_gov': 132134},
        2022: {'total_gov': 539694, 'state_gov': 0, 'local_gov': 134088},
    },
    "Wisconsin": {
        2015: {'total_gov': 2416695, 'state_gov': 0, 'local_gov': 377475},
        2017: {'total_gov': 2473083, 'state_gov': 0, 'local_gov': 377061},
        2019: {'total_gov': 2506971, 'state_gov': 0, 'local_gov': 380047},
        2020: {'total_gov': 2365625, 'state_gov': 0, 'local_gov': 364665},
        2022: {'total_gov': 2505934, 'state_gov': 0, 'local_gov': 371409},
    },
    # Key local governments (subset for brevity - full data in CSV)
    "Houston, TX": {
        2015: {'total_gov': 2024974, 'state_gov': 0, 'local_gov': 0},
        2017: {'total_gov': 1997358, 'state_gov': 0, 'local_gov': 0},
        2019: {'total_gov': 2073025, 'state_gov': 0, 'local_gov': 0},
        2020: {'total_gov': 1939150, 'state_gov': 0, 'local_gov': 0},
        2022: {'total_gov': 2080325, 'state_gov': 0, 'local_gov': 0},
    },
    "New Orleans, LA": {
        2015: {'total_gov': 165729, 'state_gov': 0, 'local_gov': 0},
        2017: {'total_gov': 167111, 'state_gov': 0, 'local_gov': 0},
        2019: {'total_gov': 173248, 'state_gov': 0, 'local_gov': 0},
        2020: {'total_gov': 144941, 'state_gov': 0, 'local_gov': 0},
        2022: {'total_gov': 159309, 'state_gov': 0, 'local_gov': 0},
    },
    "New York City, NY": {
        2015: {'total_gov': 2112535, 'state_gov': 0, 'local_gov': 0},
        2017: {'total_gov': 2196442, 'state_gov': 0, 'local_gov': 0},
        2019: {'total_gov': 2295531, 'state_gov': 0, 'local_gov': 0},
        2020: {'total_gov': 1968454, 'state_gov': 0, 'local_gov': 0},
        2022: {'total_gov': 2146292, 'state_gov': 0, 'local_gov': 0},
    },
    "Joplin, MO": {
        2015: {'total_gov': 53376, 'state_gov': 0, 'local_gov': 0},
        2017: {'total_gov': 54196, 'state_gov': 0, 'local_gov': 0},
        2019: {'total_gov': 51872, 'state_gov': 0, 'local_gov': 0},
        2020: {'total_gov': 48381, 'state_gov': 0, 'local_gov': 0},
        2022: {'total_gov': 51881, 'state_gov': 0, 'local_gov': 0},
    },
}


def get_employment_for_year(grantee: str, year: int) -> Optional[Dict[str, int]]:
    """
    Get employment data for a grantee and year.

    Parameters
    ----------
    grantee : str
        Grantee name.
    year : int
        Year to retrieve.

    Returns
    -------
    Dict or None
        Dictionary with total_gov, state_gov, local_gov employment.
    """
    if grantee not in GRANTEE_EMPLOYMENT_BY_YEAR:
        return None
    years_data = GRANTEE_EMPLOYMENT_BY_YEAR[grantee]
    if year in years_data:
        return years_data[year]
    # Try to find closest year
    available_years = sorted(years_data.keys())
    if not available_years:
        return None
    # Use closest year
    closest = min(available_years, key=lambda y: abs(y - year))
    return years_data[closest]


def compute_employment_ratio(
    grantee: str,
    disaster_type: str
) -> Optional[float]:
    """
    Compute employment ratio (employees per 1,000 population) for grantee at disaster time.

    Parameters
    ----------
    grantee : str
        Grantee name.
    disaster_type : str
        DRGR disaster type string.

    Returns
    -------
    float or None
        Employment ratio (per 1,000 population) or None if data unavailable.
    """
    disaster_year = DRGR_DISASTER_YEARS.get(disaster_type)
    if disaster_year is None:
        disaster_year = 2020

    emp_data = get_employment_for_year(grantee, disaster_year)
    if emp_data is None:
        return None

    pop = get_population_for_disaster(grantee, disaster_type)
    if pop is None or pop == 0:
        return None

    return (emp_data['total_gov'] / pop) * 1000


def get_employment_for_all_grantees() -> pd.DataFrame:
    """
    Get employment data for all grantees as a DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Grantee, Year, Total_Gov, State_Gov, Local_Gov.
    """
    records = []
    for grantee, years_data in GRANTEE_EMPLOYMENT_BY_YEAR.items():
        for year, emp in years_data.items():
            records.append({
                "Grantee": grantee,
                "Year": year,
                "Total_Gov": emp['total_gov'],
                "State_Gov": emp['state_gov'],
                "Local_Gov": emp['local_gov'],
            })
    return pd.DataFrame(records)


# Legacy function for backward compatibility
def get_population_by_decade() -> pd.DataFrame:
    """Get all population data across all decades for all grantees."""
    fips_mapping = create_grantee_to_fips_mapping()
    results = []

    for grantee, pop_by_decade in GRANTEE_POPULATION_BY_DECADE.items():
        for census_year, population in pop_by_decade.items():
            results.append({
                "Grantee": grantee,
                "Census_Year": census_year,
                "Population": population,
                "FIPS": fips_mapping.get(grantee, ""),
            })

    df = pd.DataFrame(results)
    return df.sort_values(["Grantee", "Census_Year"])
