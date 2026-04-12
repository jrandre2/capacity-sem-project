"""
Recovered Kaifa notebook analysis port.

This module ports the logic from the recovered Kaifa SEM notebook bundle
imported into:

    manuscript_kaifa_archive/data/recovered_code_bundle/

It is materially closer to the original workflow than the provisional
external reconstruction because it is based on the recovered notebook source.
However, it is still a repository-local port, not the original execution
environment. The recovered bundle is also internally inconsistent: the saved
notebook outputs reference a 573-row input, while the bundled CSV currently in
the ZIP has 577 rows and reruns to a different coefficient set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from config import DATA_WORK_DIR, PROJECT_ROOT

try:
    from semopy import Model, calc_stats

    SEMOPY_AVAILABLE = True
except ImportError:  # pragma: no cover - handled by callers/tests
    SEMOPY_AVAILABLE = False


KAIFA_RECOVERED_LABEL = "RECOVERED_NOTEBOOK_PORT"
KAIFA_RECOVERED_WARNING = (
    "Generated from a repository port of recovered Kaifa notebooks. "
    "The recovered ZIP bundle is not perfectly internally consistent: the "
    "saved notebook outputs reference a 573-row run, while the bundled CSV "
    "currently present reruns at 577 rows."
)

FIT_STATISTICS = [
    "DoF",
    "DoF Baseline",
    "chi2",
    "chi2 p-value",
    "chi2 Baseline",
    "CFI",
    "GFI",
    "AGFI",
    "NFI",
    "TLI",
    "RMSEA",
    "AIC",
    "BIC",
]

EXACT_MATCH_TOLERANCE = 1e-10
SUBSET_FORENSICS_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

MATURITY_BANDS = [
    ("<=24 months", -1.0, 24.0),
    ("24-60 months", 24.0, 60.0),
    (">60 months", 60.0, float("inf")),
]


@dataclass(frozen=True)
class RecoveredBundlePaths:
    """Locations for the recovered Kaifa notebook bundle."""

    root: Path
    zip_path: Path
    sem_notebook: Path
    partial_notebook: Path
    raw_csv: Path
    source_docx: Path
    reference_export_root: Path


RECOVERED_BUNDLE = RecoveredBundlePaths(
    root=PROJECT_ROOT / "manuscript_kaifa_archive" / "data" / "recovered_code_bundle",
    zip_path=(
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "data"
        / "recovered_code_bundle"
        / "sem_output_and_partial_dependence_plots.zip"
    ),
    sem_notebook=(
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "data"
        / "recovered_code_bundle"
        / "extracted"
        / "SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb"
    ),
    partial_notebook=(
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "data"
        / "recovered_code_bundle"
        / "extracted"
        / "sem_partial_dependence_outputs_4_v2_final.ipynb"
    ),
    raw_csv=(
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "data"
        / "recovered_code_bundle"
        / "extracted"
        / "all_state_local_fund_latent_var_4_v2.csv"
    ),
    source_docx=(
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "source_docs"
        / "SEM_Manuscript_2026-04-09_code_bundle.docx"
    ),
    reference_export_root=(
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "data"
        / "external_sem_exports"
        / "baseline_sem"
    ),
)

OFFICIAL_COUNTY_PATH = PROJECT_ROOT / "data_work" / "spatial" / "tl_2023_us_county.zip"
OFFICIAL_STATE_PATH = PROJECT_ROOT / "data_work" / "spatial" / "tl_2023_us_state.zip"
EMPLOYMENT_REFERENCE_PATH = PROJECT_ROOT / "data_work" / "employment.parquet"
QPR_STANDARDIZED_PATH = PROJECT_ROOT / "data_work" / "qpr_standardized.parquet"
QPR_QUARTERLY_PATH = PROJECT_ROOT / "data_work" / "qpr_quarterly.parquet"

BASE_STRUCTURAL_COVARIATES = [
    "state_level",
    "z_E_TOTPOP",
    "z_SPL_THEME1",
    "z_SPL_THEME2",
    "z_SPL_THEME3",
    "z_SPL_THEME4",
]


ONE_FACTOR_MODEL_SPEC = """
# Recovered Kaifa notebook port
GovCapacity =~ z_avg_employment + z_avg_payroll + z_rev_programs_per_staff + z_rev_disasters_per_staff
RecoveryPerformance =~ z_Ratio_disbursed_to_obligated + z_Ratio_expended_to_disbursed + z_Ratio_obligated_funds_fully_expended + z_Ratio_Program_Completed
RecoveryTimeliness =~ z_rev_Duration_of_completion + z_rev_Average_Duration_Program_Completion
RecoveryPerformance ~ GovCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
RecoveryTimeliness ~ GovCapacity + state_level + z_E_TOTPOP + z_SPL_THEME1 + z_SPL_THEME2 + z_SPL_THEME3 + z_SPL_THEME4
RecoveryPerformance ~~ RecoveryTimeliness
"""


TWO_FACTOR_MODEL_SPEC = """
# populated below via helper for consistency
"""


def _build_two_factor_model_spec(
    additional_predictors: list[str] | None = None,
    performance_indicators: list[str] | None = None,
    timeliness_indicators: list[str] | None = None,
    burden_indicators: list[str] | None = None,
) -> str:
    """Build a two-factor SEM specification with optional extra controls."""
    performance = performance_indicators or [
        "z_Ratio_disbursed_to_obligated",
        "z_Ratio_expended_to_disbursed",
        "z_Ratio_obligated_funds_fully_expended",
        "z_Ratio_Program_Completed",
    ]
    timeliness = timeliness_indicators or [
        "z_rev_Duration_of_completion",
        "z_rev_Average_Duration_Program_Completion",
    ]
    burden = burden_indicators or [
        "z_rev_programs_per_staff",
        "z_rev_disasters_per_staff",
    ]
    predictors = ["AdminResources", "AdminBurdenCapacity", *BASE_STRUCTURAL_COVARIATES]
    if additional_predictors:
        predictors.extend(additional_predictors)
    predictor_block = " + ".join(predictors)
    return "\n".join(
        [
            "# Recovered Kaifa notebook port",
            "AdminResources =~ z_avg_employment + z_avg_payroll",
            f"AdminBurdenCapacity =~ {' + '.join(burden)}",
            f"RecoveryPerformance =~ {' + '.join(performance)}",
            f"RecoveryTimeliness =~ {' + '.join(timeliness)}",
            f"RecoveryPerformance ~ {predictor_block}",
            f"RecoveryTimeliness ~ {predictor_block}",
            "AdminResources ~~ AdminBurdenCapacity",
            "RecoveryPerformance ~~ RecoveryTimeliness",
        ]
    )


TWO_FACTOR_MODEL_SPEC = _build_two_factor_model_spec()


TWO_FACTOR_DROP_WEAK_INDICATOR_SPEC = _build_two_factor_model_spec(
    performance_indicators=[
        "z_Ratio_disbursed_to_obligated",
        "z_Ratio_obligated_funds_fully_expended",
        "z_Ratio_Program_Completed",
    ]
)


def default_output_dir() -> Path:
    """Default location for recovered Kaifa notebook outputs."""
    return DATA_WORK_DIR / "diagnostics" / "kaifa_recovered_analysis"


def load_recovered_raw_data() -> pd.DataFrame:
    """Load the raw CSV from the recovered bundle."""
    return pd.read_csv(RECOVERED_BUNDLE.raw_csv)


def prepare_sem_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the SEM-ready dataset exactly as the recovered notebook specifies."""
    return prepare_sem_data_with_workload_mode(raw_df, workload_mode="employment_x_population")


def prepare_sem_data_with_workload_mode(
    raw_df: pd.DataFrame,
    workload_mode: str = "employment_x_population",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare SEM-ready data with a selectable workload denominator."""
    df = raw_df.copy()
    df["state_level"] = df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").astype(int)
    df["gov_level"] = df["state_level"].map({1: "state", 0: "local"})

    eps = 1e-6
    if workload_mode == "employment_x_population":
        denominator = df["avg_employment"] * df["E_TOTPOP"] + eps
    elif workload_mode == "employment_only":
        denominator = df["avg_employment"] + eps
    else:
        raise ValueError(f"Unsupported workload_mode: {workload_mode}")

    df["programs_per_staff"] = df["Num_Program"] / denominator
    df["disasters_per_staff"] = df["Num_Disaster"] / denominator
    df["rev_programs_per_staff"] = -df["programs_per_staff"]
    df["rev_disasters_per_staff"] = -df["disasters_per_staff"]
    df["rev_Duration_of_completion"] = -df["Duration_of_completion"]
    df["rev_Average_Duration_Program_Completion"] = -df[
        "Average_Duration_Program_Completion"
    ]

    sem_vars = [
        "avg_employment",
        "avg_payroll",
        "rev_programs_per_staff",
        "rev_disasters_per_staff",
        "Ratio_disbursed_to_obligated",
        "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended",
        "Ratio_Program_Completed",
        "rev_Duration_of_completion",
        "rev_Average_Duration_Program_Completion",
        "state_level",
        "E_TOTPOP",
        "SPL_THEME1",
        "SPL_THEME2",
        "SPL_THEME3",
        "SPL_THEME4",
    ]

    data = df[["Grantee"] + sem_vars].dropna().copy()

    continuous_vars = [
        "avg_employment",
        "avg_payroll",
        "rev_programs_per_staff",
        "rev_disasters_per_staff",
        "Ratio_disbursed_to_obligated",
        "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended",
        "Ratio_Program_Completed",
        "rev_Duration_of_completion",
        "rev_Average_Duration_Program_Completion",
        "E_TOTPOP",
        "SPL_THEME1",
        "SPL_THEME2",
        "SPL_THEME3",
        "SPL_THEME4",
    ]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data_z = data.copy()
    z_cols = [f"z_{col}" for col in continuous_vars]
    data_z[z_cols] = scaler.fit_transform(data[continuous_vars])

    sem_data = data_z[
        [
            "Grantee",
            "state_level",
            "z_avg_employment",
            "z_avg_payroll",
            "z_rev_programs_per_staff",
            "z_rev_disasters_per_staff",
            "z_Ratio_disbursed_to_obligated",
            "z_Ratio_expended_to_disbursed",
            "z_Ratio_obligated_funds_fully_expended",
            "z_Ratio_Program_Completed",
            "z_rev_Duration_of_completion",
            "z_rev_Average_Duration_Program_Completion",
            "z_E_TOTPOP",
            "z_SPL_THEME1",
            "z_SPL_THEME2",
            "z_SPL_THEME3",
            "z_SPL_THEME4",
        ]
    ].copy()

    return df, sem_data


def _fit_sem_model(
    model_spec: str, data: pd.DataFrame
) -> tuple[Model, pd.DataFrame, pd.DataFrame]:
    """Fit a semopy model and return parameters and fit statistics."""
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required for recovered Kaifa notebook analysis")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(model_spec)
        model.fit(data)

    params = model.inspect(std_est=True)
    raw_stats = calc_stats(model)
    stats_row = raw_stats.loc["Value"] if "Value" in raw_stats.index else raw_stats.iloc[0]
    fit_stats = stats_row.rename_axis("Statistic").reset_index(name="Value")
    return model, params, fit_stats


def _subset_parameters(params: pd.DataFrame, op: str, outcomes: list[str] | None = None) -> pd.DataFrame:
    """Select a stable subset of parameter rows for reporting."""
    mask = params["op"] == op
    if outcomes is not None:
        mask &= params["lval"].isin(outcomes)
    cols = ["lval", "op", "rval", "Estimate", "Est. Std", "p-value"]
    available = [col for col in cols if col in params.columns]
    return params.loc[mask, available].copy()


def _build_model_comparison(
    one_factor_fit: pd.DataFrame,
    two_factor_fit: pd.DataFrame,
) -> pd.DataFrame:
    """Create a one-factor vs two-factor comparison table."""
    one = one_factor_fit.rename(columns={"Value": "OneFactor"})
    two = two_factor_fit.rename(columns={"Value": "TwoFactor"})
    merged = one.merge(two, on="Statistic", how="outer")
    return merged[merged["Statistic"].isin(FIT_STATISTICS)].reset_index(drop=True)


def _load_reference_comparison() -> pd.DataFrame | None:
    """Load the previously imported reference fit table if present."""
    path = RECOVERED_BUNDLE.reference_export_root / "sem_model_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Statistic"})
    return df


def _load_reference_ready_dataset() -> pd.DataFrame | None:
    """Load the reference 573-row SEM-ready dataset if present."""
    path = RECOVERED_BUNDLE.reference_export_root / "sem_2factor_ready_dataset.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_reference_parameters() -> pd.DataFrame | None:
    """Load reference two-factor SEM parameter estimates if present."""
    path = RECOVERED_BUNDLE.reference_export_root / "sem_2factor_parameter_estimates.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_reference_admin4_ready_dataset() -> pd.DataFrame | None:
    """Load the smaller cleaned sensitivity sample if present."""
    path = (
        PROJECT_ROOT
        / "manuscript_kaifa_archive"
        / "data"
        / "external_sem_exports"
        / "baseline_sem_admin_4"
        / "sem_2factor_ready_dataset.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


def _normalize_geo_name(value: object) -> str:
    """Normalize place labels for stable county/state matching."""
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _county_aliases(value: object) -> set[str]:
    """Generate tolerant aliases for county-equivalent labels."""
    base = _normalize_geo_name(value)
    aliases = {base}
    stripped = re.sub(
        r"\b(county|parish|borough|census area|municipality|city and borough|city)\b",
        "",
        base,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped:
        aliases.add(stripped)
    return aliases


def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-score a series while tolerating degenerate input."""
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if pd.isna(std) or std <= 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - float(numeric.mean())) / std


def _build_auxiliary_sem_controls(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build extra SEM control variables used in sensitivity checks."""
    df = raw_df[["Grantee", "Num_Program", "Num_Disaster", "Duration_of_completion"]].copy()
    df["z_log_num_program"] = _zscore_series(np.log1p(df["Num_Program"]))
    df["z_log_num_disaster"] = _zscore_series(np.log1p(df["Num_Disaster"]))
    df["maturity_mid_band"] = (
        (df["Duration_of_completion"] > 24.0) & (df["Duration_of_completion"] <= 60.0)
    ).astype(int)
    df["maturity_long_band"] = (df["Duration_of_completion"] > 60.0).astype(int)
    return df[
        [
            "Grantee",
            "z_log_num_program",
            "z_log_num_disaster",
            "maturity_mid_band",
            "maturity_long_band",
        ]
    ]


def _residualize_burden_indicators(sem_data: pd.DataFrame) -> pd.DataFrame:
    """Residualize burden indicators on resource size proxies to test arithmetic coupling."""
    from sklearn.linear_model import LinearRegression

    predictors = sem_data[["z_avg_employment", "z_avg_payroll", "z_E_TOTPOP"]]
    result = sem_data.copy()
    for column in ["z_rev_programs_per_staff", "z_rev_disasters_per_staff"]:
        model = LinearRegression()
        model.fit(predictors, sem_data[column])
        resid = sem_data[column] - model.predict(predictors)
        result[f"{column}_resid"] = _zscore_series(resid)
    return result


def _fit_two_factor_spec(
    sem_data: pd.DataFrame,
    additional_predictors: list[str] | None = None,
    performance_indicators: list[str] | None = None,
    burden_indicators: list[str] | None = None,
) -> tuple[Model, pd.DataFrame, pd.DataFrame]:
    """Fit a flexible two-factor SEM on an augmented SEM-ready dataframe."""
    spec = _build_two_factor_model_spec(
        additional_predictors=additional_predictors,
        performance_indicators=performance_indicators,
        burden_indicators=burden_indicators,
    )
    return _fit_sem_model(spec, sem_data.drop(columns=["Grantee"]))


def build_measurement_appendix_table(parameters: pd.DataFrame | None) -> pd.DataFrame | None:
    """Create a compact measurement table for manuscript appendix use."""
    if parameters is None:
        return None

    df = parameters.copy()
    for col in ["Estimate", "Est. Std", "Std. Err", "p-value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    latent_by_indicator = {
        "z_avg_employment": "Administrative Resources",
        "z_avg_payroll": "Administrative Resources",
        "z_rev_programs_per_staff": "Administrative Burden Capacity",
        "z_rev_disasters_per_staff": "Administrative Burden Capacity",
        "z_Ratio_disbursed_to_obligated": "Recovery Performance",
        "z_Ratio_expended_to_disbursed": "Recovery Performance",
        "z_Ratio_obligated_funds_fully_expended": "Recovery Performance",
        "z_Ratio_Program_Completed": "Recovery Performance",
        "z_rev_Duration_of_completion": "Recovery Timeliness",
        "z_rev_Average_Duration_Program_Completion": "Recovery Timeliness",
    }
    indicator_labels = {
        "z_avg_employment": "Employment proxy",
        "z_avg_payroll": "Payroll proxy",
        "z_rev_programs_per_staff": "Reverse-coded programs per staff",
        "z_rev_disasters_per_staff": "Reverse-coded disasters per staff",
        "z_Ratio_disbursed_to_obligated": "Disbursed to obligated ratio",
        "z_Ratio_expended_to_disbursed": "Expended to disbursed ratio",
        "z_Ratio_obligated_funds_fully_expended": "Share of obligated funds fully expended",
        "z_Ratio_Program_Completed": "Program completion share",
        "z_rev_Duration_of_completion": "Reverse-coded duration of completion",
        "z_rev_Average_Duration_Program_Completion": "Reverse-coded average program duration",
    }

    rows = []
    residual_lookup = (
        df[(df["op"] == "~~") & (df["lval"] == df["rval"])]
        .set_index("lval")
        .to_dict(orient="index")
    )
    for indicator, latent in latent_by_indicator.items():
        row = df[(df["op"] == "~") & (df["lval"] == indicator)]
        if row.empty:
            continue
        record = row.iloc[0]
        residual = residual_lookup.get(indicator, {})
        note = ""
        if indicator == "z_Ratio_expended_to_disbursed":
            note = "Weak negative loading"
        elif indicator in {"z_rev_Duration_of_completion", "z_rev_Average_Duration_Program_Completion"}:
            note = "Duration-based timeliness indicator"
        rows.append(
            {
                "latent_construct": latent,
                "indicator": indicator_labels.get(indicator, indicator),
                "std_loading": record.get("Est. Std"),
                "std_error": record.get("Std. Err"),
                "std_residual_var": residual.get("Est. Std"),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def build_maturity_summary(raw_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Summarize the SEM sample by broad portfolio-maturity proxy bands."""
    if raw_df is None or raw_df.empty:
        return None

    df = raw_df.copy()
    bins = [-1.0, 24.0, 60.0, float("inf")]
    labels = [band[0] for band in MATURITY_BANDS]
    df["maturity_band"] = pd.cut(
        df["Duration_of_completion"], bins=bins, labels=labels
    )
    df["state_level"] = df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").astype(int)

    summary = (
        df.groupby("maturity_band", observed=False)
        .agg(
            jurisdictions=("Grantee", "size"),
            state_cases=("state_level", "sum"),
            local_cases=("state_level", lambda s: int((1 - s).sum())),
            median_programs=("Num_Program", "median"),
            median_disasters=("Num_Disaster", "median"),
            median_duration_months=("Duration_of_completion", "median"),
        )
        .reset_index()
    )
    return summary


def build_geography_summary(raw_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Create a compact summary of the analytical geography pipeline."""
    if raw_df is None or raw_df.empty:
        return None

    df = raw_df.copy()
    state_cases = int(df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").sum())
    local_cases = int(len(df) - state_cases)
    return pd.DataFrame(
        [
            {
                "mapping_step": "State-agency SEM unit",
                "description": "Two-letter grantee labels retained as statewide administering agencies.",
                "count_or_implication": state_cases,
            },
            {
                "mapping_step": "County-linked local SEM unit",
                "description": "Local activity or city entities aggregated to a county-linked administering jurisdiction.",
                "count_or_implication": local_cases,
            },
            {
                "mapping_step": "County map assignment",
                "description": "County-level maps inherit local-profile SEM-facing values after the county linkage step.",
                "count_or_implication": local_cases,
            },
            {
                "mapping_step": "State map assignment",
                "description": "State-level maps inherit statewide agency values rather than county-linked local scores.",
                "count_or_implication": state_cases,
            },
        ]
    )


def build_official_geography_crosswalk(
    raw_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Audit recovered SEM units against official 2023 Census state/county files."""
    if raw_df is None or raw_df.empty:
        return None, None

    import geopandas as gpd

    county_gdf = gpd.read_file(OFFICIAL_COUNTY_PATH)[["STATEFP", "GEOID", "NAME", "NAMELSAD"]]
    state_gdf = gpd.read_file(OFFICIAL_STATE_PATH)[["STATEFP", "GEOID", "STUSPS", "NAME"]]
    county_gdf = county_gdf.merge(
        state_gdf[["STATEFP", "STUSPS", "NAME"]].rename(columns={"NAME": "state_name"}),
        on="STATEFP",
        how="left",
    )

    state_lookup = state_gdf.set_index("STUSPS").to_dict(orient="index")
    county_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in county_gdf.iterrows():
        for label in [row["NAMELSAD"], row["NAME"]]:
            for alias in _county_aliases(label):
                county_lookup[(alias, row["STUSPS"])] = row.to_dict()

    rows: list[dict[str, object]] = []
    for label in raw_df["Grantee"].tolist():
        if re.fullmatch(r"[A-Z]{2}", str(label)):
            state_record = state_lookup.get(str(label))
            rows.append(
                {
                    "Grantee": label,
                    "unit_type": "state_agency",
                    "state_abbr": label,
                    "raw_local_label": "",
                    "mapped_name": state_record["NAME"] if state_record else "",
                    "mapped_geoid": state_record["GEOID"] if state_record else "",
                    "mapping_method": "official_state_abbreviation_match",
                    "mapping_source": OFFICIAL_STATE_PATH.name,
                    "confidence": 1.0 if state_record else 0.0,
                    "matched": bool(state_record),
                }
            )
            continue

        local_label, state_abbr = [part.strip() for part in str(label).rsplit(",", 1)]
        county_record = None
        for alias in _county_aliases(local_label):
            county_record = county_lookup.get((alias, state_abbr))
            if county_record:
                break
        rows.append(
            {
                "Grantee": label,
                "unit_type": "county_linked_local",
                "state_abbr": state_abbr,
                "raw_local_label": local_label,
                "mapped_name": county_record["NAMELSAD"] if county_record else "",
                "mapped_geoid": county_record["GEOID"] if county_record else "",
                "mapping_method": "official_county_name_match",
                "mapping_source": OFFICIAL_COUNTY_PATH.name,
                "confidence": 1.0 if county_record else 0.0,
                "matched": bool(county_record),
            }
        )

    detail = pd.DataFrame(rows).sort_values(["unit_type", "Grantee"]).reset_index(drop=True)
    state_total = int((detail["unit_type"] == "state_agency").sum())
    local_total = int((detail["unit_type"] == "county_linked_local").sum())
    state_matched = int(((detail["unit_type"] == "state_agency") & detail["matched"]).sum())
    local_matched = int(((detail["unit_type"] == "county_linked_local") & detail["matched"]).sum())
    unmatched = int((~detail["matched"]).sum())

    summary = pd.DataFrame(
        [
            {
                "mapping_step": "Official Census state-agency crosswalk",
                "description": (
                    "Two-letter state-administered SEM units matched directly to the 2023 "
                    "Census state file by STUSPS abbreviation."
                ),
                "count_or_implication": f"{state_matched}/{state_total} matched ({state_matched / max(state_total, 1):.1%})",
            },
            {
                "mapping_step": "Official Census county crosswalk",
                "description": (
                    "County-linked local SEM units matched directly to the 2023 Census "
                    "county file by normalized county/parish/borough label and state."
                ),
                "count_or_implication": f"{local_matched}/{local_total} matched ({local_matched / max(local_total, 1):.1%})",
            },
            {
                "mapping_step": "Recovered-sample unresolved cases",
                "description": (
                    "Recovered SEM sample units that did not resolve to an official Census "
                    "state or county GEOID."
                ),
                "count_or_implication": unmatched,
            },
            {
                "mapping_step": "Map inheritance rule",
                "description": (
                    "County-linked local units inherit county GEOIDs; state-administered "
                    "units inherit statewide GEOIDs rather than county scores."
                ),
                "count_or_implication": len(detail),
            },
        ]
    )
    return detail, summary


def build_sensitivity_summary(
    reference_ready: pd.DataFrame | None,
    candidate_raw_subset: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Compare the main two-factor SEM to light-touch robustness variants."""
    if reference_ready is None:
        return None

    rows: list[dict[str, object]] = []

    _, default_params, default_fit = _fit_sem_model(
        TWO_FACTOR_MODEL_SPEC, reference_ready.drop(columns=["Grantee"])
    )
    rows.append(
        _sensitivity_row(
            "Primary two-factor SEM",
            default_fit,
            default_params,
            note="Main manuscript specification",
        )
    )

    _, drop_params, drop_fit = _fit_sem_model(
        TWO_FACTOR_DROP_WEAK_INDICATOR_SPEC, reference_ready.drop(columns=["Grantee"])
    )
    rows.append(
        _sensitivity_row(
            "Drop weak performance indicator",
            drop_fit,
            drop_params,
            note="Removes expended-to-disbursed from Recovery Performance",
        )
    )

    if candidate_raw_subset is not None:
        _, alt_sem_data = prepare_sem_data_with_workload_mode(
            candidate_raw_subset, workload_mode="employment_only"
        )
        _, alt_params, alt_fit = _fit_two_factor_spec(alt_sem_data)
        rows.append(
            _sensitivity_row(
                "Employment-only workload denominator",
                alt_fit,
                alt_params,
                note="Uses NumProgram/avg_employment and NumDisaster/avg_employment",
            )
        )

        auxiliary = _build_auxiliary_sem_controls(candidate_raw_subset)
        base_sem_data = reference_ready.merge(auxiliary, on="Grantee", how="left")

        _, maturity_params, maturity_fit = _fit_two_factor_spec(
            base_sem_data,
            additional_predictors=["maturity_mid_band", "maturity_long_band"],
        )
        rows.append(
            _sensitivity_row(
                "Coarse maturity-band controls",
                maturity_fit,
                maturity_params,
                note=(
                    "Adds mid- and long-duration portfolio band dummies as conservative "
                    "outcome-adjacent maturity controls."
                ),
            )
        )

        _, scale_params, scale_fit = _fit_two_factor_spec(
            base_sem_data,
            additional_predictors=["z_log_num_program", "z_log_num_disaster"],
        )
        rows.append(
            _sensitivity_row(
                "Portfolio-scale controls",
                scale_fit,
                scale_params,
                note="Adds logged program and disaster counts to separate workload density from raw portfolio scale.",
            )
        )

        residualized_sem_data = _residualize_burden_indicators(reference_ready)
        _, resid_params, resid_fit = _fit_two_factor_spec(
            residualized_sem_data,
            burden_indicators=[
                "z_rev_programs_per_staff_resid",
                "z_rev_disasters_per_staff_resid",
            ],
        )
        rows.append(
            _sensitivity_row(
                "Residualized burden indicators",
                resid_fit,
                resid_params,
                note="Residualizes burden indicators on employment, payroll, and population to test arithmetic coupling.",
            )
        )

    return pd.DataFrame(rows)


def _sensitivity_row(
    specification: str,
    fit: pd.DataFrame,
    parameters: pd.DataFrame,
    note: str,
) -> dict[str, object]:
    """Extract the sensitivity statistics used in the manuscript appendix table."""
    fit_lookup = fit.set_index("Statistic")["Value"].to_dict()
    params = parameters.copy()
    for col in ["Estimate", "Est. Std", "Std. Err", "p-value"]:
        if col in params.columns:
            params[col] = pd.to_numeric(params[col], errors="coerce")

    def std_path(outcome: str, predictor: str) -> float | None:
        row = params[
            (params["op"] == "~")
            & (params["lval"] == outcome)
            & (params["rval"] == predictor)
        ]
        if row.empty:
            return None
        return float(row.iloc[0]["Est. Std"])

    return {
        "specification": specification,
        "CFI": fit_lookup.get("CFI"),
        "RMSEA": fit_lookup.get("RMSEA"),
        "burden_to_performance_beta": std_path(
            "RecoveryPerformance", "AdminBurdenCapacity"
        ),
        "burden_to_timeliness_beta": std_path(
            "RecoveryTimeliness", "AdminBurdenCapacity"
        ),
        "resources_to_performance_beta": std_path(
            "RecoveryPerformance", "AdminResources"
        ),
        "resources_to_timeliness_beta": std_path(
            "RecoveryTimeliness", "AdminResources"
        ),
        "note": note,
    }


def build_proxy_validation_summary(
    raw_df: pd.DataFrame | None,
    sensitivity_summary: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Summarize proxy correlations and coupling checks for QCEW/workload measures."""
    if raw_df is None or raw_df.empty:
        return None

    rows: list[dict[str, object]] = []

    def corr(left: str, right: str) -> float:
        return float(pd.to_numeric(raw_df[left], errors="coerce").corr(pd.to_numeric(raw_df[right], errors="coerce")))

    rows.extend(
        [
            {
                "check": "Employment proxy vs payroll proxy",
                "sample": f"Full SEM sample (n={len(raw_df)})",
                "estimate": corr("avg_employment", "avg_payroll"),
                "note": "High co-movement is expected if both capture administrative scale.",
            },
            {
                "check": "Employment proxy vs population",
                "sample": f"Full SEM sample (n={len(raw_df)})",
                "estimate": corr("avg_employment", "E_TOTPOP"),
                "note": "Tests whether the staffing proxy mainly reflects jurisdiction size.",
            },
            {
                "check": "Payroll proxy vs population",
                "sample": f"Full SEM sample (n={len(raw_df)})",
                "estimate": corr("avg_payroll", "E_TOTPOP"),
                "note": "Population-adjusted payroll should not collapse to simple size alone.",
            },
            {
                "check": "Employment proxy vs raw program count",
                "sample": f"Full SEM sample (n={len(raw_df)})",
                "estimate": corr("avg_employment", "Num_Program"),
                "note": "Helps separate pure administrative scale from portfolio volume.",
            },
            {
                "check": "Employment proxy vs raw disaster count",
                "sample": f"Full SEM sample (n={len(raw_df)})",
                "estimate": corr("avg_employment", "Num_Disaster"),
                "note": "If this is near 1, staffing may simply proxy exposure intensity.",
            },
        ]
    )

    if EMPLOYMENT_REFERENCE_PATH.exists():
        import us
        from stages._io_utils import safe_read_parquet

        state_df = raw_df[raw_df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}")].copy()
        state_df["state_name"] = state_df["Grantee"].map(
            lambda value: us.states.lookup(value).name if us.states.lookup(value) else value
        )
        employment = safe_read_parquet(EMPLOYMENT_REFERENCE_PATH)
        employment_avg = (
            employment.groupby("Grantee")[["Total_Gov"]].mean().reset_index()
        )
        merged = state_df.merge(
            employment_avg,
            left_on="state_name",
            right_on="Grantee",
            how="left",
            suffixes=("", "_employment"),
        )
        matched = merged["Total_Gov"].notna()
        if matched.any():
            rows.extend(
                [
                    {
                        "check": "State-only external employment cross-check",
                        "sample": f"States with external government-employment match (n={int(matched.sum())})",
                        "estimate": float(merged.loc[matched, "avg_employment"].corr(merged.loc[matched, "Total_Gov"])),
                        "note": "Compares the recovered state staffing proxy to a separate BLS total-government employment series.",
                    },
                    {
                        "check": "State-only external payroll cross-check",
                        "sample": f"States with external government-employment match (n={int(matched.sum())})",
                        "estimate": float(merged.loc[matched, "avg_payroll"].corr(merged.loc[matched, "Total_Gov"])),
                        "note": "A coarse external validity check; stronger direct staffing validation remains desirable.",
                    },
                ]
            )

    if sensitivity_summary is not None and not sensitivity_summary.empty:
        residualized = sensitivity_summary[
            sensitivity_summary["specification"] == "Residualized burden indicators"
        ]
        if not residualized.empty:
            record = residualized.iloc[0]
            rows.append(
                {
                    "check": "Residualized burden -> timeliness beta",
                    "sample": "SEM coupling sensitivity",
                    "estimate": record["burden_to_timeliness_beta"],
                    "note": "If this remains positive, the main burden-timeliness association is not driven only by arithmetic coupling.",
                }
            )

    return pd.DataFrame(rows)


def build_subset_forensics_summary(
    reference_ready: pd.DataFrame | None,
    admin4_ready: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """Forensically summarize the smaller cleaned sensitivity sample."""
    if reference_ready is None or admin4_ready is None:
        return None, None, None

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    df = reference_ready.copy()
    included = df["Grantee"].isin(set(admin4_ready["Grantee"]))
    df["included_in_admin4"] = included.astype(int)

    def _state_abbr(label: str) -> str:
        if re.fullmatch(r"[A-Z]{2}", str(label)):
            return str(label)
        return str(label).rsplit(",", 1)[1].strip()

    retention_by_state = (
        df.assign(state_abbr=df["Grantee"].map(_state_abbr))
        .groupby(["state_abbr", "state_level"], as_index=False)
        .agg(
            retained=("included_in_admin4", "sum"),
            total=("included_in_admin4", "size"),
        )
    )
    retention_by_state["retention_rate"] = retention_by_state["retained"] / retention_by_state["total"]

    zcols = [column for column in df.columns if column.startswith("z_")]
    best_single: dict[str, object] | None = None
    for column in zcols:
        series = pd.to_numeric(df[column], errors="coerce")
        for threshold in SUBSET_FORENSICS_THRESHOLDS:
            for label, pred in [
                (f"|{column}| <= {threshold}", (series.abs() <= threshold)),
                (f"|{column}| > {threshold}", (series.abs() > threshold)),
            ]:
                accuracy = float((pred.astype(int) == df["included_in_admin4"]).mean())
                if best_single is None or accuracy > best_single["accuracy"]:
                    best_single = {
                        "rule": label,
                        "accuracy": accuracy,
                    }

    best_common: dict[str, object] | None = None
    for threshold in SUBSET_FORENSICS_THRESHOLDS:
        pred = (df[zcols].abs() <= threshold).all(axis=1)
        accuracy = float((pred.astype(int) == df["included_in_admin4"]).mean())
        if best_common is None or accuracy > best_common["accuracy"]:
            best_common = {
                "rule": f"all |z| <= {threshold}",
                "accuracy": accuracy,
            }

    X = df[zcols + ["state_level"]]
    y = df["included_in_admin4"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_scores = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
    tree.fit(X, y)
    tree_rules = export_text(tree, feature_names=list(X.columns))
    feature_importances = (
        pd.Series(tree.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    leading_features = ", ".join(
        feature_importances.loc[feature_importances > 0].head(3).index.tolist()
    )

    state_mask = df["state_level"] == 1
    local_mask = df["state_level"] == 0
    summary = pd.DataFrame(
        [
            {
                "finding": "Strict subset relation",
                "value": "Yes" if set(admin4_ready["Grantee"]).issubset(set(reference_ready["Grantee"])) else "No",
                "note": "The smaller cleaned sensitivity sample is nested within the 573-jurisdiction SEM sample.",
            },
            {
                "finding": "Retained jurisdictions",
                "value": f"{int(included.sum())}/{len(df)} ({included.mean():.1%})",
                "note": "Share of the 573-jurisdiction reference sample retained in the N=169 subset.",
            },
            {
                "finding": "State retention rate",
                "value": f"{int(df.loc[state_mask, 'included_in_admin4'].sum())}/{int(state_mask.sum())} ({df.loc[state_mask, 'included_in_admin4'].mean():.1%})",
                "note": "State agencies are retained at a much higher rate than local jurisdictions.",
            },
            {
                "finding": "Local retention rate",
                "value": f"{int(df.loc[local_mask, 'included_in_admin4'].sum())}/{int(local_mask.sum())} ({df.loc[local_mask, 'included_in_admin4'].mean():.1%})",
                "note": "Local jurisdictions are heavily thinned in the smaller cleaned sample.",
            },
            {
                "finding": "Best single-threshold rule",
                "value": best_single["rule"] if best_single else "-",
                "note": f"Best simple threshold accuracy = {best_single['accuracy']:.3f}" if best_single else "-",
            },
            {
                "finding": "Best common outlier screen",
                "value": best_common["rule"] if best_common else "-",
                "note": f"Best common-threshold accuracy = {best_common['accuracy']:.3f}" if best_common else "-",
            },
            {
                "finding": "Depth-3 tree CV accuracy",
                "value": f"{float(tree_scores.mean()):.3f}",
                "note": f"Leading classifier features: {leading_features}",
            },
            {
                "finding": "Forensic conclusion",
                "value": "Exact rule unrecovered",
                "note": "No simple outlier threshold reproduces the N=169 subset; it behaves like a compound or curated filter.",
            },
        ]
    )
    return summary, retention_by_state, tree_rules


def build_full_structural_reporting_table(
    parameters: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build a complete structural and latent-covariance appendix table."""
    if parameters is None or parameters.empty:
        return None

    df = parameters.copy()
    for column in ["Estimate", "Est. Std", "Std. Err", "p-value"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    latent_terms = {
        "AdminResources",
        "AdminBurdenCapacity",
        "RecoveryPerformance",
        "RecoveryTimeliness",
    }

    structural = df[(df["op"] == "~") & (df["lval"].isin(["RecoveryPerformance", "RecoveryTimeliness"]))].copy()
    structural["section"] = "Structural path"
    structural["note"] = structural["lval"].map(
        {
            "RecoveryPerformance": "Secondary exploratory outcome construct",
            "RecoveryTimeliness": "Primary substantive outcome construct",
        }
    )

    covariances = df[
        (df["op"] == "~~")
        & (df["lval"].isin(latent_terms))
        & (df["rval"].isin(latent_terms))
    ].copy()
    covariances = covariances[covariances["lval"] <= covariances["rval"]].copy()
    covariances["section"] = "Latent covariance/variance"
    covariances["note"] = covariances.apply(
        lambda row: (
            "Latent covariance"
            if row["lval"] != row["rval"]
            else "Latent residual variance"
        ),
        axis=1,
    )

    combined = pd.concat([structural, covariances], ignore_index=True)
    return combined[
        [
            "section",
            "lval",
            "op",
            "rval",
            "Estimate",
            "Est. Std",
            "Std. Err",
            "p-value",
            "note",
        ]
    ]


def build_fit_verification_table(
    reference_ready: pd.DataFrame | None,
    reference_comparison: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Verify imported fit statistics by rerunning the archived models on the 573-row SEM-ready data."""
    if reference_ready is None or reference_comparison is None:
        return None

    _, _, one_fit = _fit_sem_model(ONE_FACTOR_MODEL_SPEC, reference_ready.drop(columns=["Grantee"]))
    _, _, two_fit = _fit_sem_model(TWO_FACTOR_MODEL_SPEC, reference_ready.drop(columns=["Grantee"]))
    rerun = _build_model_comparison(one_fit, two_fit).rename(
        columns={
            "OneFactor": "OneFactor_rerun_reference_ready",
            "TwoFactor": "TwoFactor_rerun_reference_ready",
        }
    )
    merged = reference_comparison.merge(rerun, on="Statistic", how="left")
    for left, right, delta in [
        ("OneFactor", "OneFactor_rerun_reference_ready", "Delta_OneFactor"),
        ("TwoFactor", "TwoFactor_rerun_reference_ready", "Delta_TwoFactor"),
    ]:
        merged[left] = pd.to_numeric(merged[left], errors="coerce")
        merged[right] = pd.to_numeric(merged[right], errors="coerce")
        merged[delta] = merged[right] - merged[left]

    notes = {
        "AIC": "Archive and rerun values come directly from semopy calc_stats on the 573-row SEM-ready data.",
        "BIC": "Archive and rerun values come directly from semopy calc_stats on the 573-row SEM-ready data.",
    }
    merged["note"] = merged["Statistic"].map(notes).fillna("")
    return merged[
        [
            "Statistic",
            "OneFactor",
            "OneFactor_rerun_reference_ready",
            "Delta_OneFactor",
            "TwoFactor",
            "TwoFactor_rerun_reference_ready",
            "Delta_TwoFactor",
            "note",
        ]
    ]


def build_data_flow_summary(
    raw_df: pd.DataFrame | None,
    candidate_raw_subset: pd.DataFrame | None,
    reference_ready: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Summarize the recoverable path from raw QPR rows to the archived SEM sample."""
    if raw_df is None or candidate_raw_subset is None or reference_ready is None:
        return None

    from stages._io_utils import safe_read_parquet

    qpr_standardized = safe_read_parquet(QPR_STANDARDIZED_PATH)
    qpr_quarterly = safe_read_parquet(QPR_QUARTERLY_PATH)
    valid_quarter_rows = int((~qpr_standardized["QA_missing_qpr_actual_quarter"].astype(bool)).sum())

    return pd.DataFrame(
        [
            {
                "stage": "Standardized DRGR/QPR activity-quarter rows",
                "count": int(len(qpr_standardized)),
                "note": "Repository-standardized administrative rows before quarter-label exclusion.",
            },
            {
                "stage": "Rows with valid actual-quarter labels",
                "count": valid_quarter_rows,
                "note": "Excludes rows flagged with missing QPR Actual Quarter in the standardized archive.",
            },
            {
                "stage": "Collapsed grantee-disaster-quarter rows",
                "count": int(len(qpr_quarterly)),
                "note": "Quarter-collapsed standardized panel used by the repo-standardized path; not yet the recovered Kaifa SEM file.",
            },
            {
                "stage": "Recovered archived administering-jurisdiction profiles",
                "count": int(len(raw_df)),
                "note": "Row count in the recovered bundled CSV prior to excluding four extra profiles.",
            },
            {
                "stage": "Reference complete-case SEM sample",
                "count": int(len(reference_ready)),
                "note": "Archived 573-jurisdiction SEM-ready file used for the main Kaifa SEM tables (30 state, 543 local).",
            },
        ]
    )


def build_ratio_artifact_summary(
    candidate_raw_subset: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Document ratio values above 1.0 and add a capped-ratio sensitivity check."""
    if candidate_raw_subset is None or candidate_raw_subset.empty:
        return None

    rows: list[dict[str, object]] = []
    for column in [
        "Ratio_disbursed_to_obligated",
        "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended",
        "Ratio_Program_Completed",
    ]:
        numeric = pd.to_numeric(candidate_raw_subset[column], errors="coerce")
        rows.append(
            {
                "element": column,
                "value": int((numeric > 1.0).sum()),
                "note": f"Observed count above 1.0 out of {int(numeric.notna().sum())} recovered 573-sample profiles.",
            }
        )

    capped = candidate_raw_subset.copy()
    for column in [
        "Ratio_disbursed_to_obligated",
        "Ratio_expended_to_disbursed",
        "Ratio_obligated_funds_fully_expended",
        "Ratio_Program_Completed",
    ]:
        capped[column] = pd.to_numeric(capped[column], errors="coerce").clip(upper=1.0)
    _, capped_sem = prepare_sem_data(capped)
    _, capped_params, capped_fit = _fit_two_factor_spec(capped_sem)
    capped_row = _sensitivity_row(
        "Capped ratio sensitivity",
        capped_fit,
        capped_params,
        note="All observed performance ratios above 1.0 are truncated to 1.0 before SEM preparation.",
    )
    rows.extend(
        [
            {
                "element": "Capped ratio sensitivity: CFI",
                "value": capped_row["CFI"],
                "note": "Two-factor SEM refit after truncating ratio indicators at 1.0.",
            },
            {
                "element": "Capped ratio sensitivity: burden -> timeliness beta",
                "value": capped_row["burden_to_timeliness_beta"],
                "note": "Primary timeliness coefficient after ratio truncation.",
            },
            {
                "element": "Capped ratio sensitivity: resources -> performance beta",
                "value": capped_row["resources_to_performance_beta"],
                "note": "Secondary performance coefficient after ratio truncation.",
            },
        ]
    )

    return pd.DataFrame(rows)


def _build_reference_fit_comparison(
    reference_comparison: pd.DataFrame | None,
    rerun_comparison: pd.DataFrame,
) -> pd.DataFrame | None:
    """Compare rerun fit statistics to the reference exported bundle."""
    if reference_comparison is None:
        return None

    merged = reference_comparison.merge(rerun_comparison, on="Statistic", how="outer")
    for left, right, delta in [
        ("OneFactor", "OneFactor_rerun", "Delta_OneFactor"),
        ("TwoFactor", "TwoFactor_rerun", "Delta_TwoFactor"),
    ]:
        if right not in merged.columns:
            source = "OneFactor" if "OneFactor" in right else "TwoFactor"
            merged = merged.rename(columns={source: right})
        merged[left] = pd.to_numeric(merged[left], errors="coerce")
        merged[right] = pd.to_numeric(merged[right], errors="coerce")
        merged[delta] = merged[right] - merged[left]
    return merged


def identify_candidate_573_subset(
    raw_df: pd.DataFrame,
    reference_ready: pd.DataFrame | None,
) -> tuple[list[str], pd.DataFrame | None, pd.DataFrame | None]:
    """
    Identify the raw-row subset that matches the reference 573-row SEM sample.

    Returns:
        excluded_grantees, candidate_raw_subset, candidate_sem_data
    """
    if reference_ready is None:
        return [], None, None

    reference_grantees = set(reference_ready["Grantee"])
    raw_grantees = set(raw_df["Grantee"])
    excluded = sorted(raw_grantees - reference_grantees)
    if not excluded:
        return [], raw_df.copy(), prepare_sem_data(raw_df)[1]

    candidate_raw = raw_df[~raw_df["Grantee"].isin(excluded)].copy()
    _, candidate_sem = prepare_sem_data(candidate_raw)
    return excluded, candidate_raw, candidate_sem


def build_candidate_alignment_summary(
    reference_ready: pd.DataFrame | None,
    candidate_sem_data: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Summarize how closely the candidate 573-row subset matches the reference SEM-ready data."""
    if reference_ready is None or candidate_sem_data is None:
        return None

    merged = reference_ready.merge(
        candidate_sem_data,
        on="Grantee",
        suffixes=("_reference", "_candidate"),
        how="inner",
    )

    rows: list[dict[str, object]] = []
    for column in candidate_sem_data.columns:
        if column == "Grantee":
            continue
        left = f"{column}_reference"
        right = f"{column}_candidate"
        deltas = (merged[left] - merged[right]).abs()
        rows.append(
            {
                "column": column,
                "max_abs_delta": float(deltas.max()),
                "mean_abs_delta": float(deltas.mean()),
                "exact_match": bool(float(deltas.max()) <= EXACT_MATCH_TOLERANCE),
            }
        )
    return pd.DataFrame(rows).sort_values(["exact_match", "max_abs_delta"], ascending=[False, True])


def build_candidate_row_mismatch_table(
    reference_ready: pd.DataFrame | None,
    candidate_sem_data: pd.DataFrame | None,
    top_n: int = 60,
) -> pd.DataFrame | None:
    """List the largest row-level mismatches between the candidate subset and reference SEM-ready data."""
    if reference_ready is None or candidate_sem_data is None:
        return None

    merged = reference_ready.merge(
        candidate_sem_data,
        on="Grantee",
        suffixes=("_reference", "_candidate"),
        how="inner",
    )

    tracked_columns = [
        "z_Ratio_disbursed_to_obligated",
        "z_Ratio_expended_to_disbursed",
        "z_Ratio_obligated_funds_fully_expended",
        "z_rev_Duration_of_completion",
    ]

    rows: list[dict[str, object]] = []
    for column in tracked_columns:
        left = f"{column}_reference"
        right = f"{column}_candidate"
        deltas = (merged[left] - merged[right]).abs()
        top = deltas.sort_values(ascending=False).head(top_n)
        for idx, delta in top.items():
            rows.append(
                {
                    "column": column,
                    "Grantee": merged.at[idx, "Grantee"],
                    "reference_value": float(merged.at[idx, left]),
                    "candidate_value": float(merged.at[idx, right]),
                    "abs_delta": float(delta),
                }
            )

    return pd.DataFrame(rows).sort_values(["abs_delta", "column"], ascending=[False, True]).reset_index(drop=True)


def _load_notebook_json(path: Path) -> dict:
    """Load a notebook JSON document."""
    return json.loads(path.read_text())


def extract_saved_notebook_metadata() -> dict[str, int | None]:
    """Extract row-count metadata from the saved notebook outputs."""
    notebook = _load_notebook_json(RECOVERED_BUNDLE.sem_notebook)
    outputs_text: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                outputs_text.extend(output["text"])
    joined = "\n".join(outputs_text)

    def _extract(pattern: str) -> int | None:
        match = re.search(pattern, joined)
        return int(match.group(1)) if match else None

    return {
        "saved_dataset_rows": _extract(r"Dataset shape:\s*\((\d+),\s*\d+\)"),
        "saved_rows_before_dropna": _extract(r"Rows before dropping missing:\s*(\d+)"),
        "saved_rows_after_dropna": _extract(r"Rows after dropping missing:\s*(\d+)"),
        "saved_state_count": _extract(r"state\s+(\d+)"),
        "saved_local_count": _extract(r"local\s+(\d+)"),
    }


def analyze_partial_notebook() -> dict[str, object]:
    """Audit the recovered partial-dependence notebook for reproducibility issues."""
    notebook = _load_notebook_json(RECOVERED_BUNDLE.partial_notebook)
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    required_shapefiles = [
        "cb_2018_us_county_500k.zip",
        "cb_2018_us_state_500k.zip",
    ]
    missing_shapefiles = [
        name
        for name in required_shapefiles
        if not list(PROJECT_ROOT.rglob(name))
    ]

    return {
        "uses_avg_employment_only_denominator": (
            'df["Num_Program"] / (df["avg_employment"] + eps)' in source
            and 'df["avg_employment"] * df[\'E_TOTPOP\']' not in source
        ),
        "references_partial_dependence_language": "PARTIAL DEPENDENCE" in source,
        "required_shapefiles": required_shapefiles,
        "missing_shapefiles": missing_shapefiles,
    }


def _write_manifest(
    output_dir: Path,
    raw_df: pd.DataFrame,
    sem_data: pd.DataFrame,
    notebook_metadata: dict[str, int | None],
    partial_audit: dict[str, object],
    reference_fit_comparison: pd.DataFrame | None,
    excluded_grantees: list[str],
    candidate_alignment_summary: pd.DataFrame | None,
    measurement_appendix_table: pd.DataFrame | None,
    maturity_summary: pd.DataFrame | None,
    geography_summary: pd.DataFrame | None,
    sensitivity_summary: pd.DataFrame | None,
    official_geography_summary: pd.DataFrame | None,
    proxy_validation_summary: pd.DataFrame | None,
    subset_forensics_summary: pd.DataFrame | None,
    full_structural_reporting: pd.DataFrame | None,
    fit_verification_summary: pd.DataFrame | None,
    data_flow_summary: pd.DataFrame | None,
    ratio_artifact_summary: pd.DataFrame | None,
) -> None:
    """Write a manifest and README describing recovered-bundle status."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = notebook_metadata.get("saved_dataset_rows")
    manifest = {
        "label": KAIFA_RECOVERED_LABEL,
        "warning": KAIFA_RECOVERED_WARNING,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bundle_root": str(RECOVERED_BUNDLE.root),
        "source_zip": str(RECOVERED_BUNDLE.zip_path),
        "source_sem_notebook": str(RECOVERED_BUNDLE.sem_notebook),
        "source_partial_notebook": str(RECOVERED_BUNDLE.partial_notebook),
        "source_raw_csv": str(RECOVERED_BUNDLE.raw_csv),
        "source_docx": str(RECOVERED_BUNDLE.source_docx),
        "raw_csv_rows": int(len(raw_df)),
        "sem_ready_rows_rerun": int(len(sem_data)),
        "saved_notebook_rows": saved_rows,
        "row_count_delta_vs_saved_notebook": (
            int(len(raw_df)) - saved_rows if saved_rows is not None else None
        ),
        "saved_state_count": notebook_metadata.get("saved_state_count"),
        "saved_local_count": notebook_metadata.get("saved_local_count"),
        "identified_excluded_grantees": excluded_grantees,
        "partial_notebook_audit": partial_audit,
        "reference_fit_available": reference_fit_comparison is not None,
        "candidate_alignment_available": candidate_alignment_summary is not None,
        "measurement_appendix_available": measurement_appendix_table is not None,
        "maturity_summary_available": maturity_summary is not None,
        "geography_summary_available": geography_summary is not None,
        "sensitivity_summary_available": sensitivity_summary is not None,
        "official_geography_summary_available": official_geography_summary is not None,
        "proxy_validation_summary_available": proxy_validation_summary is not None,
        "subset_forensics_summary_available": subset_forensics_summary is not None,
        "full_structural_reporting_available": full_structural_reporting is not None,
        "fit_verification_summary_available": fit_verification_summary is not None,
        "data_flow_summary_available": data_flow_summary is not None,
        "ratio_artifact_summary_available": ratio_artifact_summary is not None,
        "exact_match_columns": (
            candidate_alignment_summary.loc[
                candidate_alignment_summary["exact_match"], "column"
            ].tolist()
            if candidate_alignment_summary is not None
            else []
        ),
        "drift_columns": (
            candidate_alignment_summary.loc[
                ~candidate_alignment_summary["exact_match"], "column"
            ].sort_values().tolist()
            if candidate_alignment_summary is not None
            else []
        ),
    }
    (output_dir / "recovered_notebook_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    lines = [
        "# Kaifa Recovered Notebook Analysis",
        "",
        f"Label: `{KAIFA_RECOVERED_LABEL}`",
        "",
        KAIFA_RECOVERED_WARNING,
        "",
        "This directory was generated from the recovered Kaifa notebook ZIP bundle.",
        "It is the primary recovered-code workflow for the Kaifa manuscript archive.",
        "",
        f"- Raw CSV rows rerun here: `{len(raw_df)}`",
        f"- SEM-ready rows rerun here: `{len(sem_data)}`",
    ]
    if saved_rows is not None:
        lines.append(f"- Saved notebook output rows: `{saved_rows}`")
        lines.append(
            f"- Row-count discrepancy versus saved notebook output: `{len(raw_df) - saved_rows}`"
        )
    if excluded_grantees:
        lines.append(f"- Identified extra grantees in the 577-row CSV: `{', '.join(excluded_grantees)}`")
    if candidate_alignment_summary is not None:
        exact_columns = candidate_alignment_summary.loc[
            candidate_alignment_summary["exact_match"], "column"
        ].tolist()
        drift_columns = candidate_alignment_summary.loc[
            ~candidate_alignment_summary["exact_match"], "column"
        ].tolist()
        lines.append(f"- Columns matching the 573-row reference exactly: `{len(exact_columns)}`")
        lines.append(f"- Columns still drifting versus the 573-row reference: `{', '.join(drift_columns)}`")
    if maturity_summary is not None:
        lines.append(
            f"- Portfolio-maturity summary bands written: `{len(maturity_summary)}`"
        )
    if sensitivity_summary is not None:
        lines.append(
            f"- Light-touch sensitivity specifications written: `{len(sensitivity_summary)}`"
        )
    if official_geography_summary is not None:
        lines.append(
            f"- Official Census geography audit rows written: `{len(official_geography_summary)}`"
        )
    if subset_forensics_summary is not None:
        lines.append(
            f"- N=169 forensic audit summary rows written: `{len(subset_forensics_summary)}`"
        )
    if full_structural_reporting is not None:
        lines.append(
            f"- Full structural/covariance appendix rows written: `{len(full_structural_reporting)}`"
        )
    lines.extend(
        [
            "",
            "Key caveats:",
            "- The recovered SEM notebook output references a 573-row run, but the recovered CSV currently present reruns at 577 rows.",
            "- Dropping the four extra grantees reproduces the sample membership and most capacity/context columns exactly, but several outcome columns still reflect an earlier unrecovered input version.",
            "- The recovered partial-dependence/spatial notebook uses a different workload denominator and depends on shapefiles not bundled here.",
            "- The official 2023 Census crosswalk audit is strong for the recovered 573-sample labels, but it does not reconstruct the older hierarchical geography workflow used before those labels were archived.",
            "- The smaller N=169 sensitivity sample is reproducible as a nested subset, but its original filtering rule remains unrecovered.",
            "- The data-flow table now documents only the recoverable path from repository-standardized QPR rows to the archived 573-jurisdiction SEM file; some original intermediate archival workflow counts remain unrecovered.",
            "- Treat the core SEM notebook as primary and the spatial/gap notebook as secondary until that workflow is fully reconstructed.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def build_cohort_restricted_sensitivity(
    raw_df: pd.DataFrame | None,
    reference_ready: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Run the two-factor SEM on duration- and experience-restricted subsets.

    Addresses Major 3 (maturity/cohort confounding) by testing whether the
    AdminBurdenCapacity -> RecoveryTimeliness beta survives in subsets that
    exclude young portfolios or restrict to experienced jurisdictions.
    """
    if reference_ready is None or raw_df is None:
        return None

    # Build a lookup of raw Duration and Num_Disaster by grantee
    raw_lookup = raw_df[["Grantee", "Duration_of_completion", "Num_Disaster"]].copy()
    raw_lookup["Duration_of_completion"] = pd.to_numeric(
        raw_lookup["Duration_of_completion"], errors="coerce"
    )
    raw_lookup["Num_Disaster"] = pd.to_numeric(
        raw_lookup["Num_Disaster"], errors="coerce"
    )

    merged = reference_ready.merge(raw_lookup, on="Grantee", how="left")

    subsets = [
        ("Full sample (baseline)", merged),
        (
            "Duration > 24 months",
            merged[merged["Duration_of_completion"] > 24.0],
        ),
        (
            "Duration > 48 months",
            merged[merged["Duration_of_completion"] > 48.0],
        ),
        (
            "Num_Disaster >= 2",
            merged[merged["Num_Disaster"] >= 2],
        ),
    ]

    rows: list[dict[str, object]] = []
    for label, subset_df in subsets:
        sem_cols = [c for c in reference_ready.columns if c != "Grantee"]
        sem_subset = subset_df[["Grantee"] + sem_cols].copy()
        n = len(sem_subset)
        if n < 30:
            rows.append({
                "subset": label,
                "n": n,
                "burden_to_timeliness_beta": None,
                "burden_to_performance_beta": None,
                "CFI": None,
                "RMSEA": None,
                "note": f"Insufficient sample size (n={n})",
            })
            continue
        try:
            _, params, fit = _fit_sem_model(
                TWO_FACTOR_MODEL_SPEC,
                sem_subset.drop(columns=["Grantee", "Duration_of_completion", "Num_Disaster"], errors="ignore"),
            )
            fit_lookup = fit.set_index("Statistic")["Value"].to_dict()
            for col in ["Estimate", "Est. Std", "p-value"]:
                if col in params.columns:
                    params[col] = pd.to_numeric(params[col], errors="coerce")

            def _get_beta(outcome: str, predictor: str) -> float | None:
                row = params[
                    (params["op"] == "~")
                    & (params["lval"] == outcome)
                    & (params["rval"] == predictor)
                ]
                if row.empty:
                    return None
                return float(row.iloc[0]["Est. Std"])

            def _get_pvalue(outcome: str, predictor: str) -> float | None:
                row = params[
                    (params["op"] == "~")
                    & (params["lval"] == outcome)
                    & (params["rval"] == predictor)
                ]
                if row.empty:
                    return None
                val = row.iloc[0].get("p-value")
                if pd.isna(val):
                    return None
                return float(val)

            rows.append({
                "subset": label,
                "n": n,
                "burden_to_timeliness_beta": _get_beta("RecoveryTimeliness", "AdminBurdenCapacity"),
                "burden_to_timeliness_p": _get_pvalue("RecoveryTimeliness", "AdminBurdenCapacity"),
                "burden_to_performance_beta": _get_beta("RecoveryPerformance", "AdminBurdenCapacity"),
                "resources_to_performance_beta": _get_beta("RecoveryPerformance", "AdminResources"),
                "CFI": fit_lookup.get("CFI"),
                "RMSEA": fit_lookup.get("RMSEA"),
                "note": "",
            })
        except Exception as exc:
            rows.append({
                "subset": label,
                "n": n,
                "burden_to_timeliness_beta": None,
                "burden_to_performance_beta": None,
                "CFI": None,
                "RMSEA": None,
                "note": f"Estimation failed: {exc}",
            })

    return pd.DataFrame(rows)


def build_stratified_sensitivity(
    reference_ready: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Run the two-factor SEM separately for state and local jurisdictions.

    Addresses Major 5 (pooling 30 state with 543 local without weighting).
    State model will likely be unstable; local model has adequate sample.
    """
    if reference_ready is None:
        return None

    rows: list[dict[str, object]] = []
    strata = [
        ("All jurisdictions (pooled)", reference_ready),
        ("State agencies only", reference_ready[reference_ready["state_level"] == 1]),
        ("Local jurisdictions only", reference_ready[reference_ready["state_level"] == 0]),
    ]

    for label, subset_df in strata:
        n = len(subset_df)
        if n < 20:
            rows.append({
                "stratum": label,
                "n": n,
                "burden_to_timeliness_beta": None,
                "burden_to_performance_beta": None,
                "resources_to_performance_beta": None,
                "CFI": None,
                "RMSEA": None,
                "note": f"Insufficient sample size (n={n})",
            })
            continue
        try:
            # For state-only, drop state_level from specification since it's constant
            if label.startswith("State") or label.startswith("Local"):
                covariates = [c for c in BASE_STRUCTURAL_COVARIATES if c != "state_level"]
                predictor_block = " + ".join(
                    ["AdminResources", "AdminBurdenCapacity"] + covariates
                )
                spec = "\n".join([
                    "AdminResources =~ z_avg_employment + z_avg_payroll",
                    "AdminBurdenCapacity =~ z_rev_programs_per_staff + z_rev_disasters_per_staff",
                    "RecoveryPerformance =~ z_Ratio_disbursed_to_obligated + z_Ratio_expended_to_disbursed + z_Ratio_obligated_funds_fully_expended + z_Ratio_Program_Completed",
                    "RecoveryTimeliness =~ z_rev_Duration_of_completion + z_rev_Average_Duration_Program_Completion",
                    f"RecoveryPerformance ~ {predictor_block}",
                    f"RecoveryTimeliness ~ {predictor_block}",
                    "AdminResources ~~ AdminBurdenCapacity",
                    "RecoveryPerformance ~~ RecoveryTimeliness",
                ])
            else:
                spec = TWO_FACTOR_MODEL_SPEC

            _, params, fit = _fit_sem_model(
                spec, subset_df.drop(columns=["Grantee"])
            )
            fit_lookup = fit.set_index("Statistic")["Value"].to_dict()
            for col in ["Estimate", "Est. Std", "p-value"]:
                if col in params.columns:
                    params[col] = pd.to_numeric(params[col], errors="coerce")

            def _get_beta(outcome: str, predictor: str) -> float | None:
                row = params[
                    (params["op"] == "~")
                    & (params["lval"] == outcome)
                    & (params["rval"] == predictor)
                ]
                if row.empty:
                    return None
                return float(row.iloc[0]["Est. Std"])

            def _get_pvalue(outcome: str, predictor: str) -> float | None:
                row = params[
                    (params["op"] == "~")
                    & (params["lval"] == outcome)
                    & (params["rval"] == predictor)
                ]
                if row.empty:
                    return None
                val = row.iloc[0].get("p-value")
                if pd.isna(val):
                    return None
                return float(val)

            rows.append({
                "stratum": label,
                "n": n,
                "burden_to_timeliness_beta": _get_beta("RecoveryTimeliness", "AdminBurdenCapacity"),
                "burden_to_timeliness_p": _get_pvalue("RecoveryTimeliness", "AdminBurdenCapacity"),
                "burden_to_performance_beta": _get_beta("RecoveryPerformance", "AdminBurdenCapacity"),
                "resources_to_performance_beta": _get_beta("RecoveryPerformance", "AdminResources"),
                "resources_to_timeliness_beta": _get_beta("RecoveryTimeliness", "AdminResources"),
                "CFI": fit_lookup.get("CFI"),
                "RMSEA": fit_lookup.get("RMSEA"),
                "note": "",
            })
        except Exception as exc:
            rows.append({
                "stratum": label,
                "n": n,
                "burden_to_timeliness_beta": None,
                "burden_to_performance_beta": None,
                "resources_to_performance_beta": None,
                "CFI": None,
                "RMSEA": None,
                "note": f"Estimation failed: {exc}",
            })

    return pd.DataFrame(rows)


def build_raw_workload_sensitivity(
    raw_df: pd.DataFrame | None,
    reference_ready: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Test whether the staffing-denominator coupling drives the main result.

    Addresses Major 4 by replacing programs_per_staff and disasters_per_staff
    with raw log(Num_Program) and log(Num_Disaster) as burden indicators,
    removing the shared staffing denominator entirely.
    """
    if raw_df is None or reference_ready is None:
        return None

    auxiliary = _build_auxiliary_sem_controls(raw_df)
    augmented = reference_ready.merge(auxiliary, on="Grantee", how="left").dropna()

    rows: list[dict[str, object]] = []

    # Baseline for comparison
    _, base_params, base_fit = _fit_sem_model(
        TWO_FACTOR_MODEL_SPEC, reference_ready.drop(columns=["Grantee"])
    )
    base_fit_lookup = base_fit.set_index("Statistic")["Value"].to_dict()
    base_params["Est. Std"] = pd.to_numeric(base_params["Est. Std"], errors="coerce")
    base_params["p-value"] = pd.to_numeric(base_params["p-value"], errors="coerce")

    def _extract(params: pd.DataFrame, outcome: str, predictor: str) -> tuple[float | None, float | None]:
        row = params[
            (params["op"] == "~")
            & (params["lval"] == outcome)
            & (params["rval"] == predictor)
        ]
        if row.empty:
            return None, None
        beta = float(row.iloc[0]["Est. Std"]) if not pd.isna(row.iloc[0]["Est. Std"]) else None
        pval = float(row.iloc[0]["p-value"]) if not pd.isna(row.iloc[0]["p-value"]) else None
        return beta, pval

    b_time_beta, b_time_p = _extract(base_params, "RecoveryTimeliness", "AdminBurdenCapacity")
    rows.append({
        "specification": "Primary (staffing-denominator ratios)",
        "n": len(reference_ready),
        "burden_indicators": "programs_per_staff, disasters_per_staff",
        "burden_to_timeliness_beta": b_time_beta,
        "burden_to_timeliness_p": b_time_p,
        "CFI": base_fit_lookup.get("CFI"),
        "RMSEA": base_fit_lookup.get("RMSEA"),
    })

    # Raw workload indicators
    try:
        raw_spec = _build_two_factor_model_spec(
            burden_indicators=["z_log_num_program", "z_log_num_disaster"],
        )
        _, raw_params, raw_fit = _fit_sem_model(
            raw_spec, augmented.drop(columns=["Grantee"])
        )
        raw_fit_lookup = raw_fit.set_index("Statistic")["Value"].to_dict()
        raw_params["Est. Std"] = pd.to_numeric(raw_params["Est. Std"], errors="coerce")
        raw_params["p-value"] = pd.to_numeric(raw_params["p-value"], errors="coerce")
        r_time_beta, r_time_p = _extract(raw_params, "RecoveryTimeliness", "AdminBurdenCapacity")
        rows.append({
            "specification": "Raw portfolio counts (no staffing denominator)",
            "n": len(augmented),
            "burden_indicators": "log(Num_Program), log(Num_Disaster)",
            "burden_to_timeliness_beta": r_time_beta,
            "burden_to_timeliness_p": r_time_p,
            "CFI": raw_fit_lookup.get("CFI"),
            "RMSEA": raw_fit_lookup.get("RMSEA"),
        })
    except Exception as exc:
        rows.append({
            "specification": "Raw portfolio counts (no staffing denominator)",
            "n": len(augmented),
            "burden_indicators": "log(Num_Program), log(Num_Disaster)",
            "burden_to_timeliness_beta": None,
            "burden_to_timeliness_p": None,
            "CFI": None,
            "RMSEA": None,
            "note": f"Estimation failed: {exc}",
        })

    return pd.DataFrame(rows)


def build_distribution_summary(
    raw_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Report pre-standardization distributions for all proxy variables.

    Addresses Major 4 by disclosing zeros, suppression rates, skew,
    and state-vs-local breakdowns for the QCEW staffing proxies.
    """
    if raw_df is None or raw_df.empty:
        return None

    proxy_vars = [
        "avg_employment",
        "avg_payroll",
        "Num_Program",
        "Num_Disaster",
        "Duration_of_completion",
        "E_TOTPOP",
    ]

    df = raw_df.copy()
    df["gov_type"] = df["Grantee"].astype(str).str.fullmatch(r"[A-Z]{2}").map(
        {True: "state", False: "local"}
    )

    rows: list[dict[str, object]] = []
    for var in proxy_vars:
        if var not in df.columns:
            continue
        series = pd.to_numeric(df[var], errors="coerce")
        for group_label, group_mask in [
            ("All", pd.Series(True, index=df.index)),
            ("State", df["gov_type"] == "state"),
            ("Local", df["gov_type"] == "local"),
        ]:
            s = series[group_mask].dropna()
            if s.empty:
                continue
            rows.append({
                "variable": var,
                "group": group_label,
                "n": len(s),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "pct_zero": float((s == 0).mean()) * 100,
                "pct_at_min": float((s == s.min()).mean()) * 100 if len(s.unique()) > 1 else 100.0,
                "skewness": float(s.skew()) if len(s) > 2 else None,
            })

    return pd.DataFrame(rows)


def run_recovered_analysis(
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> dict[str, object]:
    """
    Run the recovered Kaifa notebook logic on the imported bundle.

    This ports the recovered notebook source into the repository and reruns it
    on the CSV currently bundled in the recovered ZIP. It also compares the
    rerun to the saved 573-row reference outputs already imported elsewhere in
    the archive.
    """
    raw_df = load_recovered_raw_data()
    processed_df, sem_data = prepare_sem_data(raw_df)
    reference_ready = _load_reference_ready_dataset()
    excluded_grantees, candidate_raw_subset, candidate_sem_data = identify_candidate_573_subset(
        raw_df, reference_ready
    )

    _, one_factor_parameters, one_factor_fit = _fit_sem_model(
        ONE_FACTOR_MODEL_SPEC, sem_data.drop(columns=["Grantee"])
    )
    _, two_factor_parameters, two_factor_fit = _fit_sem_model(
        TWO_FACTOR_MODEL_SPEC, sem_data.drop(columns=["Grantee"])
    )
    candidate_one_factor_fit = None
    candidate_two_factor_fit = None
    candidate_model_comparison = None
    candidate_reference_fit_comparison = None
    if candidate_sem_data is not None:
        _, _, candidate_one_factor_fit = _fit_sem_model(
            ONE_FACTOR_MODEL_SPEC, candidate_sem_data.drop(columns=["Grantee"])
        )
        _, _, candidate_two_factor_fit = _fit_sem_model(
            TWO_FACTOR_MODEL_SPEC, candidate_sem_data.drop(columns=["Grantee"])
        )
        candidate_model_comparison = _build_model_comparison(
            candidate_one_factor_fit, candidate_two_factor_fit
        ).rename(
            columns={
                "OneFactor": "OneFactor_candidate_573",
                "TwoFactor": "TwoFactor_candidate_573",
            }
        )

    rerun_comparison = _build_model_comparison(one_factor_fit, two_factor_fit)
    rerun_comparison = rerun_comparison.rename(
        columns={"OneFactor": "OneFactor_rerun", "TwoFactor": "TwoFactor_rerun"}
    )
    reference_comparison = _load_reference_comparison()
    reference_fit_comparison = _build_reference_fit_comparison(
        reference_comparison, rerun_comparison
    )
    if reference_comparison is not None and candidate_model_comparison is not None:
        candidate_reference_fit_comparison = reference_comparison.merge(
            candidate_model_comparison, on="Statistic", how="outer"
        )
        for left, right, delta in [
            ("OneFactor", "OneFactor_candidate_573", "Delta_OneFactor_candidate_573"),
            ("TwoFactor", "TwoFactor_candidate_573", "Delta_TwoFactor_candidate_573"),
        ]:
            candidate_reference_fit_comparison[left] = pd.to_numeric(
                candidate_reference_fit_comparison[left], errors="coerce"
            )
            candidate_reference_fit_comparison[right] = pd.to_numeric(
                candidate_reference_fit_comparison[right], errors="coerce"
            )
            candidate_reference_fit_comparison[delta] = (
                candidate_reference_fit_comparison[right]
                - candidate_reference_fit_comparison[left]
            )
    notebook_metadata = extract_saved_notebook_metadata()
    partial_audit = analyze_partial_notebook()
    candidate_alignment_summary = build_candidate_alignment_summary(
        reference_ready, candidate_sem_data
    )
    candidate_row_mismatch_table = build_candidate_row_mismatch_table(
        reference_ready, candidate_sem_data
    )
    reference_parameters = _load_reference_parameters()
    admin4_ready = _load_reference_admin4_ready_dataset()
    measurement_appendix_table = build_measurement_appendix_table(reference_parameters)
    full_structural_reporting = build_full_structural_reporting_table(reference_parameters)
    fit_verification_summary = build_fit_verification_table(
        reference_ready, reference_comparison
    )
    maturity_summary = build_maturity_summary(candidate_raw_subset)
    geography_summary = build_geography_summary(candidate_raw_subset)
    official_geography_crosswalk, official_geography_summary = build_official_geography_crosswalk(
        candidate_raw_subset
    )
    sensitivity_summary = build_sensitivity_summary(reference_ready, candidate_raw_subset)
    data_flow_summary = build_data_flow_summary(
        raw_df, candidate_raw_subset, reference_ready
    )
    ratio_artifact_summary = build_ratio_artifact_summary(candidate_raw_subset)
    proxy_validation_summary = build_proxy_validation_summary(
        candidate_raw_subset, sensitivity_summary
    )
    cohort_restricted_sensitivity = build_cohort_restricted_sensitivity(
        candidate_raw_subset, reference_ready
    )
    stratified_sensitivity = build_stratified_sensitivity(reference_ready)
    raw_workload_sensitivity = build_raw_workload_sensitivity(
        candidate_raw_subset, reference_ready
    )
    distribution_summary = build_distribution_summary(candidate_raw_subset)
    subset_forensics_summary, subset_retention_by_state, subset_tree_rules = (
        build_subset_forensics_summary(reference_ready, admin4_ready)
    )

    output_path = Path(output_dir) if output_dir else default_output_dir()

    one_factor_paths = _subset_parameters(
        one_factor_parameters, "~", ["RecoveryPerformance", "RecoveryTimeliness"]
    )
    two_factor_paths = _subset_parameters(
        two_factor_parameters, "~", ["RecoveryPerformance", "RecoveryTimeliness"]
    )
    one_factor_measurement = one_factor_parameters.loc[
        one_factor_parameters["rval"] == "GovCapacity"
    ].copy()
    two_factor_measurement = two_factor_parameters.loc[
        two_factor_parameters["rval"].isin(["AdminResources", "AdminBurdenCapacity"])
    ].copy()

    if write_outputs:
        output_path.mkdir(parents=True, exist_ok=True)
        sem_data.to_csv(output_path / "recovered_notebook_sem_ready_dataset.csv", index=False)
        processed_df.to_csv(output_path / "recovered_notebook_processed_raw_data.csv", index=False)
        one_factor_parameters.to_csv(
            output_path / "recovered_notebook_one_factor_parameter_estimates.csv",
            index=False,
        )
        two_factor_parameters.to_csv(
            output_path / "recovered_notebook_two_factor_parameter_estimates.csv",
            index=False,
        )
        one_factor_fit.to_csv(
            output_path / "recovered_notebook_one_factor_fit_statistics.csv",
            index=False,
        )
        two_factor_fit.to_csv(
            output_path / "recovered_notebook_two_factor_fit_statistics.csv",
            index=False,
        )
        rerun_comparison.to_csv(
            output_path / "recovered_notebook_model_comparison.csv",
            index=False,
        )
        one_factor_paths.to_csv(
            output_path / "recovered_notebook_one_factor_structural_paths.csv",
            index=False,
        )
        two_factor_paths.to_csv(
            output_path / "recovered_notebook_two_factor_structural_paths.csv",
            index=False,
        )
        one_factor_measurement.to_csv(
            output_path / "recovered_notebook_one_factor_measurement_rows.csv",
            index=False,
        )
        two_factor_measurement.to_csv(
            output_path / "recovered_notebook_two_factor_measurement_rows.csv",
            index=False,
        )
        if candidate_raw_subset is not None:
            candidate_raw_subset.to_csv(
                output_path / "recovered_notebook_candidate_573_raw_subset.csv",
                index=False,
            )
        if candidate_sem_data is not None:
            candidate_sem_data.to_csv(
                output_path / "recovered_notebook_candidate_573_sem_ready_dataset.csv",
                index=False,
            )
        if reference_ready is not None:
            reference_ready.to_csv(
                output_path / "recovered_notebook_reference_573_sem_ready_dataset.csv",
                index=False,
            )
        if excluded_grantees:
            pd.DataFrame({"excluded_grantee": excluded_grantees}).to_csv(
                output_path / "recovered_notebook_candidate_573_excluded_grantees.csv",
                index=False,
            )
        if candidate_alignment_summary is not None:
            candidate_alignment_summary.to_csv(
                output_path / "recovered_notebook_candidate_573_alignment_summary.csv",
                index=False,
            )
        if candidate_row_mismatch_table is not None:
            candidate_row_mismatch_table.to_csv(
                output_path / "recovered_notebook_candidate_573_row_mismatches.csv",
                index=False,
            )
        if candidate_one_factor_fit is not None:
            candidate_one_factor_fit.to_csv(
                output_path / "recovered_notebook_candidate_573_one_factor_fit_statistics.csv",
                index=False,
            )
        if candidate_two_factor_fit is not None:
            candidate_two_factor_fit.to_csv(
                output_path / "recovered_notebook_candidate_573_two_factor_fit_statistics.csv",
                index=False,
            )
        if candidate_model_comparison is not None:
            candidate_model_comparison.to_csv(
                output_path / "recovered_notebook_candidate_573_model_comparison.csv",
                index=False,
            )
        if reference_fit_comparison is not None:
            reference_fit_comparison.to_csv(
                output_path / "recovered_notebook_reference_fit_comparison.csv",
                index=False,
            )
        if candidate_reference_fit_comparison is not None:
            candidate_reference_fit_comparison.to_csv(
                output_path / "recovered_notebook_candidate_573_reference_fit_comparison.csv",
                index=False,
            )
        if measurement_appendix_table is not None:
            measurement_appendix_table.to_csv(
                output_path / "recovered_notebook_measurement_appendix_table.csv",
                index=False,
            )
        if full_structural_reporting is not None:
            full_structural_reporting.to_csv(
                output_path / "recovered_notebook_full_structural_reporting.csv",
                index=False,
            )
        if fit_verification_summary is not None:
            fit_verification_summary.to_csv(
                output_path / "recovered_notebook_fit_verification_summary.csv",
                index=False,
            )
        if maturity_summary is not None:
            maturity_summary.to_csv(
                output_path / "recovered_notebook_maturity_summary.csv",
                index=False,
            )
        if geography_summary is not None:
            geography_summary.to_csv(
                output_path / "recovered_notebook_geography_summary.csv",
                index=False,
            )
        if official_geography_crosswalk is not None:
            official_geography_crosswalk.to_csv(
                output_path / "recovered_notebook_official_geography_crosswalk.csv",
                index=False,
            )
        if official_geography_summary is not None:
            official_geography_summary.to_csv(
                output_path / "recovered_notebook_official_geography_summary.csv",
                index=False,
            )
        if sensitivity_summary is not None:
            sensitivity_summary.to_csv(
                output_path / "recovered_notebook_sensitivity_summary.csv",
                index=False,
            )
        if proxy_validation_summary is not None:
            proxy_validation_summary.to_csv(
                output_path / "recovered_notebook_proxy_validation_summary.csv",
                index=False,
            )
        if subset_forensics_summary is not None:
            subset_forensics_summary.to_csv(
                output_path / "recovered_notebook_subset_169_forensics_summary.csv",
                index=False,
            )
        if data_flow_summary is not None:
            data_flow_summary.to_csv(
                output_path / "recovered_notebook_data_flow_summary.csv",
                index=False,
            )
        if ratio_artifact_summary is not None:
            ratio_artifact_summary.to_csv(
                output_path / "recovered_notebook_ratio_artifact_summary.csv",
                index=False,
            )
        if cohort_restricted_sensitivity is not None:
            cohort_restricted_sensitivity.to_csv(
                output_path / "recovered_notebook_cohort_restricted_sensitivity.csv",
                index=False,
            )
        if stratified_sensitivity is not None:
            stratified_sensitivity.to_csv(
                output_path / "recovered_notebook_stratified_sensitivity.csv",
                index=False,
            )
        if raw_workload_sensitivity is not None:
            raw_workload_sensitivity.to_csv(
                output_path / "recovered_notebook_raw_workload_sensitivity.csv",
                index=False,
            )
        if distribution_summary is not None:
            distribution_summary.to_csv(
                output_path / "recovered_notebook_distribution_summary.csv",
                index=False,
            )
        if subset_retention_by_state is not None:
            subset_retention_by_state.to_csv(
                output_path / "recovered_notebook_subset_169_retention_by_state.csv",
                index=False,
            )
        if subset_tree_rules is not None:
            (output_path / "recovered_notebook_subset_169_tree_rules.txt").write_text(
                subset_tree_rules + "\n"
            )
        _write_manifest(
            output_path,
            raw_df=raw_df,
            sem_data=sem_data,
            notebook_metadata=notebook_metadata,
            partial_audit=partial_audit,
            reference_fit_comparison=reference_fit_comparison,
            excluded_grantees=excluded_grantees,
            candidate_alignment_summary=candidate_alignment_summary,
            measurement_appendix_table=measurement_appendix_table,
            maturity_summary=maturity_summary,
            geography_summary=geography_summary,
            sensitivity_summary=sensitivity_summary,
            official_geography_summary=official_geography_summary,
            proxy_validation_summary=proxy_validation_summary,
            subset_forensics_summary=subset_forensics_summary,
            full_structural_reporting=full_structural_reporting,
            fit_verification_summary=fit_verification_summary,
            data_flow_summary=data_flow_summary,
            ratio_artifact_summary=ratio_artifact_summary,
        )

    return {
        "label": KAIFA_RECOVERED_LABEL,
        "warning": KAIFA_RECOVERED_WARNING,
        "raw_df": raw_df,
        "processed_df": processed_df,
        "sem_data": sem_data,
        "one_factor_parameters": one_factor_parameters,
        "two_factor_parameters": two_factor_parameters,
        "one_factor_fit": one_factor_fit,
        "two_factor_fit": two_factor_fit,
        "rerun_comparison": rerun_comparison,
        "reference_fit_comparison": reference_fit_comparison,
        "candidate_one_factor_fit": candidate_one_factor_fit,
        "candidate_two_factor_fit": candidate_two_factor_fit,
        "candidate_model_comparison": candidate_model_comparison,
        "candidate_reference_fit_comparison": candidate_reference_fit_comparison,
        "notebook_metadata": notebook_metadata,
        "partial_audit": partial_audit,
        "reference_ready": reference_ready,
        "excluded_grantees": excluded_grantees,
        "candidate_raw_subset": candidate_raw_subset,
        "candidate_sem_data": candidate_sem_data,
        "candidate_alignment_summary": candidate_alignment_summary,
        "candidate_row_mismatch_table": candidate_row_mismatch_table,
        "measurement_appendix_table": measurement_appendix_table,
        "full_structural_reporting": full_structural_reporting,
        "fit_verification_summary": fit_verification_summary,
        "maturity_summary": maturity_summary,
        "geography_summary": geography_summary,
        "sensitivity_summary": sensitivity_summary,
        "data_flow_summary": data_flow_summary,
        "ratio_artifact_summary": ratio_artifact_summary,
        "official_geography_crosswalk": official_geography_crosswalk,
        "official_geography_summary": official_geography_summary,
        "proxy_validation_summary": proxy_validation_summary,
        "cohort_restricted_sensitivity": cohort_restricted_sensitivity,
        "stratified_sensitivity": stratified_sensitivity,
        "raw_workload_sensitivity": raw_workload_sensitivity,
        "distribution_summary": distribution_summary,
        "subset_forensics_summary": subset_forensics_summary,
        "subset_retention_by_state": subset_retention_by_state,
        "subset_tree_rules": subset_tree_rules,
        "output_dir": output_path,
    }
