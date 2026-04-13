#!/usr/bin/env python3
"""
Local CENTAUR configuration for the Capacity-SEM repository.

The imported framework is namespaced under `src/centaur`, but it uses a
project-like layout inside this repo so the full GUI/stage/manuscript stack can
run without colliding with the active Capacity-SEM analysis pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _find_project_root() -> Path:
    """Find the host project root for this vendored framework."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").exists() and (parent / "doc").exists():
            return parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _find_project_root()

# =============================================================================
# PATHS
# =============================================================================

DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_WORK_DIR = PROJECT_ROOT / "data_work" / "centaur"
DIAGNOSTICS_DIR = DATA_WORK_DIR / "diagnostics"

FIGURES_DIR = PROJECT_ROOT / "figures" / "centaur"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript_quarto"
MANUSCRIPT_FIGURES_DIR = MANUSCRIPT_DIR / "figures"

DOC_DIR = PROJECT_ROOT / "doc" / "centaur"
REVIEWS_DIR = DOC_DIR / "reviews"
RAW_GUIDELINES_DIR = DOC_DIR / "journal_guidelines"

QA_REPORTS_DIR = DATA_WORK_DIR / "quality"
CACHE_DIR = DATA_WORK_DIR / ".cache"
SPATIAL_DATA_DIR = DATA_WORK_DIR / "spatial"
SPATIAL_CACHE_DIR = CACHE_DIR / "spatial"

PANEL_FILE = "panel.parquet"
ESTIMATES_FILE = "estimates.parquet"
ROBUSTNESS_FILE = "robustness.parquet"
PANEL_PATH = DATA_WORK_DIR / PANEL_FILE
ESTIMATES_PATH = DATA_WORK_DIR / ESTIMATES_FILE
ROBUSTNESS_PATH = DATA_WORK_DIR / ROBUSTNESS_FILE


# =============================================================================
# QUALITY ASSURANCE
# =============================================================================

ENABLE_QA_REPORTS = True
QA_THRESHOLDS = {
    "max_missing_pct": 5.0,
    "min_row_count": 10,
    "max_duplicate_pct": 1.0,
}


# =============================================================================
# CACHING / PARALLEL
# =============================================================================

CACHE_ENABLED = True
CACHE_MAX_AGE_HOURS = 168
CACHE_COMPRESSION = False

PARALLEL_ENABLED = True
PARALLEL_MAX_WORKERS = None
PARALLEL_STAGES = ["s03_estimation", "s04_robustness", "s05_figures"]


# =============================================================================
# SPATIAL
# =============================================================================

SPATIAL_ENABLED = True
SPATIAL_DEFAULT_CRS = "EPSG:4326"
SPATIAL_CV_N_GROUPS = 5
SPATIAL_GROUPING_METHOD = "kmeans"
SPATIAL_SENSITIVITY_METHODS = [
    "kmeans",
    "balanced_kmeans",
    "geographic_bands",
    "longitude_bands",
    "spatial_blocks",
]
GEOCODING_PROVIDER = "census"
GEOCODING_CACHE_TTL_DAYS = 30


# =============================================================================
# TUNING / CV
# =============================================================================

RANDOM_STATE = 42
TUNING_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
TUNING_ENET_ALPHAS = [0.01, 0.1, 1.0, 10.0]
TUNING_ENET_L1_RATIOS = [0.1, 0.5, 0.9]
TUNING_RF_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}
TUNING_ET_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}
TUNING_GB_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}
TUNING_INNER_FOLDS = 3
REPEATED_CV_N_SPLITS = 5
REPEATED_CV_N_REPEATS = 10


# =============================================================================
# MANUSCRIPT / REVIEWS
# =============================================================================

MANUSCRIPTS = {
    "velocity": {
        "name": "Velocity Manuscript (Archived)",
        "title": "When Velocity Doesn't Matter: Reassessing Spending Rate and CDBG-DR Program Completion",
        "dir": PROJECT_ROOT / "manuscript_velocity",
        "reviews_dir": PROJECT_ROOT / "doc" / "reviews" / "velocity",
        "archive_dir": PROJECT_ROOT / "doc" / "reviews" / "velocity" / "archive",
        "description": "Archived null-results draft — superseded by manuscript_quarto/ rewrite",
    },
    "kaifa": {
        "name": "Kaifa SEM Manuscript Archive",
        "title": "State and Local Governmental Capacity in Disaster Recovery: Evidence from CDBG-DR",
        "dir": PROJECT_ROOT / "manuscript_kaifa_archive",
        "reviews_dir": PROJECT_ROOT / "doc" / "reviews" / "kaifa",
        "archive_dir": PROJECT_ROOT / "doc" / "reviews" / "kaifa" / "archive",
        "description": "Archived SEM manuscript lineage and source-document review workspace",
        "source_docx": PROJECT_ROOT / "manuscript_kaifa_archive" / "source_docs" / "SEM_Manuscript_2026-04-09_full_revision.docx",
    },
    "main": {
        "name": "Main Manuscript",
        "title": "Administrative Throughput in Disaster Recovery: Evidence from Cross-Sectional SEM of CDBG-DR Fund Management",
        "dir": MANUSCRIPT_DIR,
        "reviews_dir": REVIEWS_DIR / "main",
        "archive_dir": REVIEWS_DIR / "main" / "archive",
        "description": "Primary PAR-targeted SEM manuscript with survival comparison",
    },
    "quarto": {
        "name": "Quarto Manuscript",
        "title": "Administrative Throughput in Disaster Recovery: Evidence from Cross-Sectional SEM of CDBG-DR Fund Management",
        "dir": MANUSCRIPT_DIR,
        "reviews_dir": REVIEWS_DIR / "main",
        "archive_dir": REVIEWS_DIR / "main" / "archive",
        "description": "Alias for 'main' — primary PAR-targeted SEM manuscript with survival comparison",
    },
}
DEFAULT_MANUSCRIPT = "quarto"
DRAFTS_DIR = MANUSCRIPT_DIR / "drafts"

REVIEW_DEFAULT_SOURCE_TYPE = "synthetic"
REVIEW_GIT_TAGGING_ENABLED = True
REVIEW_GIT_TAG_FORMAT = "review-{manuscript}-{cycle:02d}-{status}"
REVIEW_RESPONSE_DEFAULT_FORMAT = "markdown"
REVIEW_DIFF_DEFAULT_FORMAT = "markdown"
REVIEW_DIFF_CONTEXT_LINES = 3
REVIEW_USE_DIRECTORY_ARCHIVE = False


# =============================================================================
# ENGINES / LLM
# =============================================================================

ANALYSIS_ENGINE = os.getenv("CENTAUR_ANALYSIS_ENGINE", "python")
R_EXECUTABLE = os.getenv("R_EXECUTABLE", "Rscript")
EXTERNAL_PROCESS_TIMEOUT = int(os.getenv("EXTERNAL_PROCESS_TIMEOUT", "3600"))
SPECIFICATIONS_FILE = DOC_DIR / "specifications.yml"

LLM_PROVIDER = os.getenv("CENTAUR_LLM_PROVIDER", "anthropic")
LLM_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4-turbo-preview",
}
LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, "claude-sonnet-4-20250514")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096


# =============================================================================
# METHODS / VALIDATION
# =============================================================================

DEFAULT_SPECIFICATION = "baseline"
SPECIFICATIONS = {
    "baseline": "Main specification with all controls",
    "no_fe": "No fixed effects",
    "with_controls": "With additional control variables",
    "unit_fe_only": "Only unit fixed effects",
}

SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
MIN_OBSERVATIONS = 30


def ensure_directories() -> None:
    """Create framework directories used by the vendored import."""
    for path in (
        DATA_RAW_DIR,
        DATA_WORK_DIR,
        DIAGNOSTICS_DIR,
        FIGURES_DIR,
        DOC_DIR,
        REVIEWS_DIR,
        RAW_GUIDELINES_DIR,
        QA_REPORTS_DIR,
        CACHE_DIR,
        SPATIAL_DATA_DIR,
        SPATIAL_CACHE_DIR,
        MANUSCRIPT_DIR,
        MANUSCRIPT_FIGURES_DIR,
        DRAFTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for path in (
        MANUSCRIPT_DIR / "journal_configs",
        MANUSCRIPT_DIR / "variants",
        MANUSCRIPT_DIR / "code",
        MANUSCRIPT_DIR / "data",
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_config() -> bool:
    """Validate the vendored configuration."""
    errors = []

    if not PROJECT_ROOT.exists():
        errors.append(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")
    if SIGNIFICANCE_LEVEL <= 0 or SIGNIFICANCE_LEVEL >= 1:
        errors.append(f"SIGNIFICANCE_LEVEL must be between 0 and 1: {SIGNIFICANCE_LEVEL}")
    if BOOTSTRAP_ITERATIONS < 1:
        errors.append(f"BOOTSTRAP_ITERATIONS must be positive: {BOOTSTRAP_ITERATIONS}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


def get_manuscript_config(manuscript: Optional[str] = None) -> dict:
    """Get configuration for a manuscript."""
    manuscript = manuscript or DEFAULT_MANUSCRIPT
    if manuscript not in MANUSCRIPTS:
        available = ", ".join(MANUSCRIPTS.keys())
        raise ValueError(f"Unknown manuscript '{manuscript}'. Available: {available}")
    return MANUSCRIPTS[manuscript]


def get_project_root() -> Path:
    """Compatibility wrapper used by helper modules."""
    return PROJECT_ROOT


def get_data_dir(subdir: str = "work") -> Path:
    """Compatibility wrapper used by helper modules."""
    if subdir == "raw":
        return DATA_RAW_DIR
    if subdir == "work":
        return DATA_WORK_DIR
    if subdir == "diagnostics":
        return DIAGNOSTICS_DIR
    return PROJECT_ROOT / f"data_{subdir}"
