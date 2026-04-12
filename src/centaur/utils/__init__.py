"""
Shared utilities from the vendored CENTAUR framework.
"""

from .cache import CacheManager
from .docx_feedback import extract_feedback, format_as_feedback_tracker
from .figure_style import apply_style, get_figure_double, get_figure_single
from .helpers import ensure_dir, format_coefficient, get_data_dir, get_project_root
from .spatial_cv import SpatialCVManager, compare_spatial_vs_random_cv
from .synthetic_data import SyntheticDataGenerator
from .validation import DataValidator

__all__ = [
    "CacheManager",
    "DataValidator",
    "SpatialCVManager",
    "SyntheticDataGenerator",
    "apply_style",
    "ensure_dir",
    "extract_feedback",
    "format_coefficient",
    "format_as_feedback_tracker",
    "get_data_dir",
    "get_figure_double",
    "get_figure_single",
    "get_project_root",
    "compare_spatial_vs_random_cv",
]
