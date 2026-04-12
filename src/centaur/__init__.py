"""
Vendored CENTAUR framework components for this repository.

The working analysis pipeline remains in the top-level `src/` tree. Reusable
CENTAUR modules live under `src/centaur/` so the framework can coexist with
project-specific code without collisions.
"""

from . import agents, analysis, utils
from .config import (
    ANALYSIS_ENGINE,
    CACHE_DIR,
    DATA_RAW_DIR,
    DATA_WORK_DIR,
    DIAGNOSTICS_DIR,
    DOC_DIR,
    FIGURES_DIR,
    MANUSCRIPT_DIR,
    PROJECT_ROOT,
    R_EXECUTABLE,
    SPECIFICATIONS_FILE,
    ensure_directories,
)

__all__ = [
    "ANALYSIS_ENGINE",
    "CACHE_DIR",
    "DATA_RAW_DIR",
    "DATA_WORK_DIR",
    "DIAGNOSTICS_DIR",
    "DOC_DIR",
    "FIGURES_DIR",
    "MANUSCRIPT_DIR",
    "PROJECT_ROOT",
    "R_EXECUTABLE",
    "SPECIFICATIONS_FILE",
    "agents",
    "analysis",
    "ensure_directories",
    "utils",
]
