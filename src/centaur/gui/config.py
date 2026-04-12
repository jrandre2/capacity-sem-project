"""
GUI-specific configuration.

Extends the main config.py with settings specific to the web interface.
"""
from pathlib import Path

# Import main project config
from ..config import (
    DATA_WORK_DIR,
    DOC_DIR,
    MANUSCRIPT_DIR,
    MANUSCRIPTS,
    PROJECT_ROOT,
    QA_REPORTS_DIR,
    REVIEWS_DIR,
)

# GUI Settings
GUI_HOST = "127.0.0.1"
GUI_PORT = 8000
GUI_DEBUG = True

# Template and static file paths
GUI_DIR = Path(__file__).parent
TEMPLATES_DIR = GUI_DIR / "templates"
STATIC_DIR = GUI_DIR / "static"

# Re-export main config items for convenience
__all__ = [
    'GUI_HOST',
    'GUI_PORT',
    'GUI_DEBUG',
    'GUI_DIR',
    'TEMPLATES_DIR',
    'STATIC_DIR',
    'PROJECT_ROOT',
    'DATA_WORK_DIR',
    'DOC_DIR',
    'MANUSCRIPT_DIR',
    'QA_REPORTS_DIR',
    'MANUSCRIPTS',
    'REVIEWS_DIR',
]
