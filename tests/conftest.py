"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for all tests
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_work_dir(project_root):
    """Return data_work directory."""
    return project_root / "data_work"


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for testing."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 50

    data = pd.DataFrame({
        'Grantee': np.random.choice(['State A', 'State B', 'Local C'], n),
        'Disaster Type': np.random.choice(['DR-001', 'DR-002'], n),
        'N_Quarters': np.random.randint(4, 20, n),
        'Total_Obligated': np.random.uniform(1e6, 1e8, n),
        'Total_Expended': np.random.uniform(5e5, 9e7, n),
        'Ratio_disbursed_to_obligated': np.random.uniform(0.5, 1.0, n),
        'Ratio_expended_to_disbursed': np.random.uniform(0.5, 1.0, n),
        'Duration_log': np.random.uniform(2, 5, n),
        'Spending_CV': np.random.uniform(0.1, 0.8, n),
        'Population_scaled': np.random.normal(0, 1, n),
        'Experience_Index_scaled': np.random.normal(0, 1, n),
    })

    return data
