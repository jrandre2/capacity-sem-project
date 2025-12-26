"""
Tests for the Capacity-SEM pipeline.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


class TestConfig:
    """Test configuration module."""

    def test_config_imports(self):
        """Test that config module imports without error."""
        from config import (
            PROJECT_ROOT,
            DATA_RAW_DIR,
            DATA_WORK_DIR,
            FIGURES_DIR,
            STATE_GOVERNMENTS,
            LOCAL_GOVERNMENTS,
        )

        assert PROJECT_ROOT.exists()
        assert len(STATE_GOVERNMENTS) > 0
        assert len(LOCAL_GOVERNMENTS) > 0

    def test_grantee_classifications(self):
        """Test grantee classification lists."""
        from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS, ALL_GRANTEES

        # Check no overlap
        state_set = set(STATE_GOVERNMENTS)
        local_set = set(LOCAL_GOVERNMENTS)
        assert len(state_set & local_set) == 0, "Overlap between state and local"

        # Check all grantees is union
        assert len(ALL_GRANTEES) == len(STATE_GOVERNMENTS) + len(LOCAL_GOVERNMENTS)

    def test_fit_thresholds(self):
        """Test that fit thresholds are defined."""
        from config import FIT_THRESHOLDS

        assert 'cfi_good' in FIT_THRESHOLDS
        assert FIT_THRESHOLDS['cfi_good'] == 0.95


class TestStageImports:
    """Test that stage modules import correctly."""

    def test_s00_ingest_imports(self):
        """Test s00_ingest module imports."""
        from stages import s00_ingest
        assert hasattr(s00_ingest, 'main')

    def test_s01_link_imports(self):
        """Test s01_link module imports."""
        from stages import s01_link
        assert hasattr(s01_link, 'main')

    def test_s02_features_imports(self):
        """Test s02_features module imports."""
        from stages import s02_features
        assert hasattr(s02_features, 'main')

    def test_s03_estimation_imports(self):
        """Test s03_estimation module imports."""
        from stages import s03_estimation
        assert hasattr(s03_estimation, 'main')

    def test_s04_robustness_imports(self):
        """Test s04_robustness module imports."""
        from stages import s04_robustness
        assert hasattr(s04_robustness, 'main')

    def test_s05_figures_imports(self):
        """Test s05_figures module imports."""
        from stages import s05_figures
        assert hasattr(s05_figures, 'main')


class TestModelSpecifications:
    """Test SEM model specifications."""

    def test_model_registry_exists(self):
        """Test that model registry is populated."""
        from capacity_sem.models.sem_specifications import MODEL_REGISTRY

        assert len(MODEL_REGISTRY) > 0
        assert 'full' in MODEL_REGISTRY
        assert 'exp_optimal_v1' in MODEL_REGISTRY

    def test_get_model_spec(self):
        """Test getting model specification."""
        from capacity_sem.models.sem_specifications import get_model_spec

        spec = get_model_spec('exp_optimal_v1')
        assert 'gov_cap' in spec
        assert 'recovery_outcome' in spec

    def test_invalid_model_raises(self):
        """Test that invalid model type raises error."""
        from capacity_sem.models.sem_specifications import get_model_spec

        with pytest.raises(ValueError):
            get_model_spec('nonexistent_model')


class TestExternalData:
    """Test external data module."""

    def test_population_data_exists(self):
        """Test that population data is embedded."""
        from capacity_sem.data.external_data import GRANTEE_POPULATION_BY_DECADE

        assert len(GRANTEE_POPULATION_BY_DECADE) > 0

    def test_severity_data_exists(self):
        """Test that severity data is embedded."""
        from capacity_sem.data.external_data import DISASTER_SEVERITY_INDEX

        assert len(DISASTER_SEVERITY_INDEX) > 0

    def test_employment_data_exists(self):
        """Test that employment data is embedded."""
        from capacity_sem.data.external_data import GRANTEE_EMPLOYMENT_BY_YEAR

        assert len(GRANTEE_EMPLOYMENT_BY_YEAR) > 0


class TestFeatures:
    """Test feature computation functions."""

    def test_timeliness_imports(self):
        """Test timeliness module imports."""
        from capacity_sem.features.timeliness import (
            calculate_duration_of_completion,
            calculate_timeliness,
            calculate_quarter_variance,
        )

    def test_experience_imports(self):
        """Test experience module imports."""
        from capacity_sem.features.experience_indicators import (
            compute_years_of_experience,
            compute_experience_index,
            build_experience_dataset,
        )

    def test_stratification_imports(self):
        """Test stratification module imports."""
        from capacity_sem.features.program_stratification import (
            PROGRAM_TYPE_MAPPING,
            map_activity_to_program_type,
        )

        assert 'Housing' in PROGRAM_TYPE_MAPPING
        assert 'Infrastructure' in PROGRAM_TYPE_MAPPING


class TestPipeline:
    """Test main pipeline module."""

    def test_pipeline_imports(self):
        """Test pipeline module imports."""
        import pipeline

        assert hasattr(pipeline, 'main')
        assert hasattr(pipeline, 'cmd_ingest_data')
        assert hasattr(pipeline, 'cmd_run_estimation')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
