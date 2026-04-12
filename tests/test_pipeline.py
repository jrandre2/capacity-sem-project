"""
Tests for the Capacity-SEM pipeline.
"""

import pytest
import subprocess
import sys
import importlib.util
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

    def test_s03c_kaifa_external_replication_imports(self):
        """Test provisional external Kaifa replication stage imports."""
        from stages import s03c_kaifa_external_replication
        assert hasattr(s03c_kaifa_external_replication, 'main')

    def test_s03d_kaifa_recovered_analysis_imports(self):
        """Test recovered Kaifa notebook analysis stage imports."""
        from stages import s03d_kaifa_recovered_analysis

        assert hasattr(s03d_kaifa_recovered_analysis, 'main')

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


class TestCapacityAlternatives:
    """Test alternative capacity measures."""

    def test_capacity_alternative_measures(self):
        """Ensure alternative capacity measures and quartiles are created."""
        import pandas as pd
        from stages.s02_features import add_capacity_alternative_measures

        df = pd.DataFrame({
            'Ratio_disbursed_to_obligated': [0.1, 0.2, 0.3, 0.4],
            'Ratio_expended_to_disbursed': [0.2, 0.3, 0.4, 0.5],
            'Disbursed_log': [1.0, 1.1, 1.2, 1.3],
            'Expended_log': [0.8, 0.9, 1.0, 1.1],
            'Disbursement_Velocity': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity': [0.005, 0.01, 0.015, 0.02],
            'Disbursement_Velocity_early_2q': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity_early_2q': [0.005, 0.01, 0.015, 0.02],
            'Disbursement_Velocity_early_3q': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity_early_3q': [0.005, 0.01, 0.015, 0.02],
            'Disbursement_Velocity_early_4q': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity_early_4q': [0.005, 0.01, 0.015, 0.02],
            'Disbursement_Velocity_early_6q': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity_early_6q': [0.005, 0.01, 0.015, 0.02],
            'Disbursement_Velocity_fixed_12m': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity_fixed_12m': [0.005, 0.01, 0.015, 0.02],
            'Disbursement_Velocity_fixed_18m': [0.01, 0.02, 0.03, 0.04],
            'Expenditure_Velocity_fixed_18m': [0.005, 0.01, 0.015, 0.02],
        })

        df['Capacity_Index'] = df[['Ratio_disbursed_to_obligated', 'Ratio_expended_to_disbursed']].mean(axis=1)
        out = add_capacity_alternative_measures(df)

        assert 'Capacity_Absolute_Index' in out.columns
        assert 'Capacity_Velocity_Index' in out.columns
        assert 'Capacity_Velocity_Index_pp' in out.columns
        assert 'Capacity_Velocity_Index_winsor' in out.columns
        assert 'Capacity_Velocity_Index_scaled' in out.columns
        assert 'Capacity_Velocity_Index_early_4q' in out.columns
        assert 'Capacity_Velocity_Index_early_4q_pp' in out.columns
        assert 'Capacity_Velocity_Index_early_4q_scaled' in out.columns
        assert 'Capacity_Velocity_Index_early_6q' in out.columns
        assert 'Capacity_Velocity_Index_early_6q_pp' in out.columns
        assert 'Capacity_Velocity_Index_early_6q_scaled' in out.columns
        assert 'Capacity_Velocity_Index_early_2q' in out.columns
        assert 'Capacity_Velocity_Index_early_2q_pp' in out.columns
        assert 'Capacity_Velocity_Index_early_2q_scaled' in out.columns
        assert 'Capacity_Velocity_Index_early_3q' in out.columns
        assert 'Capacity_Velocity_Index_early_3q_pp' in out.columns
        assert 'Capacity_Velocity_Index_early_3q_scaled' in out.columns
        assert 'Capacity_Velocity_Index_fixed_12m' in out.columns
        assert 'Capacity_Velocity_Index_fixed_12m_pp' in out.columns
        assert 'Capacity_Velocity_Index_fixed_12m_scaled' in out.columns
        assert 'Capacity_Velocity_Index_fixed_18m' in out.columns
        assert 'Capacity_Velocity_Index_fixed_18m_pp' in out.columns
        assert 'Capacity_Velocity_Index_fixed_18m_scaled' in out.columns
        assert 'Disbursement_Velocity_pp' in out.columns
        assert 'Expenditure_Velocity_pp' in out.columns
        assert 'Disbursement_Velocity_winsor' in out.columns
        assert 'Expenditure_Velocity_winsor' in out.columns
        assert 'Disbursement_Velocity_scaled' in out.columns
        assert 'Expenditure_Velocity_scaled' in out.columns
        assert 'Ratio_disbursed_to_obligated_c' in out.columns
        assert 'Ratio_disbursed_to_obligated_high' in out.columns
        assert 'Ratio_disbursed_to_obligated_above' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q25' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q33' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q67' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q75' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q25' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q33' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q67' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q75' in out.columns
        assert 'Disbursement_Velocity_pp_c' in out.columns
        assert 'Capacity_Velocity_Index_pp_c' in out.columns
        assert 'Ratio_disbursed_to_obligated_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q25_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q25_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q33_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q33_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q33_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q33_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q67_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q67_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q67_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q67_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q75_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_high_q75_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q25_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q25_x_Capacity_Velocity_Index_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q75_x_Disbursement_Velocity_pp' in out.columns
        assert 'Ratio_disbursed_to_obligated_above_q75_x_Capacity_Velocity_Index_pp' in out.columns


class TestCentaurFramework:
    """Test vendored CENTAUR framework integration."""

    def test_centaur_package_imports(self):
        """Vendored CENTAUR package should import and expose engines."""
        import centaur
        from centaur.analysis import list_engines

        assert centaur.PROJECT_ROOT.exists()

        engines = list_engines()
        assert 'python' in engines

    def test_centaur_agents_generate_mapping_and_plan(self, tmp_path):
        """Project analysis and migration planning should work on a minimal repo."""
        from centaur.agents import analyze_project, generate_migration_plan, map_project

        (tmp_path / "src").mkdir()
        (tmp_path / "doc").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "load_data.py").write_text(
            '"""Load data."""\n\ndef main():\n    return None\n'
        )
        (tmp_path / "src" / "fit_model.py").write_text(
            '"""Estimate model."""\n\ndef estimate():\n    return None\n'
        )
        (tmp_path / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")

        analysis = analyze_project(tmp_path)
        mapping = map_project(analysis)
        plan = generate_migration_plan(analysis, mapping, str(tmp_path / "target"))

        assert analysis.has_tests
        assert len(mapping.rules) > 0
        assert len(plan.steps) > 0

    def test_centaur_cli_help(self):
        """Host CLI should expose the vendored CENTAUR command group."""
        result = subprocess.run(
            [sys.executable, str(src_dir / "pipeline.py"), "centaur", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "analyze_project" in result.stdout
        assert "plan_migration" in result.stdout
        assert "review_diff" in result.stdout
        assert "review_response" in result.stdout
        assert "review_ingest_docx" in result.stdout

    def test_centaur_stage_discovery_imports_without_fastapi(self):
        """Stage discovery should not require optional FastAPI dependencies."""
        from centaur.gui.services.pipeline_service import get_pipeline_service

        stages = get_pipeline_service().discover_stages()

        assert any(stage.name == "s03_estimation" for stage in stages)
        assert any(stage.name == "s06_manuscript" for stage in stages)

    def test_centaur_cli_list_stages(self):
        """Host CLI should list vendored stages without importing the web stack."""
        result = subprocess.run(
            [sys.executable, str(src_dir / "pipeline.py"), "centaur", "list_stages"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Vendored CENTAUR Stages" in result.stdout
        assert "s03_estimation" in result.stdout

    def test_host_cli_review_help(self):
        """Host CLI should expose the unified review commands."""
        result = subprocess.run(
            [sys.executable, str(src_dir / "pipeline.py"), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "review_diff" in result.stdout
        assert "review_response" in result.stdout
        assert "review_ingest_docx" in result.stdout

    def test_review_ingest_docx_dry_run_for_kaifa(self):
        """Kaifa DOCX import should be wired through the host CLI."""
        result = subprocess.run(
            [
                sys.executable,
                str(src_dir / "pipeline.py"),
                "review_ingest_docx",
                "--manuscript",
                "kaifa",
                "--dry-run",
                "--force",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Revision Tracker: Imported DOCX Feedback" in result.stdout
        assert "Dry run summary: comments=0, track_changes=0" in result.stdout

    def test_review_management_module_is_centaur_shim(self):
        """Legacy review_management imports should expose the unified review workspace."""
        import review_management

        assert "kaifa" in review_management.MANUSCRIPTS
        assert "par_general" in review_management.FOCUS_PROMPTS


class TestPipeline:
    """Test main pipeline module."""

    def test_pipeline_imports(self):
        """Test pipeline module imports."""
        import pipeline

        assert hasattr(pipeline, 'main')
        assert hasattr(pipeline, 'cmd_ingest_data')
        assert hasattr(pipeline, 'cmd_run_estimation')
        assert hasattr(pipeline, 'cmd_run_all_legacy')

    def test_pipeline_help_lists_kaifa_replication_commands(self):
        """Host CLI should expose both provisional and recovered Kaifa workflows."""
        result = subprocess.run(
            [sys.executable, str(src_dir / "pipeline.py"), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "run_kaifa_external_replication" in result.stdout
        assert "run_kaifa_recovered_analysis" in result.stdout


class TestKaifaExternalReplication:
    """Test provisional external Kaifa SEM reconstruction."""

    def test_external_replication_registry_contains_imported_bundles(self):
        """The reconstruction module should know about imported Kaifa SEM bundles."""
        from capacity_sem.models.sem_external_replication import (
            EXTERNAL_SEM_BUNDLES,
            KAIFA_EXTERNAL_LABEL,
        )

        assert KAIFA_EXTERNAL_LABEL == "PROVISIONAL_EXTERNAL_RECONSTRUCTION"
        assert "baseline_sem" in EXTERNAL_SEM_BUNDLES
        assert "baseline_sem_admin_4" in EXTERNAL_SEM_BUNDLES

    def test_external_replication_writes_prefixed_outputs(self, tmp_path):
        """External reconstruction should write clearly labeled generated artifacts."""
        pytest.importorskip("semopy")

        from capacity_sem.models.sem_external_replication import run_external_replication

        result = run_external_replication(
            bundle_name="baseline_sem",
            output_dir=tmp_path,
            write_outputs=True,
        )

        assert result["output_dir"] == tmp_path
        assert (tmp_path / "external_reconstruction_manifest.json").exists()
        assert (tmp_path / "external_reconstruction_fit_comparison.csv").exists()
        assert (tmp_path / "external_reconstruction_two_factor_parameter_estimates.csv").exists()

        fit_comparison = result["fit_comparison"]
        assert "Delta_TwoFactor" in fit_comparison.columns

        cfi_row = fit_comparison.loc[fit_comparison["Statistic"] == "CFI"].iloc[0]
        assert abs(float(cfi_row["Delta_TwoFactor"])) < 0.05


class TestKaifaRecoveredAnalysis:
    """Test recovered Kaifa notebook analysis support."""

    def test_recovered_bundle_registry_points_to_imported_files(self):
        """Recovered bundle metadata should point at imported ZIP artifacts."""
        from capacity_sem.models.kaifa_recovered_analysis import (
            KAIFA_RECOVERED_LABEL,
            RECOVERED_BUNDLE,
        )

        assert KAIFA_RECOVERED_LABEL == "RECOVERED_NOTEBOOK_PORT"
        assert RECOVERED_BUNDLE.zip_path.exists()
        assert RECOVERED_BUNDLE.sem_notebook.exists()
        assert RECOVERED_BUNDLE.raw_csv.exists()

    def test_recovered_analysis_captures_row_count_mismatch(self, tmp_path):
        """Recovered analysis should preserve the saved-vs-rerun row-count discrepancy."""
        pytest.importorskip("semopy")

        from capacity_sem.models.kaifa_recovered_analysis import run_recovered_analysis

        result = run_recovered_analysis(output_dir=tmp_path, write_outputs=True)

        assert result["raw_df"].shape[0] == 577
        assert result["sem_data"].shape[0] == 577
        assert result["notebook_metadata"]["saved_dataset_rows"] == 573
        assert result["excluded_grantees"] == [
            "Collier County, FL",
            "KY",
            "Leon County, FL",
            "Nash County, NC",
        ]
        assert result["partial_audit"]["uses_avg_employment_only_denominator"] is True
        assert "cb_2018_us_county_500k.zip" in result["partial_audit"]["missing_shapefiles"]
        alignment = result["candidate_alignment_summary"]
        assert alignment is not None
        exact_cols = set(alignment.loc[alignment["exact_match"], "column"])
        assert "z_avg_employment" in exact_cols
        assert "z_avg_payroll" in exact_cols
        assert "z_Ratio_disbursed_to_obligated" not in exact_cols
        assert (tmp_path / "recovered_notebook_manifest.json").exists()
        assert (tmp_path / "recovered_notebook_model_comparison.csv").exists()
        assert (tmp_path / "recovered_notebook_reference_fit_comparison.csv").exists()
        assert (tmp_path / "recovered_notebook_candidate_573_raw_subset.csv").exists()
        assert (tmp_path / "recovered_notebook_candidate_573_alignment_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_measurement_appendix_table.csv").exists()
        assert (tmp_path / "recovered_notebook_full_structural_reporting.csv").exists()
        assert (tmp_path / "recovered_notebook_fit_verification_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_maturity_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_data_flow_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_geography_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_sensitivity_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_ratio_artifact_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_official_geography_crosswalk.csv").exists()
        assert (tmp_path / "recovered_notebook_official_geography_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_proxy_validation_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_subset_169_forensics_summary.csv").exists()
        assert (tmp_path / "recovered_notebook_subset_169_tree_rules.txt").exists()

        official_geography = result["official_geography_summary"]
        assert official_geography is not None
        county_row = official_geography.loc[
            official_geography["mapping_step"] == "Official Census county crosswalk"
        ].iloc[0]
        assert county_row["count_or_implication"] == "543/543 matched (100.0%)"

        proxy_validation = result["proxy_validation_summary"]
        assert proxy_validation is not None
        assert "Employment proxy vs payroll proxy" in set(proxy_validation["check"])

        subset_forensics = result["subset_forensics_summary"]
        assert subset_forensics is not None
        conclusion = subset_forensics.loc[
            subset_forensics["finding"] == "Forensic conclusion", "value"
        ].iloc[0]
        assert conclusion == "Exact rule unrecovered"


class TestKaifaWordRevision:
    """Test standalone-language safeguards for the full Kaifa Word manuscript."""

    def test_full_revision_generator_blocks_metacommentary_phrases(self):
        """The generator should validate against revision-memo language."""
        script_path = Path(__file__).parent.parent / "manuscript_kaifa_archive" / "code" / "revise_full_sem_manuscript.py"
        spec = importlib.util.spec_from_file_location("revise_full_sem_manuscript", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        module.validate_standalone_language()

        lowered = "\n".join(module.PARAGRAPH_REPLACEMENTS.values()).lower()
        for phrase in module.FORBIDDEN_STANDALONE_PHRASES:
            assert phrase not in lowered

    def test_full_revision_docx_omits_revision_memo_language(self):
        """The generated full Word manuscript should read as a standalone article."""
        docx = pytest.importorskip("docx")

        manuscript_path = (
            Path(__file__).parent.parent
            / "manuscript_kaifa_archive"
            / "source_docs"
            / "SEM_Manuscript_2026-04-09_full_revision.docx"
        )
        document = docx.Document(manuscript_path)
        text = "\n".join(p.text for p in document.paragraphs).lower()

        forbidden = [
            "revised manuscript",
            "revised paper",
            "revised study",
            "revised analysis",
            "revised design",
            "revised results",
            "earlier draft",
            "current word manuscript",
            "raw archive",
            "project repository",
            "not currently recovered",
        ]
        for phrase in forbidden:
            assert phrase not in text

    def test_full_revision_docx_includes_updated_sem_reporting(self):
        """The generated manuscript should expose DoF and confidence-interval reporting."""
        docx = pytest.importorskip("docx")

        manuscript_path = (
            Path(__file__).parent.parent
            / "manuscript_kaifa_archive"
            / "source_docs"
            / "SEM_Manuscript_2026-04-09_full_revision.docx"
        )
        document = docx.Document(manuscript_path)

        table0 = document.tables[0]
        assert table0.cell(0, 1).text == "chi-square (DoF)"
        assert table0.cell(1, 1).text == "741.72 (101)"
        assert table0.cell(2, 1).text == "469.82 (98)"

        table1 = document.tables[1]
        assert table1.cell(0, 3).text == "95% CI / p-value"
        assert table1.cell(1, 0).text == "Recovery Timeliness"
        assert "p<0.001" in table1.cell(2, 3).text
        assert "[-0.004, 0.095]" in table1.cell(1, 3).text

    def test_full_revision_docx_appends_supporting_appendix_tables(self):
        """The generated manuscript should include the new appendix support tables."""
        docx = pytest.importorskip("docx")

        manuscript_path = (
            Path(__file__).parent.parent
            / "manuscript_kaifa_archive"
            / "source_docs"
            / "SEM_Manuscript_2026-04-09_full_revision.docx"
        )
        document = docx.Document(manuscript_path)
        text = "\n".join(p.text for p in document.paragraphs)

        assert "Appendix Table A1. Two-factor SEM measurement diagnostics." in text
        assert "Appendix Table A2. Complete structural and latent-covariance reporting for the two-factor SEM." in text
        assert "Appendix Table A3. Imported-versus-rerun fit verification for the archived SEM models." in text
        assert "Appendix Table A4. Recoverable data flow from standardized QPR rows to the archived 573-jurisdiction SEM sample." in text
        assert "Appendix Table A5. Portfolio maturity proxy summary for the 573-jurisdiction SEM sample." in text
        assert "Appendix Table A6. Light-touch SEM sensitivity checks." in text
        assert "Appendix Table A7. Ratio-artifact frequency and capped-ratio sensitivity summary." in text
        assert "Appendix Table A8. Official 2023 Census geography crosswalk audit for SEM-facing units." in text
        assert "Appendix Table A9. Proxy-validation and coupling diagnostics for QCEW-based resource and workload measures." in text
        assert "Appendix Table A10. Forensic summary of the smaller cleaned N = 169 sensitivity sample." in text
        assert "Table 3. Top 20 jurisdictions with the highest disaster recovery governance risk index." not in text
        assert "Appendix Figure A6." in text
        assert "Appendix Figure A7." in text
        assert "Figure 5(a)" not in text
        assert "Figure 5(b)" not in text
        assert "130,490 valid jurisdiction-quarter records" not in text
        assert len(document.tables) >= 12


class TestQuarterlyCollapse:
    """Test quarter-level collapsing utilities."""

    def test_collapse_to_quarterly_panel_recomputes_velocity_from_quarter_end_ratios(self):
        """Repeated activity rows within a quarter should collapse to one quarter-level row."""
        import pandas as pd
        from utils.quarterly_panel import collapse_to_quarterly_panel

        df = pd.DataFrame({
            'Grantee': ['A'] * 5,
            'Disaster Type': ['Storm'] * 5,
            'QPR Actual Quarter': ['2020 Q1', '2020 Q1', '2020 Q2', '2020 Q2', '2020 Q3'],
            'QPR_Date': pd.to_datetime(['2020-03-31', '2020-03-31', '2020-06-30', '2020-06-30', '2020-09-30']),
            'QPR Fund Obligated $': [100, 100, 100, 100, 100],
            'QPR Fund Disbursed $': [10, 10, 20, 20, 30],
            'QPR Fund Expended $': [0, 5, 10, 10, 20],
            'Ratio_Disbursed_Std': [0.10, 0.10, 0.20, 0.20, 0.30],
            'Ratio_Expended_Std': [0.00, 0.05, 0.10, 0.10, 0.20],
            'Velocity_Disb_Std_pp_winsor': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Velocity_Exp_Std_pp_winsor': [0.0, 5.0, 0.0, 0.0, 0.0],
        })

        out = collapse_to_quarterly_panel(df, rolling_windows=[2])

        assert len(out) == 3
        assert list(out['QPR Actual Quarter']) == ['2020 Q1', '2020 Q2', '2020 Q3']
        assert pd.isna(out.loc[0, 'Velocity_Disb_Std_pp'])
        assert out.loc[1, 'Velocity_Disb_Std_pp'] == pytest.approx(10.0)
        assert out.loc[2, 'Velocity_Disb_Std_pp'] == pytest.approx(10.0)
        assert pd.isna(out.loc[0, 'Velocity_Exp_Std_pp'])
        assert out.loc[1, 'Velocity_Exp_Std_pp'] == pytest.approx(5.0)
        assert out.loc[2, 'Velocity_Exp_Std_pp'] == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
