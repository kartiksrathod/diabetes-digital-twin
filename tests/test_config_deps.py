"""Configuration, Dependency, and Environment Verification Tests.

Verifies declared dependencies in requirements files, file resolution from project root,
and environment setup.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_DASHBOARD_PATH = PROJECT_ROOT / "requirements-dashboard.txt"


class TestConfigurationAndPaths:
    """Verifies repository directory layout and required data/model files."""

    def test_project_root_contains_key_components(self):
        assert (PROJECT_ROOT / "dashboard_api.py").exists()
        assert (PROJECT_ROOT / "mcp_server.py").exists()
        assert (PROJECT_ROOT / "digital_twin" / "azure_client.py").exists()
        assert (PROJECT_ROOT / "dashboard" / "index.html").exists()

    def test_source_controlled_metrics_and_fixtures_exist(self):
        """Verifies that all tracked metric files and test fixtures exist in the repository."""
        assert (PROJECT_ROOT / "data" / "processed" / "final_test_results.csv").exists()
        assert (PROJECT_ROOT / "data" / "processed" / "model_b_threshold_results.csv").exists()
        assert (PROJECT_ROOT / "digital_twin" / "models" / "PatientDigitalTwin.json").exists()
        assert (PROJECT_ROOT / "tests" / "fixtures" / "fixture_model_b.joblib").exists()
        assert (PROJECT_ROOT / "tests" / "fixtures" / "fixture_features_v2.csv").exists()
        assert (PROJECT_ROOT / "tests" / "fixtures" / "fixture_train.csv").exists()
        assert (PROJECT_ROOT / "tests" / "fixtures" / "fixture_validation.csv").exists()
        assert (PROJECT_ROOT / "tests" / "fixtures" / "fixture_test.csv").exists()

    def test_fixture_patient_splits_exist_and_non_empty(self):
        """Verifies that all fixture split files are present and non-empty."""
        for split_file in ("fixture_train.csv", "fixture_validation.csv", "fixture_test.csv"):
            path = PROJECT_ROOT / "tests" / "fixtures" / split_file
            assert path.exists() and path.stat().st_size > 0

    def test_production_artifacts_when_available(self):
        """Validates heavy local production model and full datasets only when present."""
        prod_model = PROJECT_ROOT / "data" / "processed" / "model_b_random_forest.joblib"
        if not prod_model.exists():
            pytest.skip("Full production model not present in clean checkout")
        assert (PROJECT_ROOT / "data" / "processed" / "shanghai_t2dm_features_v2.csv").exists()
        for split in ("train.csv", "validation.csv", "test.csv"):
            f = PROJECT_ROOT / "data" / "processed" / split
            assert f.exists() and f.stat().st_size > 0


class TestDeclaredDependencies:
    """Verifies that requirements files specify the packages necessary to run the project."""

    def test_core_ml_dependencies_declared(self):
        content = REQUIREMENTS_PATH.read_text(encoding="utf-8")
        assert "pandas" in content
        assert "numpy" in content
        assert "scikit-learn" in content
        assert "joblib" in content

    def test_dashboard_dependencies_declared(self):
        content = REQUIREMENTS_DASHBOARD_PATH.read_text(encoding="utf-8")
        assert "fastapi" in content
        assert "uvicorn" in content

    def test_azure_and_mcp_dependencies_declared(self):
        """Regression test for missing azure and mcp package declarations in requirements.txt."""
        combined_requirements = (
            REQUIREMENTS_PATH.read_text(encoding="utf-8")
            + "\n"
            + REQUIREMENTS_DASHBOARD_PATH.read_text(encoding="utf-8")
        )
        assert "azure-digitaltwins-core" in combined_requirements, "azure-digitaltwins-core must be declared in requirements"
        assert "azure-identity" in combined_requirements, "azure-identity must be declared in requirements"
        assert "mcp" in combined_requirements, "mcp must be declared in requirements"
