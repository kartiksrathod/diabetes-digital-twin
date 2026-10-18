"""Pytest configuration and shared fixtures for GlycoTwin test suite."""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_twin_payload():
    """Realistic Azure Digital Twin document for P001 matching dtmi:glucosetwin:PatientDigitalTwin;1."""
    return {
        "$dtId": "P001",
        "$etag": 'W/"12917acd-76c6-4a40-8900-cb932f5aa8cf"',
        "age": 69,
        "bmi": 26.54,
        "currentGlucose": 163.8,
        "glucoseTrend": "Rising",
        "hba1c": 57.382,
        "patientId": "2001",
        "predictedEvent": "Glucose spike",
        "predictionWindow": "Next 2 hours",
        "riskLevel": "HIGH",
        "riskScore": 0.6774855477201854,
        "lastUpdated": "2026-10-17T05:28:48.612779+00:00",
        "$metadata": {
            "$model": "dtmi:glucosetwin:PatientDigitalTwin;1",
            "$lastUpdateTime": "2026-10-17T05:28:48.8212302Z",
        },
    }


@pytest.fixture
def mock_azure_client(mock_twin_payload):
    """Mock DigitalTwinsClient that returns the mock twin payload."""
    client = MagicMock()
    client.get_digital_twin.return_value = mock_twin_payload
    client.update_digital_twin.return_value = None
    return client


@pytest.fixture(autouse=True)
def configure_dashboard_test_artifacts(monkeypatch):
    """Ensure dashboard_api replay endpoints resolve against deterministic test fixtures and clean caches across all tests."""
    import dashboard_api
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixture_features = fixtures_dir / "fixture_features_v2.csv"
    fixture_model = fixtures_dir / "fixture_model_b.joblib"

    monkeypatch.setattr(dashboard_api, "FEATURES_PATH", fixture_features)
    monkeypatch.setattr(dashboard_api, "MODEL_PATH", fixture_model)

    dashboard_api._load_dataset.cache_clear()
    dashboard_api._load_model.cache_clear()
    if hasattr(dashboard_api, "_replay_controller"):
        dashboard_api._replay_controller._running = False
        dashboard_api._replay_controller._stop_event.set()
        dashboard_api._replay_controller = dashboard_api.ReplayController()
    try:
        yield
    finally:
        dashboard_api._load_dataset.cache_clear()
        dashboard_api._load_model.cache_clear()
        if hasattr(dashboard_api, "_replay_controller"):
            dashboard_api._replay_controller._running = False
            dashboard_api._replay_controller._stop_event.set()
            dashboard_api._replay_controller = dashboard_api.ReplayController()

