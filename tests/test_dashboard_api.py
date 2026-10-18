"""FastAPI Backend Integration and Route Tests for dashboard_api.py.

All tests mock external Azure calls to ensure deterministic, isolated,
and credentials-free execution.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from dashboard_api import app, FINAL_THRESHOLD


@pytest.fixture
def client():
    """TestClient for FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_dashboard_api_fixtures(monkeypatch):
    """Ensure dashboard_api replay endpoints resolve against deterministic test fixtures and clean caches."""
    import dashboard_api
    from pathlib import Path

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixture_features = fixtures_dir / "fixture_features_v2.csv"
    fixture_model = fixtures_dir / "fixture_model_b.joblib"

    monkeypatch.setattr(dashboard_api, "FEATURES_PATH", fixture_features)
    monkeypatch.setattr(dashboard_api, "MODEL_PATH", fixture_model)

    dashboard_api._load_dataset.cache_clear()
    dashboard_api._load_model.cache_clear()
    try:
        yield
    finally:
        dashboard_api._load_dataset.cache_clear()
        dashboard_api._load_model.cache_clear()


class TestHealthEndpoint:
    """Tests for /api/health."""

    def test_health_returns_ok_and_threshold(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["service"] == "glycotwin-dashboard-api"
        assert data["mode"] == "local-hackathon-demo"
        assert data["threshold"] == FINAL_THRESHOLD
        assert data["threshold"] == 0.57


class TestSystemStatusEndpoint:
    """Tests for /api/system-status."""

    def test_system_status_azure_connected(self, client, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            response = client.get("/api/system-status")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["azure_twin_reachable"] is True
            assert data["azure_error"] is None
            assert data["feature_csv_exists"] is True
            assert data["model_file_exists"] is True
            assert "Separate stdio server" in data["mcp_server"]
            assert "historical fixture" in data["replay_source"].lower()

    def test_system_status_azure_unreachable(self, client):
        mock_broken_client = MagicMock()
        mock_broken_client.get_digital_twin.side_effect = ConnectionError("Azure network unreachable")
        with patch("dashboard_api._azure_client", return_value=mock_broken_client):
            response = client.get("/api/system-status")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["azure_twin_reachable"] is False
            assert data["azure_error"] == "ConnectionError"


class TestTwinStateEndpoint:
    """Tests for /api/twin-state/{twin_id}."""

    def test_twin_state_success_p001(self, client, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            response = client.get("/api/twin-state/P001")
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["twin_id"] == "P001"
            assert data["patient_id"] == "2001"
            assert isinstance(data["current_glucose"], float)
            assert data["current_glucose"] == pytest.approx(163.8, rel=1e-3)
            assert 0.0 <= data["risk_score"] <= 1.0
            assert data["risk_level"] == "HIGH"
            assert data["prediction_window"] == "Next 2 hours"
            assert data["glucose_trend"] == "Rising"

            # Provenance must be explicitly simulated replay, never live wearable
            assert data["data_provenance"] == "Simulated replay of historical CGM data"
            assert "wearable" not in data["data_provenance"].lower()

            # Azure does not store source_timestamp or delta_30m; API must report None rather than inventing
            assert data["source_timestamp"] is None
            assert data["delta_30m"] is None

    def test_twin_state_low_risk_classification(self, client):
        low_risk_twin = {
            "$dtId": "P001",
            "patientId": "2001",
            "currentGlucose": 95.0,
            "glucoseTrend": "Stable",
            "riskScore": 0.22,
            "riskLevel": None,  # Test fallback classification
            "predictedEvent": "No glucose spike",
            "lastUpdated": "2026-10-17T05:00:00+00:00",
        }
        mock_client = MagicMock()
        mock_client.get_digital_twin.return_value = low_risk_twin
        with patch("dashboard_api._azure_client", return_value=mock_client):
            response = client.get("/api/twin-state/P001")
            assert response.status_code == 200
            data = response.json()
            assert data["risk_score"] == 0.22
            assert data["risk_level"] == "LOW"

    def test_twin_state_azure_read_error_handled(self, client):
        mock_broken_client = MagicMock()
        mock_broken_client.get_digital_twin.side_effect = RuntimeError("Failed Azure query")
        with patch("dashboard_api._azure_client", return_value=mock_broken_client):
            response = client.get("/api/twin-state/P001")
            assert response.status_code == 502
            data = response.json()
            assert "Azure twin read failed" in data["detail"]
            # Sensitive tokens must not leak into response
            assert "Bearer" not in str(data)

    def test_twin_state_client_init_error_handled(self, client):
        with patch("dashboard_api._azure_client", side_effect=Exception("Credential missing")):
            response = client.get("/api/twin-state/P001")
            assert response.status_code in (502, 503)


class TestReplayHistoryEndpoint:
    """Tests for /api/replay-history."""

    def test_replay_history_default_params(self, client):
        response = client.get("/api/replay-history?patient_id=2001&session_id=2001_0_20201102&limit=5")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["patient_id"] == "2001"
        assert data["session_id"] == "2001_0_20201102"
        assert data["count"] == 5
        assert len(data["rows"]) == 5
        assert data["threshold"] == 0.57
        assert data["prediction_window"] == "Next 2 hours"
        assert "Historical ShanghaiT2DM rows" in data["data_provenance"]

        # Check structure of rows
        for row in data["rows"]:
            assert "time" in row
            assert "timestamp" in row
            assert isinstance(row["glucose"], (int, float))
            assert isinstance(row["risk"], float)
            assert 0.0 <= row["risk"] <= 1.0
            assert row["trend"] in ("Rising", "Falling", "Stable")
            assert row["level"] in ("HIGH", "LOW")
            if row["level"] == "HIGH":
                assert row["risk"] >= 0.57
                assert row["predicted_event"] == "Glucose spike"
            else:
                assert row["risk"] < 0.57
                assert row["predicted_event"] == "No glucose spike"

    def test_replay_history_unknown_patient_404(self, client):
        response = client.get("/api/replay-history?patient_id=999999&session_id=none&limit=5")
        assert response.status_code == 404
        assert "No rows found" in response.json()["detail"]

    def test_replay_history_invalid_limit_validation(self, client):
        # Limit must be between 2 and 300
        res_low = client.get("/api/replay-history?limit=1")
        assert res_low.status_code == 422  # FastAPI validation error

        res_high = client.get("/api/replay-history?limit=301")
        assert res_high.status_code == 422


class TestRoutingAndCors:
    """Tests for root route 404 and CORS configuration."""

    def test_root_returns_404(self, client):
        # Root is intentionally not mapped by backend (served by frontend HTTP server on port 5500)
        response = client.get("/")
        assert response.status_code == 404

    def test_cors_headers_for_dashboard_origins(self, client):
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"
