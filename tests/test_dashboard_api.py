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

    def test_replay_history_default_returns_20_records(self, client):
        """Regression test: default limit is 20 to match active replay session configuration."""
        response = client.get("/api/replay-history")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 20
        assert len(data["rows"]) == 20

        # Monotonically increasing timestamps in chronological order
        timestamps = [r["timestamp"] for r in data["rows"]]
        assert timestamps == sorted(timestamps)
        assert len(timestamps) == len(set(timestamps))

    def test_replay_history_handles_10_and_20_records(self, client):
        """Verifies both 10-record and 20-record requests return expected slices."""
        res_10 = client.get("/api/replay-history?limit=10")
        assert res_10.status_code == 200
        data_10 = res_10.json()
        assert data_10["count"] == 10
        assert len(data_10["rows"]) == 10

        res_20 = client.get("/api/replay-history?limit=20")
        assert res_20.status_code == 200
        data_20 = res_20.json()
        assert data_20["count"] == 20
        assert len(data_20["rows"]) == 20

        # First 10 rows of 20-row request must match the 10-row request exactly
        assert [r["timestamp"] for r in data_10["rows"]] == [r["timestamp"] for r in data_20["rows"][:10]]


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


class TestReplayControllerEndpoints:
    """Tests for the automatic historical replay controller and management endpoints."""

    def test_replay_status_initial_state(self, client):
        response = client.get("/api/replay/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["running"] is False
        assert data["busy"] is False
        assert data["completed"] is False
        assert data["current_index"] == 0
        assert data["total_rows"] == 0
        assert data["last_error"] is None
        assert data["last_record"] is None
        assert data["threshold"] == FINAL_THRESHOLD
        assert "Simulated replay" in data["data_provenance"]

    def test_replay_lifecycle_start_pause_step_reset(self, client, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            # 1. Start replay with a long delay so it doesn't auto-run while testing
            start_payload = {
                "patient_id": "2001",
                "session_id": "2001_0_20201102",
                "twin_id": "P001",
                "limit": 5,
                "delay_seconds": 60.0,
            }
            start_res = client.post("/api/replay/start", json=start_payload)
            assert start_res.status_code == 200
            start_data = start_res.json()
            assert start_data["success"] is True
            assert start_data["running"] is True
            assert start_data["total_rows"] == 5
            assert start_data["twin_id"] == "P001"

            # 2. Duplicate start is rejected with 409 Conflict
            dup_res = client.post("/api/replay/start", json=start_payload)
            assert dup_res.status_code == 409
            assert "already running" in dup_res.json()["detail"].lower()

            # 3. Pause replay
            pause_res = client.post("/api/replay/pause")
            assert pause_res.status_code == 200
            pause_data = pause_res.json()
            assert pause_data["running"] is False
            assert pause_data["paused"] is True

            # Record index after pause (might be 0 or 1 depending on whether first row was processed)
            idx_before = pause_data["current_index"]

            # 4. Step one record manually
            step_res = client.post("/api/replay/step")
            assert step_res.status_code == 200
            step_data = step_res.json()
            assert step_data["current_index"] == idx_before + 1
            assert step_data["last_record"] is not None
            assert step_data["last_record"]["current_glucose"] > 0
            assert 0.0 <= step_data["last_record"]["risk_score"] <= 1.0
            assert step_data["last_record"]["risk_level"] in ("HIGH", "LOW")

            # 5. Reset replay cursor
            reset_res = client.post("/api/replay/reset")
            assert reset_res.status_code == 200
            reset_data = reset_res.json()
            assert reset_data["current_index"] == 0
            assert reset_data["last_record"] is None
            assert reset_data["completed"] is False

    def test_replay_step_failure_honest_reporting(self, client, mock_azure_client):
        # Start replay, pause it, then simulate an Azure write error during step
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            start_payload = {
                "patient_id": "2001",
                "session_id": "2001_0_20201102",
                "twin_id": "P001",
                "limit": 5,
                "delay_seconds": 60.0,
            }
            client.post("/api/replay/start", json=start_payload)
            client.post("/api/replay/pause")

        status_before = client.get("/api/replay/status").json()
        idx_before = status_before["current_index"]

        # Now mock azure client failing on update
        broken_azure = MagicMock()
        broken_azure.get_digital_twin.return_value = {"$dtId": "P001"}
        broken_azure.update_digital_twin.side_effect = RuntimeError("Azure update timeout")

        with patch("dashboard_api._azure_client", return_value=broken_azure):
            step_res = client.post("/api/replay/step")
            assert step_res.status_code == 502
            assert "failed" in step_res.json()["detail"].lower()

        # Verify cursor did NOT advance
        status_after = client.get("/api/replay/status").json()
        assert status_after["current_index"] == idx_before
        assert status_after["last_error"] is not None
        assert "RuntimeError" in status_after["last_error"]

    def test_replay_step_and_reset_uninitialized_raises_409(self, client):
        step_res = client.post("/api/replay/step")
        assert step_res.status_code == 409
        assert "not been initialized" in step_res.json()["detail"]

        reset_res = client.post("/api/replay/reset")
        assert reset_res.status_code == 409
        assert "not been initialized" in reset_res.json()["detail"]

    def test_step_and_reset_while_running_raises_409(self, client, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            start_payload = {
                "patient_id": "2001",
                "session_id": "2001_0_20201102",
                "twin_id": "P001",
                "limit": 5,
                "delay_seconds": 60.0,
            }
            client.post("/api/replay/start", json=start_payload)

            step_res = client.post("/api/replay/step")
            assert step_res.status_code == 409
            assert "pause replay" in step_res.json()["detail"].lower()

            reset_res = client.post("/api/replay/reset")
            assert reset_res.status_code == 409
            assert "pause replay" in reset_res.json()["detail"].lower()

            client.post("/api/replay/pause")

    def test_reset_does_not_mutate_or_delete_azure_twin(self, client, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            start_payload = {
                "patient_id": "2001",
                "session_id": "2001_0_20201102",
                "twin_id": "P001",
                "limit": 5,
                "delay_seconds": 60.0,
            }
            client.post("/api/replay/start", json=start_payload)
            client.post("/api/replay/pause")

            # Reset call count on mock
            mock_azure_client.update_digital_twin.reset_mock()
            mock_azure_client.delete_digital_twin.reset_mock()

            reset_res = client.post("/api/replay/reset")
            assert reset_res.status_code == 200
            assert reset_res.json()["current_index"] == 0

            # Verify reset NEVER touched Azure
            mock_azure_client.update_digital_twin.assert_not_called()
            mock_azure_client.delete_digital_twin.assert_not_called()

    def test_replay_resumes_after_pause(self, client, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            start_payload = {
                "patient_id": "2001",
                "session_id": "2001_0_20201102",
                "twin_id": "P001",
                "limit": 5,
                "delay_seconds": 60.0,
            }
            client.post("/api/replay/start", json=start_payload)
            pause_res = client.post("/api/replay/pause").json()
            idx_after_pause = pause_res["current_index"]

            # Step manually once
            step_res = client.post("/api/replay/step").json()
            assert step_res["current_index"] == idx_after_pause + 1

            # Resume by calling start again without reset_cursor
            resume_res = client.post("/api/replay/start", json=start_payload)
            assert resume_res.status_code == 200
            resume_data = resume_res.json()
            assert resume_data["running"] is True
            assert resume_data["current_index"] == idx_after_pause + 1

            # Clean up
            client.post("/api/replay/pause")

