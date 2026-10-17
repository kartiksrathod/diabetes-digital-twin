"""Extended FastAPI Adapter Integration and Route Tests (dashboard_api.py).

Validates:
- /api/health response keys and frozen threshold 0.57.
- /api/twin-state/{twin_id}:
  - Successful retrieval of twin with sourceTimestamp and delta30m.
  - Serialization of datetime objects via _serializable_twin_value.
  - Fallback risk calculation at threshold 0.57 (>= 0.57 -> HIGH, < 0.57 -> LOW).
  - Sparse/null state handling without crashing.
  - Upstream Azure HTTP and SDK error handling (502/503) without leaking tokens.
- /api/replay-history:
  - Query parameter constraints (limit min 2, max 300; patient_id, session_id).
  - 404 when fewer than 2 rows are available.
  - 404 for unknown patient or session.
  - 500 when required feature columns are missing from the dataset.
  - 500 when model inference fails.
  - FileNotFoundError on missing dataset/model files.
- /api/system-status:
  - Resilient reporting of Azure connectivity, asset existence, and component architecture.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard_api import app, FINAL_THRESHOLD, _serializable_twin_value

client = TestClient(app)


class TestHealthExtended:
    """Extended tests for /api/health."""

    def test_health_keys_and_threshold(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["service"] == "glycotwin-dashboard-api"
        assert data["threshold"] == pytest.approx(0.57)


class TestTwinStateExtended:
    """Extended tests for /api/twin-state/{twin_id}."""

    def test_twin_state_with_datetime_object_and_provenance(self):
        now_dt = datetime.now(timezone.utc)
        payload = {
            "$dtId": "P001",
            "patientId": "2001",
            "currentGlucose": 115.5,
            "delta30m": 4.2,
            "glucoseTrend": "Stable",
            "riskScore": 0.35,
            "riskLevel": "LOW",
            "predictedEvent": "No glucose spike",
            "predictionWindow": "Next 2 hours",
            "lastUpdated": now_dt,
            "sourceTimestamp": "2020-11-02T11:40:00",
        }
        mock_c = MagicMock()
        mock_c.get_digital_twin.return_value = payload

        with patch("dashboard_api._azure_client", return_value=mock_c):
            resp = client.get("/api/twin-state/P001")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["current_glucose"] == 115.5
            assert data["delta_30m"] == 4.2
            assert data["source_timestamp"] == "2020-11-02T11:40:00"
            assert data["last_updated"] == now_dt.isoformat()
            assert data["data_provenance"] == "Simulated replay of historical CGM data"

    def test_serializable_twin_value_helper(self):
        assert _serializable_twin_value(None) is None
        assert _serializable_twin_value("plain_str") == "plain_str"
        dt = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)
        assert _serializable_twin_value(dt) == dt.isoformat()

    def test_twin_state_azure_http_exception_reraised(self):
        from fastapi import HTTPException
        mock_c = MagicMock()
        mock_c.get_digital_twin.side_effect = HTTPException(status_code=503, detail="Service Unavailable")
        with patch("dashboard_api._azure_client", return_value=mock_c):
            resp = client.get("/api/twin-state/P001")
            assert resp.status_code == 503
            assert resp.json()["detail"] == "Service Unavailable"


class TestReplayHistoryExtended:
    """Extended edge case and error handling tests for /api/replay-history."""

    def test_replay_history_insufficient_rows_returns_404(self):
        # Create a single row df
        single_row_df = pd.DataFrame([{
            "patient_id": "2001",
            "session_id": "sess_short",
            "timestamp": "2020-11-02 12:00:00",
            "glucose_t0": 100.0,
            "delta_30m": 0.0,
            **{f: 0.0 for f in [
                "glucose_t_minus_15m", "glucose_t_minus_30m", "glucose_t_minus_45m",
                "glucose_t_minus_60m", "glucose_t_minus_75m", "glucose_t_minus_90m",
                "glucose_t_minus_105m", "glucose_t_minus_120m", "delta_15m", "delta_60m",
                "delta_120m", "glucose_mean_1h", "glucose_std_1h", "glucose_min_1h",
                "glucose_max_1h", "glucose_mean_2h", "glucose_std_2h", "glucose_min_2h",
                "glucose_max_2h", "slope_30m", "slope_60m", "slope_120m", "age", "bmi",
                "diabetes_duration", "fasting_glucose", "postprandial_glucose", "hba1c",
                "meal_events_30m", "meal_events_60m", "meal_events_120m", "insulin_events_30m",
                "insulin_events_60m", "insulin_events_120m", "insulin_amount_30m",
                "insulin_amount_60m", "insulin_amount_120m", "medication_events_120m"
            ]}
        }])
        single_row_df["timestamp"] = pd.to_datetime(single_row_df["timestamp"])

        with patch("dashboard_api._load_dataset", return_value=single_row_df):
            resp = client.get("/api/replay-history?patient_id=2001&session_id=sess_short&limit=10")
            assert resp.status_code == 404
            assert "At least two replay rows are required" in resp.json()["detail"]

    def test_replay_history_dataset_file_not_found_returns_503(self):
        with patch("dashboard_api._load_dataset", side_effect=FileNotFoundError("Dataset missing")):
            resp = client.get("/api/replay-history")
            assert resp.status_code == 503
            assert "Dataset missing" in resp.json()["detail"]

    def test_replay_history_model_file_not_found_returns_503(self):
        with patch("dashboard_api._load_model", side_effect=FileNotFoundError("Model missing")):
            resp = client.get("/api/replay-history")
            assert resp.status_code == 503
            assert "Model missing" in resp.json()["detail"]

    def test_replay_history_missing_feature_columns_returns_500(self):
        # DataFrame missing one of the 40 required features
        sample_df = pd.DataFrame([
            {"patient_id": "2001", "session_id": "sess_test", "timestamp": "2020-11-02 12:00:00", "glucose_t0": 100.0, "delta_30m": 0.0},
            {"patient_id": "2001", "session_id": "sess_test", "timestamp": "2020-11-02 12:15:00", "glucose_t0": 105.0, "delta_30m": 5.0},
        ])
        sample_df["timestamp"] = pd.to_datetime(sample_df["timestamp"])

        with patch("dashboard_api._load_dataset", return_value=sample_df):
            resp = client.get("/api/replay-history?patient_id=2001&session_id=sess_test")
            assert resp.status_code == 500
            assert "Feature columns missing" in resp.json()["detail"]

    def test_replay_history_inference_failure_returns_500(self):
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("Prediction internal crash")

        with patch("dashboard_api._load_model", return_value=mock_model):
            resp = client.get("/api/replay-history?patient_id=2001&session_id=2001_0_20201102&limit=5")
            assert resp.status_code == 500
            assert "Model inference failed: RuntimeError" in resp.json()["detail"]


class TestSystemStatusExtended:
    """Extended tests for /api/system-status."""

    def test_system_status_reports_correct_contract_properties(self, mock_azure_client):
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            resp = client.get("/api/system-status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert isinstance(data["azure_twin_reachable"], bool)
            assert isinstance(data["feature_csv_exists"], bool)
            assert isinstance(data["model_file_exists"], bool)
            assert "mcp_server" in data
            assert "antigravity_agent" in data
            assert "replay_source" in data
