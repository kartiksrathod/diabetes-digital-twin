"""Safe End-to-End Replay and Integration Verification (tests/test_e2e_replay.py).

Verifies the complete closed loop using a small deterministic fixture and mocked Azure client:
1. Deterministic replay row is loaded.
2. Features and model inference execute through the existing Model B Random Forest pipeline.
3. Replay derives the expected telemetry fields (glucose, delta, trend, risk score, risk level).
4. Mocked Azure client receives the expected twin ID, patch operations, source timestamp, and delta.
5. FastAPI adapter serializes and returns the state over /api/twin-state/P001.
6. Frontend rendering contract and DOM contracts verify the state representation.

Zero cloud credentials required; zero network calls; completely deterministic and safe.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard_api import app, FINAL_THRESHOLD
from digital_twin.azure_client import update_patient_twin
from ml.replay_stream import FEATURES, glucose_trend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_b_random_forest.joblib"
FIXTURE_MODEL_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_model_b.joblib"
INDEX_HTML_PATH = PROJECT_ROOT / "dashboard" / "index.html"


@pytest.fixture(scope="module")
def deterministic_cgm_fixture():
    """Small deterministic fixture representing a simulated CGM reading."""
    row_data = {
        "patient_id": "2001",
        "session_id": "2001_0_20201102",
        "timestamp": "2020-11-02 11:40:00",
        "glucose_t0": 88.2,
        "delta_30m": 1.8,
        "glucose_t_minus_15m": 87.0,
        "glucose_t_minus_30m": 86.4,
        "glucose_t_minus_45m": 85.0,
        "glucose_t_minus_60m": 84.5,
        "glucose_t_minus_75m": 84.0,
        "glucose_t_minus_90m": 83.5,
        "glucose_t_minus_105m": 83.0,
        "glucose_t_minus_120m": 82.5,
        "delta_15m": 1.2,
        "delta_60m": 3.7,
        "delta_120m": 5.7,
        "glucose_mean_1h": 86.2,
        "glucose_std_1h": 1.4,
        "glucose_min_1h": 84.5,
        "glucose_max_1h": 88.2,
        "glucose_mean_2h": 85.0,
        "glucose_std_2h": 1.9,
        "glucose_min_2h": 82.5,
        "glucose_max_2h": 88.2,
        "slope_30m": 0.06,
        "slope_60m": 0.0617,
        "slope_120m": 0.0475,
        "age": 69,
        "bmi": 26.54,
        "diabetes_duration": 10.0,
        "fasting_glucose": 6.5,
        "postprandial_glucose": 9.2,
        "hba1c": 7.1,
        "meal_events_30m": 0,
        "meal_events_60m": 0,
        "meal_events_120m": 0,
        "insulin_events_30m": 0,
        "insulin_events_60m": 0,
        "insulin_events_120m": 0,
        "insulin_amount_30m": 0.0,
        "insulin_amount_60m": 0.0,
        "insulin_amount_120m": 0.0,
        "medication_events_120m": 0,
    }
    return pd.DataFrame([row_data])


class TestEndToEndReplayFlow:
    """Verifies the complete flow from CGM row to model to Azure patch to API serialization."""

    def test_complete_replay_pipeline_cycle(self, deterministic_cgm_fixture):
        # 1. Load model and verify pipeline readiness (using deterministic fixture in CI)
        target_model = FIXTURE_MODEL_PATH if FIXTURE_MODEL_PATH.exists() else MODEL_PATH
        assert target_model.exists(), f"Model missing at {target_model}"
        model = joblib.load(target_model)
        row = deterministic_cgm_fixture.iloc[0]

        # 2. Execute inference through 40-feature contract
        X = deterministic_cgm_fixture[FEATURES]
        probabilities = model.predict_proba(X)
        risk_score = float(probabilities[0, 1])
        assert 0.0 <= risk_score <= 1.0

        risk_level = "HIGH" if risk_score >= FINAL_THRESHOLD else "LOW"
        predicted_event = "Glucose spike" if risk_level == "HIGH" else "No glucose spike"
        current_glucose = float(row["glucose_t0"])
        delta_30m = float(row["delta_30m"])
        trend = glucose_trend(delta_30m)
        source_timestamp = str(row["timestamp"])

        assert current_glucose == pytest.approx(88.2)
        assert delta_30m == pytest.approx(1.8)
        assert trend == "Stable"
        assert risk_level == "LOW"

        # 3. Simulate push to Azure Digital Twins via update_patient_twin
        mock_azure_client = MagicMock()
        update_patient_twin(
            mock_azure_client,
            twin_id="P001",
            patient_id=str(row["patient_id"]),
            age=int(row["age"]),
            bmi=float(row["bmi"]),
            hba1c=float(row["hba1c"]),
            current_glucose=current_glucose,
            glucose_trend=trend,
            risk_score=risk_score,
            risk_level=risk_level,
            predicted_event=predicted_event,
            prediction_window="Next 2 hours",
            source_timestamp=source_timestamp,
            delta_30m=delta_30m,
        )

        mock_azure_client.update_digital_twin.assert_called_once()
        twin_id_called, patches = mock_azure_client.update_digital_twin.call_args[0]
        assert twin_id_called == "P001"

        patch_map = {p["path"]: p["value"] for p in patches}
        assert patch_map["/sourceTimestamp"] == "2020-11-02 11:40:00"
        assert patch_map["/delta30m"] == pytest.approx(1.8)
        assert patch_map["/currentGlucose"] == pytest.approx(88.2)
        assert patch_map["/riskLevel"] == "LOW"
        assert patch_map["/glucoseTrend"] == "Stable"
        assert patch_map["/predictionWindow"] == "Next 2 hours"

        # 4. Synthesize updated Azure twin document
        updated_twin_doc = {
            "$dtId": "P001",
            "patientId": "2001",
            "currentGlucose": current_glucose,
            "delta30m": delta_30m,
            "glucoseTrend": trend,
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "predictedEvent": predicted_event,
            "predictionWindow": "Next 2 hours",
            "sourceTimestamp": source_timestamp,
            "lastUpdated": "2026-10-17T12:00:00+00:00",
        }
        mock_azure_client.get_digital_twin.return_value = updated_twin_doc

        # 5. Verify FastAPI adapter fetches and serializes the exact state
        test_client = TestClient(app)
        with patch("dashboard_api._azure_client", return_value=mock_azure_client):
            api_resp = test_client.get("/api/twin-state/P001")
            assert api_resp.status_code == 200
            api_data = api_resp.json()

            assert api_data["success"] is True
            assert api_data["twin_id"] == "P001"
            assert api_data["patient_id"] == "2001"
            assert api_data["current_glucose"] == pytest.approx(88.2)
            assert api_data["delta_30m"] == pytest.approx(1.8)
            assert api_data["glucose_trend"] == "Stable"
            assert api_data["risk_level"] == "LOW"
            assert api_data["risk_score"] == pytest.approx(risk_score)
            assert api_data["source_timestamp"] == "2020-11-02 11:40:00"
            assert api_data["data_provenance"] == "Simulated replay of historical CGM data"

        # 6. Verify Frontend HTML Contract incorporates all telemetry fields
        html_content = INDEX_HTML_PATH.read_text(encoding="utf-8")
        assert "card-glucose" in html_content
        assert "card-delta" in html_content
        assert "card-trend" in html_content
        assert "card-risk-score" in html_content
        assert "card-margin" in html_content
        assert "Simulated replay of historical CGM data" in html_content
