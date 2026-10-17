"""Regression, Security, and Boundary Value Tests.

Verifies boundary behavior for risk scores, absence of credential leakage in error details,
safe handling of null fields, and accuracy of documented model evaluation metrics.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard_api import app, FINAL_THRESHOLD, _trend


@pytest.fixture
def client():
    return TestClient(app)


class TestRiskScoreBoundaries:
    """Tests risk score boundaries at 0.0, 0.57, 1.0, and trend calculations."""

    @pytest.mark.parametrize("score,expected_level", [
        (0.0, "LOW"),
        (0.5699, "LOW"),
        (0.57, "HIGH"),
        (0.5701, "HIGH"),
        (1.0, "HIGH"),
    ])
    def test_risk_level_threshold_boundary(self, client, score, expected_level):
        twin_payload = {
            "$dtId": "P001",
            "patientId": "2001",
            "currentGlucose": 120.0,
            "glucoseTrend": "Stable",
            "riskScore": score,
            "riskLevel": None,  # let backend evaluate boundary
            "predictedEvent": None,
            "lastUpdated": "2026-10-17T05:00:00+00:00",
        }
        mock_client = MagicMock()
        mock_client.get_digital_twin.return_value = twin_payload
        with patch("dashboard_api._azure_client", return_value=mock_client):
            response = client.get("/api/twin-state/P001")
            assert response.status_code == 200
            data = response.json()
            assert data["risk_score"] == pytest.approx(score, rel=1e-5)
            assert data["risk_level"] == expected_level

    @pytest.mark.parametrize("delta,expected_trend", [
        (10.0, "Rising"),
        (15.2, "Rising"),
        (-10.0, "Falling"),
        (-18.5, "Falling"),
        (9.9, "Stable"),
        (-9.9, "Stable"),
        (0.0, "Stable"),
    ])
    def test_trend_calculation_boundaries(self, delta, expected_trend):
        assert _trend(delta) == expected_trend


class TestSecurityAndDataSanitization:
    """Tests security invariants: no credential leakage in errors and sanitized outputs."""

    def test_no_credential_leakage_on_azure_exceptions(self, client):
        mock_client = MagicMock()
        # Simulate exception that might contain an internal token string
        mock_client.get_digital_twin.side_effect = Exception("Auth failed: Bearer secret_token_xyz123 expired")

        with patch("dashboard_api._azure_client", return_value=mock_client):
            response = client.get("/api/twin-state/P001")
            assert response.status_code == 502
            body_str = response.text
            # Only type(exc).__name__ should be returned, never exception message containing secrets
            assert "secret_token_xyz123" not in body_str
            assert "Bearer" not in body_str
            assert "Exception" in body_str

    def test_null_timestamps_and_deltas_do_not_crash(self, client):
        """Twin payload with missing optional properties must serialize cleanly as None."""
        sparse_twin = {
            "$dtId": "P001",
            "patientId": "2001",
            "currentGlucose": None,
            "glucoseTrend": None,
            "riskScore": None,
            "riskLevel": None,
            "predictedEvent": None,
            "lastUpdated": None,
        }
        mock_client = MagicMock()
        mock_client.get_digital_twin.return_value = sparse_twin
        with patch("dashboard_api._azure_client", return_value=mock_client):
            response = client.get("/api/twin-state/P001")
            assert response.status_code == 200
            data = response.json()
            assert data["current_glucose"] is None
            assert data["risk_score"] is None
            assert data["delta_30m"] is None
            assert data["source_timestamp"] is None


class TestModelMetricsVerification:
    """Verifies that final model metrics match recorded evaluation artifacts."""

    def test_final_test_results_match_ground_truth(self):
        from pathlib import Path
        results_file = Path(__file__).resolve().parent.parent / "data" / "processed" / "final_test_results.csv"
        assert results_file.exists()

        df = pd.read_csv(results_file)
        row = df.iloc[0]

        assert row["model"] == "Model B Random Forest"
        assert row["threshold"] == FINAL_THRESHOLD
        assert row["test_patients"] == 15
        assert row["test_rows"] == 16012

        # Model B test precision is ~43.45% (not Model A's 29.25%)
        assert row["precision"] == pytest.approx(0.4345, rel=1e-3)
        assert row["accuracy"] == pytest.approx(0.8189, rel=1e-3)
        assert row["recall"] == pytest.approx(0.3504, rel=1e-3)
        assert row["f1"] == pytest.approx(0.3879, rel=1e-3)
        assert row["roc_auc"] == pytest.approx(0.7632, rel=1e-3)
