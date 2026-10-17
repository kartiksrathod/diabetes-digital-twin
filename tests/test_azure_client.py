"""Unit Tests for Azure Digital Twins client wrapper (digital_twin/azure_client.py).

All Azure calls are strictly mocked to verify JSON Patch payload construction,
property types, and error handling without mutating cloud state.
"""

from unittest.mock import MagicMock, patch
import pytest

from digital_twin.azure_client import (
    ADT_ENDPOINT,
    PATIENT_TWIN_ID,
    get_client,
    get_patient_twin,
    update_patient_twin,
)


class TestAzureClientWrapper:
    """Tests for Azure client instantiation and methods."""

    def test_constants_and_endpoint(self):
        assert ADT_ENDPOINT == "https://glucosetwin-dt.api.krc.digitaltwins.azure.net"
        assert PATIENT_TWIN_ID == "P001"

    @patch("digital_twin.azure_client.DigitalTwinsClient")
    @patch("digital_twin.azure_client.DefaultAzureCredential")
    def test_get_client_instantiation(self, mock_cred_cls, mock_dt_cls):
        mock_cred = MagicMock()
        mock_cred_cls.return_value = mock_cred
        mock_dt = MagicMock()
        mock_dt_cls.return_value = mock_dt

        client = get_client()
        mock_cred_cls.assert_called_once()
        mock_dt_cls.assert_called_once_with(ADT_ENDPOINT, mock_cred)
        assert client == mock_dt

    def test_get_patient_twin_invokes_sdk(self, mock_azure_client, mock_twin_payload):
        twin = get_patient_twin(mock_azure_client, "P001")
        mock_azure_client.get_digital_twin.assert_called_once_with("P001")
        assert twin == mock_twin_payload

    def test_update_patient_twin_json_patch_format(self, mock_azure_client):
        """Verifies that update_patient_twin produces valid JSON Patch 'add' operations."""
        update_patient_twin(
            mock_azure_client,
            "P001",
            patient_id="2001",
            age=65,
            bmi=27.4,
            hba1c=7.1,
            current_glucose=145.0,
            glucose_trend="Rising",
            risk_score=0.62,
            risk_level="HIGH",
            predicted_event="Glucose spike",
            prediction_window="Next 2 hours",
            source_timestamp="2020-11-02T11:40:00",
            delta_30m=1.8,
        )

        mock_azure_client.update_digital_twin.assert_called_once()
        twin_id_called, patches = mock_azure_client.update_digital_twin.call_args[0]
        assert twin_id_called == "P001"

        # Check patch operations format
        patch_map = {p["path"]: p for p in patches}

        # Verify mandatory properties
        assert patch_map["/patientId"]["value"] == "2001"
        assert patch_map["/patientId"]["op"] == "add"
        assert patch_map["/currentGlucose"]["value"] == 145.0
        assert patch_map["/glucoseTrend"]["value"] == "Rising"
        assert patch_map["/riskScore"]["value"] == 0.62
        assert patch_map["/riskLevel"]["value"] == "HIGH"
        assert patch_map["/predictedEvent"]["value"] == "Glucose spike"
        assert patch_map["/predictionWindow"]["value"] == "Next 2 hours"
        assert "/lastUpdated" in patch_map

        # Verify optional clinical properties
        assert patch_map["/age"]["value"] == 65
        assert patch_map["/bmi"]["value"] == 27.4
        assert patch_map["/hba1c"]["value"] == 7.1

        # Verify source timestamp and 30m delta
        assert patch_map["/sourceTimestamp"]["value"] == "2020-11-02T11:40:00"
        assert patch_map["/delta30m"]["value"] == 1.8

    def test_update_patient_twin_omits_none_clinical_values(self, mock_azure_client):
        """When age, bmi, or hba1c are None, they should not be included in patch operations."""
        update_patient_twin(
            mock_azure_client,
            "P001",
            patient_id="2001",
            age=None,
            bmi=None,
            hba1c=None,
            current_glucose=110.0,
            glucose_trend="Stable",
            risk_score=0.15,
            risk_level="LOW",
            predicted_event="No glucose spike",
        )

        mock_azure_client.update_digital_twin.assert_called_once()
        _, patches = mock_azure_client.update_digital_twin.call_args[0]
        paths = {p["path"] for p in patches}

        assert "/age" not in paths
        assert "/bmi" not in paths
        assert "/hba1c" not in paths
        assert "/currentGlucose" in paths
        assert "/riskScore" in paths
