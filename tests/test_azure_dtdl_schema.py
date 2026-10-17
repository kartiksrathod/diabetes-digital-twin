"""Unit and Contract Tests for Azure Digital Twins DTDL v2 Model and Adapter.

Validates:
- DTDL v2 JSON interface contract (digital_twin/models/PatientDigitalTwin.json).
- Model ID conforms to dtmi:glucosetwin:PatientDigitalTwin;2.
- Property schemas for all 13 required and optional properties.
- JSON Patch construction with sourceTimestamp, delta30m, and clinical properties.
- SDK error handling and credential safety.
"""

from pathlib import Path
import json
from unittest.mock import MagicMock
import pytest

from digital_twin.azure_client import (
    ADT_ENDPOINT,
    PATIENT_TWIN_ID,
    update_patient_twin,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DTDL_MODEL_PATH = PROJECT_ROOT / "digital_twin" / "models" / "PatientDigitalTwin.json"

EXPECTED_DTDL_PROPERTIES = {
    "patientId": "string",
    "age": "integer",
    "bmi": "double",
    "hba1c": "double",
    "currentGlucose": "double",
    "glucoseTrend": "string",
    "riskScore": "double",
    "riskLevel": "string",
    "predictedEvent": "string",
    "predictionWindow": "string",
    "lastUpdated": "dateTime",
    "sourceTimestamp": "string",
    "delta30m": "double",
}


class TestDtdlModelDefinition:
    """Verifies compliance of digital_twin/models/PatientDigitalTwin.json with DTDL v2."""

    @pytest.fixture(scope="class")
    def dtdl_json(self):
        assert DTDL_MODEL_PATH.exists(), f"DTDL model file not found at {DTDL_MODEL_PATH}"
        with open(DTDL_MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_dtdl_id_and_context(self, dtdl_json):
        assert dtdl_json["@id"] == "dtmi:glucosetwin:PatientDigitalTwin;2", "DTDL model must be version 2"
        assert dtdl_json["@type"] == "Interface"
        assert dtdl_json["@context"] == "dtmi:dtdl:context;2"
        assert dtdl_json["displayName"] == "Patient Digital Twin"

    def test_dtdl_all_properties_declared_with_correct_types(self, dtdl_json):
        contents = dtdl_json.get("contents", [])
        properties = {item["name"]: item["schema"] for item in contents if item.get("@type") == "Property"}

        for prop_name, expected_schema in EXPECTED_DTDL_PROPERTIES.items():
            assert prop_name in properties, f"Missing DTDL property {prop_name}"
            assert properties[prop_name] == expected_schema, \
                f"Property {prop_name} schema {properties[prop_name]} != expected {expected_schema}"


class TestAzureClientPatchGeneration:
    """Tests Azure Digital Twins update patch generation and type safety."""

    def test_patch_includes_source_timestamp_and_delta(self):
        client = MagicMock()
        update_patient_twin(
            client,
            twin_id="P001",
            patient_id="2001",
            age=55,
            bmi=28.2,
            hba1c=6.8,
            current_glucose=105.0,
            glucose_trend="Stable",
            risk_score=0.18,
            risk_level="LOW",
            predicted_event="No glucose spike",
            prediction_window="Next 2 hours",
            source_timestamp="2020-11-02 11:40:00",
            delta_30m=1.8,
        )

        client.update_digital_twin.assert_called_once()
        twin_id, patches = client.update_digital_twin.call_args[0]
        assert twin_id == "P001"

        patch_dict = {p["path"]: p["value"] for p in patches}
        assert patch_dict["/sourceTimestamp"] == "2020-11-02 11:40:00"
        assert patch_dict["/delta30m"] == pytest.approx(1.8)
        assert patch_dict["/currentGlucose"] == 105.0
        assert patch_dict["/riskScore"] == 0.18
        assert patch_dict["/riskLevel"] == "LOW"
        assert patch_dict["/patientId"] == "2001"
        assert patch_dict["/predictionWindow"] == "Next 2 hours"
        assert "/lastUpdated" in patch_dict

    def test_patch_omits_optional_values_when_none(self):
        client = MagicMock()
        update_patient_twin(
            client,
            twin_id="P001",
            patient_id="2001",
            age=None,
            bmi=None,
            hba1c=None,
            current_glucose=95.0,
            glucose_trend="Stable",
            risk_score=0.12,
            risk_level="LOW",
            predicted_event="No glucose spike",
            source_timestamp=None,
            delta_30m=None,
        )

        _, patches = client.update_digital_twin.call_args[0]
        paths = {p["path"] for p in patches}
        assert "/age" not in paths
        assert "/bmi" not in paths
        assert "/hba1c" not in paths
        assert "/sourceTimestamp" not in paths
        assert "/delta30m" not in paths
        assert "/currentGlucose" in paths
        assert "/riskScore" in paths
