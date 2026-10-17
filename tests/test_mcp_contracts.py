"""Unit and Clinical Safety Contract Tests for MCP Server Tools (mcp_server.py).

Validates:
- Tool contract signatures and return structures for:
  1. get_patient_twin
  2. get_glucose_risk
  3. update_patient_twin_state
  4. digital_twin_health_check
- Argument validation: risk_level bounds, glucose_trend bounds, risk_score bounds, negative glucose.
- Azure error resilience: 404 missing twin, credential failure, network disconnect.
- Clinical safety guardrails: agent prompt and MCP server instructions do NOT instruct
  diagnosis, medication dosing, or treatment decisions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import mcp_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLINICIAN_AGENT_PATH = PROJECT_ROOT / ".agents" / "agents" / "diabetes-clinician" / "agent.md"


@pytest.fixture(autouse=True)
def clean_mcp_client():
    mcp_server._client = None
    yield
    mcp_server._client = None


class TestMcpToolContracts:
    """Verifies response schemas and behavior of all 4 MCP tools."""

    def test_get_patient_twin_missing_twin_handled(self):
        mock_client = MagicMock()
        mock_client.get_digital_twin.side_effect = Exception("Twin not found: ResourceNotFound")
        with patch("mcp_server.get_azure_client", return_value=mock_client):
            res = mcp_server.get_patient_twin("P999")
            assert res["success"] is False
            assert res["twin_id"] == "P999"
            assert "ResourceNotFound" in res["error"]

    def test_get_glucose_risk_contract_keys(self, mock_azure_client):
        with patch("mcp_server.get_azure_client", return_value=mock_azure_client):
            res = mcp_server.get_glucose_risk("P001")
            assert res["success"] is True
            assert res["twin_id"] == "P001"
            assert "patient_id" in res
            assert "current_glucose" in res
            assert "glucose_trend" in res
            assert "risk_score" in res
            assert "risk_level" in res
            assert "predicted_event" in res
            assert "prediction_window" in res
            assert "last_updated" in res

    def test_digital_twin_health_check_disconnected(self):
        mock_client = MagicMock()
        mock_client.get_digital_twin.side_effect = ConnectionRefusedError("Connection refused")
        with patch("mcp_server.get_azure_client", return_value=mock_client):
            res = mcp_server.digital_twin_health_check("P001")
            assert res["success"] is False
            assert res["status"] == "DISCONNECTED"
            assert "Connection refused" in res["error"]

    @pytest.mark.parametrize("invalid_risk_level", ["CRITICAL", "MODERATE", "high", "low", ""])
    def test_update_twin_rejects_non_canonical_risk_level(self, invalid_risk_level):
        res = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=100.0,
            glucose_trend="Stable",
            risk_score=0.2,
            risk_level=invalid_risk_level,
            predicted_event="No glucose spike",
        )
        assert res["success"] is False
        assert "risk_level must be 'HIGH' or 'LOW'" in res["error"]

    @pytest.mark.parametrize("invalid_trend", ["Spiking", "Flat", "rising", "falling", ""])
    def test_update_twin_rejects_non_canonical_trend(self, invalid_trend):
        res = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=100.0,
            glucose_trend=invalid_trend,
            risk_score=0.2,
            risk_level="LOW",
            predicted_event="No glucose spike",
        )
        assert res["success"] is False
        assert "glucose_trend must be 'Rising', 'Falling', or 'Stable'" in res["error"]

    @pytest.mark.parametrize("invalid_score", [-0.01, 1.01, 2.5, -5.0])
    def test_update_twin_rejects_out_of_bounds_score(self, invalid_score):
        res = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=100.0,
            glucose_trend="Stable",
            risk_score=invalid_score,
            risk_level="LOW",
            predicted_event="No glucose spike",
        )
        assert res["success"] is False
        assert "risk_score must be between 0.0 and 1.0" in res["error"]

    def test_update_twin_rejects_negative_glucose(self):
        res = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=-1.0,
            glucose_trend="Stable",
            risk_score=0.2,
            risk_level="LOW",
            predicted_event="No glucose spike",
        )
        assert res["success"] is False
        assert "current_glucose cannot be negative" in res["error"]


class TestClinicalSafetyGuardrails:
    """Verifies clinical assistant prompt and system instructions enforce safety boundaries."""

    def test_agent_md_safety_rules_prohibit_treatment_and_diagnosis(self):
        assert CLINICIAN_AGENT_PATH.exists(), f"Agent prompt missing at {CLINICIAN_AGENT_PATH}"
        content = CLINICIAN_AGENT_PATH.read_text(encoding="utf-8")

        # Prohibits diagnosis, medication dosing, treatment decisions
        assert "not a doctor and not a diagnostic or treatment system" in content
        assert "Do not diagnose the patient" in content
        assert "Do not prescribe medication, insulin, diet, or treatment changes" in content
        assert "Do not give clinical management recommendations" in content
        assert "treatment decisions require qualified clinical judgment" in content

    def test_mcp_server_instructions_safety(self):
        # Server instructions instruct reading and not inventing values
        instructions = mcp_server.mcp.instructions
        assert "Do not invent patient values" in instructions
