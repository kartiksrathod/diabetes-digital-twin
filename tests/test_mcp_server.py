"""Unit Tests for MCP Server Tools (mcp_server.py).

All Azure interactions are mocked to test tool logic, parameter validation,
error containment, and response structures safely.
"""

from unittest.mock import MagicMock, patch
import pytest

import mcp_server


@pytest.fixture(autouse=True)
def reset_mcp_client():
    """Reset the cached _client in mcp_server between tests."""
    mcp_server._client = None
    yield
    mcp_server._client = None


class TestMcpToolRegistration:
    """Verifies tool registration on the MCP server instance."""

    def test_registered_tool_names(self):
        # In fastmcp / MCPServer, registered tools can be accessed or inspected
        assert hasattr(mcp_server, "get_patient_twin")
        assert hasattr(mcp_server, "get_glucose_risk")
        assert hasattr(mcp_server, "update_patient_twin_state")
        assert hasattr(mcp_server, "digital_twin_health_check")


class TestGetPatientTwinTool:
    """Tests for get_patient_twin tool."""

    def test_get_patient_twin_success(self, mock_azure_client, mock_twin_payload):
        with patch("mcp_server.get_azure_client", return_value=mock_azure_client):
            result = mcp_server.get_patient_twin("P001")
            assert result["success"] is True
            assert result["twin_id"] == "P001"
            assert result["twin"]["patientId"] == "2001"
            assert result["twin"]["currentGlucose"] == 163.8

    def test_get_patient_twin_error_handling(self):
        mock_client = MagicMock()
        mock_client.get_digital_twin.side_effect = RuntimeError("Azure ADT down")
        with patch("mcp_server.get_azure_client", return_value=mock_client):
            result = mcp_server.get_patient_twin("P001")
            assert result["success"] is False
            assert result["twin_id"] == "P001"
            assert "Azure ADT down" in result["error"]


class TestGetGlucoseRiskTool:
    """Tests for get_glucose_risk tool."""

    def test_get_glucose_risk_success(self, mock_azure_client):
        with patch("mcp_server.get_azure_client", return_value=mock_azure_client):
            result = mcp_server.get_glucose_risk("P001")
            assert result["success"] is True
            assert result["twin_id"] == "P001"
            assert result["patient_id"] == "2001"
            assert result["current_glucose"] == 163.8
            assert result["glucose_trend"] == "Rising"
            assert result["risk_score"] == pytest.approx(0.6775, rel=1e-3)
            assert result["risk_level"] == "HIGH"
            assert result["predicted_event"] == "Glucose spike"
            assert result["prediction_window"] == "Next 2 hours"
            assert "last_updated" in result

    def test_get_glucose_risk_handles_exception(self):
        mock_client = MagicMock()
        mock_client.get_digital_twin.side_effect = ConnectionResetError("Lost socket")
        with patch("mcp_server.get_azure_client", return_value=mock_client):
            result = mcp_server.get_glucose_risk("P001")
            assert result["success"] is False
            assert "Lost socket" in result["error"]


class TestUpdatePatientTwinStateTool:
    """Tests for update_patient_twin_state tool validation and execution."""

    def test_validation_rejects_invalid_risk_level(self):
        result = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=120.0,
            glucose_trend="Stable",
            risk_score=0.5,
            risk_level="CRITICAL",  # Invalid, must be HIGH or LOW
            predicted_event="Glucose spike",
        )
        assert result["success"] is False
        assert "risk_level must be 'HIGH' or 'LOW'" in result["error"]

    def test_validation_rejects_invalid_trend(self):
        result = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=120.0,
            glucose_trend="Spiking",  # Invalid, must be Rising, Falling, or Stable
            risk_score=0.5,
            risk_level="HIGH",
            predicted_event="Glucose spike",
        )
        assert result["success"] is False
        assert "glucose_trend must be 'Rising', 'Falling', or 'Stable'" in result["error"]

    def test_validation_rejects_out_of_bounds_risk_score(self):
        result = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=120.0,
            glucose_trend="Stable",
            risk_score=1.5,  # Out of [0, 1]
            risk_level="HIGH",
            predicted_event="Glucose spike",
        )
        assert result["success"] is False
        assert "risk_score must be between 0.0 and 1.0" in result["error"]

    def test_validation_rejects_negative_glucose(self):
        result = mcp_server.update_patient_twin_state(
            twin_id="P001",
            patient_id="2001",
            current_glucose=-10.0,
            glucose_trend="Stable",
            risk_score=0.3,
            risk_level="LOW",
            predicted_event="No glucose spike",
        )
        assert result["success"] is False
        assert "current_glucose cannot be negative" in result["error"]

    def test_valid_update_executes_and_reads_back(self, mock_azure_client, mock_twin_payload):
        with patch("mcp_server.get_azure_client", return_value=mock_azure_client), \
             patch("mcp_server.update_patient_twin") as mock_update:

            result = mcp_server.update_patient_twin_state(
                twin_id="P001",
                patient_id="2001",
                current_glucose=140.0,
                glucose_trend="Rising",
                risk_score=0.65,
                risk_level="HIGH",
                predicted_event="Glucose spike",
            )

            assert result["success"] is True
            assert result["message"] == "Digital Twin updated successfully."
            mock_update.assert_called_once()
            mock_azure_client.get_digital_twin.assert_called_once_with("P001")


class TestDigitalTwinHealthCheckTool:
    """Tests for digital_twin_health_check tool."""

    def test_health_check_connected(self, mock_azure_client):
        with patch("mcp_server.get_azure_client", return_value=mock_azure_client):
            result = mcp_server.digital_twin_health_check("P001")
            assert result["success"] is True
            assert result["status"] == "CONNECTED"
            assert result["patient_id"] == "2001"

    def test_health_check_disconnected_on_failure(self):
        mock_client = MagicMock()
        mock_client.get_digital_twin.side_effect = TimeoutError("ADT timed out")
        with patch("mcp_server.get_azure_client", return_value=mock_client):
            result = mcp_server.digital_twin_health_check("P001")
            assert result["success"] is False
            assert result["status"] == "DISCONNECTED"
            assert "ADT timed out" in result["error"]
