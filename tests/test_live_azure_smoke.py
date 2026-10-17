"""Optional Live Read-Only Smoke Tests against Azure Digital Twins.

This module is strictly isolated and marked with @pytest.mark.live_azure.
It is NOT executed during standard local unit test runs. It performs strictly
read-only queries and NEVER writes, patches, or mutates cloud state.
"""

import pytest
from digital_twin.azure_client import get_client, PATIENT_TWIN_ID
import mcp_server


@pytest.mark.live_azure
class TestLiveAzureReadOnlySmoke:
    """Read-only smoke tests against real Azure Digital Twins instance."""

    def test_live_azure_digital_twin_fetch_p001(self):
        try:
            client = get_client()
            twin = client.get_digital_twin(PATIENT_TWIN_ID)
        except Exception as exc:
            pytest.skip(f"Live Azure credentials not available or network error: {type(exc).__name__}")

        assert twin["$dtId"] == PATIENT_TWIN_ID
        assert twin["patientId"] == "2001"
        assert "currentGlucose" in twin
        assert "riskScore" in twin
        assert 0.0 <= float(twin["riskScore"]) <= 1.0
        assert twin["predictionWindow"] == "Next 2 hours"
        assert twin["$metadata"]["$model"] == "dtmi:glucosetwin:PatientDigitalTwin;1"

    def test_live_mcp_health_check(self):
        result = mcp_server.digital_twin_health_check(PATIENT_TWIN_ID)
        if not result.get("success"):
            pytest.skip(f"Live Azure health check returned disconnected: {result.get('error')}")

        assert result["success"] is True
        assert result["status"] == "CONNECTED"
        assert result["patient_id"] == "2001"
