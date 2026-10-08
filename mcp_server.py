"""
MCP server for the Type-2 Diabetes Digital Twin.

Exposes Azure Digital Twin P001 to an MCP host such as Antigravity.

Run from Twin/:

    python mcp_server.py

Development / Inspector:

    mcp dev mcp_server.py
"""

from typing import Optional
import sys
from pathlib import Path


# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# MCP
# ------------------------------------------------------------

from mcp.server import MCPServer


# ------------------------------------------------------------
# Azure Digital Twin
# ------------------------------------------------------------

from digital_twin.azure_client import (
    get_client,
    update_patient_twin,
)


# ------------------------------------------------------------
# Server
# ------------------------------------------------------------

mcp = MCPServer(
    "Diabetes Digital Twin",
    instructions=(
        "You are connected to an Azure Digital Twin for "
        "Type-2 Diabetes early glucose spike prediction. "
        "Use the available tools to inspect the patient's "
        "current glucose state, risk score, trend, and "
        "prediction. Do not invent patient values."
    ),
)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_TWIN_ID = "P001"


# ------------------------------------------------------------
# Azure client
# ------------------------------------------------------------

_client = None


def get_azure_client():
    """
    Lazily create the Azure Digital Twins client.

    This avoids connecting to Azure when the MCP server is
    merely being imported or inspected.
    """
    global _client

    if _client is None:
        _client = get_client()

    return _client


# ------------------------------------------------------------
# Tool 1: Get complete twin
# ------------------------------------------------------------

@mcp.tool()
def get_patient_twin(
    twin_id: str = DEFAULT_TWIN_ID,
) -> dict:
    """
    Read the complete current state of a patient Digital Twin.

    Example:
        get_patient_twin("P001")
    """

    try:
        client = get_azure_client()

        twin = client.get_digital_twin(
            twin_id
        )

        return {
            "success": True,
            "twin_id": twin_id,
            "twin": dict(twin),
        }

    except Exception as exc:
        return {
            "success": False,
            "twin_id": twin_id,
            "error": str(exc),
        }


# ------------------------------------------------------------
# Tool 2: Get glucose risk summary
# ------------------------------------------------------------

@mcp.tool()
def get_glucose_risk(
    twin_id: str = DEFAULT_TWIN_ID,
) -> dict:
    """
    Get the clinically relevant glucose prediction state
    from the Digital Twin.

    Returns:
        current glucose
        glucose trend
        risk score
        risk level
        predicted event
        prediction window
        last update time
    """

    try:
        client = get_azure_client()

        twin = client.get_digital_twin(
            twin_id
        )

        return {
            "success": True,
            "twin_id": twin_id,
            "patient_id": twin.get("patientId"),
            "current_glucose": twin.get("currentGlucose"),
            "glucose_trend": twin.get("glucoseTrend"),
            "risk_score": twin.get("riskScore"),
            "risk_level": twin.get("riskLevel"),
            "predicted_event": twin.get("predictedEvent"),
            "prediction_window": twin.get("predictionWindow"),
            "last_updated": twin.get("lastUpdated"),
        }

    except Exception as exc:
        return {
            "success": False,
            "twin_id": twin_id,
            "error": str(exc),
        }


# ------------------------------------------------------------
# Tool 3: Update twin state
# ------------------------------------------------------------

@mcp.tool()
def update_patient_twin_state(
    twin_id: str,
    patient_id: str,
    current_glucose: float,
    glucose_trend: str,
    risk_score: float,
    risk_level: str,
    predicted_event: str,
    prediction_window: str = "Next 2 hours",
    age: Optional[int] = None,
    bmi: Optional[float] = None,
    hba1c: Optional[float] = None,
) -> dict:
    """
    Update the prediction-related state of a patient Digital Twin.

    This is intentionally restricted to the fields used by the
    existing ML -> Azure pipeline.

    risk_level should normally be HIGH or LOW.
    glucose_trend should normally be Rising, Falling, or Stable.
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if risk_level not in {
        "HIGH",
        "LOW",
    }:
        return {
            "success": False,
            "error": (
                "risk_level must be 'HIGH' or 'LOW'."
            ),
        }

    if glucose_trend not in {
        "Rising",
        "Falling",
        "Stable",
    }:
        return {
            "success": False,
            "error": (
                "glucose_trend must be "
                "'Rising', 'Falling', or 'Stable'."
            ),
        }

    if not 0.0 <= risk_score <= 1.0:
        return {
            "success": False,
            "error": (
                "risk_score must be between 0.0 and 1.0."
            ),
        }

    if current_glucose < 0:
        return {
            "success": False,
            "error": (
                "current_glucose cannot be negative."
            ),
        }

    # --------------------------------------------------------
    # Azure update
    # --------------------------------------------------------

    try:
        client = get_azure_client()

        update_patient_twin(
            client,
            twin_id,
            patient_id=str(patient_id),
            age=age,
            bmi=bmi,
            hba1c=hba1c,
            current_glucose=float(current_glucose),
            glucose_trend=glucose_trend,
            risk_score=float(risk_score),
            risk_level=risk_level,
            predicted_event=predicted_event,
            prediction_window=prediction_window,
        )

        # Read back to verify the update.
        updated = client.get_digital_twin(
            twin_id
        )

        return {
            "success": True,
            "message": "Digital Twin updated successfully.",
            "twin_id": twin_id,
            "state": {
                "patient_id": updated.get("patientId"),
                "current_glucose": updated.get(
                    "currentGlucose"
                ),
                "glucose_trend": updated.get(
                    "glucoseTrend"
                ),
                "risk_score": updated.get(
                    "riskScore"
                ),
                "risk_level": updated.get(
                    "riskLevel"
                ),
                "predicted_event": updated.get(
                    "predictedEvent"
                ),
                "prediction_window": updated.get(
                    "predictionWindow"
                ),
                "last_updated": updated.get(
                    "lastUpdated"
                ),
            },
        }

    except Exception as exc:
        return {
            "success": False,
            "twin_id": twin_id,
            "error": str(exc),
        }


# ------------------------------------------------------------
# Tool 4: Health check
# ------------------------------------------------------------

@mcp.tool()
def digital_twin_health_check(
    twin_id: str = DEFAULT_TWIN_ID,
) -> dict:
    """
    Check whether the Azure Digital Twin is reachable.
    """

    try:
        client = get_azure_client()

        twin = client.get_digital_twin(
            twin_id
        )

        return {
            "success": True,
            "status": "CONNECTED",
            "twin_id": twin_id,
            "patient_id": twin.get("patientId"),
        }

    except Exception as exc:
        return {
            "success": False,
            "status": "DISCONNECTED",
            "twin_id": twin_id,
            "error": str(exc),
        }


# ------------------------------------------------------------
# Start server
# ------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()