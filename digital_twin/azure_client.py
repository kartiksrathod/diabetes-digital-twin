"""
Reusable Azure Digital Twins client for the glucose Digital Twin project.
"""

from datetime import datetime, timezone
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.digitaltwins.core import DigitalTwinsClient


ADT_ENDPOINT = "https://glucosetwin-dt.api.krc.digitaltwins.azure.net"
PATIENT_TWIN_ID = "P001"


def get_client() -> DigitalTwinsClient:
    credential = DefaultAzureCredential()
    return DigitalTwinsClient(ADT_ENDPOINT, credential)


def update_patient_twin(
    client: DigitalTwinsClient,
    twin_id: str,
    *,
    patient_id: str,
    age: int | None,
    bmi: float | None,
    hba1c: float | None,
    current_glucose: float,
    glucose_trend: str,
    risk_score: float,
    risk_level: str,
    predicted_event: str,
    prediction_window: str = "Next 2 hours",
) -> None:
    """
    Update the patient twin using JSON Patch.
    risk_score is stored as probability in [0, 1].
    """

    now = datetime.now(timezone.utc).isoformat()

    updates: list[dict[str, Any]] = [
        {"op": "add", "path": "/patientId", "value": str(patient_id)},
        {"op": "add", "path": "/currentGlucose", "value": float(current_glucose)},
        {"op": "add", "path": "/glucoseTrend", "value": str(glucose_trend)},
        {"op": "add", "path": "/riskScore", "value": float(risk_score)},
        {"op": "add", "path": "/riskLevel", "value": str(risk_level)},
        {"op": "add", "path": "/predictedEvent", "value": str(predicted_event)},
        {"op": "add", "path": "/predictionWindow", "value": str(prediction_window)},
        {"op": "add", "path": "/lastUpdated", "value": now},
    ]

    if age is not None:
        updates.append({"op": "add", "path": "/age", "value": int(age)})

    if bmi is not None:
        updates.append({"op": "add", "path": "/bmi", "value": float(bmi)})

    if hba1c is not None:
        updates.append({"op": "add", "path": "/hba1c", "value": float(hba1c)})

    client.update_digital_twin(twin_id, updates)


def get_patient_twin(
    client: DigitalTwinsClient,
    twin_id: str,
):
    return client.get_digital_twin(twin_id)

