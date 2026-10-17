"""Local FastAPI adapter for the GlycoTwin dashboard.

Place this file in the root of the existing Twin project, alongside
`digital_twin/`, `data/`, and `ml/`.

This is a hackathon/local-demo API, not a public or production service.
It reads the existing Azure Digital Twin, feature CSV, and trained model;
it does not change or retrain the model and never writes to Azure.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "shanghai_t2dm_features_v2.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_b_random_forest.joblib"
FINAL_THRESHOLD = 0.57

FEATURES = [
    "glucose_t0",
    "glucose_t_minus_15m",
    "glucose_t_minus_30m",
    "glucose_t_minus_45m",
    "glucose_t_minus_60m",
    "glucose_t_minus_75m",
    "glucose_t_minus_90m",
    "glucose_t_minus_105m",
    "glucose_t_minus_120m",
    "delta_15m",
    "delta_30m",
    "delta_60m",
    "delta_120m",
    "glucose_mean_1h",
    "glucose_std_1h",
    "glucose_min_1h",
    "glucose_max_1h",
    "glucose_mean_2h",
    "glucose_std_2h",
    "glucose_min_2h",
    "glucose_max_2h",
    "slope_30m",
    "slope_60m",
    "slope_120m",
    "age",
    "bmi",
    "diabetes_duration",
    "fasting_glucose",
    "postprandial_glucose",
    "hba1c",
    "meal_events_30m",
    "meal_events_60m",
    "meal_events_120m",
    "insulin_events_30m",
    "insulin_events_60m",
    "insulin_events_120m",
    "insulin_amount_30m",
    "insulin_amount_60m",
    "insulin_amount_120m",
    "medication_events_120m",
]

app = FastAPI(
    title="GlycoTwin Dashboard API",
    version="0.1.0",
    description="Local/hackathon adapter for Azure Digital Twins and historical replay charts.",
)

# This is deliberately scoped to local dashboard development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "null",  # permits testing from file:// if the browser sends a null origin
    ],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


def _azure_client():
    # Reuse the project's known-working Azure client configuration.
    try:
        from digital_twin.azure_client import get_client

        return get_client()
    except Exception as exc:  # keep internal stack traces out of API responses
        raise HTTPException(status_code=503, detail=f"Azure client initialization failed: {type(exc).__name__}") from exc


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    required = {"patient_id", "session_id", "timestamp", "glucose_t0", "delta_30m", *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Feature CSV is missing columns: " + ", ".join(missing))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"]).copy()
    return df


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _trend(delta_30m: float) -> str:
    if delta_30m >= 10:
        return "Rising"
    if delta_30m <= -10:
        return "Falling"
    return "Stable"


def _serializable_twin_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "success": True,
        "service": "glycotwin-dashboard-api",
        "mode": "local-hackathon-demo",
        "threshold": FINAL_THRESHOLD,
    }


@app.get("/api/twin-state/{twin_id}")
def twin_state(twin_id: str = "P001") -> dict[str, Any]:
    """Fetch the latest state stored in Azure Digital Twins.

    The API is read-only: it never mutates the Azure twin.
    """
    try:
        twin = _azure_client().get_digital_twin(twin_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Azure twin read failed: {type(exc).__name__}") from exc

    score_value = twin.get("riskScore")
    glucose_value = twin.get("currentGlucose")
    delta_value = twin.get("delta30m", twin.get("delta30M"))
    score = float(score_value) if score_value is not None else None
    glucose = float(glucose_value) if glucose_value is not None else None
    delta = float(delta_value) if delta_value is not None else None

    return {
        "success": True,
        "twin_id": twin_id,
        "patient_id": str(twin.get("patientId", "")),
        "current_glucose": glucose,
        "delta_30m": delta,
        "glucose_trend": twin.get("glucoseTrend"),
        "risk_score": score,
        "risk_level": twin.get("riskLevel") or ("HIGH" if score is not None and score >= FINAL_THRESHOLD else "LOW" if score is not None else None),
        "predicted_event": twin.get("predictedEvent"),
        "prediction_window": twin.get("predictionWindow", "Next 2 hours"),
        "last_updated": _serializable_twin_value(twin.get("lastUpdated")),
        # The existing twin schema does not reliably store the original CGM row timestamp.
        # Do not invent one; the UI can show last_updated when this is null.
        "source_timestamp": twin.get("sourceTimestamp", twin.get("source_timestamp")),
        "data_provenance": "Simulated replay of historical CGM data",
    }


@app.get("/api/replay-history")
def replay_history(
    patient_id: str = Query(default="2001", min_length=1, max_length=50),
    session_id: str = Query(default="2001_0_20201102", min_length=1, max_length=100),
    limit: int = Query(default=10, ge=2, le=300),
) -> dict[str, Any]:
    """Return sequential historical CGM rows and scores from the existing model.

    This endpoint recreates the chart data from the same feature CSV/model. It
    does not read the label column to make predictions, write to Azure, or
    claim the historical records are live sensor readings.
    """
    try:
        df = _load_dataset()
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model data: {type(exc).__name__}") from exc

    rows = df[
        (df["patient_id"].astype(str).str.strip() == patient_id.strip())
        & (df["session_id"].astype(str).str.strip() == session_id.strip())
    ].copy()

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No rows found for patient {patient_id!r}, session {session_id!r}.",
        )

    rows = rows.sort_values("timestamp").head(limit).copy()
    if len(rows) < 2:
        raise HTTPException(status_code=404, detail="At least two replay rows are required for charts.")

    missing_features = [column for column in FEATURES if column not in rows.columns]
    if missing_features:
        raise HTTPException(status_code=500, detail="Feature columns missing from dataset: " + ", ".join(missing_features))

    try:
        # Only the 40 model features are passed to the classifier; spike_label is not used.
        scores = model.predict_proba(rows[FEATURES])[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {type(exc).__name__}") from exc

    result_rows: list[dict[str, Any]] = []
    for (_, row), score_value in zip(rows.iterrows(), scores):
        glucose = float(row["glucose_t0"])
        delta = float(row["delta_30m"])
        score = float(score_value)
        level = "HIGH" if score >= FINAL_THRESHOLD else "LOW"
        timestamp = row["timestamp"]
        result_rows.append({
            "time": timestamp.strftime("%H:%M"),
            "timestamp": timestamp.isoformat(),
            "glucose": glucose,
            "delta30m": delta,
            "risk": score,
            "trend": _trend(delta),
            "level": level,
            "predicted_event": "Glucose spike" if level == "HIGH" else "No glucose spike",
        })

    return {
        "success": True,
        "patient_id": patient_id,
        "session_id": session_id,
        "data_provenance": "Historical ShanghaiT2DM rows replayed as simulated inputs",
        "threshold": FINAL_THRESHOLD,
        "prediction_window": "Next 2 hours",
        "count": len(result_rows),
        "rows": result_rows,
    }


@app.get("/api/system-status")
def system_status() -> dict[str, Any]:
    """Report what this local API can actually verify; MCP/Antigravity are separate processes."""
    azure_connected = False
    azure_error = None
    try:
        _azure_client().get_digital_twin("P001")
        azure_connected = True
    except Exception as exc:
        azure_error = type(exc).__name__

    return {
        "success": True,
        "azure_twin_reachable": azure_connected,
        "azure_error": azure_error,
        "feature_csv_exists": FEATURES_PATH.exists(),
        "model_file_exists": MODEL_PATH.exists(),
        "mcp_server": "Separate stdio server; not called directly by browser",
        "antigravity_agent": "Runs in Antigravity; browser does not directly query the agent",
        "replay_source": "Historical fixture; chart endpoint is read-only",
    }
