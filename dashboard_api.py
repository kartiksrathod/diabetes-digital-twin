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
import threading
import time

from pydantic import BaseModel, Field

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
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
    allow_methods=["GET", "POST", "OPTIONS"],
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
    limit: int = Query(default=20, ge=2, le=300),
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
        "replay_source": "Historical fixture; automatic replay writes simulated updates to Azure when started",
    }


# --- Automatic historical replay controller ---------------------------------
# This controller is process-local and intended for a single local/hackathon
# demo instance. It writes simulated historical rows to the configured Azure
# Digital Twin; it does not represent live wearable telemetry.

class ReplayStartRequest(BaseModel):
    patient_id: str = Field(default="2001", min_length=1, max_length=50)
    session_id: str = Field(default="2001_0_20201102", min_length=1, max_length=100)
    twin_id: str = Field(default="P001", min_length=1, max_length=100)
    limit: int = Field(default=20, ge=2, le=300)
    delay_seconds: float = Field(default=2.0, ge=0, le=3600)
    reset_cursor: bool = Field(default=False)


class ReplayController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rows: pd.DataFrame | None = None
        self._patient_id = "2001"
        self._session_id = "2001_0_20201102"
        self._twin_id = "P001"
        self._delay_seconds = 2.0
        self._index = 0
        self._running = False
        self._busy = False
        self._last_error: str | None = None
        self._last_record: dict[str, Any] | None = None
        self._completed = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = 0 if self._rows is None else len(self._rows)
            return {
                "success": True,
                "running": self._running,
                "busy": self._busy,
                "paused": not self._running and total > 0 and not self._completed,
                "completed": self._completed,
                "patient_id": self._patient_id,
                "session_id": self._session_id,
                "twin_id": self._twin_id,
                "current_index": self._index,
                "total_rows": total,
                "processed_rows": self._index,
                "delay_seconds": self._delay_seconds,
                "threshold": FINAL_THRESHOLD,
                "prediction_window": "Next 2 hours",
                "data_provenance": "Simulated replay of historical CGM data",
                "last_error": self._last_error,
                "last_record": self._last_record,
            }

    def start(self, request: ReplayStartRequest) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise HTTPException(status_code=409, detail="Replay is already running.")
            if self._busy:
                raise HTTPException(status_code=409, detail="A replay row is currently being processed.")
            if self._worker is not None and self._worker.is_alive():
                raise HTTPException(status_code=409, detail="The previous replay worker is still stopping; retry shortly.")

        try:
            df = _load_dataset()
            _load_model()  # fail early rather than starting a worker that cannot infer
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load replay data/model: {type(exc).__name__}") from exc

        rows = df[
            (df["patient_id"].astype(str).str.strip() == request.patient_id.strip())
            & (df["session_id"].astype(str).str.strip() == request.session_id.strip())
        ].sort_values("timestamp").head(request.limit).reset_index(drop=True).copy()
        if len(rows) < 2:
            raise HTTPException(status_code=404, detail="At least two replay rows are required for automatic replay.")

        # Confirm Azure access and twin identity before starting the background worker.
        try:
            _azure_client().get_digital_twin(request.twin_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Azure twin read failed: {type(exc).__name__}") from exc

        with self._lock:
            if self._running or self._busy:
                raise HTTPException(status_code=409, detail="Replay state changed while starting; retry the request.")
            can_resume = (
                self._rows is not None
                and not request.reset_cursor
                and not self._completed
                and self._patient_id == request.patient_id.strip()
                and self._session_id == request.session_id.strip()
                and self._twin_id == request.twin_id.strip()
                and len(self._rows) == len(rows)
            )
            self._rows = rows
            self._patient_id = request.patient_id.strip()
            self._session_id = request.session_id.strip()
            self._twin_id = request.twin_id.strip()
            self._delay_seconds = float(request.delay_seconds)
            if not can_resume:
                self._index = 0
                self._last_error = None
                self._last_record = None
                self._completed = False
            self._stop_event.clear()
            self._running = True
            self._worker = threading.Thread(target=self._run, name="glycotwin-replay", daemon=True)
            self._worker.start()
        return self.status()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if not self._running:
                    return
                total = 0 if self._rows is None else len(self._rows)
                if self._index >= total:
                    self._running = False
                    self._completed = True
                    return
                delay = self._delay_seconds

            ok = self._process_next()
            if not ok:
                with self._lock:
                    self._running = False
                return
            with self._lock:
                if not self._running or self._completed:
                    return
            if self._stop_event.wait(delay):
                return

    def _process_next(self) -> bool:
        with self._lock:
            if self._rows is None:
                raise HTTPException(status_code=409, detail="Replay has not been initialized. Start replay first.")
            if self._busy:
                raise HTTPException(status_code=409, detail="A replay row is already being processed.")
            if self._index >= len(self._rows):
                self._running = False
                self._completed = True
                return False
            row = self._rows.iloc[self._index].copy()
            row_index = self._index
            self._busy = True

        try:
            model = _load_model()
            score = float(model.predict_proba(row[FEATURES].to_frame().T)[0, 1])
            level = "HIGH" if score >= FINAL_THRESHOLD else "LOW"
            current_glucose = float(row["glucose_t0"])
            delta_30m = float(row["delta_30m"])
            trend = _trend(delta_30m)
            age_value = row.get("age")
            bmi_value = row.get("bmi")
            hba1c_value = row.get("hba1c")
            age = int(round(float(age_value))) if pd.notna(age_value) else None
            bmi = float(bmi_value) if pd.notna(bmi_value) else None
            hba1c = float(hba1c_value) if pd.notna(hba1c_value) else None
            source_timestamp = row["timestamp"].isoformat()
            client = _azure_client()
            from digital_twin.azure_client import update_patient_twin
            update_patient_twin(
                client,
                self._twin_id,
                patient_id=str(self._patient_id),
                age=age,
                bmi=bmi,
                hba1c=hba1c,
                current_glucose=current_glucose,
                glucose_trend=trend,
                risk_score=score,
                risk_level=level,
                predicted_event="Glucose spike" if level == "HIGH" else "No glucose spike",
                prediction_window="Next 2 hours",
                source_timestamp=source_timestamp,
                delta_30m=delta_30m,
            )
            record = {
                "index": row_index,
                "timestamp": source_timestamp,
                "current_glucose": current_glucose,
                "delta_30m": delta_30m,
                "glucose_trend": trend,
                "risk_score": score,
                "risk_level": level,
                "predicted_event": "Glucose spike" if level == "HIGH" else "No glucose spike",
                "prediction_window": "Next 2 hours",
                "data_provenance": "Simulated replay of historical CGM data",
            }
            with self._lock:
                # Only advance after inference and Azure update both succeed.
                self._index = row_index + 1
                self._last_record = record
                self._last_error = None
                if self._rows is not None and self._index >= len(self._rows):
                    self._completed = True
                    self._running = False
            return True
        except HTTPException as exc:
            with self._lock:
                self._last_error = str(exc.detail)
            return False
        except Exception as exc:
            with self._lock:
                self._last_error = f"Replay row {row_index} failed: {type(exc).__name__}"
            return False
        finally:
            with self._lock:
                self._busy = False

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._running = False
            self._stop_event.set()
            worker = self._worker
        if worker is not None and worker.is_alive() and worker != threading.current_thread():
            worker.join(timeout=3.0)
        return self.status()

    def step(self) -> dict[str, Any]:
        with self._lock:
            if self._running or self._busy:
                raise HTTPException(status_code=409, detail="Pause replay and wait for in-flight operations before stepping.")
            if self._rows is None:
                raise HTTPException(status_code=409, detail="Replay has not been initialized. Start replay first.")
            if self._completed or self._index >= len(self._rows):
                raise HTTPException(status_code=409, detail="Replay is complete. Reset the cursor to replay again.")
        ok = self._process_next()
        if not ok:
            state = self.status()
            raise HTTPException(status_code=502, detail=state.get("last_error") or "Replay row failed.")
        return self.status()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise HTTPException(status_code=409, detail="Pause replay and wait for the current row to finish before resetting.")
            worker = self._worker
        if worker is not None and worker.is_alive() and worker != threading.current_thread():
            worker.join(timeout=3.0)
        with self._lock:
            if self._running or self._busy:
                raise HTTPException(status_code=409, detail="Pause replay and wait for the current row to finish before resetting.")
            if self._rows is None:
                raise HTTPException(status_code=409, detail="Replay has not been initialized. Start replay first.")
            self._index = 0
            self._last_error = None
            self._last_record = None
            self._completed = False
            self._stop_event.clear()
        return self.status()



_replay_controller = ReplayController()


@app.post("/api/replay/start")
def replay_start(request: ReplayStartRequest = Body(default_factory=ReplayStartRequest)) -> dict[str, Any]:
    return _replay_controller.start(request)


@app.post("/api/replay/pause")
def replay_pause() -> dict[str, Any]:
    return _replay_controller.pause()


@app.post("/api/replay/step")
def replay_step() -> dict[str, Any]:
    return _replay_controller.step()


@app.post("/api/replay/reset")
def replay_reset() -> dict[str, Any]:
    return _replay_controller.reset()


@app.get("/api/replay/status")
def replay_status() -> dict[str, Any]:
    return _replay_controller.status()
