"""
Simulated near-real-time replay for the glucose Digital Twin.

This replays real ShanghaiT2DM feature rows as if they were arriving
from a wearable/CGM stream.

Run from Twin/:

    python ml/replay_stream.py

Optional:

    python ml/replay_stream.py --patient-id 2001 --rows 20 --delay 2
    python ml/replay_stream.py --patient-id 2001 --session-id 2001_1_20201117 --rows 30 --delay 1

The model threshold is frozen at 0.57.
The spike_label column is NEVER used for prediction.
"""

from pathlib import Path
import argparse
import sys
import time

import joblib
import pandas as pd


# ------------------------------------------------------------
# Project import
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from digital_twin.azure_client import (
    get_client,
    update_patient_twin,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

V2_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shanghai_t2dm_features_v2.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_b_random_forest.joblib"
)


# ------------------------------------------------------------
# Frozen model configuration
# ------------------------------------------------------------

FINAL_THRESHOLD = 0.57

DEFAULT_PATIENT = "2001"
DEFAULT_TWIN_ID = "P001"
DEFAULT_DELAY = 2.0
DEFAULT_ROWS = 20


# ------------------------------------------------------------
# Model features
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def glucose_trend(delta_30m: float) -> str:
    if delta_30m >= 10:
        return "Rising"

    if delta_30m <= -10:
        return "Falling"

    return "Stable"


def clean_float(value):
    if pd.isna(value):
        return None

    value = float(value)

    if pd.isna(value):
        return None

    return value


def get_rows(
    df: pd.DataFrame,
    patient_id: str,
    session_id: str | None,
) -> pd.DataFrame:

    rows = df[
        df["patient_id"].astype(str).str.strip()
        == str(patient_id).strip()
    ].copy()

    if rows.empty:
        raise ValueError(
            f"Patient {patient_id!r} was not found."
        )

    if session_id is not None:
        rows = rows[
            rows["session_id"].astype(str).str.strip()
            == str(session_id).strip()
        ].copy()

        if rows.empty:
            sessions = (
                df[
                    df["patient_id"].astype(str).str.strip()
                    == str(patient_id).strip()
                ]["session_id"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"Session {session_id!r} was not found for "
                f"patient {patient_id!r}.\n"
                f"Available sessions: {sessions}"
            )

    rows["timestamp"] = pd.to_datetime(
        rows["timestamp"],
        errors="coerce",
        format="mixed",
    )

    rows = rows.dropna(
        subset=["timestamp"]
    )

    rows = rows.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    return rows


def main():

    parser = argparse.ArgumentParser(
        description="Replay real CGM feature rows into Azure Digital Twins."
    )

    parser.add_argument(
        "--patient-id",
        default=DEFAULT_PATIENT,
    )

    parser.add_argument(
        "--session-id",
        default=None,
    )

    parser.add_argument(
        "--twin-id",
        default=DEFAULT_TWIN_ID,
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Number of sequential rows to replay.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between replayed readings.",
    )

    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0.")

    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")

    print("=" * 64)
    print("SIMULATED REAL-TIME GLUCOSE STREAM")
    print("=" * 64)

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    if not V2_PATH.exists():
        raise FileNotFoundError(
            f"Missing V2 dataset:\n{V2_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing final model:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        V2_PATH,
        low_memory=False,
    )

    model = joblib.load(
        MODEL_PATH
    )

    missing_features = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Select stream
    # --------------------------------------------------------

    stream = get_rows(
        df,
        args.patient_id,
        args.session_id,
    )

    if len(stream) < args.rows:
        print(
            f"Requested {args.rows} rows but only "
            f"{len(stream)} are available."
        )

    stream = stream.head(
        args.rows
    ).copy()

    print()
    print("STREAM CONFIGURATION")
    print(f"Source patient:  {args.patient_id}")
    print(f"Twin ID:         {args.twin_id}")
    print(f"Session:         {stream['session_id'].iloc[0]}")
    print(f"Rows to replay:  {len(stream)}")
    print(f"Replay delay:    {args.delay:.1f} seconds")
    print(f"Threshold:       {FINAL_THRESHOLD:.2f}")

    print()
    print("Connecting to Azure Digital Twins...")

    client = get_client()

    # Verify the target twin exists before starting.
    client.get_digital_twin(
        args.twin_id
    )

    print(
        f"Twin {args.twin_id} is available."
    )

    print()
    print("-" * 64)
    print("REPLAY STARTED")
    print("-" * 64)

    for index, (_, row) in enumerate(
        stream.iterrows(),
        start=1,
    ):

        X = row[
            FEATURES
        ].to_frame().T

        # ----------------------------------------------------
        # ML inference
        # ----------------------------------------------------

        risk_score = float(
            model.predict_proba(X)[0, 1]
        )

        risk_level = (
            "HIGH"
            if risk_score >= FINAL_THRESHOLD
            else "LOW"
        )

        predicted_event = (
            "Glucose spike"
            if risk_level == "HIGH"
            else "No glucose spike"
        )

        current_glucose = float(
            row["glucose_t0"]
        )

        delta_30m = float(
            row["delta_30m"]
        )

        trend = glucose_trend(
            delta_30m
        )

        age_value = clean_float(
            row["age"]
        )

        age = (
            int(round(age_value))
            if age_value is not None
            else None
        )

        bmi = clean_float(
            row["bmi"]
        )

        hba1c = clean_float(
            row["hba1c"]
        )

        source_timestamp = row[
            "timestamp"
        ]

        # ----------------------------------------------------
        # Display incoming reading
        # ----------------------------------------------------

        print()
        print(
            f"[{index:02d}/{len(stream):02d}] "
            f"Incoming CGM reading"
        )

        print(
            f"  Source time: {source_timestamp}"
        )

        print(
            f"  CGM:        {current_glucose:.2f} mg/dL"
        )

        print(
            f"  30m delta:  {delta_30m:+.2f} mg/dL"
        )

        print(
            f"  Trend:      {trend}"
        )

        print(
            f"  Risk score: {risk_score:.4f} "
            f"({risk_score * 100:.2f}%)"
        )

        print(
            f"  Decision:   {risk_level} "
            f"-> {predicted_event}"
        )

        # ----------------------------------------------------
        # Update Azure Digital Twin
        # ----------------------------------------------------

        update_patient_twin(
            client,
            args.twin_id,
            patient_id=str(args.patient_id),
            age=age,
            bmi=bmi,
            hba1c=hba1c,
            current_glucose=current_glucose,
            glucose_trend=trend,
            risk_score=risk_score,
            risk_level=risk_level,
            predicted_event=predicted_event,
            prediction_window="Next 2 hours",
        )

        print(
            f"  Azure:      Pushed to twin {args.twin_id}"
        )

        # ----------------------------------------------------
        # Read back to prove update
        # ----------------------------------------------------

        updated = client.get_digital_twin(
            args.twin_id
        )

        print(
            f"  Twin state: "
            f"glucose={updated.get('currentGlucose')} | "
            f"risk={updated.get('riskLevel')} | "
            f"score={updated.get('riskScore')}"
        )

        # ----------------------------------------------------
        # Wait before next simulated reading
        # ----------------------------------------------------

        if index < len(stream):
            time.sleep(
                args.delay
            )

    print()
    print("-" * 64)
    print("REPLAY COMPLETE")
    print("-" * 64)

    print(
        f"Processed {len(stream)} real historical rows."
    )

    print(
        f"Final Azure Digital Twin: {args.twin_id}"
    )

    print(
        "The historical readings were replayed as a simulated "
        "near-real-time stream."
    )


if __name__ == "__main__":
    main()