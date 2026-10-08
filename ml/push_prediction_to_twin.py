"""
Run the final glucose-spike model on a ShanghaiT2DM feature row
and push the prediction into Azure Digital Twins.

Run from Twin/:
    python ml/push_prediction_to_twin.py
    python ml/push_prediction_to_twin.py --patient-id 2001
    python ml/push_prediction_to_twin.py --patient-id 2001 --twin-id P001
"""

from pathlib import Path
import argparse
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from digital_twin.azure_client import get_client, update_patient_twin


V2_PATH = PROJECT_ROOT / "data" / "processed" / "shanghai_t2dm_features_v2.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_b_random_forest.joblib"

DEFAULT_TWIN_ID = "P001"
DEFAULT_SOURCE_PATIENT = "2001"
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


def glucose_trend(delta_30m: float) -> str:
    if delta_30m >= 10:
        return "Rising"
    if delta_30m <= -10:
        return "Falling"
    return "Stable"


def optional_float(value):
    if pd.isna(value):
        return None
    value = float(value)
    return value if pd.notna(value) else None


def choose_latest_row(df: pd.DataFrame, source_patient: str) -> pd.DataFrame:
    patient_df = df[
        df["patient_id"].astype(str).str.strip()
        == str(source_patient).strip()
    ].copy()

    if patient_df.empty:
        examples = (
            df["patient_id"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        raise ValueError(
            f"Source patient {source_patient!r} was not found. "
            f"Example patients: {examples[:10]}"
        )

    patient_df["timestamp"] = pd.to_datetime(
        patient_df["timestamp"],
        errors="coerce",
        format="mixed",
    )
    patient_df = patient_df.dropna(subset=["timestamp"])
    patient_df = patient_df.sort_values(["timestamp", "session_id"])

    return patient_df.tail(1).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default=DEFAULT_SOURCE_PATIENT)
    parser.add_argument("--twin-id", default=DEFAULT_TWIN_ID)
    args = parser.parse_args()

    print("=" * 60)
    print("ML -> AZURE DIGITAL TWIN INFERENCE BRIDGE")
    print("=" * 60)

    if not V2_PATH.exists():
        raise FileNotFoundError(f"V2 dataset not found:\n{V2_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    df = pd.read_csv(V2_PATH, low_memory=False)
    model = joblib.load(MODEL_PATH)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing model features:\n" + "\n".join(missing)
        )

    row_df = choose_latest_row(df, args.patient_id)
    row = row_df.iloc[0]

    X = row_df[FEATURES]

    risk_score = float(model.predict_proba(X)[0, 1])
    risk_level = "HIGH" if risk_score >= FINAL_THRESHOLD else "LOW"
    predicted_event = (
        "Glucose spike" if risk_level == "HIGH"
        else "No glucose spike"
    )

    current_glucose = float(row["glucose_t0"])
    delta_30m = float(row["delta_30m"])
    trend = glucose_trend(delta_30m)

    age_value = optional_float(row["age"])
    age = int(round(age_value)) if age_value is not None else None
    bmi = optional_float(row["bmi"])
    hba1c = optional_float(row["hba1c"])

    print()
    print("SOURCE DATA")
    print(f"Source patient:  {args.patient_id}")
    print(f"Session:         {row['session_id']}")
    print(f"Timestamp:       {row['timestamp']}")
    print(f"CGM:             {current_glucose:.2f} mg/dL")
    print(f"30m delta:       {delta_30m:.2f} mg/dL")
    print(f"Trend:           {trend}")

    print()
    print("MODEL OUTPUT")
    print(f"Risk score:      {risk_score:.4f}")
    print(f"Risk percentage: {risk_score * 100:.2f}%")
    print(f"Threshold:       {FINAL_THRESHOLD:.2f}")
    print(f"Risk level:      {risk_level}")
    print(f"Prediction:      {predicted_event}")
    print("Window:          Next 2 hours")

    print()
    print("Connecting to Azure Digital Twins...")

    client = get_client()
    client.get_digital_twin(args.twin_id)

    print(f"Connected. Twin {args.twin_id} exists.")
    print(f"Updating twin {args.twin_id}...")

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

    updated = client.get_digital_twin(args.twin_id)

    print()
    print("=" * 60)
    print("AZURE DIGITAL TWIN UPDATE VERIFIED")
    print("=" * 60)
    print(f"Twin ID:          {args.twin_id}")
    print(f"patientId:        {updated.get('patientId')}")
    print(f"currentGlucose:   {updated.get('currentGlucose')}")
    print(f"glucoseTrend:     {updated.get('glucoseTrend')}")
    print(f"riskScore:        {updated.get('riskScore')}")
    print(f"riskLevel:        {updated.get('riskLevel')}")
    print(f"predictedEvent:   {updated.get('predictedEvent')}")
    print(f"predictionWindow: {updated.get('predictionWindow')}")
    print(f"lastUpdated:      {updated.get('lastUpdated')}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
