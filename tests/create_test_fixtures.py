"""Generator for deterministic, lightweight test fixtures for GlycoTwin.

Generates:
1. tests/fixtures/fixture_model_b.joblib: Lightweight sklearn Pipeline with imputer and 3-tree RF.
2. tests/fixtures/fixture_features_v2.csv: 20-row dataframe with canonical 40 features + identifiers.
3. tests/fixtures/fixture_{train,validation,test}.csv: Mutually disjoint patient splits.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

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


def generate_fixtures():
    np.random.seed(42)
    n_rows = 20

    data = {}
    for col in FEATURES:
        base = np.random.randn(n_rows)
        if col.startswith("glucose_") or "mean_" in col or "min_" in col or "max_" in col:
            data[col] = 100.0 + 10.0 * base
        elif "delta_" in col or "slope_" in col:
            data[col] = base * 0.5
        elif col == "age":
            data[col] = 55 + np.abs(np.round(5 * base))
        elif col == "bmi":
            data[col] = 26.0 + np.abs(2 * base)
        elif col == "diabetes_duration":
            data[col] = 8.0 + np.abs(base)
        elif col == "hba1c":
            data[col] = 7.0 + base * 0.5
        elif col in ("fasting_glucose", "postprandial_glucose"):
            data[col] = 110.0 + 15.0 * base
        elif "events_" in col:
            data[col] = np.abs(np.round(2 * base))
        elif "amount_" in col:
            data[col] = np.abs(2 * base)
        else:
            data[col] = np.abs(base)

    X = pd.DataFrame(data)

    # Include missing values to exercise the imputer contract.
    X.loc[2, "hba1c"] = np.nan
    X.loc[5, "glucose_t_minus_120m"] = np.nan
    X.loc[8, "insulin_amount_30m"] = np.nan

    # Deterministic binary labels with both classes represented.
    y = (
        (X["glucose_t0"].fillna(100) > 100)
        | (X["delta_30m"] > 0.4)
    ).astype(int)
    if len(np.unique(y)) < 2:
        y.iloc[0] = 0
        y.iloc[1] = 1

    # Train a small real sklearn pipeline for contract tests.
    # The explicit feature list preserves the expected 40-column order.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=3,
            max_depth=3,
            random_state=42,
        )),
    ])
    model.fit(X[FEATURES], y)
    joblib.dump(model, FIXTURES / "fixture_model_b.joblib")

    # Feature dataset includes the identifiers required by replay/API tests.
    metadata = pd.DataFrame({
        "patient_id": ["2001"] * n_rows,
        "session_id": ["2001_0_20201102"] * n_rows,
        "timestamp": pd.date_range(
            "2020-11-02 11:00:00", periods=n_rows, freq="5min"
        ).astype(str),
    })
    fixture_df = pd.concat([metadata, X], axis=1)
    fixture_df["spike_label"] = y
    fixture_df.to_csv(FIXTURES / "fixture_features_v2.csv", index=False)

    # Patient-disjoint split fixtures: no patient crosses a split boundary.
    for split_name, patient_ids in {
        "train": [f"TRAIN_{i:03d}" for i in range(10)],
        "validation": [f"VAL_{i:03d}" for i in range(10)],
        "test": [f"TEST_{i:03d}" for i in range(10)],
    }.items():
        split_df = pd.DataFrame({
            "patient_id": patient_ids,
            "session_id": [f"{pid}_sess" for pid in patient_ids],
            "timestamp": ["2020-11-02 12:00:00"] * len(patient_ids),
            "glucose_t0": [100.0] * len(patient_ids),
        })
        split_df.to_csv(FIXTURES / f"fixture_{split_name}.csv", index=False)


if __name__ == "__main__":
    generate_fixtures()
    print("Fixtures generated successfully.")
