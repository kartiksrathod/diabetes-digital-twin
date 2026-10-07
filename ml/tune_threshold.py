from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.base import clone


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")

TARGET = "spike_label"

META_COLUMNS = [
    "patient_id",
    "session_id",
    "session_number",
    "timestamp",
    "source_file",
    TARGET,
]

RANDOM_STATE = 42


def load_xy(path):
    df = pd.read_csv(path, low_memory=False)

    feature_columns = [
        c for c in df.columns
        if c not in META_COLUMNS
    ]

    X = df[feature_columns].copy()

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    ).astype(int)

    return X, y, feature_columns


def main():

    print("Loading data...")

    X_train, y_train, features = load_xy(TRAIN_PATH)
    X_val, y_val, _ = load_xy(VAL_PATH)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                features
            )
        ]
    )

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    print("\n========== PROBABILITY METRICS ==========")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\n========== THRESHOLD SEARCH ==========")

    results = []

    for threshold in np.arange(0.10, 0.91, 0.05):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        predicted_positive_pct = (
            predictions.mean() * 100
        )

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_positive_pct":
                predicted_positive_pct
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best = (
        results_df
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\n========== BEST F1 THRESHOLD ==========")

    print(
        f"Threshold: {best['threshold']:.2f}"
    )
    print(
        f"Precision: {best['precision']:.4f}"
    )
    print(
        f"Recall:    {best['recall']:.4f}"
    )
    print(
        f"F1:        {best['f1']:.4f}"
    )
    print(
        "Predicted positive rate: "
        f"{best['predicted_positive_pct']:.2f}%"
    )

    output = Path(
        "data/processed/rf_threshold_results.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved results to:\n{output}"
    )


if __name__ == "__main__":
    main()