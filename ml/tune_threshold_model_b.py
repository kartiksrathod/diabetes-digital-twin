from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

V2_PATH = Path(
    "data/processed/shanghai_t2dm_features_v2.csv"
)

VAL_PATH = Path(
    "data/processed/validation.csv"
)

MODEL_PATH = Path(
    "data/processed/model_b_random_forest.joblib"
)

OUTPUT_PATH = Path(
    "data/processed/model_b_threshold_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "MODEL B THRESHOLD TUNING"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    if not V2_PATH.exists():
        raise FileNotFoundError(
            f"Missing V2 dataset: {V2_PATH}"
        )

    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing validation split: {VAL_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing Model B: {MODEL_PATH}"
        )

    df = pd.read_csv(
        V2_PATH,
        low_memory=False,
    )

    validation_split = pd.read_csv(
        VAL_PATH,
        low_memory=False,
    )

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Patient IDs
    # --------------------------------------------------------

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
    )

    validation_split["patient_id"] = (
        validation_split["patient_id"]
        .astype(str)
        .str.strip()
    )

    validation_patients = set(
        validation_split["patient_id"]
        .dropna()
        .unique()
    )

    val_df = df[
        df["patient_id"].isin(
            validation_patients
        )
    ].copy()

    val_df["spike_label"] = pd.to_numeric(
        val_df["spike_label"],
        errors="coerce",
    )

    val_df = val_df.dropna(
        subset=["spike_label"]
    ).copy()

    val_df["spike_label"] = (
        val_df["spike_label"]
        .astype(int)
    )

    X_val = val_df[
        FEATURES
    ]

    y_val = val_df[
        "spike_label"
    ].to_numpy()

    print()
    print(
        f"Validation patients: "
        f"{len(validation_patients)}"
    )

    print(
        f"Validation rows:     "
        f"{len(val_df):,}"
    )

    print(
        f"Actual spike rate:   "
        f"{y_val.mean()*100:.2f}%"
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities,
    )

    print()
    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Threshold search
    # --------------------------------------------------------

    thresholds = np.arange(
        0.20,
        0.81,
        0.01,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_val,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "predicted_positive_pct": (
                    predictions.mean() * 100
                ),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best = (
        results_df
        .sort_values(
            by=[
                "f1",
                "precision",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    # --------------------------------------------------------
    # Print useful range
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "TOP THRESHOLDS"
    )
    print(
        "============================================================"
    )

    top = (
        results_df
        .sort_values(
            "f1",
            ascending=False,
        )
        .head(10)
    )

    print(
        top[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "predicted_positive_pct",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "BEST MODEL B THRESHOLD"
    )
    print(
        "============================================================"
    )

    print(
        f"Threshold:          "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Accuracy:           "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1:                 "
        f"{best['f1']:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{best['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:             "
        f"{best['pr_auc']:.4f}"
    )

    print(
        f"Predicted positive: "
        f"{best['predicted_positive_pct']:.2f}%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved threshold results to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "============================================================"
    )
    print(
        "DONE"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()