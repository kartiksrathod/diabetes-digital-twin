from pathlib import Path

import joblib
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

TEST_SPLIT = Path(
    "data/processed/test.csv"
)

MODEL_PATH = Path(
    "data/processed/model_b_random_forest.joblib"
)

OUTPUT_PATH = Path(
    "data/processed/final_test_results.csv"
)


# ============================================================
# FINAL DECISION
# ============================================================

FINAL_THRESHOLD = 0.57


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
        "FINAL MODEL EVALUATION - UNTOUCHED TEST SET"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not V2_PATH.exists():
        raise FileNotFoundError(
            f"V2 dataset not found: {V2_PATH}"
        )

    if not TEST_SPLIT.exists():
        raise FileNotFoundError(
            f"Test split not found: {TEST_SPLIT}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        V2_PATH,
        low_memory=False,
    )

    test_split = pd.read_csv(
        TEST_SPLIT,
        low_memory=False,
    )

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Clean patient IDs
    # --------------------------------------------------------

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
    )

    test_split["patient_id"] = (
        test_split["patient_id"]
        .astype(str)
        .str.strip()
    )

    test_patients = set(
        test_split["patient_id"]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Extract untouched test patients
    # --------------------------------------------------------

    test_df = df[
        df["patient_id"].isin(
            test_patients
        )
    ].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    test_df["spike_label"] = pd.to_numeric(
        test_df["spike_label"],
        errors="coerce",
    )

    test_df = test_df.dropna(
        subset=["spike_label"]
    ).copy()

    test_df["spike_label"] = (
        test_df["spike_label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Prepare X/Y
    # --------------------------------------------------------

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "spike_label"
    ].to_numpy()

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # FINAL THRESHOLD
    # --------------------------------------------------------

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    predicted_positive_pct = (
        predictions.mean() * 100
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "TEST SET"
    )

    print(
        f"Test patients:       "
        f"{len(test_patients)}"
    )

    print(
        f"Test rows:            "
        f"{len(test_df):,}"
    )

    print(
        f"Actual spike rate:    "
        f"{y_test.mean()*100:.2f}%"
    )

    print()
    print(
        "FINAL THRESHOLD"
    )

    print(
        f"Threshold:            "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "FINAL TEST RESULTS"
    )
    print(
        "============================================================"
    )

    print(
        f"Accuracy:             "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision:            "
        f"{precision:.4f}"
    )

    print(
        f"Recall:               "
        f"{recall:.4f}"
    )

    print(
        f"F1:                   "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC:              "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{pr_auc:.4f}"
    )

    print(
        f"Predicted positive:   "
        f"{predicted_positive_pct:.2f}%"
    )

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(
        f"True Negatives:       {tn}"
    )

    print(
        f"False Positives:      {fp}"
    )

    print(
        f"False Negatives:      {fn}"
    )

    print(
        f"True Positives:       {tp}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "Model B Random Forest",
                "threshold": FINAL_THRESHOLD,
                "test_patients": len(test_patients),
                "test_rows": len(test_df),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "predicted_positive_pct":
                    predicted_positive_pct,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved final test results to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "============================================================"
    )

    print(
        "TEST EVALUATION COMPLETE"
    )

    print(
        "The test set was used only here, after model selection."
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()