from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

V2_PATH = Path("data/processed/shanghai_t2dm_features_v2.csv")

TRAIN_SPLIT = Path("data/processed/train.csv")
VAL_SPLIT = Path("data/processed/validation.csv")
TEST_SPLIT = Path("data/processed/test.csv")

RESULTS_PATH = Path("data/processed/model_b_results.csv")
BEST_MODEL_PATH = Path("data/processed/model_b_random_forest.joblib")


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

GLUCOSE_FEATURES = [
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
]

CLINICAL_FEATURES = [
    "age",
    "bmi",
    "diabetes_duration",
    "fasting_glucose",
    "postprandial_glucose",
    "hba1c",
]

CONTEXT_FEATURES = [
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

FEATURES = (
    GLUCOSE_FEATURES
    + CLINICAL_FEATURES
    + CONTEXT_FEATURES
)


# ============================================================
# HELPERS
# ============================================================

def load_patient_ids(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Expected the existing Model A patient-level split files."
        )

    split = pd.read_csv(
        path,
        low_memory=False,
    )

    if "patient_id" not in split.columns:
        raise RuntimeError(
            f"{path} does not contain patient_id."
        )

    return set(
        split["patient_id"]
        .dropna()
        .astype(str)
        .unique()
    )


def evaluate_model(
    name,
    model,
    X,
    y,
    threshold=0.50,
):
    probabilities = (
        model
        .predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "predicted_positive_pct": (
            predictions.mean() * 100.0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_pipeline(classifier):
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                classifier,
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "MODEL B: GLUCOSE + CLINICAL + CONTEXT"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Load V2
    # --------------------------------------------------------

    if not V2_PATH.exists():
        raise FileNotFoundError(
            f"V2 feature file not found: {V2_PATH}"
        )

    df = pd.read_csv(
        V2_PATH,
        low_memory=False,
    )

    required_columns = [
        "patient_id",
        "spike_label",
        *FEATURES,
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "V2 dataset is missing columns:\n"
            + "\n".join(missing)
        )

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
    )

    df["spike_label"] = pd.to_numeric(
        df["spike_label"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "patient_id",
            "spike_label",
        ]
    ).copy()

    df["spike_label"] = (
        df["spike_label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # EXACT SAME PATIENT-LEVEL SPLIT AS MODEL A
    # --------------------------------------------------------

    train_patients = load_patient_ids(
        TRAIN_SPLIT
    )

    val_patients = load_patient_ids(
        VAL_SPLIT
    )

    test_patients = load_patient_ids(
        TEST_SPLIT
    )

    # --------------------------------------------------------
    # Check for patient leakage
    # --------------------------------------------------------

    if train_patients & val_patients:
        raise RuntimeError(
            "Train/validation patient overlap detected."
        )

    if train_patients & test_patients:
        raise RuntimeError(
            "Train/test patient overlap detected."
        )

    if val_patients & test_patients:
        raise RuntimeError(
            "Validation/test patient overlap detected."
        )

    all_split_patients = (
        train_patients
        | val_patients
        | test_patients
    )

    v2_patients = set(
        df["patient_id"].unique()
    )

    missing_patients = (
        all_split_patients
        - v2_patients
    )

    if missing_patients:
        raise RuntimeError(
            "V2 is missing patients from the original split: "
            f"{sorted(missing_patients)}"
        )

    # --------------------------------------------------------
    # Split V2 using EXACT patient IDs
    # --------------------------------------------------------

    train_df = df[
        df["patient_id"].isin(train_patients)
    ].copy()

    val_df = df[
        df["patient_id"].isin(val_patients)
    ].copy()

    test_df = df[
        df["patient_id"].isin(test_patients)
    ].copy()

    print()
    print("PATIENT SPLIT")
    print(
        f"Train patients:      "
        f"{len(train_patients)}"
    )
    print(
        f"Validation patients: "
        f"{len(val_patients)}"
    )
    print(
        f"Test patients:       "
        f"{len(test_patients)}"
    )

    print()
    print("V2 ROW SPLIT")

    print(
        f"Train:      "
        f"{len(train_df):,} rows | "
        f"spike="
        f"{train_df['spike_label'].mean()*100:.2f}%"
    )

    print(
        f"Validation: "
        f"{len(val_df):,} rows | "
        f"spike="
        f"{val_df['spike_label'].mean()*100:.2f}%"
    )

    print(
        f"Test:       "
        f"{len(test_df):,} rows | "
        f"spike="
        f"{test_df['spike_label'].mean()*100:.2f}%"
    )

    # --------------------------------------------------------
    # Prepare matrices
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ].copy()

    y_train = train_df[
        "spike_label"
    ].to_numpy()

    X_val = val_df[
        FEATURES
    ].copy()

    y_val = val_df[
        "spike_label"
    ].to_numpy()

    X_test = test_df[
        FEATURES
    ].copy()

    y_test = test_df[
        "spike_label"
    ].to_numpy()

    print()
    print(
        f"Model B features: {len(FEATURES)}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {}

    models[
        "Logistic Regression"
    ] = build_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    )

    models[
        "Random Forest"
    ] = build_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    if XGB_AVAILABLE:

        negative = max(
            1,
            int(
                (y_train == 0).sum()
            ),
        )

        positive = max(
            1,
            int(
                (y_train == 1).sum()
            ),
        )

        scale_pos_weight = (
            negative / positive
        )

        models[
            "XGBoost"
        ] = build_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )
        )

    else:

        print()
        print(
            "WARNING: XGBoost is not installed."
        )
        print(
            "Continuing with Logistic Regression "
            "+ Random Forest."
        )

    # --------------------------------------------------------
    # Train + validation evaluation
    # --------------------------------------------------------

    results = []
    trained_models = {}

    print()
    print(
        "============================================================"
    )
    print(
        "VALIDATION RESULTS"
    )
    print(
        "============================================================"
    )

    for name, model in models.items():

        print()
        print(
            f"Training {name}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        trained_models[name] = model

        metrics = evaluate_model(
            name,
            model,
            X_val,
            y_val,
            threshold=0.50,
        )

        results.append(
            metrics
        )

        print(
            f"Accuracy:           "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision:          "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall:             "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1:                 "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:            "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:             "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Predicted positive:  "
            f"{metrics['predicted_positive_pct']:.2f}%"
        )

    # --------------------------------------------------------
    # Rank models
    # --------------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            by=[
                "f1",
                "pr_auc",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save validation results
    # --------------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "MODEL B RANKING"
    )
    print(
        "============================================================"
    )

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "predicted_positive_pct",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Save best validation model
    # --------------------------------------------------------

    best_name = results_df.iloc[0]["model"]

    best_model = trained_models[
        best_name
    ]

    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
    )

    print()
    print(
        f"Best Model B by validation F1: "
        f"{best_name}"
    )

    print(
        f"Saved: {BEST_MODEL_PATH}"
    )

    # --------------------------------------------------------
    # TEST SET REMAINS UNTOUCHED
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "TEST SET"
    )
    print(
        "============================================================"
    )

    print(
        "TEST SET HAS NOT BEEN USED FOR MODEL SELECTION."
    )

    print(
        "We will evaluate it only after comparing "
        "Model A vs Model B and tuning the final threshold."
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