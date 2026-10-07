from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

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


def load_data():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    val = pd.read_csv(VAL_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    return train, val, test


def prepare_xy(df):
    """
    Separate model features from metadata and target.
    """
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


def build_preprocessor(feature_columns):
    """
    Median imputation + missing indicators + scaling.

    The fitted transformer is learned only on TRAIN.
    """

    numeric_pipeline = Pipeline(
        steps=[
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
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns
            )
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_val, y_val):
    """
    Evaluate using probabilities because PR-AUC and ROC-AUC
    require probability/ranking information.
    """

    probabilities = model.predict_proba(X_val)[:, 1]

    # Default threshold.
    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

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

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    cm = confusion_matrix(
        y_val,
        predictions
    )

    print(f"\n========== {name} ==========")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():

    print("Loading datasets...")

    train, val, test = load_data()

    X_train, y_train, feature_columns = prepare_xy(train)
    X_val, y_val, _ = prepare_xy(val)
    X_test, y_test, _ = prepare_xy(test)

    print("\n========== DATA ==========")

    print(f"Train rows:      {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")
    print(f"Features:        {len(feature_columns)}")

    train_positive = int(y_train.sum())
    train_negative = int(len(y_train) - train_positive)

    print(f"\nTrain positives: {train_positive:,}")
    print(f"Train negatives: {train_negative:,}")

    # Class weight for imbalanced data.
    scale_pos_weight = (
        train_negative / train_positive
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.3f}"
    )

    preprocessor = build_preprocessor(
        feature_columns
    )

    results = []

    # ---------------------------------------------------------
    # 1. Logistic Regression
    # ---------------------------------------------------------

    logistic = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    logistic.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic,
            X_val,
            y_val
        )
    )

    # ---------------------------------------------------------
    # 2. Random Forest
    # ---------------------------------------------------------

    forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    feature_columns
                )
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
            ),
        ]
    )

    print("\nTraining Random Forest...")
    forest.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "Random Forest",
            forest,
            X_val,
            y_val
        )
    )

    # ---------------------------------------------------------
    # 3. XGBoost
    # ---------------------------------------------------------

    xgb_model = None

    if XGBOOST_AVAILABLE:

        xgb_model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        feature_columns
                    )
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                ),
            ]
        )

        print("\nTraining XGBoost...")

        xgb_model.fit(
            X_train,
            y_train
        )

        results.append(
            evaluate_model(
                "XGBoost",
                xgb_model,
                X_val,
                y_val
            )
        )

    else:
        print(
            "\nXGBoost is not installed. "
            "Install it with:"
        )
        print("pip install xgboost")

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "pr_auc",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\n========== MODEL COMPARISON =========="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_path = Path(
        "data/processed/model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"\nSaved comparison to:\n"
        f"{results_path}"
    )

    # ---------------------------------------------------------
    # Final test evaluation will happen separately.
    # ---------------------------------------------------------

    print(
        "\nIMPORTANT:"
        "\nTest set has NOT been used yet."
        "\nWe only used validation data for model comparison."
    )


if __name__ == "__main__":
    main()