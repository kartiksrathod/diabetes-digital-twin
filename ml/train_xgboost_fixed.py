import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET = "glucose_spike_next_2h"

# NEVER use identifiers, timestamps, or future-information columns as model inputs.
EXCLUDE = {
    "subject_id",
    "session_id",
    "timestamp",
    "future_max_glucose_2h_mg_dl",
    "future_glucose_rise_2h_mg_dl",
    TARGET,
    "diabetes_type",  # all rows are T2DM in this project; constant column
}


def split_by_patient(df, group_col="subject_id"):
    """
    Patient-wise 60/20/20 split using StratifiedGroupKFold.
    This prevents the same patient appearing in multiple splits.
    """
    y = df[TARGET].astype(int)
    groups = df[group_col]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = [test_idx for _, test_idx in cv.split(df, y, groups)]

    train_idx = np.concatenate([folds[0], folds[3], folds[4]])
    val_idx = folds[1]
    test_idx = folds[2]

    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()

    a = set(train[group_col])
    b = set(val[group_col])
    c = set(test[group_col])
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)

    return train, val, test


def prepare_dataframe(path):
    df = pd.read_csv(path, low_memory=False)

    required = {TARGET, "subject_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Target cleanup.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Convert numeric-looking columns safely.
    # We do NOT blindly convert every object column; categorical text must stay categorical.
    for col in df.columns:
        if col in EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert object/string columns to plain Python strings with a real missing token.
    # This avoids pandas.NA causing "boolean value of NA is ambiguous" in sklearn.
    for col in df.columns:
        if col in EXCLUDE:
            continue
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].where(df[col].notna(), "__MISSING__").astype(str)

    return df


def detect_features(df):
    """
    Infer usable features from the processed CSV instead of relying on a hard-coded
    column list. Drop columns that are entirely missing or have only one unique value.
    """
    candidates = [c for c in df.columns if c not in EXCLUDE]

    usable = []
    dropped_all_missing = []
    dropped_constant = []

    for c in candidates:
        non_missing = df[c].notna().sum()
        nunique = df[c].nunique(dropna=True)

        if non_missing == 0:
            dropped_all_missing.append(c)
            continue
        if nunique <= 1:
            dropped_constant.append(c)
            continue

        usable.append(c)

    numeric_features = [c for c in usable if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in usable if c not in numeric_features]

    return usable, numeric_features, categorical_features, dropped_all_missing, dropped_constant


def make_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ])
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipe, categorical_features))

    if not transformers:
        raise ValueError("No usable features remain after cleaning.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate(model, X, y, name, threshold=0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }

    cm = confusion_matrix(y, pred)

    print(f"\n===== {name.upper()} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y, pred, digits=4, zero_division=0))

    return metrics, prob, pred, cm


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost for next-2-hour glucose spike prediction.")
    parser.add_argument("--input", default="processed_t2dm.csv")
    parser.add_argument("--output-dir", default="model_output")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(args.input)

    feature_cols, numeric_features, categorical_features, all_missing, constants = detect_features(df)

    train_df, val_df, test_df = split_by_patient(df)

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET]
    y_val = val_df[TARGET]
    y_test = test_df[TARGET]

    print("Dataset:")
    print(f"  Rows: {len(df):,}")
    print(f"  Patients: {df['subject_id'].nunique():,}")
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Dropped all-missing columns: {all_missing}")
    print(f"  Dropped constant columns: {constants}")

    for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(
            f"{name}: rows={len(part):,}, "
            f"patients={part['subject_id'].nunique():,}, "
            f"spike_rate={part[TARGET].mean():.2%}"
        )

    preprocessor = make_preprocessor(numeric_features, categorical_features)

    print("\nPreprocessing...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    print(f"Training positives: {pos:,}")
    print(f"Training negatives: {neg:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=700,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=60,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )

    train_metrics, _, _, _ = evaluate(model, X_train_t, y_train, "train")
    val_metrics, _, _, _ = evaluate(model, X_val_t, y_val, "validation")
    test_metrics, test_prob, test_pred, cm = evaluate(model, X_test_t, y_test, "test")

    joblib.dump(preprocessor, out / "preprocessor.joblib")
    joblib.dump(model, out / "xgboost_glucose_spike_model.joblib")

    metadata = {
        "target": TARGET,
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_columns": sorted(EXCLUDE),
        "dropped_all_missing": all_missing,
        "dropped_constant": constants,
        "train_patients": int(train_df["subject_id"].nunique()),
        "validation_patients": int(val_df["subject_id"].nunique()),
        "test_patients": int(test_df["subject_id"].nunique()),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "scale_pos_weight": float(scale_pos_weight),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["No spike", "Spike"])
    plt.yticks([0, 1], ["No spike", "Spike"])
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    pred_df = test_df[["subject_id", "session_id", "timestamp"]].copy()
    pred_df["actual_spike"] = y_test.to_numpy()
    pred_df["spike_probability"] = test_prob
    pred_df["predicted_spike"] = test_pred
    pred_df["risk_level"] = pd.cut(
        pred_df["spike_probability"],
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    print("\nDONE.")
    print(f"Model outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
