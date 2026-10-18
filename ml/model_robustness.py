import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from train_xgboost_fixed import (
    TARGET,
    EXCLUDE,
    prepare_dataframe,
    detect_features,
    split_by_patient,
)


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def build_preprocessor(numeric, categorical, add_missing_indicators=False):
    transformers = []

    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                add_indicator=add_missing_indicators
            ))
        ])
        transformers.append(("num", num_pipe, numeric))

    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_xgb(Xtr, ytr, Xv, yv, seed=42):
    pos = max(int((ytr == 1).sum()), 1)
    neg = int((ytr == 0).sum())
    weight = neg / pos

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=700,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=weight,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=60,
    )

    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_t2dm.csv")
    parser.add_argument("--output-dir", default="robustness_output")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(args.input)
    features, numeric, categorical, dropped_all_missing, dropped_constant = detect_features(df)

    train_df, val_df, test_df = split_by_patient(df)

    X_train = train_df[features]
    X_val = val_df[features]
    X_test = test_df[features]
    y_train = train_df[TARGET].astype(int)
    y_val = val_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    configs = {
        "no_missing_indicators": False,
        "with_missing_indicators": True,
    }

    rows = []

    for name, add_ind in configs.items():
        print(f"\nTraining variant: {name}")
        pre = build_preprocessor(numeric, categorical, add_missing_indicators=add_ind)

        Xt = pre.fit_transform(X_train)
        Xv = pre.transform(X_val)
        Xte = pre.transform(X_test)

        model = train_xgb(Xt, y_train, Xv, y_val)

        val_metrics = evaluate(model, Xv, y_val)
        test_metrics = evaluate(model, Xte, y_test)

        row = {"variant": name}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        rows.append(row)

        joblib.dump(pre, out / f"{name}_preprocessor.joblib")
        joblib.dump(model, out / f"{name}_model.joblib")

    results = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    results.to_csv(out / "model_robustness_comparison.csv", index=False)

    print("\n===== MODEL ROBUSTNESS COMPARISON =====")
    print(results.to_string(index=False))
    print(f"\nSaved: {out / 'model_robustness_comparison.csv'}")


if __name__ == "__main__":
    main()
