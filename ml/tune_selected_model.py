import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import precision_recall_curve

from train_xgboost_fixed import TARGET, prepare_dataframe, detect_features, split_by_patient


def at_threshold(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_t2dm.csv")
    parser.add_argument("--model-dir", default="robustness_output")
    parser.add_argument("--variant", default="no_missing_indicators")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    pre = joblib.load(model_dir / f"{args.variant}_preprocessor.joblib")
    model = joblib.load(model_dir / f"{args.variant}_model.joblib")

    df = prepare_dataframe(args.input)
    feature_cols, _, _, _, _ = detect_features(df)
    _, val_df, test_df = split_by_patient(df)

    X_val = pre.transform(val_df[feature_cols])
    X_test = pre.transform(test_df[feature_cols])
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # Choose threshold ONLY from validation data: maximum F1.
    precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    idx = int(np.nanargmax(f1_values))
    threshold = float(thresholds[idx])

    validation_result = at_threshold(y_val, val_prob, threshold)
    test_result = at_threshold(y_test, test_prob, threshold)
    test_result["roc_auc"] = float(roc_auc_score(y_test, test_prob))
    test_result["pr_auc"] = float(average_precision_score(y_test, test_prob))

    print("\n===== SELECTED MODEL =====")
    print(args.variant)

    print("\n===== VALIDATION THRESHOLD =====")
    print(f"threshold: {threshold:.4f}")
    print(f"precision: {validation_result['precision']:.4f}")
    print(f"recall:    {validation_result['recall']:.4f}")
    print(f"f1:        {validation_result['f1']:.4f}")

    print("\n===== TEST AT SELECTED THRESHOLD =====")
    for k, v in test_result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    out = model_dir / f"{args.variant}_selected_threshold.json"
    out.write_text(
        json.dumps({
            "variant": args.variant,
            "threshold_selection": "maximum validation F1",
            "threshold": threshold,
            "validation": validation_result,
            "test": test_result,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
