import argparse
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from train_xgboost_fixed import (
    EXCLUDE,
    prepare_dataframe,
    detect_features,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_t2dm.csv")
    parser.add_argument("--model-dir", default="model_output")
    parser.add_argument("--output-dir", default="feature_analysis")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(args.input)
    feature_cols, numeric_features, categorical_features, dropped_all_missing, dropped_constant = detect_features(df)

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = joblib.load(model_dir / "xgboost_glucose_spike_model.joblib")

    # Feature names after preprocessing (including one-hot categorical names).
    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    if len(names) != len(importances):
        raise RuntimeError(
            f"Feature-name count ({len(names)}) != importance count ({len(importances)})."
        )

    result = pd.DataFrame({
        "feature": names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    result.to_csv(out / "feature_importance_all.csv", index=False)

    top20 = result.head(20).sort_values("importance", ascending=True)
    top20.to_csv(out / "feature_importance_top20.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.barh(top20["feature"], top20["importance"])
    plt.xlabel("XGBoost importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Features for 2-Hour Glucose Spike Prediction")
    plt.tight_layout()
    plt.savefig(out / "feature_importance_top20.png", dpi=180)
    plt.close()

    print("\n===== TOP 20 FEATURES =====")
    print(result.head(20).to_string(index=False))
    print("\nSaved:")
    print(out / "feature_importance_all.csv")
    print(out / "feature_importance_top20.csv")
    print(out / "feature_importance_top20.png")

if __name__ == "__main__":
    main()
