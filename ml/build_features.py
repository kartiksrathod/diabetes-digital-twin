from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/shanghai_t2dm_labeled.csv")
OUTPUT_PATH = Path("data/processed/shanghai_t2dm_features_v1.csv")


# Historical glucose points:
# t, t-15, t-30, ... t-120 minutes
LAG_MINUTES = list(range(0, 121, 15))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="mixed",
    )

    df["cgm_mg_dl"] = pd.to_numeric(
        df["cgm_mg_dl"],
        errors="coerce",
    )

    # Numeric patient-level variables.
    numeric_columns = [
        "Age (years)",
        "BMI (kg/m2)",
        "Duration of diabetes (years)",
        "Fasting Plasma Glucose (mg/dl)",
        "2-hour Postprandial Plasma Glucose (mg/dl)",
        "HbA1c (mmol/mol)",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Sort first.
    df = df.sort_values(
        ["patient_id", "session_id", "timestamp"]
    ).reset_index(drop=True)

    group_cols = ["patient_id", "session_id"]

    # ------------------------------------------------------------------
    # 1. Historical glucose features
    # ------------------------------------------------------------------

    for minutes in LAG_MINUTES:
        if minutes == 0:
            df[f"glucose_t0"] = df["cgm_mg_dl"]
        else:
            periods = minutes // 15

            df[f"glucose_t_minus_{minutes}m"] = (
                df.groupby(group_cols)["cgm_mg_dl"]
                .shift(periods)
            )

    # ------------------------------------------------------------------
    # 2. Verify actual timestamp alignment for each lag
    # ------------------------------------------------------------------

    for minutes in LAG_MINUTES:
        if minutes == 0:
            continue

        expected_time = (
            df["timestamp"]
            - pd.Timedelta(minutes=minutes)
        )

        # Use a lookup table to ensure the lag really exists at the
        # expected timestamp rather than merely being N rows earlier.
        lookup = df[
            group_cols + ["timestamp", "cgm_mg_dl"]
        ].copy()

        lookup = lookup.rename(
            columns={
                "timestamp": "lookup_timestamp",
                "cgm_mg_dl": f"exact_glucose_minus_{minutes}m",
            }
        )

        temp = df[
            group_cols
        ].copy()

        temp["lookup_timestamp"] = expected_time.values

        temp["_row_id"] = np.arange(len(temp))

        temp = temp.merge(
            lookup,
            on=group_cols + ["lookup_timestamp"],
            how="left",
        )

        # Replace row-shift result with exact timestamp result.
        df[f"glucose_t_minus_{minutes}m"] = (
            temp.sort_values("_row_id")[
                f"exact_glucose_minus_{minutes}m"
            ].to_numpy()
        )

    # ------------------------------------------------------------------
    # 3. Glucose change features
    # ------------------------------------------------------------------

    df["delta_15m"] = (
        df["glucose_t0"]
        - df["glucose_t_minus_15m"]
    )

    df["delta_30m"] = (
        df["glucose_t0"]
        - df["glucose_t_minus_30m"]
    )

    df["delta_60m"] = (
        df["glucose_t0"]
        - df["glucose_t_minus_60m"]
    )

    df["delta_120m"] = (
        df["glucose_t0"]
        - df["glucose_t_minus_120m"]
    )

    # ------------------------------------------------------------------
    # 4. Rolling statistics
    # ------------------------------------------------------------------

    history_columns_1h = [
        "glucose_t_minus_60m",
        "glucose_t_minus_45m",
        "glucose_t_minus_30m",
        "glucose_t_minus_15m",
        "glucose_t0",
    ]

    history_columns_2h = [
        "glucose_t_minus_120m",
        "glucose_t_minus_105m",
        "glucose_t_minus_90m",
        "glucose_t_minus_75m",
        "glucose_t_minus_60m",
        "glucose_t_minus_45m",
        "glucose_t_minus_30m",
        "glucose_t_minus_15m",
        "glucose_t0",
    ]

    df["glucose_mean_1h"] = df[history_columns_1h].mean(axis=1)
    df["glucose_std_1h"] = df[history_columns_1h].std(axis=1)
    df["glucose_min_1h"] = df[history_columns_1h].min(axis=1)
    df["glucose_max_1h"] = df[history_columns_1h].max(axis=1)

    df["glucose_mean_2h"] = df[history_columns_2h].mean(axis=1)
    df["glucose_std_2h"] = df[history_columns_2h].std(axis=1)
    df["glucose_min_2h"] = df[history_columns_2h].min(axis=1)
    df["glucose_max_2h"] = df[history_columns_2h].max(axis=1)

    # ------------------------------------------------------------------
    # 5. Approximate glucose slopes
    # ------------------------------------------------------------------

    df["slope_30m"] = (
        df["delta_30m"] / 30.0
    )

    df["slope_60m"] = (
        df["delta_60m"] / 60.0
    )

    df["slope_120m"] = (
        df["delta_120m"] / 120.0
    )

    # ------------------------------------------------------------------
    # 6. Patient-level clinical features
    # ------------------------------------------------------------------

    clinical_map = {
        "age": "Age (years)",
        "bmi": "BMI (kg/m2)",
        "diabetes_duration": "Duration of diabetes (years)",
        "fasting_glucose": "Fasting Plasma Glucose (mg/dl)",
        "postprandial_glucose": "2-hour Postprandial Plasma Glucose (mg/dl)",
        "hba1c": "HbA1c (mmol/mol)",
    }

    for output_name, source_name in clinical_map.items():
        if source_name in df.columns:
            df[output_name] = pd.to_numeric(
                df[source_name],
                errors="coerce",
            )
        else:
            df[output_name] = np.nan

    # ------------------------------------------------------------------
    # 7. Keep only rows with complete historical input
    # ------------------------------------------------------------------

    required_history = [
        f"glucose_t_minus_{minutes}m"
        for minutes in range(15, 121, 15)
    ]

    required_history.append("glucose_t0")

    before = len(df)

    df = df.dropna(
        subset=required_history
    ).copy()

    removed = before - len(df)

    print("\n========== FEATURE DATA ==========")
    print(f"Rows before history filtering: {before:,}")
    print(f"Rows removed:                  {removed:,}")
    print(f"Rows after filtering:          {len(df):,}")

    # ------------------------------------------------------------------
    # 8. Select final feature columns
    # ------------------------------------------------------------------

    feature_columns = [
        # Current + history
        "glucose_t0",
        "glucose_t_minus_15m",
        "glucose_t_minus_30m",
        "glucose_t_minus_45m",
        "glucose_t_minus_60m",
        "glucose_t_minus_75m",
        "glucose_t_minus_90m",
        "glucose_t_minus_105m",
        "glucose_t_minus_120m",

        # Changes
        "delta_15m",
        "delta_30m",
        "delta_60m",
        "delta_120m",

        # Rolling statistics
        "glucose_mean_1h",
        "glucose_std_1h",
        "glucose_min_1h",
        "glucose_max_1h",
        "glucose_mean_2h",
        "glucose_std_2h",
        "glucose_min_2h",
        "glucose_max_2h",

        # Slopes
        "slope_30m",
        "slope_60m",
        "slope_120m",

        # Patient profile
        "age",
        "bmi",
        "diabetes_duration",
        "fasting_glucose",
        "postprandial_glucose",
        "hba1c",
    ]

    metadata_columns = [
        "patient_id",
        "session_id",
        "session_number",
        "timestamp",
        "source_file",
        "spike_label",
    ]

    final_columns = (
        metadata_columns
        + feature_columns
    )

    final_columns = [
        col for col in final_columns
        if col in df.columns
    ]

    result = df[final_columns].copy()

    # Basic safety check.
    assert "spike_label" in result.columns

    print("\n========== FEATURE SUMMARY ==========")
    print(f"Final rows:      {len(result):,}")
    print(f"Patients:        {result['patient_id'].nunique()}")
    print(f"Sessions:        {result['session_id'].nunique()}")
    print(f"Features:        {len(feature_columns)}")

    print("\nTarget distribution:")
    print(
        result["spike_label"]
        .value_counts(normalize=False)
        .sort_index()
        .to_string()
    )

    print("\nMissing feature values:")
    missing = result[feature_columns].isna().sum()
    print(
        missing[missing > 0]
        .sort_values(ascending=False)
        .to_string()
        if (missing > 0).any()
        else "None"
    )

    return result


def main():

    print("Loading labeled dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    result = build_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved feature dataset to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()