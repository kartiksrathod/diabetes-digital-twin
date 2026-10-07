from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/processed/shanghai_t2dm_master.csv")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n========== BASIC INFO ==========")
    print(f"Rows:              {len(df):,}")
    print(f"Columns:           {len(df.columns)}")
    print(f"Unique patients:   {df['patient_id'].nunique()}")
    print(f"Unique sessions:   {df['session_id'].nunique()}")

    print("\n========== CGM ==========")

    cgm = pd.to_numeric(df["cgm_mg_dl"], errors="coerce")

    print(f"Missing CGM:       {cgm.isna().sum():,}")
    print(f"Min:               {cgm.min()}")
    print(f"Max:               {cgm.max()}")
    print(f"Mean:              {cgm.mean():.2f}")
    print(f"Median:            {cgm.median():.2f}")
    print(f"Std:               {cgm.std():.2f}")

    print("\nPercentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"{p:>2}th: {cgm.quantile(p / 100):.2f}")

    print("\n========== TIMESTAMPS ==========")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    print(f"Invalid timestamps: {df['timestamp'].isna().sum():,}")
    print(f"Earliest:           {df['timestamp'].min()}")
    print(f"Latest:             {df['timestamp'].max()}")

    print("\n========== DUPLICATES ==========")

    dup = df.duplicated(
        subset=["patient_id", "session_id", "timestamp"]
    ).sum()

    print(f"Duplicate patient/session/timestamp rows: {dup:,}")

    print("\n========== SAMPLING INTERVAL ==========")

    temp = (
        df[["patient_id", "session_id", "timestamp"]]
        .dropna()
        .sort_values(["patient_id", "session_id", "timestamp"])
    )

    temp["delta_minutes"] = (
        temp.groupby(["patient_id", "session_id"])["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    intervals = temp["delta_minutes"].dropna()

    print(f"Median interval:   {intervals.median():.2f} min")
    print(f"Mean interval:     {intervals.mean():.2f} min")

    print("\nMost common intervals:")
    print(
        intervals
        .round()
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\n========== PATIENT COVERAGE ==========")

    patient_counts = (
        df.groupby("patient_id")
        .size()
        .sort_values()
    )

    print(f"Min rows/patient:  {patient_counts.min()}")
    print(f"Median:            {patient_counts.median():.0f}")
    print(f"Max:               {patient_counts.max()}")

    print("\nSmallest patients:")
    print(patient_counts.head(10).to_string())

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()