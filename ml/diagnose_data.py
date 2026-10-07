import pandas as pd

PATH = "data/processed/shanghai_t2dm_master.csv"

df = pd.read_csv(PATH, low_memory=False)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    format="mixed"
)

print("\n========== TIMESTAMP DIAGNOSTICS ==========")

invalid = df[df["timestamp"].isna()].copy()

print(f"Total rows:              {len(df):,}")
print(f"Invalid timestamps:      {len(invalid):,}")

if not invalid.empty:
    print("\nInvalid rows by source file:")
    print(
        invalid["source_file"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nInvalid rows by patient:")
    print(
        invalid["patient_id"]
        .value_counts()
        .head(20)
        .to_string()
    )

print("\n========== DUPLICATE CHECK ==========")

valid = df.dropna(subset=["timestamp"]).copy()

dupes = valid.duplicated(
    subset=["patient_id", "session_id", "timestamp"],
    keep=False
)

print(
    f"True duplicate rows among valid timestamps: "
    f"{dupes.sum():,}"
)

if dupes.any():
    print("\nDuplicate rows by source file:")
    print(
        valid.loc[dupes, "source_file"]
        .value_counts()
        .head(20)
        .to_string()
    )

print("\n========== ROW COUNT BY SOURCE ==========")

counts = (
    df.groupby("source_file")
      .size()
      .sort_values()
)

print(f"Number of source files: {len(counts)}")
print(f"Minimum rows/file:      {counts.min()}")
print(f"Median rows/file:       {counts.median():.0f}")
print(f"Maximum rows/file:      {counts.max()}")

print("\nSmallest files:")
print(counts.head(10).to_string())

print("\nLargest files:")
print(counts.tail(10).to_string())

print("\n========== DONE ==========")