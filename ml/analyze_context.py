from pathlib import Path
import pandas as pd

PATH = Path("data/processed/shanghai_t2dm_master.csv")

df = pd.read_csv(PATH, low_memory=False)

columns = [
    "dietary_intake",
    "diet_notes",
    "进食量",
    "insulin_sc",
    "non_insulin_agents",
    "csii_bolus_insulin_iu",
    "csii_basal_insulin_iu_h",
    "insulin_iv",
]

print("\n========== CONTEXT DATA ANALYSIS ==========")

for col in columns:
    if col not in df.columns:
        print(f"\n{col}: NOT FOUND")
        continue

    s = df[col]

    print(f"\n---------- {col} ----------")
    print(f"Non-null: {s.notna().sum():,}")
    print(f"Empty/blank: {s.astype(str).str.strip().eq('').sum():,}")

    print("Sample values:")
    print(
        s.dropna()
         .astype(str)
         .str.strip()
         .replace("", pd.NA)
         .dropna()
         .drop_duplicates()
         .head(15)
         .to_string(index=False)
    )

print("\n========== DONE ==========")