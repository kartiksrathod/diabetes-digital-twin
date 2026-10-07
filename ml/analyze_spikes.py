from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/processed/shanghai_t2dm_master.csv")


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="mixed"
    )

    df["cgm_mg_dl"] = pd.to_numeric(
        df["cgm_mg_dl"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp", "cgm_mg_dl"])
          .sort_values(["patient_id", "session_id", "timestamp"])
          .reset_index(drop=True)
    )

    # Look 8 observations ahead = 2 hours at 15-minute sampling.
    group_cols = ["patient_id", "session_id"]

    g = df.groupby(group_cols)["cgm_mg_dl"]

    future_values = {
        f"future_{i}": g.shift(-i)
        for i in range(1, 9)
    }

    future_df = pd.DataFrame(future_values)

    valid = future_df.notna().all(axis=1)

    current = df.loc[valid, "cgm_mg_dl"].to_numpy()
    future = future_df.loc[valid]

    future_max = future.max(axis=1).to_numpy()
    future_min = future.min(axis=1).to_numpy()

    increase = future_max - current

    print("\n========== 2-HOUR FUTURE ANALYSIS ==========")

    print(f"Valid prediction windows: {len(current):,}")

    print("\n---------- Future Maximum ----------")
    print(f"Median: {np.median(future_max):.2f} mg/dL")
    print(f"Mean:   {np.mean(future_max):.2f} mg/dL")
    print(f"95th:   {np.percentile(future_max, 95):.2f} mg/dL")
    print(f"99th:   {np.percentile(future_max, 99):.2f} mg/dL")

    print("\n---------- Future Increase ----------")
    for threshold in [20, 30, 40, 50, 60]:
        count = np.sum(increase >= threshold)
        pct = count / len(increase) * 100
        print(
            f"Increase >= {threshold:>2} mg/dL: "
            f"{count:,} ({pct:.2f}%)"
        )

    print("\n---------- Crossing 180 mg/dL ----------")
    for threshold in [140, 160, 180, 200, 220, 250]:
        count = np.sum(future_max >= threshold)
        pct = count / len(future_max) * 100
        print(
            f"Future max >= {threshold:>3} mg/dL: "
            f"{count:,} ({pct:.2f}%)"
        )

    print("\n---------- Combined Candidates ----------")

    candidates = {
        "future_max >= 180": future_max >= 180,

        "future_max >= 180 AND increase >= 20":
            (future_max >= 180) & (increase >= 20),

        "future_max >= 180 AND increase >= 30":
            (future_max >= 180) & (increase >= 30),

        "future_max >= 180 AND increase >= 40":
            (future_max >= 180) & (increase >= 40),

        "increase >= 30":
            increase >= 30,

        "increase >= 40":
            increase >= 40,
    }

    for name, mask in candidates.items():
        count = np.sum(mask)
        pct = count / len(mask) * 100

        print(
            f"{name:<45} "
            f"{count:>8,} ({pct:>6.2f}%)"
        )

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()