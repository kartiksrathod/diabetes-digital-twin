from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/shanghai_t2dm_master.csv")
OUTPUT_PATH = Path("data/processed/shanghai_t2dm_labeled.csv")

PREDICTION_HOURS = 2
GLUCOSE_THRESHOLD = 180.0
MIN_INCREASE = 30.0


def future_max_within_window(
    timestamps: np.ndarray,
    values: np.ndarray,
    window_hours: int = 2,
) -> np.ndarray:
    """
    For every timestamp t, calculate the maximum valid CGM value
    strictly after t and within the next `window_hours`.

    The implementation uses a monotonic deque so it remains efficient
    for large time-series datasets.
    """

    n = len(timestamps)
    result = np.full(n, np.nan)

    window = pd.Timedelta(hours=window_hours)

    # Convert timestamps to nanoseconds for fast search/comparison.
    ts_ns = timestamps.astype("datetime64[ns]").astype(np.int64)

    window_ns = window.value

    # Monotonic decreasing deque of indices.
    dq = deque()

    right = 0

    for left in range(n):

        # Remove indices that are no longer inside the future window.
        while dq and dq[0] <= left:
            dq.popleft()

        # Extend right boundary.
        target_time = ts_ns[left] + window_ns

        while right < n and ts_ns[right] <= target_time:
            # We don't include the current observation itself.
            if right > left and not np.isnan(values[right]):

                while dq and values[dq[-1]] <= values[right]:
                    dq.pop()

                dq.append(right)

            right += 1

        # Remove current/future-invalid entries.
        while dq and dq[0] <= left:
            dq.popleft()

        if dq:
            result[left] = values[dq[0]]

    return result


def process_session(group: pd.DataFrame) -> pd.DataFrame:
    """
    Generate spike labels independently for one patient/session.
    """

    group = (
        group.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    timestamps = group["timestamp"].to_numpy()
    values = group["cgm_mg_dl"].to_numpy(dtype=float)

    future_max = future_max_within_window(
        timestamps=timestamps,
        values=values,
        window_hours=PREDICTION_HOURS,
    )

    increase = future_max - values

    # A spike requires BOTH:
    # 1. Future CGM reaches >= 180 mg/dL
    # 2. Future maximum rises >= 30 mg/dL above current glucose
    spike = (
        (future_max >= GLUCOSE_THRESHOLD)
        & (increase >= MIN_INCREASE)
    )

    # If there is no usable future observation, the window is invalid.
    valid_window = ~np.isnan(future_max)

    group["future_max_2h"] = future_max
    group["future_increase_2h"] = increase

    group["spike_label"] = np.where(
        valid_window,
        spike.astype(int),
        np.nan,
    )

    group["valid_prediction_window"] = valid_window

    return group


def main():
    print("Loading master dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    # Parse required fields.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="mixed",
    )

    df["cgm_mg_dl"] = pd.to_numeric(
        df["cgm_mg_dl"],
        errors="coerce",
    )

    # Remove rows where we cannot reason about the current observation.
    df = df.dropna(
        subset=[
            "patient_id",
            "session_id",
            "timestamp",
            "cgm_mg_dl",
        ]
    ).copy()

    df = df.sort_values(
        [
            "patient_id",
            "session_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    print(f"Input rows: {len(df):,}")
    print(f"Patients:    {df['patient_id'].nunique()}")
    print(f"Sessions:    {df['session_id'].nunique()}")

    print("\nGenerating 2-hour labels...")

    labeled_parts = []

    for (patient_id, session_id), group in df.groupby(
        ["patient_id", "session_id"],
        sort=False,
    ):
        labeled = process_session(group)
        labeled_parts.append(labeled)

    labeled_df = pd.concat(
        labeled_parts,
        ignore_index=True,
    )

    # Keep only windows that actually have future data.
    valid = labeled_df[
        labeled_df["valid_prediction_window"]
    ].copy()

    valid["spike_label"] = (
        valid["spike_label"]
        .astype(int)
    )

    print("\n========== LABEL SUMMARY ==========")

    total_windows = len(valid)
    positive = int(
        (valid["spike_label"] == 1).sum()
    )
    negative = int(
        (valid["spike_label"] == 0).sum()
    )

    positive_pct = (
        positive / total_windows * 100
        if total_windows
        else 0
    )

    negative_pct = (
        negative / total_windows * 100
        if total_windows
        else 0
    )

    print(
        f"Valid prediction windows: {total_windows:,}"
    )

    print(
        f"Spike = 1:                "
        f"{positive:,} ({positive_pct:.2f}%)"
    )

    print(
        f"No spike = 0:             "
        f"{negative:,} ({negative_pct:.2f}%)"
    )

    print("\n---------- Spike definition ----------")

    print(
        "Future maximum glucose >= "
        f"{GLUCOSE_THRESHOLD:.0f} mg/dL"
    )

    print(
        "AND future glucose increase >= "
        f"{MIN_INCREASE:.0f} mg/dL"
    )

    print(
        f"Prediction horizon: "
        f"next {PREDICTION_HOURS} hours"
    )

    print("\n---------- Future values ----------")

    print(
        f"Median future max: "
        f"{valid['future_max_2h'].median():.2f}"
    )

    print(
        f"Mean future max:   "
        f"{valid['future_max_2h'].mean():.2f}"
    )

    print(
        f"Median increase:   "
        f"{valid['future_increase_2h'].median():.2f}"
    )

    print(
        f"95th pct increase:  "
        f"{valid['future_increase_2h'].quantile(0.95):.2f}"
    )

    # Save.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved labeled dataset to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()