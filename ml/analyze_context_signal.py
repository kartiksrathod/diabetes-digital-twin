from pathlib import Path
import numpy as np
import pandas as pd


PATH = Path(
    "data/processed/shanghai_t2dm_labeled.csv"
)


def has_event(series: pd.Series) -> pd.Series:
    """
    Treat non-empty values as an event.
    """
    s = series.astype("string").fillna("").str.strip()
    return s.ne("") & s.ne("nan")


def main():

    print("Loading labeled dataset...")

    df = pd.read_csv(
        PATH,
        low_memory=False
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="mixed"
    )

    df = df.sort_values(
        ["patient_id", "session_id", "timestamp"]
    ).reset_index(drop=True)

    group_cols = ["patient_id", "session_id"]

    # ============================================================
    # EVENT DEFINITIONS
    # ============================================================

    df["meal_event"] = (
        has_event(df["dietary_intake"])
        | has_event(df["diet_notes"])
        | has_event(df["进食量"])
    ).astype(int)

    df["insulin_event"] = (
        has_event(df["insulin_sc"])
        | has_event(df["csii_bolus_insulin_iu"])
    ).astype(int)

    df["medication_event"] = has_event(
        df["non_insulin_agents"]
    ).astype(int)

    # ============================================================
    # RECENT EVENT WINDOWS
    # ============================================================

    for col, prefix in [
        ("meal_event", "meal"),
        ("insulin_event", "insulin"),
        ("medication_event", "medication"),
    ]:

        shifted = (
            df.groupby(group_cols)[col]
            .shift(1)
            .fillna(0)
        )

        grouped = shifted.groupby(
            [df["patient_id"], df["session_id"]]
        )

        df[f"{prefix}_events_30m"] = grouped.transform(
            lambda x: x.rolling(
                2,
                min_periods=1
            ).sum()
        )

        df[f"{prefix}_events_60m"] = grouped.transform(
            lambda x: x.rolling(
                4,
                min_periods=1
            ).sum()
        )

        df[f"{prefix}_events_120m"] = grouped.transform(
            lambda x: x.rolling(
                8,
                min_periods=1
            ).sum()
        )

    df = df.dropna(
        subset=["spike_label"]
    ).copy()

    print("\n========== CONTEXT SIGNAL ==========")

    target_rate = df["spike_label"].mean()

    print(
        f"Overall spike rate: "
        f"{target_rate * 100:.2f}%"
    )

    checks = [
        ("Meal in previous 30m", "meal_events_30m"),
        ("Meal in previous 60m", "meal_events_60m"),
        ("Meal in previous 120m", "meal_events_120m"),

        ("Insulin in previous 30m", "insulin_events_30m"),
        ("Insulin in previous 60m", "insulin_events_60m"),
        ("Insulin in previous 120m", "insulin_events_120m"),

        (
            "Medication in previous 120m",
            "medication_events_120m"
        ),
    ]

    for name, col in checks:

        event_present = df[col] > 0

        present_count = int(
            event_present.sum()
        )

        absent_count = int(
            (~event_present).sum()
        )

        present_rate = (
            df.loc[
                event_present,
                "spike_label"
            ].mean()
            if present_count
            else np.nan
        )

        absent_rate = (
            df.loc[
                ~event_present,
                "spike_label"
            ].mean()
            if absent_count
            else np.nan
        )

        print(f"\n{name}")

        print(
            f"  Rows with event: "
            f"{present_count:,}"
        )

        print(
            f"  Spike rate with event: "
            f"{present_rate * 100:.2f}%"
        )

        print(
            f"  Rows without event: "
            f"{absent_count:,}"
        )

        print(
            f"  Spike rate without event: "
            f"{absent_rate * 100:.2f}%"
        )

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()