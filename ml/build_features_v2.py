from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/shanghai_t2dm_labeled.csv"
)

OUTPUT_PATH = Path(
    "data/processed/shanghai_t2dm_features_v2.csv"
)


# ============================================================
# HELPERS
# ============================================================

NULL_TEXT = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "未记录",
    "无",
    "-",
    "--",
}


def nonempty_event(series: pd.Series) -> pd.Series:
    """
    Convert a raw event column into a binary event indicator.

    1 = a real value/event is recorded
    0 = blank, missing, zero, or an explicit "not recorded" value

    Important:
    Numeric/text zero is NOT treated as an event.
    """

    s = (
        series
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    # Pure numeric values such as 0 / 0.0 are not events.
    numeric = pd.to_numeric(s, errors="coerce")

    event = (
        ~s.isin(NULL_TEXT)
        & ~(
            numeric.notna()
            & np.isclose(numeric, 0.0)
        )
    )

    return event.astype(np.int8)


def parse_insulin_units(value) -> float:
    """
    Extract insulin units from values such as:

        Gansulin 40R, 6 IU
        Humulin 70/30 8 IU
        insulin aspart 70/30, 18 IU

    Numeric values are returned directly.

    Returns 0.0 when no usable dose is present.
    """

    if pd.isna(value):
        return 0.0

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        if np.isfinite(float(value)):
            return float(value)
        return 0.0

    text = str(value).strip()

    if not text:
        return 0.0

    # Correct regex: optional decimal part, optional spaces before IU.
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*IU\b",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        # Some files may contain a plain numeric string.
        numeric = pd.to_numeric(text, errors="coerce")
        if pd.notna(numeric) and np.isfinite(float(numeric)):
            return float(numeric)
        return 0.0

    return float(sum(float(x) for x in matches))


# ============================================================
# HISTORICAL CONTEXT WINDOW
# ============================================================

def calculate_prior_window(
    timestamps_ns: np.ndarray,
    events: np.ndarray,
    minutes: int,
    amounts: np.ndarray | None = None,
):
    """
    Calculate event counts / amounts in the historical window:

        [current_time - minutes, current_time)

    The current row is excluded.

    This implementation deliberately works from the timestamps of
    actual events rather than a rolling calculation over every CGM row.
    This prevents 30m / 60m / 120m windows from accidentally becoming
    identical or counting unrelated rows.

    Returns:
        counts

    or:
        counts, amount_sums
    """

    timestamps_ns = np.asarray(
        timestamps_ns,
        dtype=np.int64,
    )

    events = np.asarray(
        events,
        dtype=bool,
    )

    n = len(timestamps_ns)

    if n == 0:
        if amounts is None:
            return np.array([], dtype=float)

        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Only actual event timestamps are used.
    event_times = timestamps_ns[events]

    if len(event_times) == 0:
        counts = np.zeros(n, dtype=float)

        if amounts is None:
            return counts

        amount_sums = np.zeros(n, dtype=float)
        return counts, amount_sums

    # The input is sorted by timestamp and should not contain duplicate
    # timestamp rows within a patient/session. We still sort defensively.
    event_times = np.asarray(
        np.sort(event_times),
        dtype=np.int64,
    )

    window_ns = np.int64(
        minutes * 60 * 1_000_000_000
    )

    # For every current timestamp t:
    #
    # left  = first event >= t - window
    # right = first event >= t
    #
    # Therefore event_times[left:right] is exactly:
    #
    # [t - window, t)
    left = np.searchsorted(
        event_times,
        timestamps_ns - window_ns,
        side="left",
    )

    right = np.searchsorted(
        event_times,
        timestamps_ns,
        side="left",
    )

    counts = (
        right - left
    ).astype(float)

    if amounts is None:
        return counts

    amounts = np.asarray(
        amounts,
        dtype=float,
    )

    event_amounts = amounts[events]

    # Sort amount values by the same timestamp ordering.
    sort_order = np.argsort(
        timestamps_ns[events],
        kind="stable",
    )

    event_amounts = event_amounts[sort_order]

    amount_prefix = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(event_amounts),
        ]
    )

    amount_sums = (
        amount_prefix[right]
        - amount_prefix[left]
    )

    return counts, amount_sums


# ============================================================
# PROCESS ONE SESSION
# ============================================================

def process_group(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build context features for one patient/session.
    """

    group = (
        group
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # ========================================================
    # EVENT FLAGS
    # ========================================================

    # A meal event exists when any of the available dietary fields
    # contains a real value.
    group["meal_event"] = (
        (
            nonempty_event(
                group["dietary_intake"]
            )
            |
            nonempty_event(
                group["diet_notes"]
            )
            |
            nonempty_event(
                group["进食量"]
            )
        )
        .astype(np.int8)
    )

    # ========================================================
    # INSULIN
    # ========================================================

    bolus_numeric = (
        pd.to_numeric(
            group["csii_bolus_insulin_iu"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    sc_units = (
        group["insulin_sc"]
        .apply(parse_insulin_units)
        .astype(float)
        .clip(lower=0.0)
    )

    group["insulin_units"] = (
        sc_units.to_numpy()
        + bolus_numeric.to_numpy()
    )

    # IMPORTANT:
    # Do NOT treat a medication-name string itself as an insulin
    # administration event. A positive dose is the event.
    group["insulin_event"] = (
        group["insulin_units"] > 0.0
    ).astype(np.int8)

    # ========================================================
    # NON-INSULIN MEDICATION
    # ========================================================

    group["medication_event"] = (
        nonempty_event(
            group["non_insulin_agents"]
        )
        .astype(np.int8)
    )

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    timestamps_ns = (
        group["timestamp"]
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
    )

    # ========================================================
    # MEAL WINDOWS
    # ========================================================

    meal_events = group["meal_event"].to_numpy()

    group["meal_events_30m"] = calculate_prior_window(
        timestamps_ns,
        meal_events,
        30,
    )

    group["meal_events_60m"] = calculate_prior_window(
        timestamps_ns,
        meal_events,
        60,
    )

    group["meal_events_120m"] = calculate_prior_window(
        timestamps_ns,
        meal_events,
        120,
    )

    # ========================================================
    # INSULIN WINDOWS
    # ========================================================

    insulin_events = (
        group["insulin_event"].to_numpy()
    )

    insulin_amounts = (
        group["insulin_units"].to_numpy(
            dtype=float
        )
    )

    (
        insulin_counts_30,
        insulin_amount_30,
    ) = calculate_prior_window(
        timestamps_ns,
        insulin_events,
        30,
        insulin_amounts,
    )

    (
        insulin_counts_60,
        insulin_amount_60,
    ) = calculate_prior_window(
        timestamps_ns,
        insulin_events,
        60,
        insulin_amounts,
    )

    (
        insulin_counts_120,
        insulin_amount_120,
    ) = calculate_prior_window(
        timestamps_ns,
        insulin_events,
        120,
        insulin_amounts,
    )

    group["insulin_events_30m"] = insulin_counts_30
    group["insulin_events_60m"] = insulin_counts_60
    group["insulin_events_120m"] = insulin_counts_120

    group["insulin_amount_30m"] = insulin_amount_30
    group["insulin_amount_60m"] = insulin_amount_60
    group["insulin_amount_120m"] = insulin_amount_120

    # ========================================================
    # MEDICATION WINDOW
    # ========================================================

    group["medication_events_120m"] = calculate_prior_window(
        timestamps_ns,
        group["medication_event"].to_numpy(),
        120,
    )

    return group


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading labeled dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "patient_id",
        "session_id",
        "timestamp",
        "cgm_mg_dl",
        "spike_label",
        "dietary_intake",
        "diet_notes",
        "进食量",
        "insulin_sc",
        "csii_bolus_insulin_iu",
        "non_insulin_agents",
        "Age (years)",
        "BMI (kg/m2)",
        "Duration of diabetes (years)",
        "Fasting Plasma Glucose (mg/dl)",
        "2-hour Postprandial Plasma Glucose (mg/dl)",
        "HbA1c (mmol/mol)",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    # ========================================================
    # CLEAN
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="mixed",
    )

    df["cgm_mg_dl"] = pd.to_numeric(
        df["cgm_mg_dl"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "patient_id",
            "session_id",
            "timestamp",
            "cgm_mg_dl",
            "spike_label",
        ]
    ).copy()

    df = (
        df
        .sort_values(
            [
                "patient_id",
                "session_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Input rows: {len(df):,}"
    )

    print(
        f"Patients:   {df['patient_id'].nunique()}"
    )

    print(
        f"Sessions:   {df['session_id'].nunique()}"
    )

    # ========================================================
    # CONTEXT FEATURES
    # ========================================================

    print(
        "\nBuilding corrected timestamp-based context features..."
    )

    parts = []

    for _, group in df.groupby(
        [
            "patient_id",
            "session_id",
        ],
        sort=False,
    ):
        parts.append(
            process_group(group)
        )

    df = pd.concat(
        parts,
        ignore_index=True,
    )

    # ========================================================
    # CLINICAL FEATURES
    # ========================================================

    clinical_columns = {
        "age":
            "Age (years)",

        "bmi":
            "BMI (kg/m2)",

        "diabetes_duration":
            "Duration of diabetes (years)",

        "fasting_glucose":
            "Fasting Plasma Glucose (mg/dl)",

        "postprandial_glucose":
            "2-hour Postprandial Plasma Glucose (mg/dl)",

        "hba1c":
            "HbA1c (mmol/mol)",
    }

    for output_name, source_name in clinical_columns.items():
        df[output_name] = pd.to_numeric(
            df[source_name],
            errors="coerce",
        )

    # ========================================================
    # GLUCOSE HISTORY
    # ========================================================

    group_cols = [
        "patient_id",
        "session_id",
    ]

    for minutes in range(
        0,
        121,
        15,
    ):

        periods = minutes // 15

        if periods == 0:

            df["glucose_t0"] = (
                df["cgm_mg_dl"]
            )

        else:

            df[
                f"glucose_t_minus_{minutes}m"
            ] = (
                df
                .groupby(group_cols)["cgm_mg_dl"]
                .shift(periods)
            )

    # ========================================================
    # GLUCOSE CHANGES
    # ========================================================

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

    # ========================================================
    # GLUCOSE STATISTICS
    # ========================================================

    history_1h = [
        "glucose_t_minus_60m",
        "glucose_t_minus_45m",
        "glucose_t_minus_30m",
        "glucose_t_minus_15m",
        "glucose_t0",
    ]

    history_2h = [
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

    df["glucose_mean_1h"] = (
        df[history_1h].mean(axis=1)
    )

    df["glucose_std_1h"] = (
        df[history_1h].std(axis=1)
    )

    df["glucose_min_1h"] = (
        df[history_1h].min(axis=1)
    )

    df["glucose_max_1h"] = (
        df[history_1h].max(axis=1)
    )

    df["glucose_mean_2h"] = (
        df[history_2h].mean(axis=1)
    )

    df["glucose_std_2h"] = (
        df[history_2h].std(axis=1)
    )

    df["glucose_min_2h"] = (
        df[history_2h].min(axis=1)
    )

    df["glucose_max_2h"] = (
        df[history_2h].max(axis=1)
    )

    # ========================================================
    # SLOPES
    # ========================================================

    df["slope_30m"] = (
        df["delta_30m"] / 30.0
    )

    df["slope_60m"] = (
        df["delta_60m"] / 60.0
    )

    df["slope_120m"] = (
        df["delta_120m"] / 120.0
    )

    # ========================================================
    # HISTORY REQUIREMENT
    # ========================================================

    required_history = [
        f"glucose_t_minus_{m}m"
        for m in range(
            15,
            121,
            15,
        )
    ]

    required_history.append(
        "glucose_t0"
    )

    before = len(df)

    df = df.dropna(
        subset=required_history
    ).copy()

    print(
        f"\nRows before history filtering: "
        f"{before:,}"
    )

    print(
        f"Rows removed:                  "
        f"{before - len(df):,}"
    )

    print(
        f"Rows after filtering:          "
        f"{len(df):,}"
    )

    # ========================================================
    # FEATURE LIST
    # ========================================================

    glucose_features = [
        "glucose_t0",
        "glucose_t_minus_15m",
        "glucose_t_minus_30m",
        "glucose_t_minus_45m",
        "glucose_t_minus_60m",
        "glucose_t_minus_75m",
        "glucose_t_minus_90m",
        "glucose_t_minus_105m",
        "glucose_t_minus_120m",

        "delta_15m",
        "delta_30m",
        "delta_60m",
        "delta_120m",

        "glucose_mean_1h",
        "glucose_std_1h",
        "glucose_min_1h",
        "glucose_max_1h",

        "glucose_mean_2h",
        "glucose_std_2h",
        "glucose_min_2h",
        "glucose_max_2h",

        "slope_30m",
        "slope_60m",
        "slope_120m",
    ]

    clinical_features = [
        "age",
        "bmi",
        "diabetes_duration",
        "fasting_glucose",
        "postprandial_glucose",
        "hba1c",
    ]

    context_features = [
        "meal_events_30m",
        "meal_events_60m",
        "meal_events_120m",

        "insulin_events_30m",
        "insulin_events_60m",
        "insulin_events_120m",

        "insulin_amount_30m",
        "insulin_amount_60m",
        "insulin_amount_120m",

        "medication_events_120m",
    ]

    feature_columns = (
        glucose_features
        + clinical_features
        + context_features
    )

    metadata_columns = [
        "patient_id",
        "session_id",
        "session_number",
        "timestamp",
        "source_file",
        "spike_label",
    ]

    # Preserve metadata only when present.
    metadata_columns = [
        col
        for col in metadata_columns
        if col in df.columns
    ]

    result = df[
        metadata_columns
        + feature_columns
    ].copy()

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    forbidden = {
        "future_max_2h",
        "future_increase_2h",
    }

    leakage = (
        forbidden
        .intersection(result.columns)
    )

    if leakage:
        raise RuntimeError(
            "Future leakage columns found: "
            f"{sorted(leakage)}"
        )

    # ========================================================
    # CONTEXT SANITY CHECK
    # ========================================================

    print(
        "\n========== CONTEXT SANITY CHECK =========="
    )

    for minutes in [30, 60, 120]:

        meal_col = (
            f"meal_events_{minutes}m"
        )

        insulin_col = (
            f"insulin_events_{minutes}m"
        )

        print(
            f"Meal {minutes:>3}m:    "
            f"min={result[meal_col].min():.0f}  "
            f"max={result[meal_col].max():.0f}  "
            f"mean={result[meal_col].mean():.4f}"
        )

        print(
            f"Insulin {minutes:>3}m: "
            f"min={result[insulin_col].min():.0f}  "
            f"max={result[insulin_col].max():.0f}  "
            f"mean={result[insulin_col].mean():.4f}"
        )

    # Window counts must be monotonic.
    if (
        result["meal_events_30m"]
        > result["meal_events_60m"]
    ).any():

        raise RuntimeError(
            "Meal 30m count exceeds Meal 60m count."
        )

    if (
        result["meal_events_60m"]
        > result["meal_events_120m"]
    ).any():

        raise RuntimeError(
            "Meal 60m count exceeds Meal 120m count."
        )

    if (
        result["insulin_events_30m"]
        > result["insulin_events_60m"]
    ).any():

        raise RuntimeError(
            "Insulin 30m count exceeds Insulin 60m count."
        )

    if (
        result["insulin_events_60m"]
        > result["insulin_events_120m"]
    ).any():

        raise RuntimeError(
            "Insulin 60m count exceeds Insulin 120m count."
        )

    # The shorter and longer windows should not collapse into the
    # exact same vector unless there genuinely are no events that
    # fall in the extra historical time.
    meal_equal_30_60 = np.array_equal(
        result["meal_events_30m"].to_numpy(),
        result["meal_events_60m"].to_numpy(),
    )

    insulin_equal_30_60 = np.array_equal(
        result["insulin_events_30m"].to_numpy(),
        result["insulin_events_60m"].to_numpy(),
    )

    if meal_equal_30_60:
        print(
            "WARNING: Meal 30m and 60m vectors are identical."
        )

    if insulin_equal_30_60:
        print(
            "WARNING: Insulin 30m and 60m vectors are identical."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n========== V2 FEATURE SUMMARY =========="
    )

    print(
        f"Final rows:  {len(result):,}"
    )

    print(
        f"Patients:    "
        f"{result['patient_id'].nunique()}"
    )

    print(
        f"Sessions:    "
        f"{result['session_id'].nunique()}"
    )

    print(
        f"Features:    "
        f"{len(feature_columns)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        result["spike_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n========== CONTEXT FEATURE STATISTICS =========="
    )

    print(
        result[context_features]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
        .to_string()
    )

    print(
        "\n========== MISSING CONTEXT VALUES =========="
    )

    print(
        result[
            context_features
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved V2 feature dataset to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n========== DONE =========="
    )


if __name__ == "__main__":
    main()
