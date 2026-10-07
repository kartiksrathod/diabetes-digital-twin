from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path(
    "data/processed/shanghai_t2dm_features_v1.csv"
)

TRAIN_PATH = Path(
    "data/processed/train.csv"
)

VAL_PATH = Path(
    "data/processed/validation.csv"
)

TEST_PATH = Path(
    "data/processed/test.csv"
)


RANDOM_STATE = 42


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    # One unique patient should belong to exactly one split.
    patients = (
        df["patient_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(f"Total patients: {len(patients)}")

    # 70% train, 30% temporary.
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    # Split remaining 30% equally:
    # 15% validation, 15% test.
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    df_patient = df["patient_id"].astype(str)

    train_df = df[df_patient.isin(train_set)].copy()
    val_df = df[df_patient.isin(val_set)].copy()
    test_df = df[df_patient.isin(test_set)].copy()

    # Sanity checks.
    assert set(train_df["patient_id"].astype(str)).isdisjoint(
        set(val_df["patient_id"].astype(str))
    )

    assert set(train_df["patient_id"].astype(str)).isdisjoint(
        set(test_df["patient_id"].astype(str))
    )

    assert set(val_df["patient_id"].astype(str)).isdisjoint(
        set(test_df["patient_id"].astype(str))
    )

    print("\n========== PATIENT SPLIT ==========")

    print(
        f"Train patients:       {len(train_set)}"
    )
    print(
        f"Validation patients:  {len(val_set)}"
    )
    print(
        f"Test patients:        {len(test_set)}"
    )

    print("\n========== ROW SPLIT ==========")

    print(
        f"Train rows:            {len(train_df):,}"
    )
    print(
        f"Validation rows:       {len(val_df):,}"
    )
    print(
        f"Test rows:             {len(test_df):,}"
    )

    def show_target(name, data):
        counts = data["spike_label"].value_counts()
        total = len(data)

        positive = int(counts.get(1, 0))
        negative = int(counts.get(0, 0))

        print(
            f"\n{name}"
            f"\n  No spike: {negative:,} "
            f"({negative / total * 100:.2f}%)"
            f"\n  Spike:    {positive:,} "
            f"({positive / total * 100:.2f}%)"
        )

    print("\n========== TARGET DISTRIBUTION ==========")

    show_target("TRAIN", train_df)
    show_target("VALIDATION", val_df)
    show_target("TEST", test_df)

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    val_df.to_csv(
        VAL_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\n========== SAVED ==========")

    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()