import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="EDA for processed T2DM glucose-spike dataset")
    parser.add_argument("--input", default="processed_t2dm.csv", help="Path to processed CSV")
    parser.add_argument("--output-dir", default="eda_output", help="Folder for reports and plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Basic cleanup only for EDA
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    lines = []
    lines.append("T2DM EDA REPORT")
    lines.append("=" * 60)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    if "subject_id" in df.columns:
        lines.append(f"Unique patients: {df['subject_id'].nunique():,}")
    if "session_id" in df.columns:
        lines.append(f"Unique sessions: {df['session_id'].nunique():,}")
    lines.append("")

    # Duplicate checks
    lines.append("DUPLICATES")
    lines.append("-" * 60)
    lines.append(f"Exact duplicate rows: {df.duplicated().sum():,}")
    if all(c in df.columns for c in ["subject_id", "session_id", "timestamp"]):
        lines.append(
            "Duplicate patient-session-timestamp rows: "
            f"{df.duplicated(['subject_id', 'session_id', 'timestamp']).sum():,}"
        )
    lines.append("")

    # Missing values
    missing = (
        df.isna().sum()
        .to_frame("missing_count")
        .assign(missing_pct=lambda x: 100 * x["missing_count"] / len(df))
        .sort_values("missing_pct", ascending=False)
    )
    lines.append("MISSING VALUES")
    lines.append("-" * 60)
    lines.append(missing.to_string())
    lines.append("")

    # Target distribution
    if "glucose_spike_next_2h" in df.columns:
        counts = df["glucose_spike_next_2h"].value_counts(dropna=False).sort_index()
        lines.append("TARGET DISTRIBUTION")
        lines.append("-" * 60)
        lines.append(counts.to_string())
        valid_target = df["glucose_spike_next_2h"].dropna()
        if len(valid_target):
            pos_rate = 100 * (valid_target == 1).mean()
            lines.append(f"Positive spike rate: {pos_rate:.2f}%")
        lines.append("")

        plt.figure(figsize=(7, 5))
        counts.rename(index={0: "No spike", 1: "Spike"}).plot(kind="bar")
        plt.title("Glucose Spike Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of observations")
        plt.tight_layout()
        plt.savefig(out / "01_target_distribution.png", dpi=160)
        plt.close()

    # Glucose statistics
    glucose_col = "cgm_mg_dl" if "cgm_mg_dl" in df.columns else None
    if glucose_col:
        g = df[glucose_col].dropna()
        lines.append("CGM GLUCOSE")
        lines.append("-" * 60)
        if len(g):
            lines.append(f"Count: {len(g):,}")
            lines.append(f"Mean: {g.mean():.2f}")
            lines.append(f"Median: {g.median():.2f}")
            lines.append(f"Std: {g.std():.2f}")
            lines.append(f"Min: {g.min():.2f}")
            lines.append(f"Max: {g.max():.2f}")
            lines.append("")

            plt.figure(figsize=(8, 5))
            plt.hist(g, bins=50)
            plt.title("CGM Glucose Distribution")
            plt.xlabel("CGM glucose (mg/dL)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(out / "02_glucose_distribution.png", dpi=160)
            plt.close()

    # Patient observation counts
    if "subject_id" in df.columns:
        patient_counts = df.groupby("subject_id").size().sort_values(ascending=False)
        lines.append("PATIENT OBSERVATION COUNTS")
        lines.append("-" * 60)
        lines.append(patient_counts.describe().to_string())
        lines.append("")
        plt.figure(figsize=(9, 5))
        plt.hist(patient_counts, bins=30)
        plt.title("Observations per Patient")
        plt.xlabel("Observations")
        plt.ylabel("Number of patients")
        plt.tight_layout()
        plt.savefig(out / "03_observations_per_patient.png", dpi=160)
        plt.close()

    # Spike rate by patient
    if "subject_id" in df.columns and "glucose_spike_next_2h" in df.columns:
        patient_spike = df.groupby("subject_id")["glucose_spike_next_2h"].mean().dropna()
        lines.append("PATIENT-LEVEL SPIKE RATE")
        lines.append("-" * 60)
        lines.append(patient_spike.describe().to_string())
        lines.append("")
        plt.figure(figsize=(8, 5))
        plt.hist(100 * patient_spike, bins=30)
        plt.title("Patient-Level Spike Rate")
        plt.xlabel("Spike rate (%)")
        plt.ylabel("Number of patients")
        plt.tight_layout()
        plt.savefig(out / "04_patient_spike_rate.png", dpi=160)
        plt.close()

    # Correlation heatmap for top numeric features
    numeric = df.select_dtypes(include=np.number).copy()
    leakage_cols = {
        "future_max_glucose_2h_mg_dl",
        "future_glucose_rise_2h_mg_dl",
        "glucose_spike_next_2h",
    }
    corr_cols = [c for c in numeric.columns if c not in leakage_cols]

    if "glucose_spike_next_2h" in numeric.columns:
        corr_with_target = (
            numeric[corr_cols]
            .corrwith(numeric["glucose_spike_next_2h"])
            .abs()
            .sort_values(ascending=False)
        )
        top = corr_with_target.head(12).index.tolist()
        if len(top) >= 2:
            corr = numeric[top].corr()
            plt.figure(figsize=(10, 8))
            plt.imshow(corr, aspect="auto")
            plt.colorbar(label="Correlation")
            plt.xticks(range(len(top)), top, rotation=90)
            plt.yticks(range(len(top)), top)
            plt.title("Top Numeric Feature Correlations")
            plt.tight_layout()
            plt.savefig(out / "05_feature_correlation.png", dpi=160)
            plt.close()

            lines.append("TOP NUMERIC FEATURES CORRELATED WITH TARGET")
            lines.append("-" * 60)
            lines.append(corr_with_target.head(20).to_string())
            lines.append("")

    # Leakage warning
    lines.append("MODEL LEAKAGE CHECK")
    lines.append("-" * 60)
    for c in sorted(leakage_cols):
        lines.append(f"{c}: {'PRESENT - EXCLUDE FROM FEATURES' if c in df.columns else 'not present'}")
    lines.append("")

    # Columns that are almost entirely missing
    high_missing = missing[missing["missing_pct"] >= 95]
    lines.append("COLUMNS >= 95% MISSING")
    lines.append("-" * 60)
    lines.append(high_missing.to_string())
    lines.append("")

    report_path = out / "eda_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"EDA complete.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    if "subject_id" in df.columns:
        print(f"Unique patients: {df['subject_id'].nunique():,}")
    if "glucose_spike_next_2h" in df.columns:
        print(f"Spike-positive rows: {(df['glucose_spike_next_2h'] == 1).sum():,}")
        print(f"Spike rate: {100 * (df['glucose_spike_next_2h'] == 1).mean():.2f}%")
    print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {out}")

if __name__ == "__main__":
    main()
