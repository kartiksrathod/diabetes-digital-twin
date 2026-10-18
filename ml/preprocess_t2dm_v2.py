#!/usr/bin/env python3
"""Preprocess Shanghai T2DM dataset into a clean modeling table.

Input:  diabetes_datasets.zip
Output: processed_t2dm.csv

The script:
1. Extracts the ZIP.
2. Reads Shanghai_T2DM_Summary.xlsx for patient/session-level EHR data.
3. Reads every Shanghai_T2DM .xls/.xlsx CGM session.
4. Keeps T2DM only, merges EHR onto each CGM timestamp.
5. Creates history-only glucose/insulin/diet features.
6. Creates a candidate next-2-hour glucose-spike label.

IMPORTANT: The spike definition is a research choice:
future glucose >= 180 mg/dL AND future rise from current glucose >= 30 mg/dL.
Change it when your project/mentor finalizes the label definition.
"""

import argparse
import re
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ZIP = "diabetes_datasets.zip"
DEFAULT_OUTPUT = "processed_t2dm.csv"
HORIZON_MINUTES = 120
SPIKE_ABSOLUTE_THRESHOLD = 180.0
SPIKE_RISE_THRESHOLD = 30.0
TIMESTAMP_COL = "timestamp"
CGM_COL = "cgm_mg_dl"


def clean_column_name(name) -> str:
    text = str(name).strip().lower()
    text = text.replace("–", "-").replace("—", "-").replace("，", ",")
    text = re.sub(r"\s+", "_", text)
    text = text.replace("/", "_")
    text = re.sub(r"[^a-z0-9_.,()%-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def read_excel_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The ZIP contains legacy .xls files. Run: pip install xlrd"
            ) from exc
        return pd.read_excel(path, engine="xlrd")
    if suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported Excel format: {path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        return
    # Fresh extraction avoids stale files from earlier attempts.
    for child in extract_dir.iterdir():
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")


def find_summary_file(root: Path) -> Path:
    matches = list(root.rglob("Shanghai_T2DM_Summary.xlsx"))
    if not matches:
        raise FileNotFoundError("Shanghai_T2DM_Summary.xlsx not found in ZIP")
    return matches[0]


def normalize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary.columns = [clean_column_name(c) for c in summary.columns]

    rename = {
        "patient_number": "session_id",
        "gender_(female=1,_male=2)": "gender",
        "age_(years)": "age_years",
        "height_(m)": "height_m",
        "weight_(kg)": "weight_kg",
        "bmi_(kg_m2)": "bmi",
        "smoking_history_(pack_year)": "smoking_pack_year",
        "alcohol_drinking_history_(drinker_non-drinker)": "alcohol_status",
        "type_of_diabetes": "diabetes_type",
        "duration_of_diabetes_(years)": "diabetes_duration_years",
        "acute_diabetic_complications": "acute_diabetic_complications",
        "diabetic_macrovascular_complications": "macrovascular_complications",
        "diabetic_microvascular_complications": "microvascular_complications",
        "comorbidities": "comorbidities",
        "hypoglycemic_agents": "hypoglycemic_agents",
        "other_agents": "other_agents",
        "fasting_plasma_glucose_(mg_dl)": "fasting_plasma_glucose_mg_dl",
        "2-hour_postprandial_plasma_glucose_(mg_dl)": "postprandial_2h_glucose_mg_dl",
        "fasting_c-peptide_(nmol_l)": "fasting_c_peptide_nmol_l",
        "2-hour_postprandial_c-peptide_(nmol_l)": "postprandial_2h_c_peptide_nmol_l",
        "fasting_insulin_(pmol_l)": "fasting_insulin_pmol_l",
        "2-hour_postprandial_insulin_(pmol_l)": "postprandial_2h_insulin_pmol_l",
        "hba1c_(mmol_mol)": "hba1c_mmol_mol",
        "glycated_albumin_(%)": "glycated_albumin_pct",
        "total_cholesterol_(mmol_l)": "total_cholesterol_mmol_l",
        "triglyceride_(mmol_l)": "triglyceride_mmol_l",
        "high-density_lipoprotein_cholesterol_(mmol_l)": "hdl_mmol_l",
        "low-density_lipoprotein_cholesterol_(mmol_l)": "ldl_mmol_l",
        "creatinine_(umol_l)": "creatinine_umol_l",
        "estimated_glomerular_filtration_rate_(ml_min_1.73m2)": "egfr_ml_min_1_73m2",
        "uric_acid_(mmol_l)": "uric_acid_mmol_l",
        "blood_urea_nitrogen_(mmol_l)": "bun_mmol_l",
        "hypoglycemia_(yes_no)": "hypoglycemia_history",
    }
    summary = summary.rename(columns=rename)

    if "session_id" not in summary.columns:
        raise KeyError("Patient Number column was not found in the T2DM summary")

    summary["session_id"] = summary["session_id"].astype("string").str.strip()
    summary["subject_id"] = summary["session_id"].str.split("_", n=1).str[0]

    # Keep original EHR text fields; convert only these known numeric fields.
    numeric_ehr = [
        "gender", "age_years", "height_m", "weight_kg", "bmi",
        "smoking_pack_year", "diabetes_duration_years",
        "fasting_plasma_glucose_mg_dl", "postprandial_2h_glucose_mg_dl",
        "fasting_c_peptide_nmol_l", "postprandial_2h_c_peptide_nmol_l",
        "fasting_insulin_pmol_l", "postprandial_2h_insulin_pmol_l",
        "hba1c_mmol_mol", "glycated_albumin_pct", "total_cholesterol_mmol_l",
        "triglyceride_mmol_l", "hdl_mmol_l", "ldl_mmol_l",
        "creatinine_umol_l", "egfr_ml_min_1_73m2", "uric_acid_mmol_l", "bun_mmol_l"
    ]
    for col in numeric_ehr:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    # No duplicate session IDs are expected in the summary.
    if summary["session_id"].duplicated().any():
        dup = summary.loc[summary["session_id"].duplicated(), "session_id"].tolist()
        raise ValueError(f"Duplicate session IDs in summary: {dup[:10]}")

    return summary


def normalize_time_series(raw: pd.DataFrame, session_id: str) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    time_candidates = [c for c in ["date", "datetime", "timestamp"] if c in df.columns]
    if not time_candidates:
        time_candidates = [c for c in df.columns if "date" in c or "time" in c]
    if not time_candidates:
        raise KeyError(f"No date/time column found in {session_id}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[time_candidates[0]], errors="coerce")

    aliases = {
        "cgm_mg_dl": ["cgm_(mg_dl)", "cgm_(mg/dl)", "cgm_mg_dl", "cgm"],
        "cbg_mg_dl": ["cbg_(mg_dl)", "cbg_(mg/dl)", "cbg_mg_dl", "cbg"],
        "blood_ketone_mmol_l": ["blood_ketone_(mmol_l)", "blood_ketone_mmol_l"],
        "insulin_sc": ["insulin_dose_-_s.c.", "insulin_dose_s_c", "insulin_sc"],
        "insulin_iv": ["insulin_dose_-_i.v.", "insulin_dose_i_v", "insulin_iv"],
        "csii_bolus_insulin_iu": [
            "csii_-_bolus_insulin_(novolin_r,_iu)",
            "csii_bolus_insulin_(novolin_r,_iu)",
            "csii_bolus_insulin_iu",
        ],
        "csii_basal_insulin_iu_h": [
            "csii_-_basal_insulin_(novolin_r,_iu_h)",
            "csii_basal_insulin_(novolin_r,_iu_h)",
            "csii_basal_insulin_iu_h",
        ],
    }
    text_aliases = {
        "dietary_intake": ["dietary_intake"],
        "non_insulin_hypoglycemic_agents": [
            "non-insulin_hypoglycemic_agents",
            "non_insulin_hypoglycemic_agents",
        ],
    }

    for dst, choices in aliases.items():
        found = next((c for c in choices if c in df.columns), None)
        df[dst] = pd.to_numeric(df[found], errors="coerce") if found else np.nan

    for dst, choices in text_aliases.items():
        found = next((c for c in choices if c in df.columns), None)
        df[dst] = df[found].astype("string") if found else pd.Series(pd.NA, index=df.index, dtype="string")

    df["has_diet_event"] = (
        df["dietary_intake"].notna() & df["dietary_intake"].str.strip().ne("")
    ).astype("int8")
    df["has_non_insulin_agent_event"] = (
        df["non_insulin_hypoglycemic_agents"].notna()
        & df["non_insulin_hypoglycemic_agents"].str.strip().ne("")
    ).astype("int8")

    df["session_id"] = str(session_id).strip()
    df["subject_id"] = str(session_id).split("_", 1)[0]

    keep = [
        "subject_id", "session_id", TIMESTAMP_COL, "cgm_mg_dl", "cbg_mg_dl",
        "blood_ketone_mmol_l", "insulin_sc", "insulin_iv",
        "csii_bolus_insulin_iu", "csii_basal_insulin_iu_h",
        "has_diet_event", "has_non_insulin_agent_event",
        "dietary_intake", "non_insulin_hypoglycemic_agents",
    ]
    df = df[keep]
    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (_, _), g in df.groupby(["subject_id", "session_id"], sort=False):
        g = g.sort_values(TIMESTAMP_COL).copy()
        for steps, minutes in [(1, 15), (2, 30), (4, 60)]:
            g[f"glucose_lag_{minutes}m"] = g[CGM_COL].shift(steps)
            g[f"glucose_change_{minutes}m"] = g[CGM_COL] - g[f"glucose_lag_{minutes}m"]
        g["time_since_prev_min"] = g[TIMESTAMP_COL].diff().dt.total_seconds() / 60.0
        g["glucose_slope_30m"] = g["glucose_change_30m"] / 30.0
        g["glucose_slope_60m"] = g["glucose_change_60m"] / 60.0

        indexed = g.set_index(TIMESTAMP_COL)[CGM_COL]
        for window, mins in [("30min", 30), ("60min", 60)]:
            g[f"glucose_mean_{mins}m"] = indexed.rolling(window, min_periods=2).mean().to_numpy()
            g[f"glucose_std_{mins}m"] = indexed.rolling(window, min_periods=2).std().to_numpy()
        g["glucose_min_60m"] = indexed.rolling("60min", min_periods=2).min().to_numpy()
        g["glucose_max_60m"] = indexed.rolling("60min", min_periods=2).max().to_numpy()

        for col in ["insulin_sc", "insulin_iv", "csii_bolus_insulin_iu"]:
            s = pd.Series(pd.to_numeric(g[col], errors="coerce").fillna(0.0).to_numpy(), index=g[TIMESTAMP_COL])
            g[f"{col}_sum_2h"] = s.rolling("120min", min_periods=1).sum().to_numpy()
        for col in ["has_diet_event", "has_non_insulin_agent_event"]:
            s = pd.Series(g[col].astype(float).to_numpy(), index=g[TIMESTAMP_COL])
            g[f"{col}_count_2h"] = s.rolling("120min", min_periods=1).sum().to_numpy()
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def add_future_label(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (_, _), g in df.groupby(["subject_id", "session_id"], sort=False):
        g = g.sort_values(TIMESTAMP_COL).copy()
        ts = g[TIMESTAMP_COL].to_numpy(dtype="datetime64[ns]")
        glucose = g[CGM_COL].to_numpy(dtype=float)
        n = len(g)
        future_max = np.full(n, np.nan)
        future_rise = np.full(n, np.nan)
        labels = np.full(n, np.nan)
        q = deque()
        right = 0
        for i in range(n):
            end = ts[i] + np.timedelta64(HORIZON_MINUTES, "m")
            if right < i + 1:
                right = i + 1
            while right < n and ts[right] <= end:
                v = glucose[right]
                if np.isfinite(v):
                    while q and glucose[q[-1]] <= v:
                        q.pop()
                    q.append(right)
                right += 1
            while q and q[0] <= i:
                q.popleft()
            if q and np.isfinite(glucose[i]):
                mx = float(glucose[q[0]])
                future_max[i] = mx
                future_rise[i] = mx - glucose[i]
                labels[i] = int(mx >= SPIKE_ABSOLUTE_THRESHOLD and (mx - glucose[i]) >= SPIKE_RISE_THRESHOLD)
        g["future_max_glucose_2h_mg_dl"] = future_max
        g["future_glucose_rise_2h_mg_dl"] = future_rise
        g["glucose_spike_next_2h"] = labels
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def build_dataset(zip_path: Path, output: Path, extract_dir: Path) -> pd.DataFrame:
    extract_zip(zip_path, extract_dir)
    summary = normalize_summary(read_excel_file(find_summary_file(extract_dir)))
    t2_dir = extract_dir / "Shanghai_T2DM"
    if not t2_dir.exists():
        raise FileNotFoundError("Shanghai_T2DM directory not found")

    session_files = sorted(p for p in t2_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".xls", ".xlsx"} and not p.name.startswith("~$"))
    expected = set(summary["session_id"].astype(str))

    sessions, errors = [], []
    for i, path in enumerate(session_files, 1):
        try:
            sid = path.stem
            sessions.append(normalize_time_series(read_excel_file(path), sid))
            print(f"[{i:03d}/{len(session_files):03d}] {path.name}")
        except Exception as exc:
            errors.append((path.name, str(exc)))
            print(f"ERROR {path.name}: {exc}")

    if not sessions:
        raise RuntimeError("No T2DM session files could be read")

    ts = pd.concat(sessions, ignore_index=True)
    ts = add_temporal_features(ts)
    ts = add_future_label(ts)

    merged = ts.merge(summary, on=["session_id", "subject_id"], how="left", validate="many_to_one")

    if "age_years" not in merged.columns:
        raise RuntimeError("EHR merge failed: age_years column is missing")
    unmatched = merged["age_years"].isna().mean()
    print(f"EHR unmatched row rate: {unmatched:.2%}")
    if unmatched > 0:
        print("WARNING: some CGM rows did not match the EHR summary")

    required = [CGM_COL, "glucose_mean_60m", "glucose_std_60m", "glucose_spike_next_2h"]
    processed = merged.dropna(subset=required).copy()

    # IMPORTANT: only known numeric columns are converted.
    numeric_cols = [
        "cgm_mg_dl", "cbg_mg_dl", "blood_ketone_mmol_l", "insulin_sc", "insulin_iv",
        "csii_bolus_insulin_iu", "csii_basal_insulin_iu_h", "has_diet_event",
        "has_non_insulin_agent_event", "time_since_prev_min", "glucose_lag_15m",
        "glucose_lag_30m", "glucose_lag_60m", "glucose_change_15m", "glucose_change_30m",
        "glucose_change_60m", "glucose_slope_30m", "glucose_slope_60m", "glucose_mean_30m",
        "glucose_std_30m", "glucose_mean_60m", "glucose_std_60m", "glucose_min_60m",
        "glucose_max_60m", "insulin_sc_sum_2h", "insulin_iv_sum_2h", "csii_bolus_insulin_iu_sum_2h",
        "has_diet_event_count_2h", "has_non_insulin_agent_event_count_2h",
        "future_max_glucose_2h_mg_dl", "future_glucose_rise_2h_mg_dl", "glucose_spike_next_2h",
        "gender", "age_years", "height_m", "weight_kg", "bmi", "smoking_pack_year",
        "diabetes_duration_years", "fasting_plasma_glucose_mg_dl", "postprandial_2h_glucose_mg_dl",
        "fasting_c_peptide_nmol_l", "postprandial_2h_c_peptide_nmol_l", "fasting_insulin_pmol_l",
        "postprandial_2h_insulin_pmol_l", "hba1c_mmol_mol", "glycated_albumin_pct",
        "total_cholesterol_mmol_l", "triglyceride_mmol_l", "hdl_mmol_l", "ldl_mmol_l",
        "creatinine_umol_l", "egfr_ml_min_1_73m2", "uric_acid_mmol_l", "bun_mmol_l",
    ]
    for col in numeric_cols:
        if col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors="coerce")

    processed[TIMESTAMP_COL] = pd.to_datetime(processed[TIMESTAMP_COL], errors="coerce")
    processed = processed.sort_values(["subject_id", "session_id", TIMESTAMP_COL]).reset_index(drop=True)
    processed[TIMESTAMP_COL] = processed[TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M:%S")

    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False, encoding="utf-8-sig")

    positives = int(processed["glucose_spike_next_2h"].sum())
    print("\n================ RESULT ================")
    print(f"Session files found:       {len(session_files)}")
    print(f"Sessions read:             {len(sessions)}")
    print(f"Session errors:            {len(errors)}")
    print(f"Rows after preprocessing:  {len(processed):,}")
    print(f"Unique patients:            {processed['subject_id'].nunique():,}")
    print(f"Positive spike rows:        {positives:,}")
    print(f"Positive spike rate:        {positives / len(processed):.2%}")
    for c in ["diabetes_type", "hypoglycemic_agents", "other_agents", "alcohol_status", "acute_diabetic_complications"]:
        if c in processed.columns:
            print(f"Non-missing {c}:           {processed[c].notna().sum():,} ({processed[c].notna().mean():.1%})")
    print(f"Output:                     {output.resolve()}")
    if errors:
        print("\nFirst 10 file errors:")
        for name, err in errors[:10]:
            print(f"- {name}: {err}")
    return processed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--extract-dir", default="extracted_t2dm_v2")
    args = p.parse_args()
    build_dataset(Path(args.zip).resolve(), Path(args.output).resolve(), Path(args.extract_dir).resolve())


if __name__ == "__main__":
    main()
