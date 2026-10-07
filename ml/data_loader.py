import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd

T2DM_DIR_NAME = "Shanghai_T2DM"
SUMMARY_NAME = "Shanghai_T2DM_Summary.xlsx"

CANONICAL_COLUMNS = {
    "timestamp": "timestamp",
    "cgm_mg_dl": "cgm_mg_dl",
    "cbg_mg_dl": "cbg_mg_dl",
    "blood_ketone_mmol_l": "blood_ketone_mmol_l",
    "dietary_intake": "dietary_intake",
    "diet_notes": "diet_notes",
    "insulin_sc": "insulin_sc",
    "non_insulin_agents": "non_insulin_agents",
    "csii_bolus_insulin_iu": "csii_bolus_insulin_iu",
    "csii_basal_insulin_iu_h": "csii_basal_insulin_iu_h",
    "insulin_iv": "insulin_iv",
}


def clean_header(value) -> str:
    """Normalize a column header for matching."""
    s = str(value).strip()
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s


def header_key(value) -> str:
    """Create a tolerant comparison key."""
    s = clean_header(value).lower()

    # Normalize common textual variations.
    replacements = {
        "／": "/",
        "－": "-",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    # Remove spaces for tolerant matching.
    s = s.replace(" ", "")
    return s


def normalize_dynamic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ShanghaiT2DM raw-file headers.

    The dataset contains older .xls files with slightly different
    header formatting, so matching is intentionally tolerant.
    """

    rename_map = {}

    for original in df.columns:
        key = header_key(original)
        text = clean_header(original)

        # Timestamp
        if key in {"date", "datetime", "timestamp", "time", "date/time"}:
            rename_map[original] = "timestamp"

       # CGM
        elif key == "cgm" or (
            key.startswith("cgm") and ("mg/dl" in key or "mgdl" in key)
        ):
            rename_map[original] = "cgm_mg_dl"

          # CBG
        elif key == "cbg" or (
            key.startswith("cbg") and ("mg/dl" in key or "mgdl" in key)
        ):
            rename_map[original] = "cbg_mg_dl"

        # Blood ketone
        elif "bloodketone" in key:
            rename_map[original] = "blood_ketone_mmol_l"

        # Dietary intake
        elif key == "dietaryintake":
            rename_map[original] = "dietary_intake"

        # Chinese diet column
        elif "进食量" in text:
            rename_map[original] = "进食量"

        elif text == "饮食":
            rename_map[original] = "diet_notes"

        # Insulin
        elif "insulindose" in key and ("s.c." in text.lower() or "sc" in key):
            rename_map[original] = "insulin_sc"

        elif "non-insulin" in text.lower() or "noninsulin" in key:
            rename_map[original] = "non_insulin_agents"

        elif "csii" in key and "bolus" in key:
            rename_map[original] = "csii_bolus_insulin_iu"

        elif "csii" in key and "basal" in key:
            rename_map[original] = "csii_basal_insulin_iu_h"

        elif "insulindose" in key and ("i.v." in text.lower() or "iv" in key):
            rename_map[original] = "insulin_iv"

    return df.rename(columns=rename_map)


def parse_ids(stem: str) -> tuple[str, str, str]:
    """
    Parse filenames like:
        2001_0_20201102.xlsx

    Returns:
        session_id = 2001_0_20201102
        patient_id = 2001
        session_number = 0
    """

    m = re.match(r"^(\d+)_(\d+)_(\d{8})$", stem)

    if not m:
        raise ValueError(f"Unexpected filename format: {stem}")

    return stem, m.group(1), m.group(2)


def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    dataset_dir = extract_dir / T2DM_DIR_NAME

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Missing '{T2DM_DIR_NAME}' directory inside ZIP."
        )

    return dataset_dir


def read_one_excel(file_path: Path) -> pd.DataFrame:
    """Read either .xlsx or legacy .xls."""
    suffix = file_path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")

    if suffix == ".xls":
        return pd.read_excel(file_path, engine="xlrd")

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_dynamic_files(dataset_dir: Path) -> pd.DataFrame:

    files = sorted(
        p
        for p in dataset_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls"}
        and not p.name.startswith("~$")
    )

    if not files:
        raise FileNotFoundError(
            f"No .xlsx/.xls files found in {dataset_dir}"
        )

    frames = []
    failures = []

    for file_path in files:

        try:
            session_id, patient_id, session_number = parse_ids(
                file_path.stem
            )

            df = read_one_excel(file_path)

            # Normalize source headers.
            df = normalize_dynamic_columns(df)

            # Required columns.
            if "timestamp" not in df.columns:
                raise ValueError(
                    f"Timestamp column not found. "
                    f"Columns: {list(df.columns)}"
                )

            if "cgm_mg_dl" not in df.columns:
                raise ValueError(
                    f"CGM column not found. "
                    f"Columns: {list(df.columns)}"
                )

            # Convert numeric fields.
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

            df["cgm_mg_dl"] = pd.to_numeric(
                df["cgm_mg_dl"],
                errors="coerce"
            )

            for col in [
                "cbg_mg_dl",
                "blood_ketone_mmol_l",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )
                else:
                    df[col] = pd.NA

            # Metadata.
            df["session_id"] = session_id
            df["patient_id"] = patient_id
            df["session_number"] = session_number
            df["source_file"] = file_path.name

            frames.append(df)

        except Exception as exc:
            failures.append(
                (file_path.name, type(exc).__name__, str(exc))
            )

    # VERY IMPORTANT:
    # Print failures before deciding whether to raise.
    print("\n========== FILE LOAD REPORT ==========")
    print(f"Files discovered: {len(files)}")
    print(f"Files loaded:     {len(frames)}")
    print(f"Files failed:     {len(failures)}")

    if failures:
        print("\n---------- FAILED FILES ----------")
        for name, error_type, message in failures:
            print(f"\n{name}")
            print(f"  {error_type}: {message}")

    if not frames:
        raise RuntimeError(
            "\nNo dynamic files could be loaded. "
            "See the detailed failures printed above."
        )

    master = pd.concat(
        frames,
        ignore_index=True
    )

    master = master.sort_values(
        ["patient_id", "session_id", "timestamp"],
        kind="stable"
    )

    print("\n========== DATA SUMMARY ==========")
    print(f"Loaded rows:       {len(master):,}")
    print(f"Unique patients:   {master['patient_id'].nunique()}")
    print(f"Unique sessions:   {master['session_id'].nunique()}")
    print(
        f"Missing CGM rows:  "
        f"{master['cgm_mg_dl'].isna().sum():,}"
    )

    return master


def read_summary(summary_path: Path) -> pd.DataFrame:

    df = pd.read_excel(
        summary_path,
        engine="openpyxl"
    )

    if "Patient Number" not in df.columns:
        raise ValueError(
            "Summary file does not contain 'Patient Number'."
        )

    df["session_id"] = (
        df["Patient Number"]
        .astype(str)
        .str.strip()
    )

    df["patient_id"] = (
        df["session_id"]
        .str.extract(r"^(\d+)_", expand=False)
    )

    return df


def merge_summary(
    dynamic_df: pd.DataFrame,
    summary_df: pd.DataFrame
) -> pd.DataFrame:

    merged = dynamic_df.merge(
        summary_df,
        on=["session_id", "patient_id"],
        how="left",
        suffixes=("", "_summary")
    )

    match_column = "Gender (Female=1, Male=2)"

    if match_column in merged.columns:
        matched = merged[match_column].notna().sum()
        print(
            f"Summary rows matched: "
            f"{matched:,}/{len(merged):,} "
            f"({matched / len(merged):.1%})"
        )
    else:
        print(
            "WARNING: Could not find the expected "
            "summary match column."
        )

    return merged


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Load ShanghaiT2DM raw CGM files "
            "into one master dataset."
        )
    )

    parser.add_argument(
        "--zip",
        required=True,
        help="Path to diabetes_datasets.zip"
    )

    parser.add_argument(
        "--out",
        default="data/processed/shanghai_t2dm_master.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found: {zip_path}"
        )

    extract_dir = (
        out_path.parent / "_extracted"
    )

    dataset_dir = extract_dataset(
        zip_path,
        extract_dir
    )

    # Extract summary file from ZIP root.
    summary_path = (
        extract_dir / SUMMARY_NAME
    )

    with zipfile.ZipFile(zip_path, "r") as zf:

        with zf.open(SUMMARY_NAME) as src:

            summary_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                summary_path,
                "wb"
            ) as dst:
                dst.write(src.read())

    # Load dynamic CGM files.
    dynamic_df = load_dynamic_files(
        dataset_dir
    )

    # Load EHR summary.
    summary_df = read_summary(
        summary_path
    )

    # Merge.
    master = merge_summary(
        dynamic_df,
        summary_df
    )

    # Save.
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        out_path,
        index=False
    )

    print(
        f"\nSaved master dataset to:\n"
        f"{out_path}"
    )


if __name__ == "__main__":
    main()