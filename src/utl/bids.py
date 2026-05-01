import json
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

def parse_bids_entities(eeg_file: Path) -> Dict[str, Any]:
    """
    Parse BIDS entities (subject, task) from filename.
    """
    entities: Dict[str, Any] = {"participant_id": None, "task": None}

    stem = eeg_file.name.rsplit(".", 1)[0]
    for token in stem.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key == "sub":
            entities["participant_id"] = f"sub-{value}"
        elif key == "task":
            entities["task"] = value

    if entities["participant_id"] is None:
        entities["participant_id"] = next(
            (p for p in eeg_file.parts if p.startswith("sub-")), None
        )

    return entities

def load_bids_dataset(dataset_path: str) -> Tuple[pd.DataFrame, Tuple[List[str], List[str], List[str]]]:
    """
    Load resting-state EEG BIDS dataset metadata.

    Returns:
        df: DataFrame with all relevant EEG files and participant info
        subject_groups: Tuple of (control_subjects, ad_subjects, ftd_subjects)
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return pd.DataFrame(), ([], [], [])

    eeg_suffixes = {".set", ".edf", ".bdf", ".eeg", ".vhdr", ".fif"}
    # Include files in both root and derivatives for discovery
    eeg_files = [
        f for f in dataset_dir.rglob("*_eeg.*")
        if f.is_file() and f.suffix.lower() in eeg_suffixes
    ]


    records = []
    for eeg_file in sorted(eeg_files):
        entities = parse_bids_entities(eeg_file)

        base = eeg_file.name.rsplit("_eeg", 1)[0]
        channels_tsv = eeg_file.with_name(f"{base}_channels.tsv")
        eeg_json = eeg_file.with_name(f"{base}_eeg.json")

        json_meta = {}
        if eeg_json.exists():
            with open(eeg_json, "r", encoding="utf-8") as f:
                try:
                    json_meta = json.load(f)
                except json.JSONDecodeError:
                    pass

        records.append({
            **entities,
            "eeg_file": str(eeg_file),
            "channels_file": str(channels_tsv) if channels_tsv.exists() else None,
            "eeg_json": str(eeg_json) if eeg_json.exists() else None,
            "file_format": eeg_file.suffix.lower().lstrip("."),
            "SamplingFrequency": json_meta.get("SamplingFrequency"),
        })

    df = pd.DataFrame(records)
    
    # Deduplicate: if same subject and task have multiple files (e.g. raw and derivatives),
    # prioritize the one in derivatives.
    if not df.empty:
        df['is_derivative'] = df['eeg_file'].apply(lambda x: 'derivatives' in x)
        df = df.sort_values(['participant_id', 'task', 'is_derivative'], ascending=[True, True, False])
        df = df.drop_duplicates(subset=['participant_id', 'task'], keep='first')

    # Merge participant metadata from participants.tsv

    participants_tsv = dataset_dir / "participants.tsv"
    if not participants_tsv.exists() and "processed" in dataset_dir.parts:
        # Fallback to the raw directory for participants.tsv
        raw_dataset_dir = Path(*(part if part != "processed" else "raw" for part in dataset_dir.parts))
        fallback_tsv = raw_dataset_dir / "participants.tsv"
        if fallback_tsv.exists():
            participants_tsv = fallback_tsv

    if participants_tsv.exists() and not df.empty:
        participants_df = pd.read_csv(participants_tsv, sep="\t", encoding="utf-8-sig")
        participants_df.columns = [c.strip() for c in participants_df.columns]
        df = df.merge(participants_df, on="participant_id", how="left")

    if df.empty:
        return df, ([], [], [])

    # Identify subject groups
    unique_subjects = df["participant_id"].dropna().unique()
    control_subjects = []
    ad_subjects = []
    ftd_subjects = []

    for subject in unique_subjects:
        subject_df = df[df["participant_id"] == subject]
        group = subject_df["Group"].iloc[0] if "Group" in subject_df.columns else None

        if group == "C":
            control_subjects.append(subject)
        elif group == "A":
            ad_subjects.append(subject)
        elif group == "F":
            ftd_subjects.append(subject)

    df = df.sort_values(["participant_id", "task"]).reset_index(drop=True)

    return df, (control_subjects, ad_subjects, ftd_subjects)
