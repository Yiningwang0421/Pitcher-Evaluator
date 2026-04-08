import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


FEATURE_COLUMNS = [
    "effective_speed",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "plate_x",
    "release_spin_rate",
    "spin_axis",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "arm_angle",
    "pitch_number",
    "n_thruorder_pitcher",
    "xops_sigmoid",
    "xFIP",
]

FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "FC"}
BREAKING_TYPES = {"SL", "ST", "SV", "CU", "KC", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "KN", "EP", "SC"}


def normalize_pitch_type(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    pitch_type = str(value).strip().upper()
    if pitch_type == "PO":
        return None
    if pitch_type == "FA":
        return "FF"
    return pitch_type


def pitch_family(pitch_type: str) -> str:
    if pitch_type in FASTBALL_TYPES:
        return "fastball"
    if pitch_type in BREAKING_TYPES:
        return "breaking"
    if pitch_type in OFFSPEED_TYPES:
        return "offspeed"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean merged pitch data into model-ready regression dataset.")
    parser.add_argument(
        "--input-file",
        default="data/interim/merged_with_xops_xfip.csv",
        help="Merged and enriched input CSV.",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/players_processed_bk/data_merged_processed.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def map_label(row: pd.Series):
    desc = row["description"]
    event = row["events"]

    if event in ["single", "double", "triple", "home_run"]:
        return event
    if desc in ["ball", "blocked_ball", "hit_by_pitch"]:
        return "ball"
    if desc == "called_strike":
        return "called_strike"
    if desc in ["swinging_strike", "swinging_strike_blocked"]:
        return "swinging_strike"
    if desc in ["foul", "foul_tip"]:
        return "foul"
    if event in [
        "field_out",
        "force_out",
        "double_play",
        "sac_fly",
        "grounded_into_double_play",
        "field_error",
        "fielders_choice",
        "fielders_choice_out",
        "strikeout_double_play",
        "sac_fly_double_play",
        "triple_play",
    ]:
        return "field_out"
    return None


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    required_cols = ["pitcher", "game_year", "pitch_type", "description", "events", *FEATURE_COLUMNS]
    header = pd.read_csv(input_file, nrows=0, engine="python")
    missing_input = [c for c in required_cols if c not in header.columns]
    if missing_input: 
        raise ValueError(f"Missing required columns in input: {missing_input}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    label_counts = None
    label_percent = None
    total_rows = 0

    chunk_iter = pd.read_csv(
        input_file,
        usecols=required_cols,
        engine="c",
        chunksize=200000,
        on_bad_lines="skip",
    )

    first_chunk = True
    for chunk in chunk_iter:
        chunk = chunk.copy()
        chunk["pitch_type"] = chunk["pitch_type"].apply(normalize_pitch_type)
        chunk = chunk[chunk["pitch_type"].notna()].copy()
        chunk = chunk[~chunk["description"].astype(str).str.contains("bunt", na=False)]
        chunk["label"] = chunk.apply(map_label, axis=1)
        chunk = chunk[chunk["label"].notna()].copy()

        chunk["pitch_family"] = chunk["pitch_type"].apply(pitch_family)

        if label_counts is None:
            label_counts = chunk["label"].value_counts()
        else:
            label_counts = label_counts.add(chunk["label"].value_counts(), fill_value=0)
            label_percent = None 

        df_model = chunk[["pitcher", "game_year", "pitch_type", "pitch_family"] + FEATURE_COLUMNS + ["label"]].copy()
        df_model.to_csv(output_file, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False
        total_rows += len(df_model)

    if label_counts is None:
        raise ValueError("No rows were retained after cleaning.")

    print("Label counts:\n", label_counts)
    label_percent = (label_counts / max(float(total_rows), 1.0) * 100).round(2)
    print("\nLabel percentages (%):\n", label_percent)
    print(f"Saved cleaned dataset: {output_file} ({total_rows} rows)")


if __name__ == "__main__":
    main()