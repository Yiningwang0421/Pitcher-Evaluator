import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-pitcher CSV files into one CSV.")
    parser.add_argument(
        "--input-folder",
        default="data/raw/statcast/players_bk",
        help="Folder that contains per-pitcher CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default="data/interim/merged.csv",
        help="Path for merged output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder = Path(args.input_folder)
    output_file = Path(args.output_file)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    csv_files = sorted(input_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Loaded {file_path.name}: {len(df)} rows")
        except Exception as exc:
            print(f"Skip {file_path.name} due to read error: {exc}")

    if not dfs:
        raise RuntimeError("No valid CSV files were loaded.")

    merged_df = pd.concat(dfs, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} files -> {output_file} ({len(merged_df)} rows)")


if __name__ == "__main__":
    main()