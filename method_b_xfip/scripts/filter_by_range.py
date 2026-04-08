import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rows by numeric range for one column.")
    parser.add_argument("--input-file", default="data/processed/players_processed_bk/data_merged_processed_256plus.csv")
    parser.add_argument("--output-file", default="data/processed/players_processed_bk/filtered.csv")
    parser.add_argument("--filter-column", default="xFIP")
    parser.add_argument("--min-value", type=float, default=2.5)
    parser.add_argument("--max-value", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    df = pd.read_csv(input_file)
    if args.filter_column not in df.columns:
        raise ValueError(f"Column '{args.filter_column}' not found in {input_file}")

    before_count = len(df)
    df_filtered = df[(df[args.filter_column] >= args.min_value) & (df[args.filter_column] <= args.max_value)]
    after_count = len(df_filtered)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_file, index=False)
    print(f"Filtered rows: {before_count} -> {after_count}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()