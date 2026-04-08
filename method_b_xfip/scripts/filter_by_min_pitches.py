import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Filter seasons by minimum pitch count.")
	parser.add_argument("--input-file", default="data/processed/players_processed_bk/data_merged_processed.csv")
	parser.add_argument("--output-file", default="data/processed/players_processed_bk/data_merged_processed_256plus.csv")
	parser.add_argument("--group-col", default="xFIP")
	parser.add_argument("--min-pitches", type=int, default=256)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_file = Path(args.input_file)
	output_file = Path(args.output_file)

	if not input_file.exists():
		raise FileNotFoundError(f"Input CSV not found: {input_file}")

	df = pd.read_csv(input_file)
	if args.group_col not in df.columns:
		raise ValueError(f"Column '{args.group_col}' not found in {input_file}")

	pitch_counts = df.groupby(args.group_col).size()
	valid_groups = pitch_counts[pitch_counts >= args.min_pitches].index
	df_filtered = df[df[args.group_col].isin(valid_groups)]

	output_file.parent.mkdir(parents=True, exist_ok=True)
	df_filtered.to_csv(output_file, index=False)

	print(f"Saved filtered CSV: {output_file}")
	print(f"Total groups: {len(pitch_counts)}")
	print(f"Groups with >= {args.min_pitches} pitches: {len(valid_groups)}")
	print(f"Rows after filter: {len(df_filtered)}")


if __name__ == "__main__":
	main()