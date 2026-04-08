import argparse
from pathlib import Path

import pandas as pd


def split_xops_by_year(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    if "year" not in df.columns:
        raise ValueError("Input xOPS CSV must contain a 'year' column")

    output_dir.mkdir(parents=True, exist_ok=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    for year in sorted(df["year"].dropna().unique()):
        year = int(year)
        out_path = output_dir / f"xops_{year}.csv"
        df[df["year"] == year].to_csv(out_path, index=False)
        print(f"Saved {out_path}")


def merge_year_with_refs(year: int, merged_file: Path, year_col: str, refs_dir: Path, output_dir: Path) -> None:
    xops = refs_dir / f"xops_{year}.csv"
    xfip = refs_dir / f"fangraphs_pitching_data_{year}_xfip.csv"

    for p in [merged_file, xops, xfip]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df_all = pd.read_csv(merged_file)
    if year_col not in df_all.columns:
        raise ValueError(f"Column '{year_col}' not found in {merged_file}")

    df = df_all[df_all[year_col] == year].copy()
    if df.empty:
        raise ValueError(f"No rows found for year={year} in {merged_file}")

    df_xops = pd.read_csv(xops)
    df_xfip = pd.read_csv(xfip)

    original_rows = len(df)

    df = df.merge(df_xops[["player_id", "xops_sigmoid"]], left_on="batter", right_on="player_id", how="left")
    df = df.drop(columns=["player_id"])

    df = df.merge(df_xfip[["xMLBAMID", "xFIP"]], left_on="pitcher", right_on="xMLBAMID", how="left")
    df = df.drop(columns=["xMLBAMID"])

    df = df.dropna(subset=["xops_sigmoid", "xFIP"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pitches_with_xops_xfip_{year}.csv"
    df.to_csv(out_path, index=False)

    print(f"Merged year {year}")
    print(f"Rows: {original_rows} -> {len(df)} (dropped {original_rows - len(df)})")
    print(f"Saved {out_path}")


def concat_years(input_dir: Path, output_csv: Path, years: list[int]) -> None:
    dfs = []
    for year in years:
        p = input_dir / f"pitches_with_xops_xfip_{year}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing yearly file: {p}")
        dfs.append(pd.read_csv(p))

    merged = pd.concat(dfs, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Saved merged enriched CSV: {output_csv} ({len(merged)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for adding xOPS and xFIP to pitch-level data.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split-xops", help="Split a master xOPS CSV into per-year files")
    p_split.add_argument("--input-csv", default="data/raw/external/batter_stats_normalized_22_to_24_sigmoid.csv")
    p_split.add_argument("--output-dir", default="data/raw/external")

    p_merge = sub.add_parser("merge-year", help="Merge one year pitch CSV with xOPS and xFIP")
    p_merge.add_argument("--year", type=int, required=True)
    p_merge.add_argument("--merged-file", default="data/interim/merged.csv")
    p_merge.add_argument("--year-col", default="game_year")
    p_merge.add_argument("--refs-dir", default="data/raw/external")
    p_merge.add_argument("--output-dir", default="data/interim/enriched")

    p_concat = sub.add_parser("concat-years", help="Concatenate enriched yearly CSV files")
    p_concat.add_argument("--input-dir", default="data/interim/enriched")
    p_concat.add_argument("--output-csv", default="data/interim/merged_with_xops_xfip.csv")
    p_concat.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "split-xops":
        split_xops_by_year(Path(args.input_csv), Path(args.output_dir))
    elif args.cmd == "merge-year":
        merge_year_with_refs(args.year, Path(args.merged_file), args.year_col, Path(args.refs_dir), Path(args.output_dir))
    elif args.cmd == "concat-years":
        concat_years(Path(args.input_dir), Path(args.output_csv), args.years)


if __name__ == "__main__":
    main()
