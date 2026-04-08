import argparse
import subprocess
import sys
from pathlib import Path

import requests


def run_step(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def module_cmd(py: str, module_name: str, args: list[str]) -> list[str]:
    return [py, "-m", module_name, *args]


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    print(f"Downloaded {url} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full data build pipeline for model training.")

    parser.add_argument("--python", default=sys.executable, help="Python executable used to run sub-scripts.")

    parser.add_argument("--pitcher-list-url", default=None, help="Optional URL to download pitcher_list_bk.csv.")
    parser.add_argument("--pitcher-list-path", default="data/raw/external/pitcher_list_bk.csv")

    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--pitch-types", default="ALL", help="Pitch type filter for fetch step.")
    parser.add_argument("--force-refresh-fetch", action="store_true")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--min-pitches", type=int, default=256)

    parser.add_argument("--xfip-min", type=float, default=2.5)
    parser.add_argument("--xfip-max", type=float, default=6.0)
    parser.add_argument("--apply-range-filter", action="store_true")
    parser.add_argument("--train-method-b", action="store_true")

    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-min-filter", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pitcher_list_path = Path(args.pitcher_list_path)
    if args.pitcher_list_url:
        download_file(args.pitcher_list_url, pitcher_list_path)

    py = args.python

    if not args.skip_fetch:
        fetch_cmd = module_cmd(
            py,
            "method_b_xfip.scripts.fetch_statcast_data",
            [
            "--pitcher-list",
            str(pitcher_list_path),
            "--players-dir",
            "data/raw/statcast/players_bk",
            "--done-file",
            "data/raw/statcast/done_pitchers_bk.txt",
            "--output-file",
            "data/interim/random_1000_pitchers_all_pitches.csv",
            "--num-samples",
            str(args.num_samples),
            "--pitch-types",
            args.pitch_types,
            ],
        )
        if args.force_refresh_fetch:
            fetch_cmd.append("--force-refresh")
        run_step(fetch_cmd)

    if not args.skip_merge:
        run_step(
            module_cmd(
                py,
                "method_b_xfip.scripts.merge_pitcher_files",
                [
                "--input-folder",
                "data/raw/statcast/players_bk",
                "--output-file",
                "data/interim/merged.csv",
                ],
            )
        )

    if not args.skip_enrich:
        for year in args.years:
            run_step(
                module_cmd(
                    py,
                    "method_b_xfip.scripts.enrich_pitch_data",
                    [
                    "merge-year",
                    "--year",
                    str(year),
                    "--merged-file",
                    "data/interim/merged.csv",
                    "--year-col",
                    "game_year",
                    "--refs-dir",
                    "data/raw/external",
                    "--output-dir",
                    "data/interim/enriched",
                    ],
                )
            )

        run_step(
            module_cmd(
                py,
                "method_b_xfip.scripts.enrich_pitch_data",
                [
                "concat-years",
                "--input-dir",
                "data/interim/enriched",
                "--output-csv",
                "data/interim/merged_with_xops_xfip.csv",
                "--years",
                *[str(y) for y in args.years],
                ],
            )
        )

    if not args.skip_build:
        run_step(
            module_cmd(
                py,
                "method_b_xfip.scripts.build_regression_dataset",
                [
                "--input-file",
                "data/interim/merged_with_xops_xfip.csv",
                "--output-file",
                "data/processed/players_processed_bk/data_merged_processed.csv",
                ],
            )
        )

    if args.train_method_b:
        run_step(
            module_cmd(
                py,
                "method_b_xfip.scripts.train_method_b_xfip",
                [
                "--input-file",
                "data/processed/players_processed_bk/data_merged_processed.csv",
                "--artifact-dir",
                "artifacts/method_b_xfip",
                "--report-dir",
                "reports/method_b_xfip",
                ],
            )
        )

    if not args.skip_min_filter:
        run_step(
            module_cmd(
                py,
                "method_b_xfip.scripts.filter_by_min_pitches",
                [
                "--input-file",
                "data/processed/players_processed_bk/data_merged_processed.csv",
                "--output-file",
                "data/processed/players_processed_bk/data_merged_processed_256plus.csv",
                "--group-col",
                "xFIP",
                "--min-pitches",
                str(args.min_pitches),
                ],
            )
        )

    if args.apply_range_filter:
        run_step(
            module_cmd(
                py,
                "method_b_xfip.scripts.filter_by_range",
                [
                "--input-file",
                "data/processed/players_processed_bk/data_merged_processed_256plus.csv",
                "--output-file",
                "data/processed/players_processed_bk/filtered.csv",
                "--filter-column",
                "xFIP",
                "--min-value",
                str(args.xfip_min),
                "--max-value",
                str(args.xfip_max),
                ],
            )
        )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
