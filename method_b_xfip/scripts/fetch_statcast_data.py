import argparse
import time
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ALL_PITCH_TYPES = [
    "FF", "SI", "FT", "FC", "SL", "ST", "SV", "CH", "CU", "KC", "CS", "FS", "FO", "KN", "EP", "SC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Statcast pitch data per pitcher.")
    parser.add_argument("--pitcher-list", default="data/raw/external/pitcher_list_bk.csv")
    parser.add_argument("--players-dir", default="data/raw/statcast/players_bk")
    parser.add_argument("--done-file", default="data/raw/statcast/done_pitchers_bk.txt")
    parser.add_argument("--output-file", default="data/interim/random_1000_pitchers_all_pitches.csv")
    parser.add_argument(
        "--pitch-types",
        default="ALL",
        help=(
            "Pitch type filter. Use 'ALL' for all pitches, or comma-separated Statcast codes "
            "like 'FF,SI,SL,CH,CU'."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download pitcher CSVs even if local files already exist.",
    )
    return parser.parse_args()


def _build_hfpt_param(pitch_types: str) -> str:
    if pitch_types.strip().upper() == "ALL":
        return quote("|".join(ALL_PITCH_TYPES) + "|", safe="")
    tokens = [t.strip().upper() for t in pitch_types.split(",") if t.strip()]
    if not tokens:
        return ""
    return quote("|".join(tokens) + "|", safe="")


def build_url(player_id: str, pitch_types: str) -> str:
    hfpt = _build_hfpt_param(pitch_types)
    return (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        f"?hfPT={hfpt}"
        "&hfAB="
        "&hfGT=R%7C"
        "&hfPR="
        "&hfZ="
        "&hfStadium="
        "&hfBBL="
        "&hfNewZones="
        "&hfPull="
        "&hfC="
        "&hfSea=2022%7C2023%7C2024%7C"
        "&hfSit="
        "&player_type=pitcher"
        "&hfOuts="
        "&hfOpponent="
        "&pitcher_throws="
        "&batter_stands="
        "&hfSA="
        "&game_date_gt="
        "&game_date_lt="
        "&hfMo="
        "&hfTeam="
        "&home_road="
        "&hfRO="
        "&position="
        "&hfInfield="
        "&hfOutfield="
        "&hfInn="
        "&hfBBT="
        "&hfFlag="
        "&metric_1="
        "&group_by=name"
        "&min_pitches=1000"
        "&min_results=0"
        "&min_pas=0"
        "&sort_col=pitches"
        "&player_event_sort=api_p_release_speed"
        "&sort_order=desc"
        "&type=details"
        f"&player_id={player_id}"
        "&minors=false"
    )


def main() -> None:
    args = parse_args()
    pitcher_list = Path(args.pitcher_list)
    players_dir = Path(args.players_dir)
    done_file = Path(args.done_file)
    output_file = Path(args.output_file)

    if not pitcher_list.exists():
        raise FileNotFoundError(
            f"Pitcher list not found: {pitcher_list}. "
            "Create this CSV with at least columns ['player_id', 'player_name']."
        )

    players_dir.mkdir(parents=True, exist_ok=True)
    done_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_pitchers = pd.read_csv(pitcher_list)
    print(f"Loaded {len(df_pitchers)} pitchers from {pitcher_list}")

    sample_n = min(args.num_samples, len(df_pitchers))
    sample_pitchers = df_pitchers.sample(n=sample_n, random_state=args.random_state)

    completed_ids = set()
    if done_file.exists():
        completed_ids = set(done_file.read_text().splitlines())
    print(f"{len(completed_ids)} pitchers already completed")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    all_data = []
    for _, row in sample_pitchers.iterrows():
        pid = str(row["player_id"])
        name = row["player_name"]
        csv_path = players_dir / f"{pid}_bk.csv"

        if csv_path.exists() and not args.force_refresh:
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    all_data.append(df)
                print(f"Reuse existing data for {name} ({pid})")
            except Exception as exc:
                print(f"Failed reading {csv_path.name}: {exc}")
            continue

        if pid in completed_ids and not args.force_refresh:
            print(f"Skip {name} ({pid}): marked completed")
            continue

        if csv_path.exists() and args.force_refresh:
            try:
                csv_path.unlink()
            except OSError:
                pass

        success = False
        for attempt in range(1, args.max_retries + 1):
            try:
                print(f"Fetching {name} ({pid}), attempt {attempt}")
                resp = requests.get(build_url(pid, args.pitch_types), headers=headers, timeout=30)
                if resp.status_code == 200 and resp.content:
                    df = pd.read_csv(StringIO(resp.text))
                    if not df.empty:
                        df["pitcher_name"] = name
                        df.to_csv(csv_path, index=False)
                        all_data.append(df)
                        print(f"Saved {len(df)} rows to {csv_path}")
                    success = True
                    break
                print(f"HTTP {resp.status_code} for {name}")
            except Exception as exc:
                print(f"Request failed for {name}: {exc}")

            if attempt < args.max_retries:
                time.sleep(args.retry_wait)

        if success:
            if pid not in completed_ids:
                completed_ids.add(pid)
                with done_file.open("a", encoding="utf-8") as f:
                    f.write(pid + "\n")

        time.sleep(args.sleep_seconds)

    if not all_data:
        print("No data fetched or loaded")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Saved merged fetched CSV: {output_file} ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
