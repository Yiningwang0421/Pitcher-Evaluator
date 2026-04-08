from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pandas.api.types import CategoricalDtype


STANDARD_COLUMN_ALIASES = {
    "pitcher": "pitcher_id",
    "pitcher_id": "pitcher_id",
    "game_year": "season",
    "season": "season",
    "xFIP": "xFIP",
    "target_xFIP": "xFIP",
}


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def standardize_input_dataframe(
    df: pd.DataFrame,
    pitcher_col: str = "pitcher_id",
    season_col: str = "season",
    pitch_type_col: str = "pitch_type",
    target_col: str = "xFIP",
) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}

    for raw_name, standard_name in STANDARD_COLUMN_ALIASES.items():
        if raw_name in out.columns and standard_name not in out.columns:
            rename_map[raw_name] = standard_name

    if rename_map:
        out = out.rename(columns=rename_map)

    if pitcher_col not in out.columns or season_col not in out.columns or pitch_type_col not in out.columns or target_col not in out.columns:
        missing = [c for c in [pitcher_col, season_col, pitch_type_col, target_col] if c not in out.columns]
        raise ValueError(f"Missing required columns after standardization: {missing}")

    out[pitcher_col] = out[pitcher_col].astype(str)
    out[season_col] = pd.to_numeric(out[season_col], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out[pitch_type_col] = out[pitch_type_col].astype(str).str.strip().str.upper()
    out = out.dropna(subset=[pitcher_col, season_col, pitch_type_col, target_col]).copy()
    out[season_col] = out[season_col].astype(int)
    return out


def unique_pitcher_season_frame(df: pd.DataFrame, pitcher_col: str, season_col: str) -> pd.DataFrame:
    out: pd.DataFrame = pd.DataFrame(df.loc[:, [pitcher_col, season_col]].drop_duplicates().reset_index(drop=True))
    return out


def split_pitcher_seasons(
    df: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitFrames:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must both be > 0")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    groups = unique_pitcher_season_frame(df, pitcher_col, season_col)
    groups["_group_key"] = groups[pitcher_col].astype(str) + "__" + groups[season_col].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(groups, groups=groups["_group_key"]))
    trainval_groups = groups.iloc[trainval_idx].copy()
    test_groups = groups.iloc[test_idx].copy()

    relative_val_fraction = validation_fraction / (1.0 - test_fraction)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_fraction, random_state=seed + 1)
    train_idx, val_idx = next(splitter_val.split(trainval_groups, groups=trainval_groups["_group_key"]))
    train_groups = trainval_groups.iloc[train_idx].copy()
    val_groups = trainval_groups.iloc[val_idx].copy()

    train_keys = set(train_groups["_group_key"])
    val_keys = set(val_groups["_group_key"])
    test_keys = set(test_groups["_group_key"])

    key_series = df[pitcher_col].astype(str) + "__" + df[season_col].astype(str)
    train_df = pd.DataFrame(df.loc[key_series.isin(train_keys)].copy())
    val_df = pd.DataFrame(df.loc[key_series.isin(val_keys)].copy())
    test_df = pd.DataFrame(df.loc[key_series.isin(test_keys)].copy())

    return SplitFrames(train=train_df, validation=val_df, test=test_df)


def infer_feature_spec(
    df: pd.DataFrame,
    exclude_columns: Iterable[str],
    max_categorical_levels: int = 8,
) -> FeatureSpec:
    exclude = set(exclude_columns)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in exclude or str(col).startswith("Unnamed:"):
            continue
        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            continue
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(series) or isinstance(series.dtype, CategoricalDtype):
            if 2 <= nunique <= max_categorical_levels:
                categorical_cols.append(col)

    return FeatureSpec(numeric=numeric_cols, categorical=categorical_cols)


def season_target_table(
    df: pd.DataFrame,
    pitcher_col: str,
    season_col: str,
    target_col: str,
) -> pd.DataFrame:
    return (
        df.groupby([pitcher_col, season_col], as_index=False)
        .agg(true_xFIP=(target_col, "mean"), total_pitch_count=(target_col, "size"))
        .rename(columns={pitcher_col: "pitcher_id", season_col: "season"})
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
