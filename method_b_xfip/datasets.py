from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class GroupRecord:
    features: np.ndarray
    target: float
    group_key: str
    pitch_count: int
    pitcher_id: str
    season: int
    pitch_type: str


class PitchTypeGroupDataset(Dataset):
    def __init__(self, records: Sequence[GroupRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "x": torch.tensor(record.features, dtype=torch.float32),
            "y": torch.tensor(record.target, dtype=torch.float32),
            "group_key": record.group_key,
            "pitch_count": record.pitch_count,
            "pitcher_id": record.pitcher_id,
            "season": record.season,
            "pitch_type": record.pitch_type,
        }


def collate_group_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")

    max_len = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    batch_size = len(batch)

    x = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y = torch.stack([item["y"] for item in batch]).float()

    group_keys = []
    pitch_counts = []
    pitcher_ids = []
    seasons = []
    pitch_types = []

    for row_idx, item in enumerate(batch):
        n = item["x"].shape[0]
        x[row_idx, :n] = item["x"]
        mask[row_idx, :n] = True
        group_keys.append(item["group_key"])
        pitch_counts.append(int(item["pitch_count"]))
        pitcher_ids.append(item["pitcher_id"])
        seasons.append(int(item["season"]))
        pitch_types.append(item["pitch_type"])

    return {
        "x": x,
        "mask": mask,
        "y": y,
        "group_keys": group_keys,
        "pitch_counts": torch.tensor(pitch_counts, dtype=torch.long),
        "pitcher_ids": pitcher_ids,
        "seasons": seasons,
        "pitch_types": pitch_types,
    }


def build_group_records(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    pitcher_col: str,
    season_col: str,
    pitch_type_col: str,
    min_group_size: int,
    group_keys: set[str] | None = None,
    group_name: str | None = None,
) -> list[GroupRecord]:
    records: list[GroupRecord] = []
    grouped = df.groupby([pitcher_col, season_col, pitch_type_col], sort=False)

    for (pitcher_id, season, pitch_type), group in grouped:
        group_key = f"{pitcher_id}__{season}__{pitch_type}"
        if group_keys is not None and group_key not in group_keys:
            continue
        if len(group) < min_group_size:
            continue
        features = group.loc[:, feature_cols].to_numpy(dtype=np.float32)
        target = float(group[target_col].iloc[0])
        records.append(
            GroupRecord(
                features=features,
                target=target,
                group_key=group_key,
                pitch_count=int(len(group)),
                pitcher_id=str(pitcher_id),
                season=int(season),
                pitch_type=str(pitch_type),
            )
        )
    return records
