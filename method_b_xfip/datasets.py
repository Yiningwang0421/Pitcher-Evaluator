from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pitch_families import FAMILY_ORDER


@dataclass
class PitcherSeasonFamilyRecord:
    pitcher_id: str
    season: int
    target: float
    family_features: dict[str, np.ndarray]
    family_counts: dict[str, int]
    family_usage_weights: np.ndarray


@dataclass
class SamplingStats:
    total_records: int = 0
    empty_fastball: int = 0
    empty_breaking: int = 0
    empty_offspeed: int = 0
    padded_fastball: int = 0
    padded_breaking: int = 0
    padded_offspeed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_records": int(self.total_records),
            "empty_fastball": int(self.empty_fastball),
            "empty_breaking": int(self.empty_breaking),
            "empty_offspeed": int(self.empty_offspeed),
            "padded_fastball": int(self.padded_fastball),
            "padded_breaking": int(self.padded_breaking),
            "padded_offspeed": int(self.padded_offspeed),
        }


class PitcherSeasonFamilyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        pitcher_col: str,
        season_col: str,
        family_col: str,
        family_sample_sizes: Mapping[str, int],
        sampling_mode_train: str,
        sampling_mode_eval: str,
        short_family_strategy: str,
        random_seed: int,
        is_train: bool,
    ):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.pitcher_col = pitcher_col
        self.season_col = season_col
        self.family_col = family_col
        self.family_sample_sizes = {family: int(family_sample_sizes[family]) for family in FAMILY_ORDER}
        self.sampling_mode_train = sampling_mode_train
        self.sampling_mode_eval = sampling_mode_eval
        self.short_family_strategy = short_family_strategy
        self.random_seed = int(random_seed)
        self.is_train = is_train

        self.records = self._build_records(df)
        self.sampling_stats = self._build_sampling_stats()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        sampled: dict[str, dict[str, torch.Tensor]] = {}

        for family in FAMILY_ORDER:
            sampled_x, sampled_mask = self._sample_family(
                record=record,
                family=family,
                target_size=self.family_sample_sizes[family],
                record_idx=idx,
            )
            sampled[family] = {
                "x": torch.tensor(sampled_x, dtype=torch.float32),
                "mask": torch.tensor(sampled_mask, dtype=torch.bool),
            }

        return {
            "fastball_x": sampled["fastball"]["x"],
            "fastball_mask": sampled["fastball"]["mask"],
            "breaking_x": sampled["breaking"]["x"],
            "breaking_mask": sampled["breaking"]["mask"],
            "offspeed_x": sampled["offspeed"]["x"],
            "offspeed_mask": sampled["offspeed"]["mask"],
            "family_usage_weights": torch.tensor(record.family_usage_weights, dtype=torch.float32),
            "y": torch.tensor(record.target, dtype=torch.float32),
            "pitcher_id": record.pitcher_id,
            "season": int(record.season),
            "family_counts": torch.tensor([record.family_counts[family] for family in FAMILY_ORDER], dtype=torch.long),
        }

    def get_sampling_stats(self) -> SamplingStats:
        return self.sampling_stats

    def _build_records(self, df: pd.DataFrame) -> list[PitcherSeasonFamilyRecord]:
        records: list[PitcherSeasonFamilyRecord] = []
        grouped = df.groupby([self.pitcher_col, self.season_col], sort=False)

        for (pitcher_id, season), group in grouped:
            family_features: dict[str, np.ndarray] = {}
            family_counts: dict[str, int] = {}
            total_count = 0

            for family in FAMILY_ORDER:
                family_group = group.loc[group[self.family_col] == family, self.feature_cols]
                values = family_group.to_numpy(dtype=np.float32)
                family_features[family] = values
                family_counts[family] = int(len(values))
                total_count += int(len(values))

            if total_count == 0:
                continue

            usage = np.asarray(
                [family_counts[family] / float(total_count) for family in FAMILY_ORDER],
                dtype=np.float32,
            )

            records.append(
                PitcherSeasonFamilyRecord(
                    pitcher_id=str(pitcher_id),
                    season=int(season),
                    target=float(group[self.target_col].iloc[0]),
                    family_features=family_features,
                    family_counts=family_counts,
                    family_usage_weights=usage,
                )
            )
        return records

    def _build_sampling_stats(self) -> SamplingStats:
        stats = SamplingStats(total_records=len(self.records))
        for record in self.records:
            for family in FAMILY_ORDER:
                count = record.family_counts[family]
                target_size = self.family_sample_sizes[family]
                if count == 0:
                    setattr(stats, f"empty_{family}", getattr(stats, f"empty_{family}") + 1)
                if count < target_size:
                    setattr(stats, f"padded_{family}", getattr(stats, f"padded_{family}") + 1)
        return stats

    def _rng_for_record(self, record: PitcherSeasonFamilyRecord, family: str, record_idx: int) -> np.random.Generator:
        key = f"{record.pitcher_id}__{record.season}__{family}__{record_idx}__{self.random_seed}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2**32)
        return np.random.default_rng(seed)

    def _sample_family(
        self,
        record: PitcherSeasonFamilyRecord,
        family: str,
        target_size: int,
        record_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = record.family_features[family]
        current_n = int(features.shape[0])
        feat_dim = int(len(self.feature_cols))

        if current_n == 0:
            return np.zeros((target_size, feat_dim), dtype=np.float32), np.zeros((target_size,), dtype=np.bool_)

        if current_n >= target_size:
            if self.is_train and self.sampling_mode_train == "random_without_replacement":
                rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
                selected_idx = rng.choice(current_n, size=target_size, replace=False)
            else:
                selected_idx = np.arange(target_size)
            sampled = features[selected_idx]
            mask = np.ones((target_size,), dtype=np.bool_)
            return sampled.astype(np.float32), mask

        if self.short_family_strategy == "sample_with_replacement":
            if self.is_train:
                rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
            else:
                rng = self._rng_for_record(record, family, record_idx)
            selected_idx = rng.choice(current_n, size=target_size, replace=True)
            sampled = features[selected_idx]
            mask = np.ones((target_size,), dtype=np.bool_)
            return sampled.astype(np.float32), mask

        out = np.zeros((target_size, feat_dim), dtype=np.float32)
        mask = np.zeros((target_size,), dtype=np.bool_)
        out[:current_n] = features.astype(np.float32)
        mask[:current_n] = True
        return out, mask


def collate_pitcher_season_family_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")

    return {
        "fastball_x": torch.stack([item["fastball_x"] for item in batch], dim=0),
        "fastball_mask": torch.stack([item["fastball_mask"] for item in batch], dim=0),
        "breaking_x": torch.stack([item["breaking_x"] for item in batch], dim=0),
        "breaking_mask": torch.stack([item["breaking_mask"] for item in batch], dim=0),
        "offspeed_x": torch.stack([item["offspeed_x"] for item in batch], dim=0),
        "offspeed_mask": torch.stack([item["offspeed_mask"] for item in batch], dim=0),
        "family_usage_weights": torch.stack([item["family_usage_weights"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "pitcher_ids": [item["pitcher_id"] for item in batch],
        "seasons": [item["season"] for item in batch],
        "family_counts": torch.stack([item["family_counts"] for item in batch], dim=0),
    }
