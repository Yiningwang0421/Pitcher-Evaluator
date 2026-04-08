from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MethodBConfig:
    input_path: Path = Path("data/processed/players_processed_bk/data_merged_processed.csv")
    artifact_dir: Path = Path("artifacts/method_b_xfip")
    report_dir: Path = Path("reports/method_b_xfip")

    pitcher_col: str = "pitcher_id"
    season_col: str = "season"
    pitch_type_col: str = "pitch_type"
    pitch_group_mode: str = "family"
    target_col: str = "xFIP"

    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42

    min_pitch_type_rows: int = 15000
    min_pitch_type_groups: int = 120
    min_group_size: int = 10
    max_missing_feature_frac: float = 0.4
    max_categorical_levels: int = 8

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip_norm: float = 5.0

    aggregation: str = "mean"
    calibrate: bool = False
    calibration_lr: float = 1e-2
    calibration_epochs: int = 150
    calibration_l2: float = 1e-4

    device: str = "auto"
    num_workers: int = 0
    verbose: bool = True
