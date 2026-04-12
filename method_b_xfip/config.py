from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .pitch_families import DEFAULT_PITCH_FAMILY_MAPPING


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

    pitch_family_mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PITCH_FAMILY_MAPPING))
    drop_unknown_pitch_types: bool = True
    unknown_pitch_family_fallback: str = "offspeed"

    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42

    max_missing_feature_frac: float = 0.4
    max_categorical_levels: int = 8

    family_sample_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "fastball": 128,
            "breaking": 96,
            "offspeed": 96,
        }
    )
    sampling_mode_train: str = "random_without_replacement"
    sampling_mode_eval: str = "deterministic"
    short_family_strategy: str = "zero_pad_with_mask"
    use_data_cache: bool = True
    data_cache_dir: Path = Path("artifacts/method_b_xfip_cache")

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    family_encoder_type: str = "mean_mlp"
    family_embed_dim: int = 64
    family_encoder_num_heads: int = 4
    family_encoder_num_layers: int = 1
    family_encoder_dropout: float = 0.15
    family_pooling: str = "attention"
    family_score_hidden_dim: int = 32
    loss_type: str = "mse"
    huber_delta: float = 0.8
    use_residual_correction: bool = False
    residual_model_type: str = "hgbt"
    residual_max_iter: int = 200
    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip_norm: float = 5.0

    aggregation: str = "mean"
    fusion_mode: str = "usage_weighted_sum"
    use_family_reliability: bool = False
    reliability_count_c: float = 20.0
    normalize_reliability_weights: bool = True
    calibrate: bool = False
    collapse_reg_weight: float = 1e-3
    calibration_lr: float = 1e-2
    calibration_epochs: int = 150
    calibration_l2: float = 1e-4

    device: str = "auto"
    num_workers: int = 0
    verbose: bool = True
    random_seed: int = 42
