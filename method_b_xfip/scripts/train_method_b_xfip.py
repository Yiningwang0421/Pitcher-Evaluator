from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from method_b_xfip.config import MethodBConfig
from method_b_xfip.data_utils import (
    FeatureSpec,
    ensure_dir,
    infer_feature_spec,
    split_pitcher_seasons,
    standardize_input_dataframe,
)
from method_b_xfip.datasets import (
    PitcherSeasonFamilyDataset,
    collate_pitcher_season_family_batch,
)
from method_b_xfip.evaluate import print_metrics, rank_contributions, regression_metrics
from method_b_xfip.models import PitchFamilySetModel
from method_b_xfip.pitch_families import FAMILY_ORDER, FamilyMappingStats, add_pitch_family_column


def parse_args() -> argparse.Namespace:
    defaults = MethodBConfig()
    parser = argparse.ArgumentParser(description="Train Method B xFIP model with family-wise fixed-size pitch sampling.")
    parser.add_argument("--input-file", default=str(defaults.input_path))
    parser.add_argument("--artifact-dir", default=str(defaults.artifact_dir))
    parser.add_argument("--report-dir", default=str(defaults.report_dir))

    parser.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    parser.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)

    parser.add_argument("--max-missing-feature-frac", type=float, default=defaults.max_missing_feature_frac)
    parser.add_argument("--max-categorical-levels", type=int, default=defaults.max_categorical_levels)

    parser.add_argument("--hidden-dims", default=",".join(map(str, defaults.hidden_dims)))
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--family-encoder-type", default=defaults.family_encoder_type, choices=["mean_mlp", "set_attention"])
    parser.add_argument("--family-embed-dim", type=int, default=defaults.family_embed_dim)
    parser.add_argument("--family-encoder-num-heads", type=int, default=defaults.family_encoder_num_heads)
    parser.add_argument("--family-encoder-num-layers", type=int, default=defaults.family_encoder_num_layers)
    parser.add_argument("--family-encoder-dropout", type=float, default=defaults.family_encoder_dropout)
    parser.add_argument("--family-pooling", default=defaults.family_pooling, choices=["attention", "masked_mean"])
    parser.add_argument("--family-score-hidden-dim", type=int, default=defaults.family_score_hidden_dim)
    parser.add_argument("--loss-type", default=defaults.loss_type, choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=defaults.huber_delta)
    parser.add_argument("--use-residual-correction", action="store_true", default=defaults.use_residual_correction)
    parser.add_argument("--residual-model-type", default=defaults.residual_model_type, choices=["hgbt", "ridge"])
    parser.add_argument("--residual-max-iter", type=int, default=defaults.residual_max_iter)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)

    parser.add_argument("--aggregation", default=defaults.aggregation, choices=["mean", "median", "trimmed_mean"])
    parser.add_argument(
        "--fusion-mode",
        default=defaults.fusion_mode,
        choices=["usage_weighted_sum", "calibrated_family_scores"],
    )
    parser.add_argument("--use-family-reliability", action="store_true", default=defaults.use_family_reliability)
    parser.add_argument("--reliability-count-c", type=float, default=defaults.reliability_count_c)
    parser.add_argument(
        "--normalize-reliability-weights",
        action="store_true",
        default=defaults.normalize_reliability_weights,
    )
    parser.add_argument("--no-normalize-reliability-weights", action="store_false", dest="normalize_reliability_weights")
    parser.add_argument("--calibrate", action="store_true", default=defaults.calibrate)
    parser.add_argument("--collapse-reg-weight", type=float, default=defaults.collapse_reg_weight)

    parser.add_argument("--fastball-samples", type=int, default=defaults.family_sample_sizes["fastball"])
    parser.add_argument("--breaking-samples", type=int, default=defaults.family_sample_sizes["breaking"])
    parser.add_argument("--offspeed-samples", type=int, default=defaults.family_sample_sizes["offspeed"])

    parser.add_argument(
        "--sampling-mode-train",
        default=defaults.sampling_mode_train,
        choices=["random_without_replacement"],
    )
    parser.add_argument(
        "--sampling-mode-eval",
        default=defaults.sampling_mode_eval,
        choices=["deterministic"],
    )
    parser.add_argument(
        "--short-family-strategy",
        default=defaults.short_family_strategy,
        choices=["zero_pad_with_mask", "sample_with_replacement"],
    )

    parser.add_argument("--drop-unknown-pitch-types", action="store_true", default=defaults.drop_unknown_pitch_types)
    parser.add_argument("--keep-unknown-pitch-types", action="store_false", dest="drop_unknown_pitch_types")
    parser.add_argument(
        "--unknown-pitch-family-fallback",
        default=defaults.unknown_pitch_family_fallback,
        choices=list(FAMILY_ORDER),
    )

    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--data-cache-dir", default=str(defaults.data_cache_dir))
    parser.add_argument("--use-data-cache", action="store_true", default=defaults.use_data_cache)
    parser.add_argument("--no-data-cache", action="store_false", dest="use_data_cache")
    return parser.parse_args()


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    dims = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not dims:
        raise ValueError("At least one hidden dimension is required")
    return tuple(dims)


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_feature_columns(df: pd.DataFrame, spec: FeatureSpec, max_missing_feature_frac: float) -> FeatureSpec:
    numeric: list[str] = []
    categorical: list[str] = []

    for col in spec.numeric:
        if col not in df.columns:
            continue
        valid_frac = float(df[col].notna().mean())
        if valid_frac < (1.0 - max_missing_feature_frac):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        numeric.append(col)

    for col in spec.categorical:
        if col not in df.columns:
            continue
        valid_frac = float(df[col].notna().mean())
        if valid_frac < (1.0 - max_missing_feature_frac):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        categorical.append(col)

    return FeatureSpec(numeric=numeric, categorical=categorical)


def build_preprocessor(feature_spec: FeatureSpec) -> ColumnTransformer:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers: list[tuple[str, Any, list[str]]] = []
    if feature_spec.numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_spec.numeric,
            )
        )
    if feature_spec.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot),
                    ]
                ),
                feature_spec.categorical,
            )
        )

    if not transformers:
        raise ValueError("No usable features found")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transform_frame(preprocessor: ColumnTransformer, df: pd.DataFrame, feature_spec: FeatureSpec) -> pd.DataFrame:
    arr = preprocessor.transform(df.loc[:, feature_spec.numeric + feature_spec.categorical])
    names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(arr, columns=names, index=df.index)


def to_transformed_with_meta(
    raw_split: pd.DataFrame,
    transformed_split: pd.DataFrame,
    config: MethodBConfig,
) -> pd.DataFrame:
    return transformed_split.join(
        raw_split[[config.pitcher_col, config.season_col, config.target_col, "pitch_family"]],
        how="left",
    )


def build_data_cache_key(config: MethodBConfig) -> str:
    input_path = Path(config.input_path)
    stat = input_path.stat()
    payload = {
        "input_path": str(input_path.resolve()),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "input_size": int(stat.st_size),
        "pitcher_col": config.pitcher_col,
        "season_col": config.season_col,
        "pitch_type_col": config.pitch_type_col,
        "target_col": config.target_col,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "split_seed": config.split_seed,
        "max_missing_feature_frac": config.max_missing_feature_frac,
        "max_categorical_levels": config.max_categorical_levels,
        "pitch_family_mapping": dict(sorted(config.pitch_family_mapping.items())),
        "drop_unknown_pitch_types": config.drop_unknown_pitch_types,
        "unknown_pitch_family_fallback": config.unknown_pitch_family_fallback,
    }
    key_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]


def save_prepared_data_cache(
    cache_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    feature_spec: FeatureSpec,
    family_stats: FamilyMappingStats,
) -> None:
    ensure_dir(cache_dir)
    train_df.to_pickle(cache_dir / "train_df.pkl")
    val_df.to_pickle(cache_dir / "val_df.pkl")
    test_df.to_pickle(cache_dir / "test_df.pkl")
    joblib.dump(preprocessor, cache_dir / "preprocessor.joblib")
    with (cache_dir / "feature_spec.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(feature_spec), f, indent=2)
    with (cache_dir / "family_mapping_stats.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(family_stats), f, indent=2)


def load_prepared_data_cache(
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer, FeatureSpec, FamilyMappingStats] | None:
    required_files = [
        cache_dir / "train_df.pkl",
        cache_dir / "val_df.pkl",
        cache_dir / "test_df.pkl",
        cache_dir / "preprocessor.joblib",
        cache_dir / "feature_spec.json",
        cache_dir / "family_mapping_stats.json",
    ]
    if not all(path.exists() for path in required_files):
        return None

    train_df = pd.read_pickle(cache_dir / "train_df.pkl")
    val_df = pd.read_pickle(cache_dir / "val_df.pkl")
    test_df = pd.read_pickle(cache_dir / "test_df.pkl")
    preprocessor = joblib.load(cache_dir / "preprocessor.joblib")

    feature_spec_dict = json.loads((cache_dir / "feature_spec.json").read_text(encoding="utf-8"))
    feature_spec = FeatureSpec(
        numeric=list(feature_spec_dict.get("numeric", [])),
        categorical=list(feature_spec_dict.get("categorical", [])),
    )
    family_stats_dict = json.loads((cache_dir / "family_mapping_stats.json").read_text(encoding="utf-8"))
    family_stats = FamilyMappingStats(
        total_rows=int(family_stats_dict.get("total_rows", 0)),
        mapped_rows=int(family_stats_dict.get("mapped_rows", 0)),
        dropped_rows=int(family_stats_dict.get("dropped_rows", 0)),
        fallback_rows=int(family_stats_dict.get("fallback_rows", 0)),
    )
    return train_df, val_df, test_df, preprocessor, feature_spec, family_stats


def make_loader(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: MethodBConfig,
    is_train: bool,
) -> tuple[DataLoader, PitcherSeasonFamilyDataset]:
    dataset = PitcherSeasonFamilyDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=config.target_col,
        pitcher_col=config.pitcher_col,
        season_col=config.season_col,
        family_col="pitch_family",
        family_sample_sizes=config.family_sample_sizes,
        sampling_mode_train=config.sampling_mode_train,
        sampling_mode_eval=config.sampling_mode_eval,
        short_family_strategy=config.short_family_strategy,
        random_seed=config.random_seed,
        is_train=is_train,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        collate_fn=collate_pitcher_season_family_batch,
    )
    return loader, dataset


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    out = dict(batch)
    for key in [
        "fastball_x",
        "fastball_mask",
        "breaking_x",
        "breaking_mask",
        "offspeed_x",
        "offspeed_mask",
        "family_usage_weights",
        "y",
        "family_counts",
    ]:
        out[key] = batch[key].to(device)
    return out


def run_epoch(
    model: PitchFamilySetModel,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
    collapse_reg_weight: float,
    loss_type: str,
    huber_delta: float,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "huber":
        loss_fn = nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    losses: list[float] = []
    mse_losses: list[float] = []
    collapse_losses: list[float] = []
    all_preds: list[float] = []
    all_trues: list[float] = []
    all_usage: list[np.ndarray] = []
    all_family_scores: list[np.ndarray] = []
    per_example: list[dict[str, Any]] = []

    for batch in loader:
        batch_dev = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(
                fastball_x=batch_dev["fastball_x"],
                fastball_mask=batch_dev["fastball_mask"],
                breaking_x=batch_dev["breaking_x"],
                breaking_mask=batch_dev["breaking_mask"],
                offspeed_x=batch_dev["offspeed_x"],
                offspeed_mask=batch_dev["offspeed_mask"],
                family_usage_weights=batch_dev["family_usage_weights"],
                family_counts=batch_dev["family_counts"],
            )
            pred = outputs["predicted_xfip"]
            mse_loss = loss_fn(pred, batch_dev["y"])
            collapse_loss = family_score_collapse_penalty(outputs["family_scores"])
            loss = mse_loss + (collapse_reg_weight * collapse_loss if is_train else 0.0)

            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        losses.append(float(loss.item()))
        mse_losses.append(float(mse_loss.item()))
        collapse_losses.append(float(collapse_loss.item()))

        pred_np = pred.detach().cpu().numpy()
        true_np = batch_dev["y"].detach().cpu().numpy()
        family_scores_np = outputs["family_scores"].detach().cpu().numpy()
        usage_np = batch_dev["family_usage_weights"].detach().cpu().numpy()
        counts_np = batch_dev["family_counts"].detach().cpu().numpy()

        all_preds.extend(pred_np.tolist())
        all_trues.extend(true_np.tolist())
        all_family_scores.append(family_scores_np)
        all_usage.append(usage_np)

        for idx in range(len(pred_np)):
            per_example.append(
                {
                    "pitcher_id": batch["pitcher_ids"][idx],
                    "season": int(batch["seasons"][idx]),
                    "true_xFIP": float(true_np[idx]),
                    "predicted_xFIP": float(pred_np[idx]),
                    "score_fastball": float(family_scores_np[idx, 0]),
                    "score_breaking": float(family_scores_np[idx, 1]),
                    "score_offspeed": float(family_scores_np[idx, 2]),
                    "weight_fastball": float(usage_np[idx, 0]),
                    "weight_breaking": float(usage_np[idx, 1]),
                    "weight_offspeed": float(usage_np[idx, 2]),
                    "pitch_count_fastball": int(counts_np[idx, 0]),
                    "pitch_count_breaking": int(counts_np[idx, 1]),
                    "pitch_count_offspeed": int(counts_np[idx, 2]),
                }
            )

    frame = pd.DataFrame(per_example)
    if not frame.empty:
        for family in FAMILY_ORDER:
            frame[f"contribution_{family}"] = frame[f"score_{family}"] * frame[f"weight_{family}"]

    avg_usage = (
        np.concatenate(all_usage, axis=0).mean(axis=0).tolist() if all_usage else [0.0, 0.0, 0.0]
    )

    dominant_family_counts: dict[str, int] = {}
    if not frame.empty:
        contribution_cols = [f"contribution_{family}" for family in FAMILY_ORDER]
        contribution_values = frame[contribution_cols].to_numpy(dtype=np.float64)
        dominant_idx = np.argmax(contribution_values, axis=1)
        dominant_family_counts = {
            family: int(np.sum(dominant_idx == family_idx))
            for family_idx, family in enumerate(FAMILY_ORDER)
        }

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mse_loss": float(np.mean(mse_losses)) if mse_losses else float("nan"),
        "collapse_loss": float(np.mean(collapse_losses)) if collapse_losses else float("nan"),
        "metrics": regression_metrics(all_trues, all_preds),
        "avg_usage": avg_usage,
        "frame": frame,
        "dominant_family_counts": dominant_family_counts,
        "family_score_examples": np.concatenate(all_family_scores, axis=0)[:3].tolist() if all_family_scores else [],
    }


def family_score_collapse_penalty(family_scores: torch.Tensor) -> torch.Tensor:
    if family_scores.ndim != 2 or family_scores.shape[1] != 3:
        raise ValueError("family_scores must be [batch, 3]")
    if family_scores.shape[0] <= 1:
        return torch.tensor(0.0, device=family_scores.device)

    centered = family_scores - family_scores.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    z = centered / std
    corr = (z.T @ z) / float(z.shape[0] - 1)
    off_diag = corr - torch.eye(3, device=family_scores.device)
    return (off_diag.pow(2).sum() / 6.0)


def residual_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "predicted_xFIP",
        "score_fastball",
        "score_breaking",
        "score_offspeed",
        "weight_fastball",
        "weight_breaking",
        "weight_offspeed",
        "contribution_fastball",
        "contribution_breaking",
        "contribution_offspeed",
        "pitch_count_fastball",
        "pitch_count_breaking",
        "pitch_count_offspeed",
    ]
    return [col for col in candidates if col in frame.columns]


def fit_residual_corrector(
    train_frame: pd.DataFrame,
    model_type: str,
    max_iter: int,
    random_seed: int,
) -> tuple[object, list[str]]:
    feature_cols = residual_feature_columns(train_frame)
    x_train = train_frame[feature_cols].to_numpy(dtype=np.float32)
    y_residual = (train_frame["true_xFIP"] - train_frame["predicted_xFIP"]).to_numpy(dtype=np.float32)

    if model_type == "hgbt":
        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=max_iter,
            random_state=random_seed,
        )
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unsupported residual model type: {model_type}")

    model.fit(x_train, y_residual)
    return model, feature_cols


def apply_residual_corrector(frame: pd.DataFrame, model: Any, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    x = out[feature_cols].to_numpy(dtype=np.float32)
    residual = model.predict(x)
    out["predicted_xFIP_base"] = out["predicted_xFIP"]
    out["predicted_residual"] = residual
    out["predicted_xFIP"] = out["predicted_xFIP"] + out["predicted_residual"]
    return out


def main() -> None:
    args = parse_args()

    fusion_mode = "calibrated_family_scores" if args.calibrate else args.fusion_mode

    config = MethodBConfig(
        input_path=Path(args.input_file),
        artifact_dir=Path(args.artifact_dir),
        report_dir=Path(args.report_dir),
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_missing_feature_frac=args.max_missing_feature_frac,
        max_categorical_levels=args.max_categorical_levels,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        family_encoder_type=args.family_encoder_type,
        family_embed_dim=args.family_embed_dim,
        family_encoder_num_heads=args.family_encoder_num_heads,
        family_encoder_num_layers=args.family_encoder_num_layers,
        family_encoder_dropout=args.family_encoder_dropout,
        family_pooling=args.family_pooling,
        family_score_hidden_dim=args.family_score_hidden_dim,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        use_residual_correction=args.use_residual_correction,
        residual_model_type=args.residual_model_type,
        residual_max_iter=args.residual_max_iter,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        aggregation=args.aggregation,
        fusion_mode=fusion_mode,
        use_family_reliability=args.use_family_reliability,
        reliability_count_c=args.reliability_count_c,
        normalize_reliability_weights=args.normalize_reliability_weights,
        calibrate=args.calibrate,
        collapse_reg_weight=args.collapse_reg_weight,
        family_sample_sizes={
            "fastball": int(args.fastball_samples),
            "breaking": int(args.breaking_samples),
            "offspeed": int(args.offspeed_samples),
        },
        sampling_mode_train=args.sampling_mode_train,
        sampling_mode_eval=args.sampling_mode_eval,
        short_family_strategy=args.short_family_strategy,
        drop_unknown_pitch_types=args.drop_unknown_pitch_types,
        unknown_pitch_family_fallback=args.unknown_pitch_family_fallback,
        device=args.device,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        data_cache_dir=Path(args.data_cache_dir),
        use_data_cache=args.use_data_cache,
    )

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    device = get_device(config.device)
    ensure_dir(config.artifact_dir)
    ensure_dir(config.report_dir)

    cache_key = build_data_cache_key(config)
    data_cache_dir = Path(config.data_cache_dir) / cache_key
    cached = load_prepared_data_cache(data_cache_dir) if config.use_data_cache else None

    if cached is not None:
        train_df, val_df, test_df, preprocessor, feature_spec, family_stats = cached
        print(f"[Data cache] hit: {data_cache_dir}")
        print(f"[Cached split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    else:
        print(f"[Data cache] miss: {data_cache_dir}")
        print(f"[Load] {config.input_path}")
        raw = pd.read_csv(config.input_path)
        df = standardize_input_dataframe(
            raw,
            pitcher_col=config.pitcher_col,
            season_col=config.season_col,
            pitch_type_col=config.pitch_type_col,
            target_col=config.target_col,
        )

        df, family_stats = add_pitch_family_column(
            df=df,
            pitch_type_col=config.pitch_type_col,
            out_col="pitch_family",
            mapping=config.pitch_family_mapping,
            drop_unknown_pitch_types=config.drop_unknown_pitch_types,
            unknown_pitch_family_fallback=config.unknown_pitch_family_fallback,
        )
        print(
            "[Pitch family mapping] "
            f"total={family_stats.total_rows} mapped={family_stats.mapped_rows} "
            f"dropped={family_stats.dropped_rows} fallback={family_stats.fallback_rows}"
        )

        split = split_pitcher_seasons(
            df,
            pitcher_col=config.pitcher_col,
            season_col=config.season_col,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.split_seed,
        )
        print(f"[Split] train={len(split.train):,} val={len(split.validation):,} test={len(split.test):,}")

        global_spec = infer_feature_spec(
            split.train,
            exclude_columns={
                config.pitcher_col,
                config.season_col,
                config.pitch_type_col,
                config.target_col,
                "pitch_family",
                "label",
                "events",
                "description",
            },
            max_categorical_levels=config.max_categorical_levels,
        )
        feature_spec = select_feature_columns(split.train, global_spec, config.max_missing_feature_frac)
        print(f"[Features] numeric={len(feature_spec.numeric)} categorical={len(feature_spec.categorical)}")

        preprocessor = build_preprocessor(feature_spec)
        preprocessor.fit(split.train.loc[:, feature_spec.numeric + feature_spec.categorical])

        train_x = transform_frame(preprocessor, split.train, feature_spec)
        val_x = transform_frame(preprocessor, split.validation, feature_spec)
        test_x = transform_frame(preprocessor, split.test, feature_spec)

        train_df = to_transformed_with_meta(split.train, train_x, config)
        val_df = to_transformed_with_meta(split.validation, val_x, config)
        test_df = to_transformed_with_meta(split.test, test_x, config)

        if config.use_data_cache:
            save_prepared_data_cache(
                cache_dir=data_cache_dir,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                preprocessor=preprocessor,
                feature_spec=feature_spec,
                family_stats=family_stats,
            )
            print(f"[Data cache] saved: {data_cache_dir}")

    print(
        "[Pitch family mapping] "
        f"total={family_stats.total_rows} mapped={family_stats.mapped_rows} "
        f"dropped={family_stats.dropped_rows} fallback={family_stats.fallback_rows}"
    )
    print(f"[Features] numeric={len(feature_spec.numeric)} categorical={len(feature_spec.categorical)}")

    feature_cols = list(preprocessor.get_feature_names_out())
    train_loader, train_dataset = make_loader(train_df, feature_cols, config, is_train=True)
    val_loader, val_dataset = make_loader(val_df, feature_cols, config, is_train=False)
    test_loader, test_dataset = make_loader(test_df, feature_cols, config, is_train=False)

    print(f"[Sampling stats][train] {train_dataset.get_sampling_stats().to_dict()}")
    print(f"[Sampling stats][val] {val_dataset.get_sampling_stats().to_dict()}")
    print(f"[Sampling stats][test] {test_dataset.get_sampling_stats().to_dict()}")

    input_dim = len(feature_cols)
    model = PitchFamilySetModel(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        family_encoder_type=config.family_encoder_type,
        family_embed_dim=config.family_embed_dim,
        family_encoder_num_heads=config.family_encoder_num_heads,
        family_encoder_num_layers=config.family_encoder_num_layers,
        family_encoder_dropout=config.family_encoder_dropout,
        family_pooling=config.family_pooling,
        family_score_hidden_dim=config.family_score_hidden_dim,
        aggregation=config.aggregation,
        fusion_mode=config.fusion_mode,
        use_family_reliability=config.use_family_reliability,
        reliability_count_c=config.reliability_count_c,
        normalize_reliability_weights=config.normalize_reliability_weights,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        train_out = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            collapse_reg_weight=config.collapse_reg_weight,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
        )
        val_out = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            grad_clip_norm=config.grad_clip_norm,
            collapse_reg_weight=config.collapse_reg_weight,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_out['loss']:.4f} train_mse={train_out['mse_loss']:.4f} "
            f"collapse={train_out['collapse_loss']:.6f} "
            f"val_rmse={val_out['metrics']['rmse']:.4f} "
            f"val_mae={val_out['metrics']['mae']:.4f} "
            f"val_r2={val_out['metrics']['r2']:.4f} "
            f"avg_usage={train_out['avg_usage']}"
        )
        if train_out["dominant_family_counts"]:
            print(f"[Dominant family][train] {train_out['dominant_family_counts']}")
        if config.verbose and val_out["family_score_examples"]:
            print(f"[Debug family scores] {val_out['family_score_examples']}")

        if val_out["metrics"]["rmse"] < best_val_rmse:
            best_val_rmse = val_out["metrics"]["rmse"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
            }
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[Early stopping] epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint")

    model.load_state_dict(best_state["model_state_dict"])

    train_out = run_epoch(
        model,
        train_loader,
        device,
        optimizer=None,
        grad_clip_norm=config.grad_clip_norm,
        collapse_reg_weight=config.collapse_reg_weight,
        loss_type=config.loss_type,
        huber_delta=config.huber_delta,
    )
    val_out = run_epoch(
        model,
        val_loader,
        device,
        optimizer=None,
        grad_clip_norm=config.grad_clip_norm,
        collapse_reg_weight=config.collapse_reg_weight,
        loss_type=config.loss_type,
        huber_delta=config.huber_delta,
    )
    test_out = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        grad_clip_norm=config.grad_clip_norm,
        collapse_reg_weight=config.collapse_reg_weight,
        loss_type=config.loss_type,
        huber_delta=config.huber_delta,
    )

    print_metrics("Train", train_out["metrics"])
    print_metrics("Validation", val_out["metrics"])
    print_metrics("Test", test_out["metrics"])
    if train_out["dominant_family_counts"]:
        print(f"[Dominant family][train] {train_out['dominant_family_counts']}")
    if val_out["dominant_family_counts"]:
        print(f"[Dominant family][validation] {val_out['dominant_family_counts']}")
    if test_out["dominant_family_counts"]:
        print(f"[Dominant family][test] {test_out['dominant_family_counts']}")

    train_frame = train_out["frame"].copy()
    val_frame = val_out["frame"].copy()
    test_frame = test_out["frame"].copy()

    base_metrics = {
        "train": train_out["metrics"],
        "validation": val_out["metrics"],
        "test": test_out["metrics"],
    }

    residual_feature_cols: list[str] = []
    if config.use_residual_correction:
        residual_model, residual_feature_cols = fit_residual_corrector(
            train_frame=train_frame,
            model_type=config.residual_model_type,
            max_iter=config.residual_max_iter,
            random_seed=config.random_seed,
        )
        train_frame = apply_residual_corrector(train_frame, residual_model, residual_feature_cols)
        val_frame = apply_residual_corrector(val_frame, residual_model, residual_feature_cols)
        test_frame = apply_residual_corrector(test_frame, residual_model, residual_feature_cols)

        metrics_train_corr = regression_metrics(train_frame["true_xFIP"], train_frame["predicted_xFIP"])
        metrics_val_corr = regression_metrics(val_frame["true_xFIP"], val_frame["predicted_xFIP"])
        metrics_test_corr = regression_metrics(test_frame["true_xFIP"], test_frame["predicted_xFIP"])
        print_metrics("Train (Residual Corrected)", metrics_train_corr)
        print_metrics("Validation (Residual Corrected)", metrics_val_corr)
        print_metrics("Test (Residual Corrected)", metrics_test_corr)

        train_out["metrics"] = metrics_train_corr
        val_out["metrics"] = metrics_val_corr
        test_out["metrics"] = metrics_test_corr

        joblib.dump(
            {
                "model": residual_model,
                "feature_cols": residual_feature_cols,
                "model_type": config.residual_model_type,
            },
            config.artifact_dir / "residual_corrector.joblib",
        )

    train_frame["split"] = "train"
    val_frame["split"] = "validation"
    test_frame["split"] = "test"

    all_predictions = pd.concat([train_frame, val_frame, test_frame], ignore_index=True)
    all_predictions = rank_contributions(
        all_predictions,
        contribution_cols=[f"contribution_{family}" for family in FAMILY_ORDER],
    )

    out_predictions = config.report_dir / "method_b_predictions.csv"
    out_summary = config.report_dir / "method_b_metrics.json"

    all_predictions.to_csv(out_predictions, index=False)

    model_artifact = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": config.hidden_dims,
        "dropout": config.dropout,
        "family_encoder_type": config.family_encoder_type,
        "family_embed_dim": config.family_embed_dim,
        "family_encoder_num_heads": config.family_encoder_num_heads,
        "family_encoder_num_layers": config.family_encoder_num_layers,
        "family_encoder_dropout": config.family_encoder_dropout,
        "family_pooling": config.family_pooling,
        "family_score_hidden_dim": config.family_score_hidden_dim,
        "use_residual_correction": config.use_residual_correction,
        "residual_model_type": config.residual_model_type,
        "residual_max_iter": config.residual_max_iter,
        "aggregation": config.aggregation,
        "fusion_mode": config.fusion_mode,
        "use_family_reliability": config.use_family_reliability,
        "reliability_count_c": config.reliability_count_c,
        "normalize_reliability_weights": config.normalize_reliability_weights,
    }
    torch.save(model_artifact, config.artifact_dir / "family_model.pt")
    joblib.dump(preprocessor, config.artifact_dir / "preprocessor.joblib")
    with (config.artifact_dir / "feature_spec.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(feature_spec), f, indent=2)

    config_dict = asdict(config)
    config_dict["input_path"] = str(config.input_path)
    config_dict["artifact_dir"] = str(config.artifact_dir)
    config_dict["report_dir"] = str(config.report_dir)
    config_dict["data_cache_dir"] = str(config.data_cache_dir)

    summary = {
        "config": config_dict,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_metrics": train_out["metrics"],
        "validation_metrics": val_out["metrics"],
        "test_metrics": test_out["metrics"],
        "sampling_stats": {
            "train": train_dataset.get_sampling_stats().to_dict(),
            "validation": val_dataset.get_sampling_stats().to_dict(),
            "test": test_dataset.get_sampling_stats().to_dict(),
        },
        "family_mapping_stats": asdict(family_stats),
        "model_metadata": {
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "family_encoder_type": config.family_encoder_type,
            "family_embed_dim": config.family_embed_dim,
            "family_encoder_num_heads": config.family_encoder_num_heads,
            "family_encoder_num_layers": config.family_encoder_num_layers,
            "family_encoder_dropout": config.family_encoder_dropout,
            "family_pooling": config.family_pooling,
            "family_score_hidden_dim": config.family_score_hidden_dim,
            "use_residual_correction": config.use_residual_correction,
            "residual_model_type": config.residual_model_type,
            "residual_max_iter": config.residual_max_iter,
            "loss_type": config.loss_type,
            "huber_delta": config.huber_delta,
            "aggregation": config.aggregation,
            "fusion_mode": config.fusion_mode,
            "use_family_reliability": config.use_family_reliability,
            "reliability_count_c": config.reliability_count_c,
            "normalize_reliability_weights": config.normalize_reliability_weights,
            "calibrate": config.calibrate,
            "collapse_reg_weight": config.collapse_reg_weight,
            "family_sample_sizes": config.family_sample_sizes,
            "short_family_strategy": config.short_family_strategy,
        },
        "base_metrics_before_residual": base_metrics,
        "residual_feature_cols": residual_feature_cols,
        "output_predictions": str(out_predictions),
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (config.artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {out_predictions}")
    print(f"[Saved] {out_summary}")


if __name__ == "__main__":
    main()
