from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from method_b_xfip.config import MethodBConfig
from method_b_xfip.data_utils import (
    FeatureSpec,
    ensure_dir,
    infer_feature_spec,
    season_target_table,
    split_pitcher_seasons,
    standardize_input_dataframe,
)
from method_b_xfip.datasets import PitchTypeGroupDataset, build_group_records, collate_group_batch
from method_b_xfip.evaluate import print_metrics, rank_contributions, regression_metrics
from method_b_xfip.fusion import apply_calibrator, build_season_prediction_frame, fit_calibrator
from method_b_xfip.models import PitchTypeSetModel

FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "FC"}
BREAKING_TYPES = {"SL", "ST", "SV", "CU", "KC", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "KN", "EP", "SC"}


def map_pitch_type_to_family(value: object) -> str:
    if pd.isna(value):
        return "other"
    pitch_type = str(value).strip().upper()
    if pitch_type == "FA":
        pitch_type = "FF"
    if pitch_type == "PO":
        return "other"
    if pitch_type in FASTBALL_TYPES:
        return "fastball"
    if pitch_type in BREAKING_TYPES:
        return "breaking"
    if pitch_type in OFFSPEED_TYPES:
        return "offspeed"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Method B hierarchical pitch-type xFIP models.")
    parser.add_argument("--input-file", default=str(MethodBConfig().input_path))
    parser.add_argument("--artifact-dir", default=str(MethodBConfig().artifact_dir))
    parser.add_argument("--report-dir", default=str(MethodBConfig().report_dir))
    parser.add_argument("--validation-fraction", type=float, default=MethodBConfig.validation_fraction)
    parser.add_argument("--test-fraction", type=float, default=MethodBConfig.test_fraction)
    parser.add_argument("--split-seed", type=int, default=MethodBConfig.split_seed)
    parser.add_argument("--min-pitch-type-rows", type=int, default=MethodBConfig.min_pitch_type_rows)
    parser.add_argument("--min-pitch-type-groups", type=int, default=MethodBConfig.min_pitch_type_groups)
    parser.add_argument("--min-group-size", type=int, default=MethodBConfig.min_group_size)
    parser.add_argument("--max-missing-feature-frac", type=float, default=MethodBConfig.max_missing_feature_frac)
    parser.add_argument("--max-categorical-levels", type=int, default=MethodBConfig.max_categorical_levels)
    parser.add_argument("--hidden-dims", default=",".join(map(str, MethodBConfig.hidden_dims)))
    parser.add_argument("--dropout", type=float, default=MethodBConfig.dropout)
    parser.add_argument("--batch-size", type=int, default=MethodBConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=MethodBConfig.epochs)
    parser.add_argument("--lr", type=float, default=MethodBConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=MethodBConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=MethodBConfig.patience)
    parser.add_argument("--grad-clip-norm", type=float, default=MethodBConfig.grad_clip_norm)
    parser.add_argument("--aggregation", default=MethodBConfig.aggregation, choices=["mean", "median", "trimmed_mean"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-lr", type=float, default=MethodBConfig.calibration_lr)
    parser.add_argument("--calibration-epochs", type=int, default=MethodBConfig.calibration_epochs)
    parser.add_argument("--calibration-l2", type=float, default=MethodBConfig.calibration_l2)
    parser.add_argument("--device", default=MethodBConfig.device)
    parser.add_argument("--num-workers", type=int, default=MethodBConfig.num_workers)
    parser.add_argument("--pitch-group-mode", default=MethodBConfig.pitch_group_mode, choices=["family", "raw"])
    return parser.parse_args()


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    dims = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not dims:
        raise ValueError("At least one hidden dimension is required")
    return tuple(dims)


def select_feature_columns(
    df: pd.DataFrame,
    spec: FeatureSpec,
    max_missing_feature_frac: float,
) -> FeatureSpec:
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
        raise ValueError("No usable features found for this pitch type")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transform_frame(preprocessor: ColumnTransformer, df: pd.DataFrame, feature_spec: FeatureSpec) -> pd.DataFrame:
    arr = preprocessor.transform(df.loc[:, feature_spec.numeric + feature_spec.categorical])
    names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(arr, columns=names, index=df.index)


def make_group_records(
    transformed_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: MethodBConfig,
    pitch_type: str,
    allowed_keys: set[str] | None = None,
) -> list:
    return build_group_records(
        df=transformed_df.join(raw_df[[config.pitcher_col, config.season_col, config.pitch_type_col, config.target_col]], how="left"),
        feature_cols=list(transformed_df.columns),
        target_col=config.target_col,
        pitcher_col=config.pitcher_col,
        season_col=config.season_col,
        pitch_type_col=config.pitch_type_col,
        min_group_size=config.min_group_size,
        group_keys=allowed_keys,
    )


def make_group_key_series(df: pd.DataFrame, config: MethodBConfig) -> pd.Series:
    return df[config.pitcher_col].astype(str) + "__" + df[config.season_col].astype(str)


def train_single_pitch_type(
    pitch_type: str,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    config: MethodBConfig,
    hidden_dims: tuple[int, ...],
    device: str,
    global_spec: FeatureSpec,
) -> dict[str, Any] | None:
    pitch_train = pd.DataFrame(train_raw.loc[train_raw[config.pitch_type_col] == pitch_type].copy())
    pitch_val = pd.DataFrame(val_raw.loc[val_raw[config.pitch_type_col] == pitch_type].copy())
    pitch_test = pd.DataFrame(test_raw.loc[test_raw[config.pitch_type_col] == pitch_type].copy())

    if len(pitch_train) < config.min_pitch_type_rows:
        print(f"[SKIP] {pitch_type}: only {len(pitch_train)} train rows")
        return None

    train_groups = pitch_train.groupby([config.pitcher_col, config.season_col], sort=False).size()
    if len(train_groups) < config.min_pitch_type_groups:
        print(f"[SKIP] {pitch_type}: only {len(train_groups)} train pitcher-season groups")
        return None

    pitch_spec = select_feature_columns(pitch_train, global_spec, config.max_missing_feature_frac)
    preprocessor = build_preprocessor(pitch_spec)
    preprocessor.fit(pitch_train.loc[:, pitch_spec.numeric + pitch_spec.categorical])

    train_x: pd.DataFrame = transform_frame(preprocessor, pitch_train, pitch_spec)
    val_x: pd.DataFrame = transform_frame(preprocessor, pitch_val, pitch_spec) if len(pitch_val) else pd.DataFrame(columns=list(preprocessor.get_feature_names_out()))
    test_x: pd.DataFrame = transform_frame(preprocessor, pitch_test, pitch_spec) if len(pitch_test) else pd.DataFrame(columns=list(preprocessor.get_feature_names_out()))

    train_records = make_group_records(train_x, pitch_train, config, pitch_type, None)
    val_records = make_group_records(val_x, pitch_val, config, pitch_type, None) if len(pitch_val) else []
    test_records = make_group_records(test_x, pitch_test, config, pitch_type, None) if len(pitch_test) else []

    if not train_records or not val_records:
        print(f"[SKIP] {pitch_type}: insufficient grouped train/val records")
        return None

    train_loader = DataLoader(
        PitchTypeGroupDataset(train_records),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_group_batch,
    )
    val_loader = DataLoader(
        PitchTypeGroupDataset(val_records),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_group_batch,
    )

    input_dim = train_records[0].features.shape[1]
    model = PitchTypeSetModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=config.dropout, aggregation=config.aggregation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred, _ = model(x, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_pred, val_true = score_group_loader(model, val_loader, device)
        val_metrics = regression_metrics(val_true, val_pred)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"[{pitch_type}] epoch {epoch:03d} train_loss={train_loss:.4f} val_rmse={val_metrics['rmse']:.4f}")

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "feature_spec": asdict(pitch_spec),
                "config": asdict(config),
                "aggregation": config.aggregation,
                "hidden_dims": hidden_dims,
            }
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{pitch_type}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"Pitch type {pitch_type} failed to train")

    model.load_state_dict(best_state["model_state_dict"])

    return {
        "pitch_type": pitch_type,
        "model": model,
        "preprocessor": preprocessor,
        "feature_spec": pitch_spec,
        "best_state": best_state,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "train_raw": pitch_train,
        "val_raw": pitch_val,
        "test_raw": pitch_test,
    }


@torch.no_grad()
def score_group_loader(model: PitchTypeSetModel, loader: DataLoader, device: str) -> tuple[list[float], list[float]]:
    model.eval()
    preds: list[float] = []
    trues: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)
        pred, _ = model(x, mask)
        preds.extend(pred.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())
    return preds, trues


def score_pitch_type_groups(
    model: PitchTypeSetModel,
    preprocessor: ColumnTransformer,
    feature_spec: FeatureSpec,
    raw_df: pd.DataFrame,
    config: MethodBConfig,
    pitch_type: str,
    device: str,
) -> pd.DataFrame:
    subset = pd.DataFrame(raw_df.loc[raw_df[config.pitch_type_col] == pitch_type].copy())
    if subset.empty:
        return pd.DataFrame(columns=["pitcher_id", "season", "pitch_type", "score", "pitch_count"])

    transformed = transform_frame(preprocessor, subset, feature_spec)
    features = transformed.join(subset[[config.pitcher_col, config.season_col, config.pitch_type_col, config.target_col]], how="left")

    rows = []
    for (pitcher_id, season, pt), group in features.groupby([config.pitcher_col, config.season_col, config.pitch_type_col], sort=False):
        x = torch.tensor(group[list(transformed.columns)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)
        x = x.unsqueeze(0)
        with torch.no_grad():
            score, _ = model(x, mask)
        rows.append(
            {
                "pitcher_id": str(pitcher_id),
                "season": int(season),
                "pitch_type": str(pt),
                "score": float(score.item()),
                "pitch_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_pitch_type_model_output_dir(base_dir: Path, pitch_type: str) -> Path:
    return ensure_dir(base_dir / "pitch_types" / pitch_type)


def save_pitch_type_artifacts(result: dict[str, Any], artifact_dir: Path) -> None:
    model_dir = build_pitch_type_model_output_dir(artifact_dir, result["pitch_type"])
    torch.save(result["model"].state_dict(), model_dir / "model.pt")
    joblib.dump(result["preprocessor"], model_dir / "preprocessor.joblib")
    with (model_dir / "feature_spec.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result["feature_spec"]), f, indent=2)
    meta = {
        "pitch_type": result["pitch_type"],
        "best_val_rmse": result["best_val_rmse"],
        "best_epoch": result["best_epoch"],
        "input_dim": result["best_state"]["input_dim"],
        "aggregation": result["best_state"]["aggregation"],
        "hidden_dims": list(result["best_state"]["hidden_dims"]),
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_prediction_tables_for_split(
    split_df: pd.DataFrame,
    config: MethodBConfig,
    pitch_type_models: dict[str, dict[str, Any]],
    all_pitch_types: list[str],
    device: str,
) -> pd.DataFrame:
    season_keys = season_target_table(split_df, config.pitcher_col, config.season_col, config.target_col)
    per_type_scores: dict[str, pd.DataFrame] = {}
    for pitch_type in all_pitch_types:
        result = pitch_type_models.get(pitch_type)
        if result is None:
            per_type_scores[pitch_type] = pd.DataFrame(columns=["pitcher_id", "season", "score", "pitch_count"])
            continue
        per_type_scores[pitch_type] = score_pitch_type_groups(
            model=result["model"],
            preprocessor=result["preprocessor"],
            feature_spec=result["feature_spec"],
            raw_df=split_df,
            config=config,
            pitch_type=pitch_type,
            device=device,
        )
    return build_season_prediction_frame(season_keys, all_pitch_types, per_type_scores)


def main() -> None:
    args = parse_args()
    config = MethodBConfig(
        input_path=Path(args.input_file),
        artifact_dir=Path(args.artifact_dir),
        report_dir=Path(args.report_dir),
        pitch_group_mode=args.pitch_group_mode,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        min_pitch_type_rows=args.min_pitch_type_rows,
        min_pitch_type_groups=args.min_pitch_type_groups,
        min_group_size=args.min_group_size,
        max_missing_feature_frac=args.max_missing_feature_frac,
        max_categorical_levels=args.max_categorical_levels,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        aggregation=args.aggregation,
        calibrate=args.calibrate,
        calibration_lr=args.calibration_lr,
        calibration_epochs=args.calibration_epochs,
        calibration_l2=args.calibration_l2,
        device=args.device,
        num_workers=args.num_workers,
    )

    device = get_device(config.device)
    ensure_dir(config.artifact_dir)
    ensure_dir(config.report_dir)

    print(f"[Load] {config.input_path}")
    raw = pd.read_csv(config.input_path)
    df = standardize_input_dataframe(
        raw,
        pitcher_col=config.pitcher_col,
        season_col=config.season_col,
        pitch_type_col=config.pitch_type_col,
        target_col=config.target_col,
    )

    if config.pitch_group_mode == "family":
        df["pitch_type_group"] = df[config.pitch_type_col].apply(map_pitch_type_to_family)
        config.pitch_type_col = "pitch_type_group"
    else:
        df["pitch_type_group"] = df[config.pitch_type_col].astype(str).str.strip().str.upper()

    print(f"[Load] rows={len(df):,} cols={len(df.columns)}")
    print(f"[Pitch grouping] mode={config.pitch_group_mode} model_col={config.pitch_type_col}")

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
            "pitch_type",
            "pitch_type_group",
            config.target_col,
            "pitch_family",
            "label",
            "events",
            "description",
        },
        max_categorical_levels=config.max_categorical_levels,
    )
    print(f"[Features] numeric={len(global_spec.numeric)} categorical={len(global_spec.categorical)}")

    all_pitch_types = sorted(df[config.pitch_type_col].dropna().astype(str).unique().tolist())
    print(f"[Pitch types] {all_pitch_types}")

    trained_models: dict[str, dict[str, Any]] = {}
    skipped_pitch_types: list[str] = []
    for pitch_type in all_pitch_types:
        result = train_single_pitch_type(
            pitch_type=pitch_type,
            train_raw=split.train,
            val_raw=split.validation,
            test_raw=split.test,
            config=config,
            hidden_dims=config.hidden_dims,
            device=device,
            global_spec=global_spec,
        )
        if result is None:
            skipped_pitch_types.append(pitch_type)
            continue
        trained_models[pitch_type] = result
        save_pitch_type_artifacts(result, config.artifact_dir)

    if not trained_models:
        raise RuntimeError("No pitch types were trained successfully")

    print(f"[Trained pitch types] {list(trained_models.keys())}")
    if skipped_pitch_types:
        print(f"[Skipped pitch types] {skipped_pitch_types}")

    train_pred = build_prediction_tables_for_split(split.train, config, trained_models, all_pitch_types, device)
    val_pred = build_prediction_tables_for_split(split.validation, config, trained_models, all_pitch_types, device)
    test_pred = build_prediction_tables_for_split(split.test, config, trained_models, all_pitch_types, device)

    train_pred["split"] = "train"
    val_pred["split"] = "validation"
    test_pred["split"] = "test"

    if config.calibrate:
        calibrator = fit_calibrator(
            train_pred,
            pitch_types=all_pitch_types,
            target_col="true_xFIP",
            epochs=config.calibration_epochs,
            lr=config.calibration_lr,
            l2=config.calibration_l2,
            device=device,
        )
        torch.save(calibrator.state_dict(), config.artifact_dir / "fusion_calibrator.pt")
        train_pred = apply_calibrator(train_pred, all_pitch_types, calibrator, device=device)
        val_pred = apply_calibrator(val_pred, all_pitch_types, calibrator, device=device)
        test_pred = apply_calibrator(test_pred, all_pitch_types, calibrator, device=device)
        print("[Fusion] calibration enabled")
    else:
        print("[Fusion] pure usage-weighted sum")

    all_predictions = pd.concat([train_pred, val_pred, test_pred], ignore_index=True)
    all_predictions = rank_contributions(all_predictions, [f"contribution_{pt}" for pt in all_pitch_types])

    metrics_train = regression_metrics(train_pred["true_xFIP"], train_pred["predicted_xFIP"])
    metrics_val = regression_metrics(val_pred["true_xFIP"], val_pred["predicted_xFIP"])
    metrics_test = regression_metrics(test_pred["true_xFIP"], test_pred["predicted_xFIP"])
    print_metrics("Train", metrics_train)
    print_metrics("Validation", metrics_val)
    print_metrics("Test", metrics_test)

    out_predictions = config.report_dir / "method_b_predictions.csv"
    out_summary = config.report_dir / "method_b_metrics.json"
    out_pitch_types = config.report_dir / "method_b_pitch_type_summary.csv"

    all_predictions.to_csv(out_predictions, index=False)

    pitch_type_summary_rows = []
    for pitch_type, result in trained_models.items():
        pitch_type_summary_rows.append(
            {
                "pitch_type": pitch_type,
                "best_val_rmse": result["best_val_rmse"],
                "best_epoch": result["best_epoch"],
                "train_groups": len(result["train_records"]),
                "val_groups": len(result["val_records"]),
                "test_groups": len(result["test_records"]),
                "input_dim": result["best_state"]["input_dim"],
            }
        )
    pd.DataFrame(pitch_type_summary_rows).sort_values("best_val_rmse").to_csv(out_pitch_types, index=False)

    config_dict = asdict(config)
    config_dict["input_path"] = str(config_dict["input_path"])
    config_dict["artifact_dir"] = str(config_dict["artifact_dir"])
    config_dict["report_dir"] = str(config_dict["report_dir"])

    summary = {
        "config": config_dict,
        "trained_pitch_types": list(trained_models.keys()),
        "skipped_pitch_types": skipped_pitch_types,
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "test_metrics": metrics_test,
        "output_predictions": str(out_predictions),
        "output_pitch_type_summary": str(out_pitch_types),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (config.artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {out_predictions}")
    print(f"[Saved] {out_summary}")
    print(f"[Saved] {out_pitch_types}")


if __name__ == "__main__":
    main()
