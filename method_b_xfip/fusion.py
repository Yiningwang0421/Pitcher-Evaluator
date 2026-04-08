from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .models import FusionCalibrator


def compute_usage_weights(count_row: pd.Series, pitch_types: Sequence[str]) -> dict[str, float]:
    total = float(sum(float(count_row.get(f"pitch_count_{pt}", 0.0)) for pt in pitch_types))
    if total <= 0:
        return {pt: 0.0 for pt in pitch_types}
    return {pt: float(count_row.get(f"pitch_count_{pt}", 0.0)) / total for pt in pitch_types}


def build_season_prediction_frame(
    season_keys: pd.DataFrame,
    pitch_types: Sequence[str],
    pitch_type_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = season_keys[["pitcher_id", "season", "true_xFIP", "total_pitch_count"]].copy()

    for pt in pitch_types:
        score_df = pitch_type_scores.get(pt)
        if score_df is None or score_df.empty:
            out[f"score_{pt}"] = 0.0
            out[f"pitch_count_{pt}"] = 0
            out[f"weight_{pt}"] = 0.0
            out[f"contribution_{pt}"] = 0.0
            continue

        score_df = score_df.loc[:, ["pitcher_id", "season", "score", "pitch_count"]].copy()
        score_df.columns = ["pitcher_id", "season", f"score_{pt}", f"pitch_count_{pt}"]
        out = out.merge(score_df, on=["pitcher_id", "season"], how="left")

    out = out.fillna({col: 0.0 for col in out.columns if col.startswith(("score_", "weight_", "contribution_"))})
    out = out.fillna({col: 0 for col in out.columns if col.startswith("pitch_count_")})

    weight_cols = []
    contribution_cols = []
    for pt in pitch_types:
        count_col = f"pitch_count_{pt}"
        score_col = f"score_{pt}"
        weight_col = f"weight_{pt}"
        contrib_col = f"contribution_{pt}"
        out[weight_col] = out[count_col] / out["total_pitch_count"].replace(0, np.nan)
        out[weight_col] = out[weight_col].fillna(0.0)
        out[contrib_col] = out[weight_col] * out[score_col]
        weight_cols.append(weight_col)
        contribution_cols.append(contrib_col)

    out["predicted_xFIP"] = out[contribution_cols].sum(axis=1)
    return pd.DataFrame(out)


def fit_calibrator(train_frame: pd.DataFrame, pitch_types: Sequence[str], target_col: str = "true_xFIP", epochs: int = 150, lr: float = 1e-2, l2: float = 1e-4, device: str = "cpu") -> FusionCalibrator:
    feature_cols = [f"contribution_{pt}" for pt in pitch_types]
    x = torch.tensor(train_frame[feature_cols].to_numpy(dtype=np.float32), device=device)
    y = torch.tensor(train_frame[target_col].to_numpy(dtype=np.float32), device=device)

    model = FusionCalibrator(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    return model


def apply_calibrator(frame: pd.DataFrame, pitch_types: Sequence[str], calibrator: FusionCalibrator, device: str = "cpu") -> pd.DataFrame:
    out = frame.copy()
    feature_cols = [f"contribution_{pt}" for pt in pitch_types]
    x = torch.tensor(out[feature_cols].to_numpy(dtype=np.float32), device=device)
    calibrator = calibrator.to(device)
    calibrator.eval()
    with torch.no_grad():
        out["predicted_xFIP"] = calibrator(x).cpu().numpy()
    return out
