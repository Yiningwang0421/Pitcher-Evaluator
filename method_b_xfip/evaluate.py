from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.float64)
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    metrics = {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }
    if len(y_true_arr) > 1:
        metrics["pearson"] = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
    else:
        metrics["pearson"] = float("nan")
    return metrics


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"=== {title} ===")
    print(
        " | ".join(
            [
                f"MSE: {metrics['mse']:.4f}",
                f"RMSE: {metrics['rmse']:.4f}",
                f"MAE: {metrics['mae']:.4f}",
                f"R2: {metrics['r2']:.4f}",
                f"Pearson: {metrics['pearson']:.4f}",
            ]
        )
    )


def rank_contributions(frame: pd.DataFrame, contribution_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    contrib_values = out[contribution_cols].to_numpy(dtype=np.float64)
    ranks = np.argsort(-contrib_values, axis=1)
    top_names = []
    for row_idx in range(len(out)):
        ordered = [contribution_cols[i] for i in ranks[row_idx]]
        top_names.append("|".join(ordered[:3]))
    out["top_contributors"] = top_names
    return out
