from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / denom


class PitchScoreMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.15):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims, 1]
        for left, right in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(left, right))
            if right != 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PitchTypeSetModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.15, aggregation: str = "mean"):
        super().__init__()
        self.pitch_scorer = PitchScoreMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.aggregation = aggregation

    def aggregate(self, pitch_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return masked_mean(pitch_scores, mask)
        if self.aggregation == "median":
            masked_scores = []
            for row_scores, row_mask in zip(pitch_scores, mask):
                valid = row_scores[row_mask.bool()]
                if valid.numel() == 0:
                    masked_scores.append(torch.tensor(0.0, device=pitch_scores.device))
                else:
                    masked_scores.append(valid.median())
            return torch.stack(masked_scores)
        if self.aggregation == "trimmed_mean":
            masked_scores = []
            for row_scores, row_mask in zip(pitch_scores, mask):
                valid = row_scores[row_mask.bool()]
                if valid.numel() == 0:
                    masked_scores.append(torch.tensor(0.0, device=pitch_scores.device))
                    continue
                sorted_vals, _ = torch.sort(valid)
                trim = max(int(0.1 * len(sorted_vals)), 0)
                trimmed = sorted_vals[trim: len(sorted_vals) - trim if len(sorted_vals) - trim > trim else len(sorted_vals)]
                masked_scores.append(trimmed.mean() if trimmed.numel() > 0 else sorted_vals.mean())
            return torch.stack(masked_scores)
        raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_len, feat_dim = x.shape
        flat_x = x.reshape(batch_size * max_len, feat_dim)
        flat_scores = self.pitch_scorer(flat_x).reshape(batch_size, max_len)
        group_scores = self.aggregate(flat_scores, mask)
        return group_scores, flat_scores


class FusionCalibrator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
