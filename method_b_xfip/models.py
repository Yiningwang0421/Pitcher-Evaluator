from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / denom


def masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = []
    for row_values, row_mask in zip(values, mask):
        valid = row_values[row_mask]
        if valid.numel() == 0:
            out.append(torch.tensor(0.0, device=values.device))
            continue
        out.append(valid.median())
    return torch.stack(out)


def masked_trimmed_mean(values: torch.Tensor, mask: torch.Tensor, trim_ratio: float = 0.1) -> torch.Tensor:
    out = []
    for row_values, row_mask in zip(values, mask):
        valid = row_values[row_mask]
        if valid.numel() == 0:
            out.append(torch.tensor(0.0, device=values.device))
            continue
        sorted_values, _ = torch.sort(valid)
        trim = int(trim_ratio * sorted_values.numel())
        if trim * 2 >= sorted_values.numel():
            trimmed = sorted_values
        else:
            trimmed = sorted_values[trim : sorted_values.numel() - trim]
        out.append(trimmed.mean() if trimmed.numel() else sorted_values.mean())
    return torch.stack(out)


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


class PitchEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanMLPFamilyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.15,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.pitch_scorer = PitchScoreMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.aggregation = aggregation

    def _aggregate(self, pitch_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return masked_mean(pitch_scores, mask)
        if self.aggregation == "median":
            return masked_median(pitch_scores, mask)
        if self.aggregation == "trimmed_mean":
            return masked_trimmed_mean(pitch_scores, mask)
        raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, family_size, feat_dim = x.shape
        flat_x = x.reshape(batch_size * family_size, feat_dim)
        pitch_scores = self.pitch_scorer(flat_x).reshape(batch_size, family_size)
        family_scores = self._aggregate(pitch_scores, mask.bool())
        return family_scores, pitch_scores


class SetAttentionFamilyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.15,
        pooling: str = "attention",
        score_hidden_dim: int = 32,
    ):
        super().__init__()
        self.pooling = pooling
        self.pitch_embed = PitchEmbeddingMLP(input_dim=input_dim, embed_dim=embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=embed_dim * 2,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool_attn = nn.Linear(embed_dim, 1)
        self.pitch_score_proj = nn.Linear(embed_dim, 1)
        self.family_score_head = nn.Sequential(
            nn.Linear(embed_dim, score_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden_dim, 1),
        )

    def _masked_mean_pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (encoded * mask_f).sum(dim=1) / denom

    def _attention_pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.pool_attn(encoded).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_bool = mask.bool()
        valid_counts = mask_bool.sum(dim=1)

        embedded = self.pitch_embed(x)
        encoded = self.encoder(embedded, src_key_padding_mask=~mask_bool)

        if self.pooling == "attention":
            pooled = self._attention_pool(encoded, mask_bool)
        elif self.pooling == "masked_mean":
            pooled = self._masked_mean_pool(encoded, mask_bool)
        else:
            raise ValueError(f"Unsupported family pooling: {self.pooling}")

        family_scores = self.family_score_head(pooled).squeeze(-1)
        pitch_scores = self.pitch_score_proj(encoded).squeeze(-1)

        empty = valid_counts == 0
        if empty.any():
            family_scores = family_scores.masked_fill(empty, 0.0)
            pitch_scores = pitch_scores.masked_fill(empty.unsqueeze(1), 0.0)

        return family_scores, pitch_scores


class FusionCalibrator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class PitchFamilySetModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.15,
        family_encoder_type: str = "mean_mlp",
        family_embed_dim: int = 64,
        family_encoder_num_heads: int = 4,
        family_encoder_num_layers: int = 1,
        family_encoder_dropout: float = 0.15,
        family_pooling: str = "attention",
        family_score_hidden_dim: int = 32,
        aggregation: str = "mean",
        fusion_mode: str = "usage_weighted_sum",
        use_family_reliability: bool = False,
        reliability_count_c: float = 20.0,
        normalize_reliability_weights: bool = True,
    ):
        super().__init__()
        self.family_encoder_type = family_encoder_type
        self.aggregation = aggregation
        self.fusion_mode = fusion_mode
        self.use_family_reliability = use_family_reliability
        self.reliability_count_c = float(reliability_count_c)
        self.normalize_reliability_weights = normalize_reliability_weights
        self.calibrator = FusionCalibrator(input_dim=3) if fusion_mode == "calibrated_family_scores" else None

        if self.family_encoder_type == "mean_mlp":
            self.fastball_encoder = MeanMLPFamilyEncoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                aggregation=aggregation,
            )
            self.breaking_encoder = MeanMLPFamilyEncoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                aggregation=aggregation,
            )
            self.offspeed_encoder = MeanMLPFamilyEncoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                aggregation=aggregation,
            )
        elif self.family_encoder_type == "set_attention":
            self.fastball_encoder = SetAttentionFamilyEncoder(
                input_dim=input_dim,
                embed_dim=family_embed_dim,
                num_heads=family_encoder_num_heads,
                num_layers=family_encoder_num_layers,
                dropout=family_encoder_dropout,
                pooling=family_pooling,
                score_hidden_dim=family_score_hidden_dim,
            )
            self.breaking_encoder = SetAttentionFamilyEncoder(
                input_dim=input_dim,
                embed_dim=family_embed_dim,
                num_heads=family_encoder_num_heads,
                num_layers=family_encoder_num_layers,
                dropout=family_encoder_dropout,
                pooling=family_pooling,
                score_hidden_dim=family_score_hidden_dim,
            )
            self.offspeed_encoder = SetAttentionFamilyEncoder(
                input_dim=input_dim,
                embed_dim=family_embed_dim,
                num_heads=family_encoder_num_heads,
                num_layers=family_encoder_num_layers,
                dropout=family_encoder_dropout,
                pooling=family_pooling,
                score_hidden_dim=family_score_hidden_dim,
            )
        else:
            raise ValueError(f"Unsupported family_encoder_type: {self.family_encoder_type}")

    def _family_reliability(self, family_counts: torch.Tensor) -> torch.Tensor:
        counts = family_counts.float().clamp_min(0.0)
        c = max(self.reliability_count_c, 1e-6)
        return torch.sqrt(counts / (counts + c))

    def fuse(
        self,
        family_scores: torch.Tensor,
        family_usage_weights: torch.Tensor,
        family_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fusion_mode == "usage_weighted_sum":
            weights = family_usage_weights
            if self.use_family_reliability and family_counts is not None:
                reliability = self._family_reliability(family_counts)
                weights = weights * reliability
                if self.normalize_reliability_weights:
                    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    weights = weights / denom
            return (family_scores * weights).sum(dim=1)
        if self.fusion_mode == "calibrated_family_scores":
            if self.calibrator is None:
                raise RuntimeError("Calibrator must be initialized for calibrated_family_scores")
            return self.calibrator(family_scores)
        raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

    def forward(
        self,
        fastball_x: torch.Tensor,
        fastball_mask: torch.Tensor,
        breaking_x: torch.Tensor,
        breaking_mask: torch.Tensor,
        offspeed_x: torch.Tensor,
        offspeed_mask: torch.Tensor,
        family_usage_weights: torch.Tensor,
        family_counts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fastball_score, fastball_pitch_scores = self.fastball_encoder(fastball_x, fastball_mask)
        breaking_score, breaking_pitch_scores = self.breaking_encoder(breaking_x, breaking_mask)
        offspeed_score, offspeed_pitch_scores = self.offspeed_encoder(offspeed_x, offspeed_mask)

        family_scores = torch.stack([fastball_score, breaking_score, offspeed_score], dim=1)
        predicted = self.fuse(
            family_scores=family_scores,
            family_usage_weights=family_usage_weights,
            family_counts=family_counts,
        )

        return {
            "predicted_xfip": predicted,
            "family_scores": family_scores,
            "fastball_pitch_scores": fastball_pitch_scores,
            "breaking_pitch_scores": breaking_pitch_scores,
            "offspeed_pitch_scores": offspeed_pitch_scores,
        }
