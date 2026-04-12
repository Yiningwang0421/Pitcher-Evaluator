# Method B xFIP (Pitcher-Season Family Set Model)

This project trains an interpretable pitcher-season xFIP model with three fixed pitch families:

- `fastball`: `FF`, `FT`, `SI`, `FC`
- `breaking`: `SL`, `CU`, `KC`, `SV`
- `offspeed`: `CH`, `FS`, `FO`, `SC`

Each pitcher-season sample is represented as three masked pitch sets, encoded per family, then fused by usage-weighted contributions.

## Repository Layout

- Core package: `method_b_xfip/`
- Training script: `method_b_xfip/scripts/train_method_b_xfip.py`
- Model/data outputs (local only): `artifacts/`, `reports/`, `logs/`, `model/`
- Data folders (local only): `data/`

## Recommended Training Command

```bash
python -m method_b_xfip.scripts.train_method_b_xfip \
  --artifact-dir artifacts/method_b_xfip_family_best \
  --report-dir reports/method_b_xfip_family_best \
  --epochs 60 \
  --patience 10 \
  --collapse-reg-weight 0 \
  --fastball-samples 64 \
  --breaking-samples 48 \
  --offspeed-samples 48 \
  --loss-type huber \
  --huber-delta 0.8 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 16 \
  --hidden-dims 128,64 \
  --family-encoder-type mean_mlp
```

## Main Outputs

- Metrics: `reports/<run_name>/method_b_metrics.json`
- Predictions: `reports/<run_name>/method_b_predictions.csv`
- Family summary: `reports/<run_name>/method_b_pitch_type_summary.csv`
- Model checkpoints and metadata: `artifacts/<run_name>/`

Prediction table includes `score_<family>`, `weight_<family>`, and `contribution_<family>` for interpretability.

## Keep Repo Small

This repo ignores local-heavy folders by default via `.gitignore`:

- `artifacts/`
- `reports/`
- `logs/`
- `model/`
- `data/`

If local disk usage grows during experiments, cleanup can be done with:

```bash
rm -rf artifacts/* reports/* logs/* model/* data/interim/* data/raw/*
```

Keep only the minimum data you still need in `data/processed/`.
