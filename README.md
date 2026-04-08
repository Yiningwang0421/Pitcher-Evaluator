# Method B xFIP Pipeline

This repository now uses a single modeling path: Method B hierarchical xFIP modeling.
The command entrypoint is unified under `method_b_xfip/cli.py` so root-level scripts and package code can be run through one interface.

## Unified Layout

- Core modeling code: `method_b_xfip/`
- Consolidated script entrypoints: `method_b_xfip/scripts/`
- Root-level Python script wrappers were removed.

## Core Idea

Method B predicts pitcher-season xFIP in four stages:

1. Build pitch-level physical features.
2. Train one set-to-scalar model per pitch group.
3. Aggregate each pitcher-season-pitch-group into a score.
4. Fuse scores by pitch usage into final pitcher-season xFIP.

Default pitch grouping mode is `family`:
- `fastball`
- `breaking`
- `offspeed`
- `other`

Use `raw` mode only when you want original pitch codes trained separately.

## Recommended Workflow

### 1) Build processed data

```bash
python -m method_b_xfip.cli build -- \
  --num-samples 800 \
  --years 2022 2023 2024 \
  --min-pitches 256
```

This prepares:
- `data/processed/players_processed_bk/data_merged_processed.csv`
- `data/processed/players_processed_bk/data_merged_processed_256plus.csv`

### 2) Train Method B

```bash
python -m method_b_xfip.cli train -- \
  --input-file data/processed/players_processed_bk/data_merged_processed.csv \
  --artifact-dir artifacts/method_b_xfip \
  --report-dir reports/method_b_xfip \
  --pitch-group-mode family \
  --calibrate
```

## Outputs

### Artifacts
- `artifacts/method_b_xfip/config.json`
- `artifacts/method_b_xfip/pitch_types/<pitch_group>/model.pt`
- `artifacts/method_b_xfip/pitch_types/<pitch_group>/preprocessor.joblib`
- `artifacts/method_b_xfip/fusion_calibrator.joblib` (when `--calibrate` is enabled)

### Reports
- `reports/method_b_xfip/method_b_predictions.csv`
- `reports/method_b_xfip/method_b_metrics.json`
- `reports/method_b_xfip/method_b_pitch_type_summary.csv`

## Prediction Columns

- `pitcher_id`
- `season`
- `true_xFIP`
- `predicted_xFIP`
- `score_<pitch_group>`
- `weight_<pitch_group>`
- `contribution_<pitch_group>`
- `top_contributors`

## Pipeline Options Kept

The unified CLI forwards to `pipeline.py` and `train_method_b_xfip.py`:

- `--skip-fetch`, `--skip-merge`, `--skip-enrich`, `--skip-build`, `--skip-min-filter`
- `--apply-range-filter`, `--xfip-min`, `--xfip-max`
- `--train-method-b`

## One-command full run

```bash
python -m method_b_xfip.cli full
```

## Notes

- Legacy modeling scripts were removed to avoid path confusion and mixed training logic.
- Use package entrypoints only (`python -m method_b_xfip ...`).
