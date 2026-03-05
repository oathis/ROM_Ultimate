# ROM Ultimate

ROM Ultimate is a session-based ROM workflow app for:

1. preprocessing CFD-like transient CSV snapshots
2. mode decomposition (`pod`, `dmd`)
3. offline surrogate training (`rbf`, `nn`, `projection`)
4. online prediction and test evaluation
5. interactive 3D visualization

## Key Concepts

- `mode` is for reduced-order basis generation (`pod`, `dmd`)
- `trainer` is for offline surrogate fitting on latent dynamics (`rbf`, `nn`, `projection`)
- each browser session is isolated under `artifacts/sessions/<session-id>/...`

## Current Data Flow

1. Split raw file list into train/test indices (`split_manifest.json`)
2. Preprocess raw CSV to processed snapshots
3. Train mode artifacts from train processed data
4. Train offline model from train processed data + mode artifacts
5. Evaluate on test processed data (`R2`, `1-R2`, CSV/plot outputs)

## Preprocess Behavior (Important)

In the web app Preprocess tab:

- `split-policy=apply`:
  - split manifest is used
  - train and test are generated together automatically
  - output directories become:
    - `<base>-train`
    - `<base>-test`
- `split-policy=none`:
  - no split is used
  - one processed dataset is generated

This removes manual train/test picking for the normal path.

## Artifact Layout (Session Workspace)

Within one session:

```text
artifacts/sessions/<session-id>/
  workspace/
    data/
      processed/
        catalog/
          dataset-train/
          dataset-test/
    models/
      catalog/
        <mode-profile-name>/
          pod/...
          dmd/...
          rbf/<mode>/...
          nn/<mode>/...
    predictions/
      eval/...
```

`mode` artifacts and `trainer` artifacts are separated by namespace under the selected model profile directory.

## Evaluate Tab Philosophy

Evaluate should answer one task:

`evaluate selected ROM profile on selected test dataset`

The tab is designed to:

- select one ROM profile (registered or direct)
- auto-resolve `models-dir`, `mode`, and `trainer`
- auto-suggest test dataset path
- run test evaluation
- show curves and 3D GT vs Prediction comparison

## Quick Start (CLI)

### 0) Optional: create split manifest

```bash
python scripts/run_dataset_split.py ^
  --raw-dir "examples/sample_raw/mini_case" ^
  --output-path "artifacts/splits/mini_case_split.json" ^
  --mode extrapolation ^
  --test-ratio 0.25 ^
  --min-train-samples 4
```

### 1) Preprocess with split (auto creates train/test together)

```bash
python scripts/run_preprocess.py ^
  --raw-dir "examples/sample_raw/mini_case" ^
  --processed-dir "artifacts/tmp/mini_case/dataset" ^
  --split-manifest "artifacts/splits/mini_case_split.json" ^
  --subset all
```

Outputs:

- `artifacts/tmp/mini_case/dataset-train`
- `artifacts/tmp/mini_case/dataset-test`

### 2) Mode training on train set

```bash
python scripts/run_mode_training.py ^
  --processed-dir "artifacts/tmp/mini_case/dataset-train" ^
  --models-dir "artifacts/tmp/mini_case/models" ^
  --mode pod ^
  --rank 8
```

### 3) Offline training on train set

```bash
python scripts/run_offline_training.py ^
  --processed-dir "artifacts/tmp/mini_case/dataset-train" ^
  --models-dir "artifacts/tmp/mini_case/models" ^
  --mode pod ^
  --trainer rbf ^
  --input-column time ^
  --kernel cubic ^
  --epsilon 1.0
```

### 4) Evaluate on test set

```bash
python scripts/run_test_evaluation.py ^
  --processed-test-dir "artifacts/tmp/mini_case/dataset-test" ^
  --models-dir "artifacts/tmp/mini_case/models" ^
  --output-dir "artifacts/tmp/mini_case/eval" ^
  --mode pod ^
  --trainer rbf ^
  --input-column time
```

## Web App

Run:

```bash
python -m streamlit run scripts/run_web_app.py
```

Recommended order in UI:

1. Split tab: build split manifest
2. Preprocess tab: `split-policy=apply`
3. Mode tab: train mode profile from `dataset-train`
4. Offline tab: train trainer using the same profile + train dataset
5. Evaluate tab: select ROM profile and test dataset, run evaluation
6. Viewer tab: inspect 3D prediction playback

## Example Data Included

Small runnable raw sample files are included here:

- `examples/sample_raw/mini_case/xresult-*.csv`

They are intentionally tiny to validate end-to-end flow quickly.

## Notes

- `1 - R2` is dimensionless (not percent).
- Very large `1 - R2` means poor generalization (often severe extrapolation or wrong path pairing).
- Large runtime artifacts are ignored by git; code and small examples are versioned.
