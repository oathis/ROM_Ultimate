# Run Commands

All user-facing execution commands are centralized in this folder.

By default, these scripts relaunch with conda base Python if detected
(typically `C:\Anaconda3\python.exe`).

- `python scripts/run_preprocess.py`
- `python scripts/run_dataset_split.py --mode extrapolation --test-ratio 0.2`
- `python scripts/run_preprocess.py --split-manifest artifacts/splits/split_manifest.json --subset all --processed-dir data/processed/dataset`
- `python scripts/run_preprocess.py --split-manifest artifacts/splits/split_manifest.json --subset train`
- `python scripts/run_preprocess.py --split-manifest artifacts/splits/split_manifest.json --subset test --processed-dir data/processed/test`
- `python scripts/run_mode_training.py`
- `python scripts/run_offline_training.py`
- `python scripts/run_online_prediction.py <time>`
- `python scripts/run_test_evaluation.py --processed-test-dir data/processed/test`
- `python scripts/run_pipeline.py --predict-time <time>`
- `python scripts/generate_animation.py --start <t0> --end <t1>`
- `python -m streamlit run scripts/run_web_app.py`
- `python scripts/native_viewer.py --mode pod --trainer rbf --variable velocity --start-time 0.005 --end-time 0.05`
- `python scripts/run_mode_training.py --mode dmd --dt 0 --time-column time`

Legacy aliases are kept:

- `python scripts/train_offline.py`
- `python scripts/predict_online.py <time>`

Optional overrides:

- `ROM_USE_BASE=0` to skip base relaunch
- `ROM_BASE_PYTHON=<path>` to force a specific base Python path

## Web Session UI

`run_web_app.py` provides a session-isolated control panel:

- Create/select experiment sessions under `artifacts/sessions/<session-id>/`
- Run preprocess/mode/offline/online/full pipeline from browser
- Choose registered mode/trainer/runner components from dropdowns
- Save per-run summaries/logs under each session
- Use `Viewer` tab for in-app 3D ROM playback with time slider and camera orbit
- Viewer supports playback loop cycles and frame-speed control (`frame-ms`)
- Viewer includes a desktop native renderer launcher (PyVista/VTK) for large datasets
- `reconstruction` runner supports both `pod` and `dmd` modes
- Offline trainer artifacts are saved per mode namespace:
  `models/<trainer>/<mode>/<variable>/<trainer>_model.pkl`

By default, `.streamlit/config.toml` sets:

- `server.maxMessageSize = 4096`

Optional dependency for native desktop viewer:

- `python -m pip install pyvista`

## Recommended split-first flow

1. Create split manifest from raw data.
2. Preprocess with `--subset all` to generate `-train` and `-test` directories in one run.
3. Train mode + offline model from train processed data.
4. Run `run_test_evaluation.py` on the test processed directory to export `r2_by_time.csv`, `r2_summary.csv`, `r2_plot.png`.

### Split mode details

- `extrapolation`
  - Raw files are sorted by time from filename `xresult-<time>.csv`.
  - The last contiguous block is assigned to test.
  - Example: total 100, test-ratio 0.2 -> train first 80, test last 20.
- `interpolation`
  - Raw files are sorted by time.
  - Test indices are sampled from interior time points (endpoints are kept in train when possible).
  - Sampling is approximately uniform over the interior sequence.
  - Example: total 100, test-ratio 0.2 -> about 20 interior points become test, spread over the range.
- `min-train-samples`
  - Guarantees a minimum train count.
  - If ratio would make train too small, test count is automatically reduced.

### Web app catalog workflow

- Preprocess tab:
  - Use `output-policy=registered` and save each processed dataset with a clear name.
- Mode tab:
  - Select `dataset-source=registered`.
  - Set `mode-profile-output=registered` and store mode artifacts under a named profile.
- Offline tab:
  - Select `mode-profile-source=registered` to pick a previously built mode profile.
  - `processed-link=from-profile` uses the processed directory recorded in mode metadata.
