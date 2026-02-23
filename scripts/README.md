# Run Commands

All user-facing execution commands are centralized in this folder.

By default, these scripts relaunch with conda base Python if detected
(typically `C:\Anaconda3\python.exe`).

- `python scripts/run_preprocess.py`
- `python scripts/run_mode_training.py`
- `python scripts/run_offline_training.py`
- `python scripts/run_online_prediction.py <time>`
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
