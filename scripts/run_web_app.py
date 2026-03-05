from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import importlib.util
import json
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys
import traceback

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
import numpy as np
import pandas as pd
from rom.core.workflows import (
    run_dataset_split,
    run_full_pipeline,
    run_mode_training,
    run_offline_training,
    run_online_prediction,
    run_preprocess,
    run_test_evaluation,
)
from rom.data.split import build_raw_split_manifest
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.runner_registry import REGISTRY as RUNNER_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY
from rom.runners.online_prediction import OnlinePredictionRunner
from rom.web.session_store import (
    create_session,
    ensure_session,
    list_sessions,
    record_run,
    recent_runs,
)

try:
    import plotly.graph_objects as go
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError as exc:
    raise SystemExit(
        "streamlit is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


RBF_KERNEL_OPTIONS = [
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "gaussian",
    "inverse_quadratic",
    "inverse_multiquadric",
    "multiquadric",
]

NN_ACTIVATION_OPTIONS = ["relu", "tanh", "gelu", "sigmoid", "linear"]
VISUAL_VARIABLES = ["velocity", "T", "u", "v", "w"]
OFFLINE_EVAL_OPTIONS = ["none", "interpolation", "extrapolation", "both"]
SPLIT_MODE_OPTIONS = ["interpolation", "extrapolation"]
PROCESSED_CATALOG_DIRNAME = "catalog"
MODEL_PROFILE_DIRNAME = "catalog"

MODE_DESCRIPTIONS = {
    "pod": "POD reduced-basis builder. Control dimensionality with `rank`/`energy_threshold`.",
    "dmd": "Exact DMD builder with modal dynamics (eigenvalues/modes/amplitudes).",
}

TRAINER_DESCRIPTIONS = {
    "rbf": "RBF interpolation surrogate. Good baseline for small to medium datasets.",
    "nn": "Lightweight NN-like surrogate placeholder (currently linear baseline).",
    "projection": "Projection-style surrogate with ridge stabilization.",
}

FIELD_HELP = {
    "processed_dir": "Directory containing preprocessed snapshots (`Snapshot_*.npy`) and `doe.csv`.",
    "models_dir": "Root directory where mode and trainer artifacts are saved/loaded.",
    "predictions_dir": "Directory for online prediction outputs (csv/json).",
    "raw_dir": "Directory containing raw `xresult-*.csv` files.",
    "mode": "Mode builder algorithm (registry key).",
    "trainer": "Offline trainer algorithm (registry key).",
    "runner": "Online runtime strategy (registry key).",
    "input_column": "Column name in `doe.csv` used as training input.",
    "eval_mode": "Validation profile: interpolation, extrapolation, both, or none.",
    "val_ratio": "Fraction of samples reserved for validation split.",
    "min_train_samples": "Minimum sample count retained for training during validation split.",
    "split_manifest": "Path to split manifest json (train/test raw file list).",
    "subset": "Subset from split manifest to preprocess: all, train, or test.",
    "split_mode": (
        "Raw split policy. extrapolation=뒤쪽 연속 구간을 test로 홀드아웃, "
        "interpolation=중간 시점 후보를 균일 간격으로 test 샘플링"
    ),
    "test_ratio": "총 샘플 중 test로 보낼 비율(반올림 후, train 최소 개수 제약으로 보정).",
}

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _quote(value: str | Path) -> str:
    return f"\"{value}\""


def _resolve_existing_directory(path_text: str | Path) -> Path:
    raw = str(path_text or "").strip()
    candidate = Path(raw) if raw else Path(ROOT_DIR)

    if candidate.exists() and candidate.is_file():
        candidate = candidate.parent
    elif not candidate.exists():
        # If the input looks like a file path, start from its parent.
        if candidate.suffix:
            candidate = candidate.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent

    if not candidate.exists():
        candidate = Path(ROOT_DIR)
    return candidate


def _open_directory_picker(initial_path: str) -> str | None:
    if os.name != "nt":
        st.warning("`찾아보기`는 Windows 환경에서만 지원됩니다.")
        return None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # noqa: BLE001
        st.warning(f"폴더 선택기를 열 수 없습니다: {exc}")
        return None

    start_dir = _resolve_existing_directory(initial_path)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=str(start_dir),
            title="폴더 선택",
            mustexist=False,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"폴더 선택 중 오류가 발생했습니다: {exc}")
        return None
    finally:
        if root is not None:
            root.destroy()

    if not selected:
        return None
    return str(Path(selected))


def _directory_input(
    label: str,
    value: str,
    key: str,
    help_text: str | None = None,
    *,
    in_form: bool = False,
    container=None,
) -> str:
    pending_key = f"{key}__picked_dir"
    if pending_key in st.session_state:
        st.session_state[key] = str(st.session_state.pop(pending_key))

    initial_value = str(st.session_state.get(key, value))
    host = st if container is None else container
    input_col, browse_col = host.columns([0.84, 0.16])

    with input_col:
        current_value = st.text_input(
            label,
            value=initial_value,
            key=key,
            help=help_text,
        )

    with browse_col:
        st.markdown("<div style='height: 1.85rem'></div>", unsafe_allow_html=True)
        if in_form:
            browse_clicked = st.form_submit_button(
                f"찾아보기 ({label})",
                use_container_width=True,
            )
        else:
            browse_clicked = st.button(
                "찾아보기",
                key=f"{key}_browse",
                use_container_width=True,
            )

    if browse_clicked:
        selected = _open_directory_picker(current_value)
        if selected:
            st.session_state[pending_key] = str(selected)
            st.rerun()

    return current_value


def _shell_join(args: list[str]):
    if os.name == "nt":
        # Windows-safe best-effort quoting for readable command previews.
        out = []
        for arg in args:
            if any(ch in arg for ch in (" ", "\t", '"')):
                out.append(f"\"{arg.replace('\"', '\"\"')}\"")
            else:
                out.append(arg)
        return " ".join(out)
    return " ".join(shlex.quote(arg) for arg in args)


def _build_native_viewer_command(
    models_dir: Path,
    processed_dir: Path,
    mode_name: str,
    trainer_name: str,
    variable: str,
    start_time: float,
    end_time: float,
    frames: int,
    max_points: int,
    point_size: float,
    loop_cycles: int,
    frame_duration_ms: int,
    rotate_camera: bool,
    backend: str,
):
    cmd = [
        str(sys.executable),
        str((ROOT_DIR / "scripts" / "native_viewer.py").resolve()),
        "--models-dir",
        str(Path(models_dir).resolve()),
        "--processed-dir",
        str(Path(processed_dir).resolve()),
        "--mode",
        mode_name,
        "--trainer",
        trainer_name,
        "--variable",
        variable,
        "--start-time",
        str(float(start_time)),
        "--end-time",
        str(float(end_time)),
        "--frames",
        str(int(frames)),
        "--max-points",
        str(int(max_points)),
        "--point-size",
        str(float(point_size)),
        "--loop-cycles",
        str(int(loop_cycles)),
        "--frame-ms",
        str(int(frame_duration_ms)),
        "--backend",
        str(backend),
    ]
    if rotate_camera:
        cmd.append("--rotate-camera")
    return cmd


def _launch_native_viewer(command: list[str]):
    script_path = Path(command[1])
    if not script_path.exists():
        raise FileNotFoundError(f"Native viewer script not found: {script_path}")

    popen_kwargs = {"cwd": str(ROOT_DIR)}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs)
    return {
        "launched": True,
        "pid": int(proc.pid),
        "command": _shell_join(command),
    }


def _parse_hidden_dims(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def _merge_extra_kwargs(base: dict, extra_json_text: str) -> dict:
    text = (extra_json_text or "").strip()
    if not text:
        return base
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Extra kwargs JSON must be an object.")
    merged = dict(base)
    merged.update(parsed)
    return merged


def _run_stage(session, stage: str, request: dict, action):
    started_at = _now_iso()
    log_buffer = io.StringIO()
    status = "success"
    summary = {}
    error = None
    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            summary = action() or {}
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=log_buffer)
    finished_at = _now_iso()
    record = record_run(
        session=session,
        stage=stage,
        status=status,
        request=request,
        summary=summary,
        log_text=log_buffer.getvalue(),
        error=error,
        started_at=started_at,
        finished_at=finished_at,
    )
    return record


def _render_record(record: dict):
    if record.get("status") == "success":
        st.success(f"{record.get('stage')} completed.")
    else:
        st.error(f"{record.get('stage')} failed: {record.get('error')}")

    st.json(record.get("summary", {}))
    with st.expander("Execution log", expanded=False):
        log_path = Path(record["log_path"])
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        st.code(log_text or "(no logs captured)", language="text")
    st.caption(f"Run record: `{record.get('record_path')}`")


def _session_metrics(session):
    snapshots = sorted(session.processed_dir.glob("Snapshot_*.npy"))
    mode_dirs = [d for d in session.models_dir.iterdir() if d.is_dir()] if session.models_dir.exists() else []
    prediction_files = sorted(session.predictions_dir.glob("predicted_result-*"))
    return {
        "snapshot_count": len(snapshots),
        "mode_artifact_groups": len(mode_dirs),
        "prediction_files": len(prediction_files),
    }


def _safe_read_json(path: Path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _slugify_label(text: str, fallback: str):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(text or "").strip()).strip("-_").lower()
    return slug or fallback


def _processed_catalog_root(session):
    return Path(session.processed_dir) / PROCESSED_CATALOG_DIRNAME


def _model_profile_root(session):
    return Path(session.models_dir) / MODEL_PROFILE_DIRNAME


def _discover_processed_catalog(catalog_root_text: str):
    root = Path(catalog_root_text)
    if not root.exists():
        return []

    datasets = []
    for ds_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        doe_path = ds_dir / "doe.csv"
        snapshot_paths = sorted(ds_dir.glob("Snapshot_*.npy"))
        if not doe_path.exists() or not snapshot_paths:
            continue

        manifest = _safe_read_json(ds_dir / "dataset_manifest.json") or {}
        variables = sorted([p.stem.replace("Snapshot_", "") for p in snapshot_paths])
        try:
            updated_ts = ds_dir.stat().st_mtime
        except OSError:
            updated_ts = 0.0

        datasets.append(
            {
                "name": ds_dir.name,
                "path": str(ds_dir),
                "n_snapshots": len(snapshot_paths),
                "variables": variables,
                "updated_ts": float(updated_ts),
                "manifest": manifest,
                "subset": manifest.get("subset"),
                "raw_dir": manifest.get("raw_dir"),
                "split_mode": manifest.get("split_mode"),
            }
        )

    datasets.sort(key=lambda item: item.get("updated_ts", 0.0), reverse=True)
    return datasets


def _discover_mode_profile_catalog(profile_root_text: str):
    root = Path(profile_root_text)
    if not root.exists():
        return []

    profiles = []
    mode_options = sorted(MODE_REGISTRY.keys())
    trainer_options = sorted(TRAINER_REGISTRY.keys())

    for profile_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        mode_details = {}
        for mode_name in mode_options:
            mode_root = profile_dir / mode_name
            if not mode_root.exists():
                continue

            variables = sorted([d.name for d in mode_root.iterdir() if d.is_dir()])
            if not variables:
                continue

            mode_manifest = _safe_read_json(mode_root / "mode_manifest.json") or {}
            trainer_availability = {}
            for trainer_name in trainer_options:
                trainer_root = profile_dir / trainer_name / mode_name
                if not trainer_root.exists():
                    continue
                trained_vars = []
                for var in variables:
                    model_file = trainer_root / var / f"{trainer_name}_model.pkl"
                    if model_file.exists():
                        trained_vars.append(var)
                if trained_vars:
                    trainer_availability[trainer_name] = sorted(trained_vars)

            mode_details[mode_name] = {
                "variables": variables,
                "manifest": mode_manifest,
                "processed_dir": mode_manifest.get("processed_dir"),
                "trainers": trainer_availability,
            }

        if not mode_details:
            continue

        try:
            updated_ts = profile_dir.stat().st_mtime
        except OSError:
            updated_ts = 0.0

        profiles.append(
            {
                "name": profile_dir.name,
                "path": str(profile_dir),
                "modes": sorted(mode_details.keys()),
                "mode_details": mode_details,
                "updated_ts": float(updated_ts),
            }
        )

    profiles.sort(key=lambda item: item.get("updated_ts", 0.0), reverse=True)
    return profiles


def _discover_rom_profiles(models_dir_text: str):
    models_dir = Path(models_dir_text)
    if not models_dir.exists():
        return []

    mode_options = sorted(MODE_REGISTRY.keys())
    trainer_options = sorted(TRAINER_REGISTRY.keys())
    profiles = []

    for mode_name in mode_options:
        mode_root = models_dir / mode_name
        if not mode_root.exists():
            continue

        mode_vars = sorted([d.name for d in mode_root.iterdir() if d.is_dir()])
        if not mode_vars:
            continue
        mode_var_set = set(mode_vars)
        mode_manifest = _safe_read_json(mode_root / "mode_manifest.json")

        for trainer_name in trainer_options:
            trainer_root = models_dir / trainer_name / mode_name
            if not trainer_root.exists():
                continue

            trainer_vars = []
            for var in mode_vars:
                model_path = trainer_root / var / f"{trainer_name}_model.pkl"
                if model_path.exists():
                    trainer_vars.append(var)
            common_vars = sorted(set(trainer_vars) & mode_var_set)
            if not common_vars:
                continue

            trainer_manifest = _safe_read_json(trainer_root / "trainer_manifest.json")
            try:
                updated_ts = trainer_root.stat().st_mtime
            except OSError:
                updated_ts = 0.0

            profile = {
                "mode_name": mode_name,
                "trainer_name": trainer_name,
                "variables": common_vars,
                "updated_ts": float(updated_ts),
                "mode_manifest": mode_manifest,
                "trainer_manifest": trainer_manifest,
            }
            profiles.append(profile)

    profiles.sort(key=lambda item: item.get("updated_ts", 0.0), reverse=True)
    return profiles


def _derive_subset_peer_dir(path_text: str | Path, target_subset: str):
    source = Path(path_text)
    subset = str(target_subset or "").strip().lower()
    if subset not in {"train", "test"}:
        return source

    name = source.name
    lowered = name.lower()
    for suffix in ("-train", "-test", "_train", "_test"):
        if lowered.endswith(suffix):
            root = name[: -len(suffix)]
            if root:
                return source.with_name(f"{root}-{subset}")
    return source.with_name(f"{name}-{subset}")


def _discover_registered_rom_combos(profile_root_text: str):
    combos = []
    for profile in _discover_mode_profile_catalog(profile_root_text):
        profile_path = Path(profile["path"])
        for mode_name, mode_info in (profile.get("mode_details") or {}).items():
            trainers = mode_info.get("trainers") or {}
            for trainer_name, trained_vars in trainers.items():
                trainer_root = profile_path / trainer_name / mode_name
                trainer_manifest = _safe_read_json(trainer_root / "trainer_manifest.json") or {}
                try:
                    updated_ts = trainer_root.stat().st_mtime
                except OSError:
                    updated_ts = float(profile.get("updated_ts", 0.0))

                combos.append(
                    {
                        "profile_name": profile.get("name"),
                        "models_dir": str(profile_path),
                        "mode_name": mode_name,
                        "trainer_name": trainer_name,
                        "variables": sorted(trained_vars),
                        "processed_dir": mode_info.get("processed_dir"),
                        "mode_manifest": mode_info.get("manifest") or {},
                        "trainer_manifest": trainer_manifest,
                        "updated_ts": float(updated_ts),
                    }
                )

    combos.sort(key=lambda item: item.get("updated_ts", 0.0), reverse=True)
    return combos


def _pick_latest_dataset_by_subset(datasets: list[dict], subset_name: str):
    subset = str(subset_name or "").strip().lower()
    for item in datasets:
        if str(item.get("subset") or "").strip().lower() == subset:
            return item
    return None


@st.cache_data(show_spinner=False)
def _load_snapshot_array(snapshot_path: str):
    return np.asarray(np.load(snapshot_path), dtype=np.float64)


def _snapshot_values_at_time(processed_dir: Path, variable: str, time_index: int):
    processed_dir = Path(processed_dir)
    idx = int(time_index)

    def _single(var_name: str):
        snap_path = processed_dir / f"Snapshot_{var_name}.npy"
        if not snap_path.exists():
            raise FileNotFoundError(f"Missing snapshot file: {snap_path}")
        arr = _load_snapshot_array(str(snap_path))
        if arr.ndim != 2:
            raise ValueError(f"Snapshot array must be 2D: {snap_path} shape={arr.shape}")
        if idx >= arr.shape[1]:
            if idx < arr.shape[0]:
                arr = arr.T
            else:
                raise IndexError(f"time_index={idx} out of range for {snap_path} shape={arr.shape}")
        return np.asarray(arr[:, idx], dtype=np.float64)

    if variable == "velocity":
        u = _single("u")
        v = _single("v")
        w = _single("w")
        return np.sqrt(u * u + v * v + w * w)

    raw_name = {"T": "T", "u": "u", "v": "v", "w": "w"}.get(variable, variable)
    return _single(raw_name)


def _scatter3d_field_figure(coords: np.ndarray, values: np.ndarray, title: str, cmin: float, cmax: float, point_size: float):
    marker = {
        "size": float(point_size),
        "color": np.asarray(values, dtype=np.float32),
        "colorscale": "Turbo",
        "cmin": float(cmin),
        "cmax": float(cmax),
        "opacity": 0.9,
        "colorbar": {"title": title},
    }
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=np.asarray(coords[:, 0], dtype=np.float32),
                y=np.asarray(coords[:, 1], dtype=np.float32),
                z=np.asarray(coords[:, 2], dtype=np.float32),
                mode="markers",
                marker=marker,
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 34, "b": 0},
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "data"},
    )
    return fig


def _render_mode_docs(mode_name: str):
    st.caption(MODE_DESCRIPTIONS.get(mode_name, "Mode parameter help"))
    if mode_name == "pod":
        st.info(
            "`rank`: number of retained modes\n"
            "`energy_threshold`: cumulative energy target (0~1) used when rank is auto-selected"
        )
    elif mode_name == "dmd":
        st.info(
            "`rank`: reduced basis dimension\n"
            "`dt`: time-step (set 0 or auto to infer from doe.csv)\n"
            "`time_column`: doe.csv column used for DMD time grid"
        )

def _render_trainer_docs(trainer_name: str):
    st.caption(TRAINER_DESCRIPTIONS.get(trainer_name, "Trainer parameter help"))
    if trainer_name == "rbf":
        st.info(
            "`kernel`: RBF kernel type\n"
            "`epsilon`: kernel scale parameter"
        )
    elif trainer_name == "nn":
        st.info(
            "`hidden_dims`: hidden layer widths (comma-separated)\n"
            "`activation`: activation label\n"
            "`epochs`: training epochs (metadata in current baseline)"
        )
    elif trainer_name == "projection":
        st.info(
            "`solver`: projection solver name\n"
            "`stabilization`: stabilization on/off\n"
            "`ridge`: L2 regularization strength"
        )


def _render_split_mode_docs(raw_dir_text: str, split_mode: str, test_ratio: float, min_train_samples: int):
    st.info(
        "`extrapolation`: 시간순 정렬 후 뒤쪽 연속 구간을 test로 분리합니다.\n"
        "`interpolation`: 양 끝을 train에 남기고(가능한 경우), 중간 시점 후보에서 test를 균일 간격으로 선택합니다."
    )
    with st.expander("Split 계산 규칙 자세히 보기", expanded=False):
        st.markdown(
            "1. `xresult-<time>.csv` 파일명에서 `<time>`을 파싱해 오름차순 정렬합니다.\n"
            "2. `test_count = round(n_total * test_ratio)`를 계산합니다.\n"
            "3. `test_count`는 최소 1, 최대 `n_total - min_train_samples`가 되도록 자동 보정합니다.\n"
            "4. `extrapolation`은 마지막 `test_count`개를 test로, 나머지를 train으로 둡니다.\n"
            "5. `interpolation`은 중간 후보 인덱스에서 균일 간격으로 `test_count`개를 고르고 나머지를 train으로 둡니다."
        )

    raw_dir = Path(raw_dir_text)
    if not raw_dir.exists():
        st.warning(f"raw-dir not found: {raw_dir}")
        return

    try:
        preview = build_raw_split_manifest(
            raw_dir=raw_dir,
            mode=split_mode,
            test_ratio=float(test_ratio),
            min_train_samples=int(min_train_samples),
            split_id="preview",
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Split preview is unavailable: {exc}")
        return

    p1, p2, p3 = st.columns(3)
    p1.metric("Total", preview.get("n_total", 0))
    p2.metric("Train", preview.get("n_train", 0))
    p3.metric("Test", preview.get("n_test", 0))

    train_times = preview.get("train_times") or []
    test_times = preview.get("test_times") or []
    train_min = train_times[0] if train_times else None
    train_max = train_times[-1] if train_times else None
    test_min = test_times[0] if test_times else None
    test_max = test_times[-1] if test_times else None

    st.caption(
        f"Train time range: {train_min} ~ {train_max} | "
        f"Test time range: {test_min} ~ {test_max}"
    )

    with st.expander("Preview files (head/tail)", expanded=False):
        st.write("Train files:", (preview.get("train_files") or [])[:3], "...", (preview.get("train_files") or [])[-3:])
        st.write("Test files:", (preview.get("test_files") or [])[:3], "...", (preview.get("test_files") or [])[-3:])


@st.cache_data(show_spinner=False)
def _load_points(points_path: str):
    arr = np.fromfile(points_path, dtype=np.float64)
    if arr.size % 3 != 0:
        raise ValueError(f"Invalid points.bin shape at {points_path}")
    return arr.reshape(-1, 3)


def _subsample_indices(total: int, max_points: int):
    max_points = max(1, int(max_points))
    if total <= max_points:
        return np.arange(total)
    step = max(1, total // max_points)
    return np.arange(0, total, step)


def _value_series(df: pd.DataFrame, variable: str):
    if variable == "velocity":
        ux = df["x-velocity"] if "x-velocity" in df else df.get("u")
        vy = df["y-velocity"] if "y-velocity" in df else df.get("v")
        wz = df["z-velocity"] if "z-velocity" in df else df.get("w")
        if ux is None or vy is None or wz is None:
            raise ValueError("Velocity components are missing in prediction output.")
        return np.sqrt(np.asarray(ux) ** 2 + np.asarray(vy) ** 2 + np.asarray(wz) ** 2)

    mapping = {
        "T": "temperature",
        "u": "x-velocity",
        "v": "y-velocity",
        "w": "z-velocity",
    }
    col_name = mapping.get(variable, variable)
    if col_name in df.columns:
        return np.asarray(df[col_name])
    if variable in df.columns:
        return np.asarray(df[variable])
    raise ValueError(f"Variable '{variable}' not found in prediction frame.")


def _camera_eye(frame_idx: int, total_frames: int, radius: float, z_height: float):
    angle = (2.0 * np.pi * frame_idx) / max(1, total_frames)
    return {
        "x": float(radius * np.cos(angle)),
        "y": float(radius * np.sin(angle)),
        "z": float(z_height),
    }


def _build_viewer_data(
    models_dir: Path,
    processed_dir: Path,
    mode_name: str,
    trainer_name: str,
    variable: str,
    start_time: float,
    end_time: float,
    frames: int,
    max_points: int,
):
    if mode_name not in {"pod", "dmd"}:
        raise NotImplementedError(
            f"Interactive field viewer currently supports `pod` and `dmd` reconstruction modes (got: {mode_name})."
        )

    points_path = Path(processed_dir) / "points.bin"
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points.bin at {points_path}")

    coords = _load_points(str(points_path))
    keep = _subsample_indices(len(coords), max_points=max_points)
    coords_sub = coords[keep].astype(np.float32, copy=False)

    runner = OnlinePredictionRunner(models_dir=Path(models_dir), mode_name=mode_name, trainer_name=trainer_name)
    time_values = np.linspace(float(start_time), float(end_time), int(frames))

    values = []
    vmin = np.inf
    vmax = -np.inf
    for tval in time_values:
        frame_df = runner.step(float(tval))
        frame_vals = _value_series(frame_df, variable)[keep].astype(np.float32, copy=False)
        values.append(frame_vals)
        vmin = min(vmin, float(np.min(frame_vals)))
        vmax = max(vmax, float(np.max(frame_vals)))

    return {
        "times": [float(t) for t in time_values],
        "coords": coords_sub,
        "values": values,
        "vmin": vmin,
        "vmax": vmax,
        "variable": variable,
        "points_total": int(len(coords)),
        "points_used": int(len(coords_sub)),
    }


def _estimate_viewer_payload_mb(bundle: dict):
    points = int(bundle["points_used"])
    frames = len(bundle["times"])
    # Approximate JSON payload with optimization:
    # - coords once (x,y,z)
    # - per-frame color array
    # Add overhead factor for JSON serialization metadata.
    bytes_raw = (points * 3 * 4) + (frames * points * 4)
    bytes_with_overhead = int(bytes_raw * 1.8)
    return bytes_with_overhead / (1024 * 1024)


def _is_pyvista_available():
    return importlib.util.find_spec("pyvista") is not None


def _build_viewer_figure(
    bundle: dict,
    point_size: float,
    rotate_camera: bool,
    rotation_radius: float,
    z_height: float,
    loop_cycles: int,
    frame_duration_ms: int,
):
    coords = bundle["coords"]
    x = np.asarray(coords[:, 0], dtype=np.float32)
    y = np.asarray(coords[:, 1], dtype=np.float32)
    z = np.asarray(coords[:, 2], dtype=np.float32)
    times = bundle["times"]
    values = bundle["values"]
    vmin = bundle["vmin"]
    vmax = bundle["vmax"]
    variable = bundle["variable"]

    label_map = {
        "T": "Temperature",
        "u": "X Velocity",
        "v": "Y Velocity",
        "w": "Z Velocity",
        "velocity": "Velocity Magnitude",
    }
    color_label = label_map.get(variable, variable)

    initial_marker = {
        "size": float(point_size),
        "color": np.asarray(values[0], dtype=np.float32),
        "colorscale": "Turbo",
        "cmin": vmin,
        "cmax": vmax,
        "opacity": 0.9,
        "colorbar": {"title": color_label},
    }
    base = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=initial_marker,
    )

    frames = []
    slider_steps = []
    total = len(times)
    # Keep Plotly animation target simple for Streamlit stability.
    play_target = None

    for idx, tval in enumerate(times):
        frame_marker = {
            "size": float(point_size),
            "color": np.asarray(values[idx], dtype=np.float32),
            "opacity": 0.9,
        }
        frame_layout = {}
        if rotate_camera:
            frame_layout = {
                "scene": {
                    "camera": {
                        "eye": _camera_eye(
                            frame_idx=idx,
                            total_frames=total,
                            radius=rotation_radius,
                            z_height=z_height,
                        )
                    }
                }
            }

        frames.append(
            go.Frame(
                name=str(idx),
                data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=frame_marker)],
                layout=frame_layout,
            )
        )
        slider_steps.append(
            {
                "args": [
                    [str(idx)],
                    {
                        "frame": {"duration": int(frame_duration_ms), "redraw": True},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
                "label": f"{tval:.6f}",
                "method": "animate",
            }
        )

    fig = go.Figure(data=[base], frames=frames)
    fig.update_layout(
        title=f"ROM Viewer | {color_label}",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.05,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            play_target,
                            {
                                "frame": {"duration": int(frame_duration_ms), "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "len": 0.86,
                "currentvalue": {"prefix": "time: "},
                "steps": slider_steps,
            }
        ],
    )
    if rotate_camera:
        fig.update_layout(scene_camera={"eye": _camera_eye(0, total, rotation_radius, z_height)})
    return fig


def _render_live_colorbar_viewer(fig: go.Figure, key: str, vmin: float, vmax: float):
    plot_id = f"rom_live_plot_{key}".replace(":", "_").replace(".", "_")
    min_id = f"{plot_id}_min"
    max_id = f"{plot_id}_max"
    scale_id = f"{plot_id}_scale"
    apply_id = f"{plot_id}_apply"
    reset_id = f"{plot_id}_reset"
    status_id = f"{plot_id}_status"
    colorscale_options = ["Turbo", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Jet", "RdBu"]
    fig_json = fig.to_json()
    config_json = json.dumps({"displaylogo": False, "scrollZoom": True, "responsive": True})
    colorscale_json = json.dumps(colorscale_options)
    html = f"""
<div style="border:1px solid rgba(15,42,67,0.14); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.9);">
  <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin-bottom:8px;">
    <div style="min-width:180px;">
      <label for="{min_id}" style="font-size:12px; color:#29455f;">color-min (cmin)</label>
      <input id="{min_id}" type="number" step="any" style="width:100%; padding:6px 8px;" />
    </div>
    <div style="min-width:180px;">
      <label for="{max_id}" style="font-size:12px; color:#29455f;">color-max (cmax)</label>
      <input id="{max_id}" type="number" step="any" style="width:100%; padding:6px 8px;" />
    </div>
    <div style="min-width:180px;">
      <label for="{scale_id}" style="font-size:12px; color:#29455f;">colorscale</label>
      <select id="{scale_id}" style="width:100%; padding:6px 8px;"></select>
    </div>
    <button id="{apply_id}" style="padding:6px 12px; border:1px solid #0f2a43; border-radius:8px; background:#e8f1f8; cursor:pointer;">Apply</button>
    <button id="{reset_id}" style="padding:6px 12px; border:1px solid #6b7280; border-radius:8px; background:#f8fafc; cursor:pointer;">Reset</button>
    <span id="{status_id}" style="font-size:12px; color:#29455f;"></span>
  </div>
  <div id="{plot_id}" style="height:760px;"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>
(() => {{
  const figure = {fig_json};
  const defaultMin = {float(vmin)};
  const defaultMax = {float(vmax)};
  const colorscales = {colorscale_json};
  const config = {config_json};

  const gd = document.getElementById("{plot_id}");
  const minInput = document.getElementById("{min_id}");
  const maxInput = document.getElementById("{max_id}");
  const scaleInput = document.getElementById("{scale_id}");
  const applyBtn = document.getElementById("{apply_id}");
  const resetBtn = document.getElementById("{reset_id}");
  const statusEl = document.getElementById("{status_id}");
  const plotlyRef = window.Plotly || (window.parent && window.parent.Plotly);

  function setStatus(msg, isError) {{
    statusEl.textContent = msg;
    statusEl.style.color = isError ? "#b91c1c" : "#29455f";
  }}

  function parseInputs() {{
    const cmin = Number.parseFloat(minInput.value);
    const cmax = Number.parseFloat(maxInput.value);
    if (!Number.isFinite(cmin) || !Number.isFinite(cmax)) {{
      return {{ ok: false, message: "cmin/cmax must be numeric." }};
    }}
    if (cmax <= cmin) {{
      return {{ ok: false, message: "cmax must be larger than cmin." }};
    }}
    return {{ ok: true, cmin, cmax, scale: scaleInput.value }};
  }}

  function updateFrameMarkerStyle(cmin, cmax, scale) {{
    const td = gd._transitionData;
    if (!td || !Array.isArray(td._frames)) return;
    td._frames.forEach((frame) => {{
      if (!frame || !Array.isArray(frame.data) || frame.data.length === 0) return;
      const trace0 = frame.data[0] || {{}};
      trace0.marker = trace0.marker || {{}};
      trace0.marker.cmin = cmin;
      trace0.marker.cmax = cmax;
      trace0.marker.colorscale = scale;
      frame.data[0] = trace0;
    }});
  }}

  function applyStyle() {{
    const parsed = parseInputs();
    if (!parsed.ok) {{
      setStatus(parsed.message, true);
      return;
    }}
    if (!plotlyRef) {{
      setStatus("Plotly runtime is not available in browser.", true);
      return;
    }}
    updateFrameMarkerStyle(parsed.cmin, parsed.cmax, parsed.scale);
    plotlyRef.restyle(
      gd,
      {{
        "marker.cmin": [parsed.cmin],
        "marker.cmax": [parsed.cmax],
        "marker.colorscale": [parsed.scale],
      }},
      [0]
    );
    setStatus(`Applied cmin=${{parsed.cmin.toFixed(6)}}, cmax=${{parsed.cmax.toFixed(6)}}`, false);
  }}

  let debounceTimer = null;
  function scheduleApply() {{
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(applyStyle, 120);
  }}

  colorscales.forEach((name) => {{
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === "Turbo") option.selected = true;
    scaleInput.appendChild(option);
  }});
  minInput.value = defaultMin.toFixed(6);
  maxInput.value = defaultMax.toFixed(6);

  if (!plotlyRef) {{
    setStatus("Plotly library load failed. Disable ad/script blockers or use standard chart mode.", true);
    return;
  }}

  function registerFrames() {{
    if (!Array.isArray(figure.frames) || figure.frames.length === 0) {{
      return Promise.resolve();
    }}
    return plotlyRef.addFrames(gd, figure.frames);
  }}

  plotlyRef.newPlot(gd, figure.data, figure.layout, config).then(() => registerFrames()).then(() => {{
    applyStyle();
    applyBtn.addEventListener("click", applyStyle);
    resetBtn.addEventListener("click", () => {{
      minInput.value = defaultMin.toFixed(6);
      maxInput.value = defaultMax.toFixed(6);
      scaleInput.value = "Turbo";
      applyStyle();
    }});
    minInput.addEventListener("input", scheduleApply);
    maxInput.addEventListener("input", scheduleApply);
    scaleInput.addEventListener("change", applyStyle);
  }});
}})();
</script>
    """
    components.html(html, height=860, scrolling=False)


def _inject_styles():
    st.markdown(
        """
<style>
:root {
  --ink: #0f2a43;
  --accent: #006d77;
  --accent-2: #e29578;
  --panel: rgba(255, 255, 255, 0.88);
  --line: rgba(15, 42, 67, 0.16);
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: "Aptos", "Candara", "Gill Sans", sans-serif;
  color: var(--ink);
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 12%, rgba(226,149,120,0.20), transparent 26%),
    radial-gradient(circle at 86% 8%, rgba(0,109,119,0.18), transparent 28%),
    linear-gradient(145deg, #f6f9fc 0%, #eef4f7 55%, #fdf7f2 100%);
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(0,109,119,0.14), rgba(255,255,255,0.85));
  border-right: 1px solid var(--line);
}

.panel-card {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 24px rgba(15, 42, 67, 0.08);
  animation: lift-in 280ms ease;
}

.panel-title {
  margin: 0 0 4px;
  color: var(--accent);
  font-size: 1.02rem;
  letter-spacing: 0.02em;
}

.panel-value {
  margin: 0;
  font-size: 1.6rem;
  font-weight: 700;
}

@keyframes lift-in {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.command-hint {
  border-left: 3px solid var(--accent-2);
  padding-left: 10px;
  margin: 6px 0 2px;
  font-size: 0.92rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_mode_controls(mode_name: str, key_prefix: str):
    _render_mode_docs(mode_name)
    auto_rank = st.checkbox(
        "Auto rank (rank=None)",
        value=False,
        key=f"{key_prefix}_auto_rank",
        help="Enable to pass rank=None and let the mode builder choose the rank.",
    )
    rank = None
    if not auto_rank:
        rank = int(
            st.number_input(
                "rank",
                min_value=1,
                value=32,
                step=1,
                key=f"{key_prefix}_rank",
                help="Number of retained modes.",
            )
        )

    mode_params = {"rank": rank}
    if mode_name == "pod":
        mode_params["energy_threshold"] = float(
            st.number_input(
                "energy_threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.0001,
                format="%.4f",
                key=f"{key_prefix}_energy_threshold",
                help="Cumulative energy target (0~1).",
            )
        )
    if mode_name == "dmd":
        auto_dt = st.checkbox(
            "Auto dt from doe.csv",
            value=True,
            key=f"{key_prefix}_auto_dt",
            help="Infer dt from the selected time column in doe.csv.",
        )
        if auto_dt:
            mode_params["dt"] = 0.0
        else:
            mode_params["dt"] = float(
                st.number_input(
                    "dt",
                    min_value=1e-9,
                    value=1.0,
                    step=0.1,
                    key=f"{key_prefix}_dt",
                    help="Uniform time-step used by DMD.",
                )
            )
        mode_params["time_column"] = st.text_input(
            "time_column",
            value="time",
            key=f"{key_prefix}_time_column",
            help="Column in doe.csv used to infer DMD time grid.",
        )
    extra_json = st.text_area(
        "Extra mode kwargs (JSON object)",
        value="{}",
        key=f"{key_prefix}_extra_mode_kwargs",
        help="Additional kwargs as JSON object. Example: {\"foo\": 1}",
    )
    return mode_params, extra_json


def _render_trainer_controls(trainer_name: str, key_prefix: str):
    _render_trainer_docs(trainer_name)
    trainer_params = {}
    if trainer_name == "rbf":
        trainer_params["kernel"] = st.selectbox(
            "kernel",
            RBF_KERNEL_OPTIONS,
            index=RBF_KERNEL_OPTIONS.index("cubic"),
            key=f"{key_prefix}_kernel",
            help="RBF kernel type.",
        )
        trainer_params["epsilon"] = float(
            st.number_input(
                "epsilon",
                min_value=1e-9,
                value=1.0,
                step=0.1,
                key=f"{key_prefix}_epsilon",
                help="RBF kernel scale parameter.",
            )
        )
    elif trainer_name == "nn":
        trainer_params["hidden_dims"] = st.text_input(
            "hidden_dims (comma separated)",
            value="128,128",
            key=f"{key_prefix}_hidden_dims",
            help="Example: 128,128",
        )
        trainer_params["activation"] = st.selectbox(
            "activation",
            NN_ACTIVATION_OPTIONS,
            index=0,
            key=f"{key_prefix}_activation",
            help="Activation label.",
        )
        trainer_params["epochs"] = int(
            st.number_input(
                "epochs",
                min_value=1,
                value=200,
                step=10,
                key=f"{key_prefix}_epochs",
                help="Training epochs.",
            )
        )
    elif trainer_name == "projection":
        trainer_params["solver"] = st.text_input(
            "solver",
            value="galerkin",
            key=f"{key_prefix}_solver",
            help="Projection solver name.",
        )
        trainer_params["stabilization"] = st.checkbox(
            "stabilization",
            value=False,
            key=f"{key_prefix}_stabilization",
            help="Enable stabilization.",
        )
        trainer_params["ridge"] = float(
            st.number_input(
                "ridge",
                min_value=0.0,
                value=1e-8,
                format="%.8f",
                key=f"{key_prefix}_ridge",
                help="L2 regularization strength.",
            )
        )

    extra_json = st.text_area(
        "Extra trainer kwargs (JSON object)",
        value="{}",
        key=f"{key_prefix}_extra_trainer_kwargs",
        help="Additional kwargs as JSON object. Example: {\"foo\": 1}",
    )
    return trainer_params, extra_json

def _ensure_active_session():
    sessions = list_sessions(ROOT_DIR)
    if not sessions:
        created = create_session(ROOT_DIR, session_name="session")
        st.session_state["active_session_id"] = created.session_id
        sessions = list_sessions(ROOT_DIR)

    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = sessions[0]["session_id"]

    session_ids = [item["session_id"] for item in sessions]
    if st.session_state["active_session_id"] not in session_ids:
        st.session_state["active_session_id"] = session_ids[0]

    return sessions


def main():
    st.set_page_config(
        page_title="ROM Session Deck",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    sessions = _ensure_active_session()
    session_ids = [item["session_id"] for item in sessions]
    sid = st.session_state["active_session_id"]

    st.sidebar.title("Session Control")
    selected_sid = st.sidebar.selectbox(
        "Active session",
        options=session_ids,
        index=session_ids.index(sid),
        key="active_session_select",
    )
    if selected_sid != sid:
        st.session_state["active_session_id"] = selected_sid
        st.rerun()

    new_session_name = st.sidebar.text_input("New session name", value="experiment")
    if st.sidebar.button("Create Session", use_container_width=True):
        created = create_session(ROOT_DIR, session_name=new_session_name)
        st.session_state["active_session_id"] = created.session_id
        st.rerun()

    session = ensure_session(ROOT_DIR, st.session_state["active_session_id"])

    st.sidebar.markdown("---")
    st.sidebar.caption("Session paths")
    st.sidebar.code(str(session.session_dir), language="text")

    raw_dir_input = _directory_input(
        "Raw data directory",
        value=str(session.raw_dir),
        key=f"{session.session_id}_raw_dir",
        container=st.sidebar,
    )
    if st.sidebar.button("Update Raw Path", use_container_width=True):
        session = ensure_session(ROOT_DIR, session.session_id, raw_dir=Path(raw_dir_input))
        st.sidebar.success("Raw path updated")
        st.rerun()

    st.title("ROM Session Deck")
    st.caption("Session-isolated ROM workflows with selectable mode/trainer/runner components")

    metrics = _session_metrics(session)
    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f"<div class='panel-card'><p class='panel-title'>Snapshots</p><p class='panel-value'>{metrics['snapshot_count']}</p></div>",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"<div class='panel-card'><p class='panel-title'>Model Groups</p><p class='panel-value'>{metrics['mode_artifact_groups']}</p></div>",
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"<div class='panel-card'><p class='panel-title'>Predictions</p><p class='panel-value'>{metrics['prediction_files']}</p></div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_split, tab_pre, tab_mode, tab_offline, tab_online, tab_eval, tab_pipeline, tab_viewer = st.tabs(
        ["Overview", "Split", "Preprocess", "Mode", "Offline", "Online", "Evaluate", "Pipeline", "Viewer"]
    )

    with tab_overview:
        st.subheader("Session Layout")
        st.json(session.as_dict())

        st.subheader("Recent Runs")
        runs = recent_runs(session, limit=15)
        if not runs:
            st.info("No runs yet in this session.")
        else:
            st.dataframe(
                [
                    {
                        "run_id": item.get("run_id"),
                        "stage": item.get("stage"),
                        "status": item.get("status"),
                        "started_at": item.get("started_at"),
                        "finished_at": item.get("finished_at"),
                        "record_path": item.get("record_path"),
                    }
                    for item in runs
                ],
                use_container_width=True,
                hide_index=True,
            )

            latest = runs[0]
            with st.expander("Latest run details", expanded=False):
                st.json(latest)
                log_path = Path(latest.get("log_path", ""))
                if log_path.exists():
                    st.code(log_path.read_text(encoding="utf-8"), language="text")

    with tab_split:
        st.subheader("Dataset Split")
        st.caption("Preprocess 이전에 raw CSV를 train/test로 고정 분할합니다.")
        key_base = f"{session.session_id}_split"

        default_manifest = ROOT_DIR / "artifacts/splits/split_manifest.json"
        with st.form(f"{key_base}_form"):
            raw_dir_text = _directory_input(
                "raw-dir",
                value=str(session.raw_dir),
                key=f"{key_base}_raw",
                help_text=FIELD_HELP["raw_dir"],
                in_form=True,
            )
            manifest_path_text = st.text_input(
                "output-manifest",
                value=str(default_manifest),
                key=f"{key_base}_manifest",
                help=FIELD_HELP["split_manifest"],
            )
            split_col1, split_col2, split_col3 = st.columns(3)
            with split_col1:
                split_mode = st.selectbox(
                    "split-mode",
                    options=SPLIT_MODE_OPTIONS,
                    index=SPLIT_MODE_OPTIONS.index("extrapolation"),
                    key=f"{key_base}_mode",
                    help=FIELD_HELP["split_mode"],
                )
            with split_col2:
                test_ratio = float(
                    st.number_input(
                        "test-ratio",
                        min_value=0.01,
                        max_value=0.9,
                        value=0.2,
                        step=0.01,
                        format="%.2f",
                        key=f"{key_base}_ratio",
                        help=FIELD_HELP["test_ratio"],
                    )
                )
            with split_col3:
                min_train_samples = int(
                    st.number_input(
                        "min-train-samples",
                        min_value=1,
                        value=8,
                        step=1,
                        key=f"{key_base}_min_train_samples",
                        help=FIELD_HELP["min_train_samples"],
                    )
                )
            split_id_text = st.text_input(
                "split-id (optional)",
                value="",
                key=f"{key_base}_split_id",
                help="비워두면 자동 split id를 사용합니다.",
            )
            submitted = st.form_submit_button("Create Split Manifest", use_container_width=True)

        _render_split_mode_docs(
            raw_dir_text=raw_dir_text,
            split_mode=split_mode,
            test_ratio=test_ratio,
            min_train_samples=min_train_samples,
        )

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        cmd_parts = [
            "python scripts/run_dataset_split.py",
            f"--raw-dir {_quote(raw_dir_text)}",
            f"--output-path {_quote(manifest_path_text)}",
            f"--mode {split_mode}",
            f"--test-ratio {test_ratio}",
            f"--min-train-samples {min_train_samples}",
        ]
        if split_id_text.strip():
            cmd_parts.append(f"--split-id {_quote(split_id_text.strip())}")
        st.code(" ".join(part for part in cmd_parts if part))

        if submitted:
            request = {
                "raw_dir": raw_dir_text,
                "output_path": manifest_path_text,
                "split_mode": split_mode,
                "test_ratio": test_ratio,
                "min_train_samples": min_train_samples,
                "split_id": split_id_text.strip() or None,
            }
            with st.spinner("Creating split manifest..."):
                record = _run_stage(
                    session=session,
                    stage="dataset_split",
                    request=request,
                    action=lambda: run_dataset_split(
                        raw_dir=Path(raw_dir_text),
                        output_path=Path(manifest_path_text),
                        split_mode=split_mode,
                        test_ratio=test_ratio,
                        min_train_samples=min_train_samples,
                        split_id=split_id_text.strip() or None,
                    ),
                )
            st.session_state[f"{key_base}_last_record"] = record

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

    with tab_pre:
        st.subheader("Preprocess")
        st.caption("전처리 결과를 이름으로 저장해 두고, 이후 Mode/Offline에서 선택해서 재사용할 수 있습니다.")
        st.caption("split 적용 시 train/test를 자동으로 동시에 생성합니다.")
        key_base = f"{session.session_id}_preprocess"
        default_manifest = ROOT_DIR / "artifacts/splits/split_manifest.json"
        catalog_root = _processed_catalog_root(session)
        with st.form(f"{key_base}_form"):
            raw_dir_text = _directory_input(
                "raw-dir",
                value=str(session.raw_dir),
                key=f"{key_base}_raw",
                help_text=FIELD_HELP["raw_dir"],
                in_form=True,
            )
            output_policy = st.radio(
                "output-policy",
                options=["registered", "direct"],
                index=0,
                key=f"{key_base}_output_policy",
                horizontal=True,
                help="registered=카탈로그에 이름으로 저장, direct=경로 직접 지정",
            )

            split_policy = st.radio(
                "split-policy",
                options=["apply", "none"],
                index=0 if default_manifest.exists() else 1,
                key=f"{key_base}_split_policy",
                horizontal=True,
                help="apply=split-manifest를 사용해 train/test 분할 전처리, none=분할 없이 전체 전처리",
            )
            if split_policy == "apply":
                split_manifest_text = st.text_input(
                    "split-manifest",
                    value=str(default_manifest) if default_manifest.exists() else "",
                    key=f"{key_base}_split_manifest",
                    help=FIELD_HELP["split_manifest"],
                )
                subset = "all"
                st.caption("Split is enabled: train/test datasets will be generated together.")
            else:
                split_manifest_text = ""
                subset = "all"
                st.caption("Split is disabled: raw 전체를 단일 processed dataset으로 생성합니다.")

            if output_policy == "registered":
                default_name = "dataset"
                dataset_name_text = st.text_input(
                    "dataset-name",
                    value=default_name,
                    key=f"{key_base}_dataset_name",
                    help="카탈로그에 저장할 이름 (자동 slug 변환)",
                )
                dataset_slug = _slugify_label(dataset_name_text, default_name)
                processed_dir_text = str(catalog_root / dataset_slug)
                st.caption(f"Resolved processed-dir: `{processed_dir_text}`")
                if split_policy == "apply" and split_manifest_text.strip():
                    st.caption(
                        "This will generate both: "
                        f"`{processed_dir_text}-train` and `{processed_dir_text}-test`"
                    )
            else:
                dataset_name_text = ""
                processed_dir_text = _directory_input(
                    "processed-dir",
                    value=str(session.processed_dir),
                    key=f"{key_base}_processed",
                    help_text=FIELD_HELP["processed_dir"],
                    in_form=True,
                )

            submitted = st.form_submit_button("Run Preprocess", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        cmd_parts = [
            "python scripts/run_preprocess.py",
            f"--raw-dir {_quote(raw_dir_text)}",
            f"--processed-dir {_quote(processed_dir_text)}",
        ]
        if split_policy == "apply" and split_manifest_text.strip():
            cmd_parts.extend(
                [
                    f"--split-manifest {_quote(split_manifest_text.strip())}",
                    f"--subset {subset}",
                ]
            )
        st.code(" ".join(cmd_parts))

        if submitted:
            raw_dir = Path(raw_dir_text)
            processed_dir = Path(processed_dir_text)
            if split_policy == "apply" and not split_manifest_text.strip():
                st.error("split-policy=apply 인 경우 `split-manifest` 경로를 입력해야 합니다.")
                split_manifest_path = None
                raw_dir = None
            else:
                split_manifest_path = (
                    Path(split_manifest_text.strip())
                    if split_policy == "apply" and split_manifest_text.strip()
                    else None
                )
            if raw_dir is not None:
                request = {
                    "raw_dir": str(raw_dir),
                    "processed_dir": str(processed_dir),
                    "output_policy": output_policy,
                    "dataset_name": dataset_name_text.strip() if output_policy == "registered" else None,
                    "split_policy": split_policy,
                    "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
                    "subset": subset,
                }
                with st.spinner("Running preprocess..."):
                    record = _run_stage(
                        session=session,
                        stage="preprocess",
                        request=request,
                        action=lambda: run_preprocess(
                            raw_dir=raw_dir,
                            processed_dir=processed_dir,
                            split_manifest_path=split_manifest_path,
                            subset=subset,
                        ),
                    )
                st.session_state[f"{key_base}_last_record"] = record

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

        datasets = _discover_processed_catalog(str(catalog_root))
        if datasets:
            with st.expander("Registered processed datasets", expanded=False):
                st.dataframe(
                    [
                        {
                            "dataset": item.get("name"),
                            "subset": item.get("subset"),
                            "snapshots": item.get("n_snapshots"),
                            "variables": ", ".join(item.get("variables", [])),
                            "path": item.get("path"),
                        }
                        for item in datasets
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_mode:
        st.subheader("Mode Training")
        key_base = f"{session.session_id}_mode"
        mode_options = sorted(MODE_REGISTRY.keys())

        dataset_source = st.radio(
            "dataset-source",
            options=["registered", "direct"],
            index=0,
            key=f"{key_base}_dataset_source",
            horizontal=True,
            help="registered=전처리 카탈로그에서 선택, direct=경로 직접 지정",
        )
        processed_dir_text = ""
        selected_dataset = None
        processed_catalog_root = _processed_catalog_root(session)
        if dataset_source == "registered":
            datasets = _discover_processed_catalog(str(processed_catalog_root))
            if datasets:
                labels = []
                for item in datasets:
                    subset_label = item.get("subset") or "n/a"
                    labels.append(
                        f"{item['name']} | subset={subset_label} | snaps={item['n_snapshots']} | vars={len(item.get('variables', []))}"
                    )
                selected_label = st.selectbox(
                    "processed-dataset",
                    options=labels,
                    key=f"{key_base}_dataset_label",
                )
                selected_idx = labels.index(selected_label)
                selected_dataset = datasets[selected_idx]
                processed_dir_text = selected_dataset["path"]
                st.caption(f"Resolved processed-dir: `{processed_dir_text}`")
            else:
                st.info("No registered processed dataset found. Switch to `direct` or run preprocess with `registered` output.")
                processed_dir_text = _directory_input(
                    "processed-dir",
                    value=str(session.processed_dir),
                    key=f"{key_base}_processed_fallback",
                    help_text=FIELD_HELP["processed_dir"],
                )
        else:
            processed_dir_text = _directory_input(
                "processed-dir",
                value=str(session.processed_dir),
                key=f"{key_base}_processed",
                help_text=FIELD_HELP["processed_dir"],
            )

        mode_name = st.selectbox(
            "mode",
            options=mode_options,
            key=f"{key_base}_mode_name",
            help=FIELD_HELP["mode"],
        )
        mode_params, extra_mode_json = _render_mode_controls(mode_name, key_base)

        profile_policy = st.radio(
            "mode-profile-output",
            options=["registered", "direct"],
            index=0,
            key=f"{key_base}_profile_policy",
            horizontal=True,
            help="registered=모드 프로필 카탈로그에 이름으로 저장",
        )
        model_profile_root = _model_profile_root(session)
        if profile_policy == "registered":
            default_profile_name = f"{mode_name}-profile"
            mode_profile_name = st.text_input(
                "mode-profile-name",
                value=default_profile_name,
                key=f"{key_base}_profile_name",
                help="모드 아티팩트를 저장할 프로필 이름",
            )
            mode_profile_slug = _slugify_label(mode_profile_name, "mode-profile")
            models_dir_text = str(model_profile_root / mode_profile_slug)
            st.caption(f"Resolved models-dir: `{models_dir_text}`")
        else:
            mode_profile_name = ""
            mode_profile_slug = ""
            models_dir_text = _directory_input(
                "models-dir",
                value=str(session.models_dir),
                key=f"{key_base}_models",
                help_text=FIELD_HELP["models_dir"],
            )

        submitted = st.button("Run Mode Training", key=f"{key_base}_run", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        rank_cmd = "--auto-rank" if mode_params.get("rank") is None else f"--rank {mode_params.get('rank')}"
        cmd_parts = [
            "python scripts/run_mode_training.py",
            f"--processed-dir {_quote(processed_dir_text)}",
            f"--models-dir {_quote(models_dir_text)}",
            f"--mode {mode_name}",
            rank_cmd,
        ]
        if "energy_threshold" in mode_params:
            cmd_parts.append(f"--energy-threshold {mode_params['energy_threshold']}")
        if "dt" in mode_params:
            cmd_parts.append(f"--dt {mode_params['dt']}")
        if "time_column" in mode_params:
            cmd_parts.append(f"--time-column {mode_params['time_column']}")
        st.code(" ".join(cmd_parts))

        if submitted:
            try:
                mode_kwargs = _merge_extra_kwargs(mode_params, extra_mode_json)
                request = {
                    "processed_dir": processed_dir_text,
                    "dataset_source": dataset_source,
                    "processed_dataset": None if selected_dataset is None else selected_dataset.get("name"),
                    "models_dir": models_dir_text,
                    "profile_policy": profile_policy,
                    "mode_profile_name": mode_profile_name if profile_policy == "registered" else None,
                    "mode_profile_slug": mode_profile_slug if profile_policy == "registered" else None,
                    "mode_name": mode_name,
                    "mode_params": mode_kwargs,
                }
                with st.spinner("Running mode training..."):
                    record = _run_stage(
                        session=session,
                        stage="mode_training",
                        request=request,
                        action=lambda: run_mode_training(
                            processed_dir=Path(processed_dir_text),
                            models_dir=Path(models_dir_text),
                            mode_name=mode_name,
                            mode_params=mode_kwargs,
                        ),
                    )
                st.session_state[f"{key_base}_last_record"] = record
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid mode parameters: {exc}")

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

        profiles = _discover_mode_profile_catalog(str(model_profile_root))
        if profiles:
            with st.expander("Registered mode profiles", expanded=False):
                st.dataframe(
                    [
                        {
                            "profile": item.get("name"),
                            "modes": ", ".join(item.get("modes", [])),
                            "path": item.get("path"),
                        }
                        for item in profiles
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_offline:
        st.subheader("Offline Training")
        key_base = f"{session.session_id}_offline"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())

        profile_source = st.radio(
            "mode-profile-source",
            options=["registered", "direct"],
            index=0,
            key=f"{key_base}_profile_source",
            horizontal=True,
            help="registered=Mode 탭에서 만든 프로필 선택, direct=경로 직접 지정",
        )
        model_profile_root = _model_profile_root(session)
        selected_profile = None
        if profile_source == "registered":
            profiles = _discover_mode_profile_catalog(str(model_profile_root))
            if profiles:
                profile_labels = []
                for item in profiles:
                    profile_labels.append(
                        f"{item['name']} | modes={','.join(item.get('modes', []))}"
                    )
                selected_label = st.selectbox(
                    "mode-profile",
                    options=profile_labels,
                    key=f"{key_base}_profile_label",
                )
                selected_idx = profile_labels.index(selected_label)
                selected_profile = profiles[selected_idx]
                models_dir_text = selected_profile["path"]
                available_modes = selected_profile.get("modes", []) or mode_options
                mode_name = st.selectbox(
                    "mode",
                    options=available_modes,
                    key=f"{key_base}_mode_name_registered",
                    help=FIELD_HELP["mode"],
                )
                st.caption(f"Resolved models-dir: `{models_dir_text}`")
            else:
                st.info("No registered mode profile found. Switch to `direct` or run Mode Training first.")
                models_dir_text = _directory_input(
                    "models-dir",
                    value=str(session.models_dir),
                    key=f"{key_base}_models_fallback",
                    help_text=FIELD_HELP["models_dir"],
                )
                mode_name = st.selectbox(
                    "mode",
                    options=mode_options,
                    key=f"{key_base}_mode_name_fallback",
                    help=FIELD_HELP["mode"],
                )
        else:
            models_dir_text = _directory_input(
                "models-dir",
                value=str(session.models_dir),
                key=f"{key_base}_models",
                help_text=FIELD_HELP["models_dir"],
            )
            mode_name = st.selectbox(
                "mode",
                options=mode_options,
                key=f"{key_base}_mode_name",
                help=FIELD_HELP["mode"],
            )

        inferred_processed_dir = str(session.processed_dir)
        if selected_profile is not None:
            mode_info = (selected_profile.get("mode_details") or {}).get(mode_name, {})
            from_manifest = mode_info.get("processed_dir")
            if from_manifest:
                inferred_processed_dir = str(from_manifest)

        processed_link = st.radio(
            "processed-link",
            options=["from-profile", "direct"],
            index=0,
            key=f"{key_base}_processed_link",
            horizontal=True,
            help="from-profile=mode_manifest의 processed_dir 사용",
        )
        if processed_link == "from-profile":
            processed_dir_text = _directory_input(
                "processed-dir",
                value=inferred_processed_dir,
                key=f"{key_base}_processed_profile",
                help_text=FIELD_HELP["processed_dir"],
            )
        else:
            processed_dir_text = _directory_input(
                "processed-dir",
                value=str(session.processed_dir),
                key=f"{key_base}_processed_direct",
                help_text=FIELD_HELP["processed_dir"],
            )

        trainer_name = st.selectbox(
            "trainer",
            options=trainer_options,
            key=f"{key_base}_trainer_name",
            help=FIELD_HELP["trainer"],
        )

        default_columns = ["time"]
        current_doe = Path(processed_dir_text) / "doe.csv"
        if current_doe.exists():
            try:
                default_columns = list(pd.read_csv(current_doe, nrows=1).columns)
            except Exception:  # noqa: BLE001
                default_columns = ["time"]
        default_input_column = "time" if "time" in default_columns else default_columns[0]
        input_column = st.text_input(
            "input-column",
            value=default_input_column,
            key=f"{key_base}_input_column",
            help=FIELD_HELP["input_column"],
        )
        eval_mode = "none"
        val_ratio = 0.2
        min_train_samples = 8
        st.caption(f"Detected columns in `doe.csv`: {', '.join(default_columns)}")
        st.caption("Offline 탭은 학습 전용입니다. 일반화 평가는 `Evaluate` 탭에서 test subset으로 수행하세요.")
        trainer_params, extra_trainer_json = _render_trainer_controls(trainer_name, key_base)
        submitted = st.button("Run Offline Training", key=f"{key_base}_run", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        cmd_parts = [
            "python scripts/run_offline_training.py",
            f"--processed-dir {_quote(processed_dir_text)}",
            f"--models-dir {_quote(models_dir_text)}",
            f"--mode {mode_name}",
            f"--trainer {trainer_name}",
            f"--input-column {input_column}",
        ]
        if trainer_name == "rbf":
            cmd_parts.extend([f"--kernel {trainer_params.get('kernel')}", f"--epsilon {trainer_params.get('epsilon')}"])
        if trainer_name == "nn":
            cmd_parts.extend(
                [
                    f"--hidden-dims {_quote(trainer_params.get('hidden_dims', ''))}",
                    f"--activation {trainer_params.get('activation')}",
                    f"--epochs {trainer_params.get('epochs')}",
                ]
            )
        if trainer_name == "projection":
            cmd_parts.extend(
                [
                    f"--solver {trainer_params.get('solver')}",
                    f"--ridge {trainer_params.get('ridge')}",
                    "--stabilization" if trainer_params.get("stabilization") else "",
                ]
            )
        st.code(" ".join(part for part in cmd_parts if part))

        if submitted:
            try:
                if trainer_name == "nn":
                    trainer_params["hidden_dims"] = _parse_hidden_dims(trainer_params.get("hidden_dims", ""))
                trainer_kwargs = _merge_extra_kwargs(trainer_params, extra_trainer_json)
                request = {
                    "processed_dir": processed_dir_text,
                    "mode_profile_source": profile_source,
                    "mode_profile_name": None if selected_profile is None else selected_profile.get("name"),
                    "processed_link": processed_link,
                    "models_dir": models_dir_text,
                    "mode_name": mode_name,
                    "trainer_name": trainer_name,
                    "trainer_params": trainer_kwargs,
                    "input_column": input_column,
                    "eval_mode": eval_mode,
                    "val_ratio": val_ratio,
                    "min_train_samples": min_train_samples,
                }
                with st.spinner("Running offline training..."):
                    record = _run_stage(
                        session=session,
                        stage="offline_training",
                        request=request,
                        action=lambda: run_offline_training(
                            processed_dir=Path(processed_dir_text),
                            models_dir=Path(models_dir_text),
                            mode_name=mode_name,
                            trainer_name=trainer_name,
                            trainer_params=trainer_kwargs,
                            input_column=input_column,
                            eval_mode=eval_mode,
                            val_ratio=val_ratio,
                            min_train_samples=min_train_samples,
                        ),
                    )
                st.session_state[f"{key_base}_last_record"] = record
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid trainer parameters: {exc}")

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

    with tab_online:
        st.subheader("Online Prediction")
        key_base = f"{session.session_id}_online"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())
        runner_options = sorted(RUNNER_REGISTRY.keys())
        time_value = float(
            st.number_input(
                "time",
                value=0.025,
                step=0.001,
                format="%.6f",
                key=f"{key_base}_time",
                help="Time value used for online prediction",
            )
        )
        models_dir_text = _directory_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help_text=FIELD_HELP["models_dir"],
        )
        processed_dir_text = _directory_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help_text=FIELD_HELP["processed_dir"],
        )
        output_dir_text = _directory_input(
            "output-dir",
            value=str(session.predictions_dir),
            key=f"{key_base}_output",
            help_text=FIELD_HELP["predictions_dir"],
        )
        mode_name = st.selectbox(
            "mode",
            options=mode_options,
            key=f"{key_base}_mode_name",
            help=FIELD_HELP["mode"],
        )
        trainer_name = st.selectbox(
            "trainer",
            options=trainer_options,
            key=f"{key_base}_trainer_name",
            help=FIELD_HELP["trainer"],
        )
        runner_name = st.selectbox(
            "runner",
            options=runner_options,
            key=f"{key_base}_runner_name",
            help=FIELD_HELP["runner"],
        )
        submitted = st.button("Run Online Prediction", key=f"{key_base}_run", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        st.code(
            " ".join(
                [
                    "python scripts/run_online_prediction.py",
                    f"{time_value}",
                    f"--models-dir {_quote(models_dir_text)}",
                    f"--processed-dir {_quote(processed_dir_text)}",
                    f"--output-dir {_quote(output_dir_text)}",
                    f"--mode {mode_name}",
                    f"--trainer {trainer_name}",
                    f"--runner {runner_name}",
                ]
            )
        )

        if runner_name == "reconstruction" and mode_name not in {"pod", "dmd"}:
            st.warning("`reconstruction` runner supports `pod`/`dmd` mode artifacts.")

        if submitted:
            request = {
                "time_value": time_value,
                "models_dir": models_dir_text,
                "processed_dir": processed_dir_text,
                "output_dir": output_dir_text,
                "mode_name": mode_name,
                "trainer_name": trainer_name,
                "runner_name": runner_name,
            }
            with st.spinner("Running online prediction..."):
                record = _run_stage(
                    session=session,
                    stage="online_prediction",
                    request=request,
                    action=lambda: run_online_prediction(
                        time_value=time_value,
                        models_dir=Path(models_dir_text),
                        processed_dir=Path(processed_dir_text),
                        output_dir=Path(output_dir_text),
                        mode_name=mode_name,
                        trainer_name=trainer_name,
                        runner_name=runner_name,
                    ),
                )
            st.session_state[f"{key_base}_last_record"] = record

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

    with tab_eval:
        st.subheader("Test Evaluation")
        st.caption("선택한 ROM 프로필과 test 데이터셋으로 R2/1-R2 평가를 수행합니다.")
        key_base = f"{session.session_id}_evaluate"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())

        model_profile_root = _model_profile_root(session)
        processed_catalog_root = _processed_catalog_root(session)
        processed_catalog = _discover_processed_catalog(str(processed_catalog_root))
        latest_test_dataset = _pick_latest_dataset_by_subset(processed_catalog, "test")

        profile_source = st.radio(
            "rom-profile-source",
            options=["registered", "direct"],
            index=0,
            key=f"{key_base}_profile_source",
            horizontal=True,
            help="registered=카탈로그 ROM 프로필 선택, direct=경로 직접 지정",
        )

        selected_rom = None
        models_dir_text = ""
        mode_name = None
        trainer_name = None

        if profile_source == "registered":
            rom_combos = _discover_registered_rom_combos(str(model_profile_root))
            if rom_combos:
                labels = []
                for item in rom_combos:
                    updated_local = datetime.fromtimestamp(float(item["updated_ts"])).strftime("%Y-%m-%d %H:%M:%S")
                    labels.append(
                        f"{item['profile_name']} | {item['mode_name']}+{item['trainer_name']} | "
                        f"vars={len(item['variables'])} | {updated_local}"
                    )
                selected_label = st.selectbox(
                    "rom-profile",
                    options=labels,
                    key=f"{key_base}_rom_profile",
                    help="Mode/Trainer 학습이 완료된 ROM 프로필을 선택합니다.",
                )
                selected_idx = labels.index(selected_label)
                selected_rom = rom_combos[selected_idx]
                models_dir_text = str(selected_rom["models_dir"])
                mode_name = str(selected_rom["mode_name"])
                trainer_name = str(selected_rom["trainer_name"])
                st.caption(f"Resolved models-dir: `{models_dir_text}`")
                st.code(
                    "\n".join(
                        [
                            f"Mode artifacts   : {Path(models_dir_text) / mode_name}",
                            f"Trainer artifacts: {Path(models_dir_text) / trainer_name / mode_name}",
                        ]
                    ),
                    language="text",
                )
            else:
                st.info("No registered ROM profile found. Switch to `direct` or run mode/offline training first.")
                profile_source = "direct"

        if profile_source == "direct":
            models_dir_text = _directory_input(
                "models-dir",
                value=str(session.models_dir),
                key=f"{key_base}_models",
                help_text=FIELD_HELP["models_dir"],
            )
            auto_detect_rom = st.checkbox(
                "Auto detect mode/trainer from models-dir",
                value=True,
                key=f"{key_base}_auto_detect",
                help="trainer_manifest/mode_manifest를 읽어 mode/trainer를 자동 선택합니다.",
            )
            detected = _discover_rom_profiles(models_dir_text) if auto_detect_rom else []
            if detected:
                labels = []
                for item in detected:
                    updated_local = datetime.fromtimestamp(float(item["updated_ts"])).strftime("%Y-%m-%d %H:%M:%S")
                    labels.append(
                        f"{item['mode_name']}+{item['trainer_name']} | vars={len(item['variables'])} | {updated_local}"
                    )
                selected_label = st.selectbox(
                    "detected-rom",
                    options=labels,
                    key=f"{key_base}_detected_rom",
                    help="탐지된 조합에서 mode/trainer를 고정합니다.",
                )
                selected_idx = labels.index(selected_label)
                selected_item = detected[selected_idx]
                mode_name = str(selected_item["mode_name"])
                trainer_name = str(selected_item["trainer_name"])
                st.caption(f"Detected: mode=`{mode_name}`, trainer=`{trainer_name}`")
            else:
                if auto_detect_rom:
                    st.info("No auto-detectable ROM found. Select mode/trainer manually.")
                mode_name = st.selectbox(
                    "mode",
                    options=mode_options,
                    key=f"{key_base}_mode_manual",
                    help=FIELD_HELP["mode"],
                )
                trainer_name = st.selectbox(
                    "trainer",
                    options=trainer_options,
                    key=f"{key_base}_trainer_manual",
                    help=FIELD_HELP["trainer"],
                )

        inferred_test_dir = None
        if selected_rom is not None:
            from_mode_manifest = (selected_rom.get("mode_manifest") or {}).get("processed_dir")
            train_dir = from_mode_manifest or selected_rom.get("processed_dir")
            if train_dir:
                inferred_test_dir = str(_derive_subset_peer_dir(train_dir, "test"))

        test_source = st.radio(
            "test-dataset-source",
            options=["auto", "registered", "direct"],
            index=0,
            key=f"{key_base}_test_source",
            horizontal=True,
            help="auto=프로필/카탈로그 기반 자동 추천",
        )

        processed_test_dir_text = ""
        if test_source == "auto":
            auto_candidate = None
            if inferred_test_dir:
                auto_candidate = inferred_test_dir
            elif latest_test_dataset is not None:
                auto_candidate = str(latest_test_dataset["path"])
            else:
                auto_candidate = str(session.processed_dir)

            processed_test_dir_text = _directory_input(
                "processed-test-dir",
                value=auto_candidate,
                key=f"{key_base}_processed_test_auto",
                help_text="Auto-resolved test processed directory",
            )
            if inferred_test_dir and not Path(inferred_test_dir).exists():
                st.warning(
                    "프로필의 train 경로로부터 추정한 test 경로가 아직 없습니다. "
                    "Preprocess에서 subset=all(또는 test)을 실행해 주세요."
                )
        elif test_source == "registered":
            preferred = [d for d in processed_catalog if str(d.get("subset") or "").lower() == "test"]
            if not preferred:
                preferred = [d for d in processed_catalog if str(d.get("name") or "").endswith("-test")]

            if preferred:
                labels = []
                for item in preferred:
                    subset_label = item.get("subset") or "n/a"
                    labels.append(
                        f"{item['name']} | subset={subset_label} | snaps={item['n_snapshots']} | vars={len(item.get('variables', []))}"
                    )
                default_idx = 0
                if inferred_test_dir:
                    inferred_norm = str(Path(inferred_test_dir))
                    for idx, item in enumerate(preferred):
                        if str(Path(item["path"])) == inferred_norm:
                            default_idx = idx
                            break
                selected_label = st.selectbox(
                    "processed-test-dataset",
                    options=labels,
                    index=default_idx,
                    key=f"{key_base}_test_dataset_label",
                )
                selected_idx = labels.index(selected_label)
                selected_dataset = preferred[selected_idx]
                processed_test_dir_text = str(selected_dataset["path"])
                st.caption(f"Resolved processed-test-dir: `{processed_test_dir_text}`")
            else:
                st.info("No registered test dataset found. Switch to `auto` or `direct`.")
                processed_test_dir_text = _directory_input(
                    "processed-test-dir",
                    value=str(session.processed_dir),
                    key=f"{key_base}_processed_test_fallback",
                    help_text="Test subset processed directory",
                )
        else:
            processed_test_dir_text = _directory_input(
                "processed-test-dir",
                value=inferred_test_dir or str(session.processed_dir),
                key=f"{key_base}_processed_test_direct",
                help_text="Test subset processed directory",
            )

        output_default = Path(session.predictions_dir) / "eval"
        if selected_rom is not None:
            output_default = output_default / str(selected_rom["profile_name"]) / str(mode_name) / str(trainer_name)
        output_default = output_default / Path(processed_test_dir_text).name
        output_dir_text = _directory_input(
            "output-dir",
            value=str(output_default),
            key=f"{key_base}_output",
            help_text="Directory for evaluation csv/png outputs",
        )

        default_columns = ["time"]
        test_doe = Path(processed_test_dir_text) / "doe.csv"
        if test_doe.exists():
            try:
                default_columns = list(pd.read_csv(test_doe, nrows=1).columns)
            except Exception:  # noqa: BLE001
                default_columns = ["time"]
        default_input_column = "time" if "time" in default_columns else default_columns[0]
        input_column = st.text_input(
            "input-column",
            value=default_input_column,
            key=f"{key_base}_input_column",
            help=FIELD_HELP["input_column"],
        )
        submitted = st.button("Run Test Evaluation", key=f"{key_base}_run", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        st.code(
            " ".join(
                [
                    "python scripts/run_test_evaluation.py",
                    f"--processed-test-dir {_quote(processed_test_dir_text)}",
                    f"--models-dir {_quote(models_dir_text)}",
                    f"--output-dir {_quote(output_dir_text)}",
                    f"--mode {mode_name}",
                    f"--trainer {trainer_name}",
                    f"--input-column {input_column}",
                ]
            )
        )

        if submitted:
            request = {
                "rom_profile_source": profile_source,
                "rom_profile_name": None if selected_rom is None else selected_rom.get("profile_name"),
                "test_dataset_source": test_source,
                "processed_test_dir": processed_test_dir_text,
                "models_dir": models_dir_text,
                "output_dir": output_dir_text,
                "mode_name": mode_name,
                "trainer_name": trainer_name,
                "input_column": input_column,
            }
            with st.spinner("Running test evaluation..."):
                record = _run_stage(
                    session=session,
                    stage="test_evaluation",
                    request=request,
                    action=lambda: run_test_evaluation(
                        processed_test_dir=Path(processed_test_dir_text),
                        models_dir=Path(models_dir_text),
                        mode_name=mode_name,
                        trainer_name=trainer_name,
                        output_dir=Path(output_dir_text),
                        input_column=input_column,
                    ),
                )
            st.session_state[f"{key_base}_last_record"] = record

        if f"{key_base}_last_record" in st.session_state:
            record = st.session_state[f"{key_base}_last_record"]
            _render_record(record)
            summary = record.get("summary") or {}
            r2_plot_path = Path(summary.get("r2_plot_path", ""))
            if record.get("status") == "success" and r2_plot_path.exists():
                st.image(str(r2_plot_path), caption="R2 Diagnostics", use_container_width=True)

            r2_by_time_path = Path(summary.get("r2_by_time_path", ""))
            if record.get("status") == "success" and r2_by_time_path.exists():
                try:
                    eval_df = pd.read_csv(r2_by_time_path)
                except Exception:  # noqa: BLE001
                    eval_df = pd.DataFrame()

                if not eval_df.empty and {"time", "variable", "r2", "r2_error"}.issubset(set(eval_df.columns)):
                    st.markdown("**Evaluation Curves**")
                    variable_options = sorted(eval_df["variable"].astype(str).unique().tolist())
                    selected_variable = st.selectbox(
                        "evaluation-variable",
                        options=variable_options,
                        key=f"{key_base}_curve_variable",
                    )
                    var_df = eval_df[eval_df["variable"] == selected_variable].sort_values("time")
                    curve_df = var_df.set_index("time")[["r2", "r2_error"]]
                    st.line_chart(curve_df, use_container_width=True)

            with st.expander("3D Compare: Ground Truth vs Prediction", expanded=False):
                compare_variable = st.selectbox(
                    "compare-variable",
                    options=VISUAL_VARIABLES,
                    index=VISUAL_VARIABLES.index("velocity"),
                    key=f"{key_base}_compare_variable",
                )
                max_points = int(
                    st.number_input(
                        "compare-max-points",
                        min_value=1000,
                        value=12000,
                        step=1000,
                        key=f"{key_base}_compare_max_points",
                    )
                )
                point_size = float(
                    st.number_input(
                        "compare-point-size",
                        min_value=0.5,
                        value=2.0,
                        step=0.5,
                        key=f"{key_base}_compare_point_size",
                    )
                )

                cmp_doe_path = Path(processed_test_dir_text) / "doe.csv"
                if not cmp_doe_path.exists():
                    st.info("processed-test-dir에 doe.csv가 없어 3D 비교를 표시할 수 없습니다.")
                else:
                    try:
                        cmp_doe = pd.read_csv(cmp_doe_path)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"doe.csv 읽기 실패: {exc}")
                        cmp_doe = pd.DataFrame()

                    if cmp_doe.empty or input_column not in cmp_doe.columns:
                        st.info(f"`{input_column}` 컬럼을 찾을 수 없어 3D 비교를 표시할 수 없습니다.")
                    else:
                        cmp_times = cmp_doe[input_column].to_numpy(dtype=np.float64)
                        if cmp_times.size == 0:
                            st.info("평가 데이터의 시간축이 비어 있습니다.")
                        else:
                            time_idx = int(
                                st.slider(
                                    "compare-time-index",
                                    min_value=0,
                                    max_value=int(cmp_times.size - 1),
                                    value=0,
                                    key=f"{key_base}_compare_time_idx",
                                )
                            )
                            selected_time = float(cmp_times[time_idx])
                            st.caption(f"Selected time: {selected_time:.6f}")
                            load_compare = st.button(
                                "Load 3D Compare",
                                key=f"{key_base}_load_compare",
                                use_container_width=True,
                            )

                            if load_compare:
                                try:
                                    points = _load_points(str(Path(processed_test_dir_text) / "points.bin"))
                                    keep = _subsample_indices(len(points), max_points=max_points)
                                    points_sub = points[keep].astype(np.float32, copy=False)

                                    true_vals = _snapshot_values_at_time(
                                        processed_dir=Path(processed_test_dir_text),
                                        variable=compare_variable,
                                        time_index=time_idx,
                                    )[keep]

                                    runner = OnlinePredictionRunner(
                                        models_dir=Path(models_dir_text),
                                        mode_name=str(mode_name),
                                        trainer_name=str(trainer_name),
                                    )
                                    pred_df = runner.step(selected_time)
                                    pred_vals = _value_series(pred_df, compare_variable)[keep]

                                    cmin = float(min(np.min(true_vals), np.min(pred_vals)))
                                    cmax = float(max(np.max(true_vals), np.max(pred_vals)))

                                    col_true, col_pred = st.columns(2)
                                    with col_true:
                                        fig_true = _scatter3d_field_figure(
                                            coords=points_sub,
                                            values=true_vals,
                                            title=f"Ground Truth | {compare_variable}",
                                            cmin=cmin,
                                            cmax=cmax,
                                            point_size=point_size,
                                        )
                                        st.plotly_chart(fig_true, use_container_width=True)
                                    with col_pred:
                                        fig_pred = _scatter3d_field_figure(
                                            coords=points_sub,
                                            values=pred_vals,
                                            title=f"Prediction | {compare_variable}",
                                            cmin=cmin,
                                            cmax=cmax,
                                            point_size=point_size,
                                        )
                                        st.plotly_chart(fig_pred, use_container_width=True)
                                except Exception as exc:  # noqa: BLE001
                                    st.error(f"3D compare failed: {exc}")

    with tab_pipeline:
        st.subheader("Full Pipeline")
        key_base = f"{session.session_id}_pipeline"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())
        runner_options = sorted(RUNNER_REGISTRY.keys())
        raw_dir_text = _directory_input(
            "raw-dir",
            value=str(session.raw_dir),
            key=f"{key_base}_raw",
            help_text=FIELD_HELP["raw_dir"],
        )
        processed_dir_text = _directory_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help_text=FIELD_HELP["processed_dir"],
        )
        models_dir_text = _directory_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help_text=FIELD_HELP["models_dir"],
        )
        predictions_dir_text = _directory_input(
            "predictions-dir",
            value=str(session.predictions_dir),
            key=f"{key_base}_predictions",
            help_text=FIELD_HELP["predictions_dir"],
        )
        mode_name = st.selectbox(
            "mode",
            options=mode_options,
            key=f"{key_base}_mode_name",
            help=FIELD_HELP["mode"],
        )
        trainer_name = st.selectbox(
            "trainer",
            options=trainer_options,
            key=f"{key_base}_trainer_name",
            help=FIELD_HELP["trainer"],
        )
        runner_name = st.selectbox(
            "runner",
            options=runner_options,
            key=f"{key_base}_runner_name",
            help=FIELD_HELP["runner"],
        )
        mode_params, extra_mode_json = _render_mode_controls(mode_name, key_base)
        trainer_params, extra_trainer_json = _render_trainer_controls(trainer_name, key_base)
        eval_mode = "none"
        val_ratio = 0.2
        min_train_samples = 8
        st.caption("Full Pipeline의 offline 단계는 학습 전용으로 동작합니다 (eval-mode=none).")
        run_online = st.checkbox("Run online prediction after training", value=True, key=f"{key_base}_run_online")
        predict_time = float(
            st.number_input(
                "predict-time",
                value=0.025,
                step=0.001,
                format="%.6f",
                key=f"{key_base}_predict_time",
                help="Time value used when online prediction is enabled",
            )
        )
        submitted = st.button("Run Full Pipeline", key=f"{key_base}_run", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        mode_cmd = "--auto-rank" if mode_params.get("rank") is None else f"--rank {mode_params.get('rank')}"
        cmd_parts = [
            "python scripts/run_pipeline.py",
            f"--raw-dir {_quote(raw_dir_text)}",
            f"--processed-dir {_quote(processed_dir_text)}",
            f"--models-dir {_quote(models_dir_text)}",
            f"--predictions-dir {_quote(predictions_dir_text)}",
            f"--mode {mode_name}",
            f"--trainer {trainer_name}",
            f"--runner {runner_name}",
            mode_cmd,
        ]
        if "energy_threshold" in mode_params:
            cmd_parts.append(f"--energy-threshold {mode_params['energy_threshold']}")
        if "dt" in mode_params:
            cmd_parts.append(f"--dt {mode_params['dt']}")
        if "time_column" in mode_params:
            cmd_parts.append(f"--time-column {mode_params['time_column']}")
        if trainer_name == "rbf":
            cmd_parts.extend(
                [
                    f"--kernel {trainer_params.get('kernel')}",
                    f"--epsilon {trainer_params.get('epsilon')}",
                ]
            )
        if trainer_name == "nn":
            cmd_parts.extend(
                [
                    f"--hidden-dims {_quote(trainer_params.get('hidden_dims', ''))}",
                    f"--activation {trainer_params.get('activation')}",
                    f"--epochs {trainer_params.get('epochs')}",
                ]
            )
        if trainer_name == "projection":
            cmd_parts.extend(
                [
                    f"--solver {trainer_params.get('solver')}",
                    f"--ridge {trainer_params.get('ridge')}",
                    "--stabilization" if trainer_params.get("stabilization") else "",
                ]
            )
        if run_online:
            cmd_parts.append(f"--predict-time {predict_time}")
        st.code(" ".join(part for part in cmd_parts if part))

        if submitted:
            try:
                if trainer_name == "nn":
                    trainer_params["hidden_dims"] = _parse_hidden_dims(trainer_params.get("hidden_dims", ""))

                mode_kwargs = _merge_extra_kwargs(mode_params, extra_mode_json)
                trainer_kwargs = _merge_extra_kwargs(trainer_params, extra_trainer_json)
                request = {
                    "raw_dir": raw_dir_text,
                    "processed_dir": processed_dir_text,
                    "models_dir": models_dir_text,
                    "predictions_dir": predictions_dir_text,
                    "mode_name": mode_name,
                    "trainer_name": trainer_name,
                    "runner_name": runner_name,
                    "mode_params": mode_kwargs,
                    "trainer_params": trainer_kwargs,
                    "eval_mode": eval_mode,
                    "val_ratio": val_ratio,
                    "min_train_samples": min_train_samples,
                    "predict_time": predict_time if run_online else None,
                }
                with st.spinner("Running full pipeline..."):
                    record = _run_stage(
                        session=session,
                        stage="full_pipeline",
                        request=request,
                        action=lambda: run_full_pipeline(
                            raw_dir=Path(raw_dir_text),
                            processed_dir=Path(processed_dir_text),
                            models_dir=Path(models_dir_text),
                            predictions_dir=Path(predictions_dir_text),
                            mode_name=mode_name,
                            trainer_name=trainer_name,
                            runner_name=runner_name,
                            mode_params=mode_kwargs,
                            trainer_params=trainer_kwargs,
                            predict_time=predict_time if run_online else None,
                            offline_eval_mode=eval_mode,
                            offline_val_ratio=val_ratio,
                            offline_min_train_samples=min_train_samples,
                        ),
                    )
                st.session_state[f"{key_base}_last_record"] = record
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid pipeline parameters: {exc}")

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

    with tab_viewer:
        st.subheader("Interactive ROM Viewer")
        st.caption(
            "완성된 ROM을 불러와 시간 구간 예측을 앱 내부 3D 뷰어에서 재생합니다. "
            "마우스로 각도를 직접 회전할 수 있고, 자동 카메라 회전도 지원합니다."
        )

        key_base = f"{session.session_id}_viewer"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())

        models_dir_text = _directory_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help_text=FIELD_HELP["models_dir"],
        )
        processed_dir_text = _directory_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help_text=FIELD_HELP["processed_dir"],
        )

        auto_profile = st.checkbox(
            "Auto detect mode/trainer from saved ROM artifacts",
            value=True,
            key=f"{key_base}_auto_profile",
            help="mode_manifest/trainer_manifest를 읽어 mode/trainer를 자동 선택합니다.",
        )
        discovered_profiles = _discover_rom_profiles(models_dir_text)
        selected_profile = None
        if auto_profile:
            if discovered_profiles:
                profile_labels = []
                for item in discovered_profiles:
                    updated_local = datetime.fromtimestamp(float(item["updated_ts"])).strftime("%Y-%m-%d %H:%M:%S")
                    profile_labels.append(
                        f"{item['mode_name']} + {item['trainer_name']} | vars={len(item['variables'])} | {updated_local}"
                    )
                selected_label = st.selectbox(
                    "Detected ROM profile",
                    options=profile_labels,
                    key=f"{key_base}_detected_profile",
                    help="최근 업데이트 기준으로 정렬된 ROM 프로파일",
                )
                selected_idx = profile_labels.index(selected_label)
                selected_profile = discovered_profiles[selected_idx]
                st.caption(
                    f"Selected profile: mode=`{selected_profile['mode_name']}`, "
                    f"trainer=`{selected_profile['trainer_name']}`, "
                    f"variables={', '.join(selected_profile['variables'])}"
                )
                with st.expander("Detected metadata", expanded=False):
                    st.json(
                        {
                            "mode_manifest": selected_profile.get("mode_manifest"),
                            "trainer_manifest": selected_profile.get("trainer_manifest"),
                        }
                    )
            else:
                st.info("No auto-detectable profile found under models-dir. Falling back to manual selection.")

        viewer_lock_mode_trainer = auto_profile and selected_profile is not None
        mode_default = selected_profile["mode_name"] if selected_profile else mode_options[0]
        trainer_default = selected_profile["trainer_name"] if selected_profile else trainer_options[0]
        mode_index = mode_options.index(mode_default) if mode_default in mode_options else 0
        trainer_index = trainer_options.index(trainer_default) if trainer_default in trainer_options else 0
        if viewer_lock_mode_trainer:
            st.session_state[f"{key_base}_mode"] = mode_default
            st.session_state[f"{key_base}_trainer"] = trainer_default

        c1, c2, c3 = st.columns(3)
        with c1:
            mode_name = st.selectbox(
                "mode",
                options=mode_options,
                index=mode_index,
                key=f"{key_base}_mode",
                help="현재 인터랙티브 뷰어는 POD/DMD 재구성 모드를 지원합니다.",
                disabled=viewer_lock_mode_trainer,
            )
            if selected_profile and {"u", "v", "w"}.issubset(set(selected_profile.get("variables", []))):
                default_variable = "velocity"
            elif selected_profile:
                default_variable = selected_profile["variables"][0]
            else:
                default_variable = "velocity"
            variable = st.selectbox(
                "variable",
                options=VISUAL_VARIABLES,
                index=VISUAL_VARIABLES.index(default_variable) if default_variable in VISUAL_VARIABLES else 0,
                key=f"{key_base}_variable",
                help="시각화할 물리량",
            )
        with c2:
            trainer_name = st.selectbox(
                "trainer",
                options=trainer_options,
                index=trainer_index,
                key=f"{key_base}_trainer",
                help=FIELD_HELP["trainer"],
                disabled=viewer_lock_mode_trainer,
            )
            frames = int(
                st.number_input(
                    "frames",
                    min_value=2,
                    value=40,
                    step=1,
                    key=f"{key_base}_frames",
                    help="시간 구간을 몇 프레임으로 나눌지 설정합니다.",
                )
            )
        with c3:
            max_points = int(
                st.number_input(
                    "max-points",
                    min_value=1000,
                    value=15000,
                    step=1000,
                    key=f"{key_base}_max_points",
                    help="렌더링 성능을 위해 포인트 클라우드를 이 개수 이하로 샘플링합니다.",
                )
            )
            point_size = float(
                st.number_input(
                    "point-size",
                    min_value=0.5,
                    value=2.0,
                    step=0.5,
                    key=f"{key_base}_point_size",
                    help="3D 점 크기",
                )
            )

        default_viewer_start = 0.005
        default_viewer_end = 0.05
        default_viewer_step = 0.001
        viewer_doe = Path(processed_dir_text) / "doe.csv"
        if viewer_doe.exists():
            try:
                viewer_doe_df = pd.read_csv(viewer_doe, nrows=100000)
                if "time" in viewer_doe_df.columns and not viewer_doe_df.empty:
                    tvals = viewer_doe_df["time"].to_numpy(dtype=np.float64)
                    default_viewer_start = float(np.min(tvals))
                    default_viewer_end = float(np.max(tvals))
                    if tvals.size > 1:
                        diffs = np.diff(np.sort(tvals))
                        diffs = diffs[diffs > 0]
                        if diffs.size > 0:
                            default_viewer_step = float(np.median(diffs))
            except Exception:
                pass

        t1, t2 = st.columns(2)
        with t1:
            start_time = float(
                st.number_input(
                    "start-time",
                    value=default_viewer_start,
                    step=default_viewer_step,
                    format="%.6f",
                    key=f"{key_base}_start",
                    help="예측 시작 시간",
                )
            )
        with t2:
            end_time = float(
                st.number_input(
                    "end-time",
                    value=default_viewer_end,
                    step=default_viewer_step,
                    format="%.6f",
                    key=f"{key_base}_end",
                    help="예측 종료 시간",
                )
            )

        cam1, cam2, cam3, cam4 = st.columns(4)
        with cam1:
            rotate_camera = st.checkbox(
                "Auto camera orbit",
                value=True,
                key=f"{key_base}_rotate",
                help="재생 중 카메라를 자연스럽게 공전시킵니다.",
            )
        with cam2:
            rotation_radius = float(
                st.number_input(
                    "camera-radius",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    key=f"{key_base}_cam_radius",
                    help="카메라 공전 반경",
                )
            )
        with cam3:
            z_height = float(
                st.number_input(
                    "camera-z",
                    min_value=0.1,
                    value=0.9,
                    step=0.1,
                    key=f"{key_base}_cam_z",
                    help="카메라 높이",
                )
            )
        with cam4:
            loop_cycles = int(
                st.number_input(
                    "loop-cycles",
                    min_value=1,
                    value=1,
                    step=1,
                    key=f"{key_base}_loop_cycles",
                    help="Desktop Native Viewer에서 재생 반복 횟수. 브라우저 Plotly는 1회 루프 기준으로 재생됩니다.",
                )
            )

        frame_duration_ms = int(
            st.number_input(
                "frame-ms",
                min_value=20,
                value=70,
                step=10,
                key=f"{key_base}_frame_ms",
                help="프레임 전환 시간(ms). 작을수록 빠르게 재생됩니다.",
            )
        )

        if mode_name not in {"pod", "dmd"}:
            st.warning("현재 인터랙티브 뷰어는 `mode=pod` 또는 `mode=dmd` 경로만 지원합니다.")
        if end_time <= start_time:
            st.warning("`end-time`은 `start-time`보다 커야 합니다.")

        load_viewer = st.button("Load / Refresh Viewer", key=f"{key_base}_load", use_container_width=True)
        if load_viewer:
            if end_time <= start_time:
                st.error("Invalid time range. Set `end-time` > `start-time`.")
            else:
                request = {
                    "models_dir": models_dir_text,
                    "processed_dir": processed_dir_text,
                    "mode_name": mode_name,
                    "trainer_name": trainer_name,
                    "variable": variable,
                    "start_time": start_time,
                    "end_time": end_time,
                    "frames": frames,
                    "max_points": max_points,
                }
                holder = {}

                def _prepare_viewer():
                    bundle = _build_viewer_data(
                        models_dir=Path(models_dir_text),
                        processed_dir=Path(processed_dir_text),
                        mode_name=mode_name,
                        trainer_name=trainer_name,
                        variable=variable,
                        start_time=start_time,
                        end_time=end_time,
                        frames=frames,
                        max_points=max_points,
                    )
                    holder["bundle"] = bundle
                    return {
                        "variable": variable,
                        "frames": frames,
                        "points_total": bundle["points_total"],
                        "points_used": bundle["points_used"],
                        "time_range": [start_time, end_time],
                    }

                with st.spinner("Preparing interactive viewer frames..."):
                    record = _run_stage(
                        session=session,
                        stage="viewer_prepare",
                        request=request,
                        action=_prepare_viewer,
                    )
                st.session_state[f"{key_base}_last_record"] = record
                if record.get("status") == "success" and "bundle" in holder:
                    st.session_state[f"{key_base}_bundle"] = holder["bundle"]

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

        bundle = st.session_state.get(f"{key_base}_bundle")
        if bundle:
            payload_mb = _estimate_viewer_payload_mb(bundle)
            st.success(
                f"Viewer ready: {bundle['points_used']:,} / {bundle['points_total']:,} points, "
                f"{len(bundle['times'])} frames"
            )
            st.caption(f"Estimated chart payload: ~{payload_mb:.1f} MB")
            if payload_mb > 180:
                st.warning(
                    "이 설정은 기본 200MB 메시지 제한에 근접/초과할 수 있습니다. "
                    "`max-points`, `frames`를 낮추거나 `server.maxMessageSize`를 높여주세요."
                )
            fig = _build_viewer_figure(
                bundle=bundle,
                point_size=point_size,
                rotate_camera=rotate_camera,
                rotation_radius=rotation_radius,
                z_height=z_height,
                loop_cycles=loop_cycles,
                frame_duration_ms=frame_duration_ms,
            )
            live_colorbar = st.checkbox(
                "Live colorbar controls (client-side)",
                value=True,
                key=f"{key_base}_live_colorbar",
                help="브라우저에서 앱 재실행 없이 cmin/cmax/colorscale을 즉시 변경합니다.",
            )
            if live_colorbar:
                _render_live_colorbar_viewer(
                    fig=fig,
                    key=f"{key_base}_{session.session_id}",
                    vmin=float(bundle["vmin"]),
                    vmax=float(bundle["vmax"]),
                )
            else:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                )
            st.caption(
                "Play 버튼으로 시간축 애니메이션을 재생하세요. "
                "마우스 드래그로 임의 각도에서 즉시 탐색할 수 있습니다."
            )
            if loop_cycles > 1:
                st.caption(
                    "브라우저 Plotly 재생은 1회 루프 기준으로 동작합니다. "
                    "여러 회 반복은 Desktop Native Viewer에서 정확히 반영됩니다."
                )

        st.markdown("---")
        st.subheader("Desktop Native Viewer")
        st.caption(
            "브라우저 대신 데스크톱 렌더러(PyVista/VTK)로 실행합니다. "
            "대용량 포인트에서 브라우저보다 안정적입니다."
        )
        pyvista_available = _is_pyvista_available()
        if not pyvista_available:
            st.info(
                "PyVista가 없어도 실행 가능합니다. 이 경우 Native Viewer는 matplotlib 백엔드로 실행됩니다."
            )
        native_backend_options = ["auto", "pyvista", "matplotlib"]
        native_backend_default = "pyvista" if pyvista_available else "matplotlib"
        native_backend = st.selectbox(
            "native-backend",
            options=native_backend_options,
            index=native_backend_options.index(native_backend_default),
            key=f"{key_base}_native_backend",
            help="auto는 pyvista 가능 시 pyvista, 아니면 matplotlib로 자동 선택합니다.",
        )

        native_command = _build_native_viewer_command(
            models_dir=Path(models_dir_text),
            processed_dir=Path(processed_dir_text),
            mode_name=mode_name,
            trainer_name=trainer_name,
            variable=variable,
            start_time=start_time,
            end_time=end_time,
            frames=frames,
            max_points=max_points,
            point_size=point_size,
            loop_cycles=loop_cycles,
            frame_duration_ms=frame_duration_ms,
            rotate_camera=rotate_camera,
            backend=native_backend,
        )
        st.code(_shell_join(native_command))
        launch_native = st.button(
            "Launch Native Viewer (Desktop)",
            key=f"{key_base}_launch_native",
            use_container_width=True,
        )
        if launch_native:
            if end_time <= start_time:
                st.error("Invalid time range. Set `end-time` > `start-time`.")
            else:
                request = {
                    "command": _shell_join(native_command),
                    "mode_name": mode_name,
                    "trainer_name": trainer_name,
                    "variable": variable,
                    "start_time": start_time,
                    "end_time": end_time,
                    "frames": frames,
                    "max_points": max_points,
                    "loop_cycles": loop_cycles,
                    "frame_duration_ms": frame_duration_ms,
                }
                with st.spinner("Launching native desktop viewer..."):
                    native_record = _run_stage(
                        session=session,
                        stage="viewer_launch_native",
                        request=request,
                        action=lambda: _launch_native_viewer(native_command),
                    )
                st.session_state[f"{key_base}_native_last_record"] = native_record

        if f"{key_base}_native_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_native_last_record"])


if __name__ == "__main__":
    main()




