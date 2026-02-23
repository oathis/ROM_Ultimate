from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import importlib.util
import json
from pathlib import Path
import os
import shlex
import subprocess
import sys
import traceback

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
import numpy as np
import pandas as pd
from rom.core.workflows import (
    run_full_pipeline,
    run_mode_training,
    run_offline_training,
    run_online_prediction,
    run_preprocess,
)
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
}

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _quote(value: str | Path) -> str:
    return f"\"{value}\""


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

    raw_dir_input = st.sidebar.text_input(
        "Raw data directory",
        value=str(session.raw_dir),
        key=f"{session.session_id}_raw_dir",
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

    tab_overview, tab_pre, tab_mode, tab_offline, tab_online, tab_pipeline, tab_viewer = st.tabs(
        ["Overview", "Preprocess", "Mode", "Offline", "Online", "Pipeline", "Viewer"]
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

    with tab_pre:
        st.subheader("Preprocess")
        key_base = f"{session.session_id}_preprocess"
        with st.form(f"{key_base}_form"):
            raw_dir_text = st.text_input(
                "raw-dir",
                value=str(session.raw_dir),
                key=f"{key_base}_raw",
                help=FIELD_HELP["raw_dir"],
            )
            processed_dir_text = st.text_input(
                "processed-dir",
                value=str(session.processed_dir),
                key=f"{key_base}_processed",
                help=FIELD_HELP["processed_dir"],
            )
            submitted = st.form_submit_button("Run Preprocess", use_container_width=True)

        st.markdown("<div class='command-hint'>Equivalent CLI command</div>", unsafe_allow_html=True)
        st.code(
            f"python scripts/run_preprocess.py --raw-dir {_quote(raw_dir_text)} "
            f"--processed-dir {_quote(processed_dir_text)}"
        )

        if submitted:
            raw_dir = Path(raw_dir_text)
            processed_dir = Path(processed_dir_text)
            request = {"raw_dir": str(raw_dir), "processed_dir": str(processed_dir)}
            with st.spinner("Running preprocess..."):
                record = _run_stage(
                    session=session,
                    stage="preprocess",
                    request=request,
                    action=lambda: run_preprocess(raw_dir=raw_dir, processed_dir=processed_dir),
                )
            st.session_state[f"{key_base}_last_record"] = record

        if f"{key_base}_last_record" in st.session_state:
            _render_record(st.session_state[f"{key_base}_last_record"])

    with tab_mode:
        st.subheader("Mode Training")
        key_base = f"{session.session_id}_mode"
        mode_options = sorted(MODE_REGISTRY.keys())
        processed_dir_text = st.text_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help=FIELD_HELP["processed_dir"],
        )
        models_dir_text = st.text_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help=FIELD_HELP["models_dir"],
        )
        mode_name = st.selectbox(
            "mode",
            options=mode_options,
            key=f"{key_base}_mode_name",
            help=FIELD_HELP["mode"],
        )
        mode_params, extra_mode_json = _render_mode_controls(mode_name, key_base)
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
                    "models_dir": models_dir_text,
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

    with tab_offline:
        st.subheader("Offline Training")
        key_base = f"{session.session_id}_offline"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())
        processed_dir_text = st.text_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help=FIELD_HELP["processed_dir"],
        )
        models_dir_text = st.text_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help=FIELD_HELP["models_dir"],
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
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        with eval_col1:
            eval_mode = st.selectbox(
                "eval-mode",
                options=OFFLINE_EVAL_OPTIONS,
                index=OFFLINE_EVAL_OPTIONS.index("both"),
                key=f"{key_base}_eval_mode",
                help=FIELD_HELP["eval_mode"],
            )
        with eval_col2:
            val_ratio = float(
                st.number_input(
                    "val-ratio",
                    min_value=0.01,
                    max_value=0.9,
                    value=0.2,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_base}_val_ratio",
                    help=FIELD_HELP["val_ratio"],
                )
            )
        with eval_col3:
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
        st.caption(f"Detected columns in `doe.csv`: {', '.join(default_columns)}")
        st.caption(
            "Validation policy: "
            "`interpolation`=중간 시점을 샘플링, `extrapolation`=후반 구간 홀드아웃, `both`=둘 다 계산"
        )
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
            f"--eval-mode {eval_mode}",
            f"--val-ratio {val_ratio}",
            f"--min-train-samples {min_train_samples}",
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
        models_dir_text = st.text_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help=FIELD_HELP["models_dir"],
        )
        processed_dir_text = st.text_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help=FIELD_HELP["processed_dir"],
        )
        output_dir_text = st.text_input(
            "output-dir",
            value=str(session.predictions_dir),
            key=f"{key_base}_output",
            help=FIELD_HELP["predictions_dir"],
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

    with tab_pipeline:
        st.subheader("Full Pipeline")
        key_base = f"{session.session_id}_pipeline"
        mode_options = sorted(MODE_REGISTRY.keys())
        trainer_options = sorted(TRAINER_REGISTRY.keys())
        runner_options = sorted(RUNNER_REGISTRY.keys())
        raw_dir_text = st.text_input(
            "raw-dir",
            value=str(session.raw_dir),
            key=f"{key_base}_raw",
            help=FIELD_HELP["raw_dir"],
        )
        processed_dir_text = st.text_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help=FIELD_HELP["processed_dir"],
        )
        models_dir_text = st.text_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help=FIELD_HELP["models_dir"],
        )
        predictions_dir_text = st.text_input(
            "predictions-dir",
            value=str(session.predictions_dir),
            key=f"{key_base}_predictions",
            help=FIELD_HELP["predictions_dir"],
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
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        with eval_col1:
            eval_mode = st.selectbox(
                "eval-mode",
                options=OFFLINE_EVAL_OPTIONS,
                index=OFFLINE_EVAL_OPTIONS.index("both"),
                key=f"{key_base}_eval_mode",
                help=FIELD_HELP["eval_mode"],
            )
        with eval_col2:
            val_ratio = float(
                st.number_input(
                    "val-ratio",
                    min_value=0.01,
                    max_value=0.9,
                    value=0.2,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_base}_val_ratio",
                    help=FIELD_HELP["val_ratio"],
                )
            )
        with eval_col3:
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
            f"--eval-mode {eval_mode}",
            f"--val-ratio {val_ratio}",
            f"--min-train-samples {min_train_samples}",
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

        models_dir_text = st.text_input(
            "models-dir",
            value=str(session.models_dir),
            key=f"{key_base}_models",
            help=FIELD_HELP["models_dir"],
        )
        processed_dir_text = st.text_input(
            "processed-dir",
            value=str(session.processed_dir),
            key=f"{key_base}_processed",
            help=FIELD_HELP["processed_dir"],
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            mode_name = st.selectbox(
                "mode",
                options=mode_options,
                key=f"{key_base}_mode",
                help="현재 인터랙티브 뷰어는 POD/DMD 재구성 모드를 지원합니다.",
            )
            variable = st.selectbox(
                "variable",
                options=VISUAL_VARIABLES,
                index=0,
                key=f"{key_base}_variable",
                help="시각화할 물리량",
            )
        with c2:
            trainer_name = st.selectbox(
                "trainer",
                options=trainer_options,
                key=f"{key_base}_trainer",
                help=FIELD_HELP["trainer"],
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




