from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

from rom.runners.online_prediction import OnlinePredictionRunner


VARIABLE_CHOICES = ("velocity", "T", "u", "v", "w")
BACKEND_CHOICES = ("auto", "pyvista", "matplotlib")


def _load_points(points_path: Path):
    raw = np.fromfile(points_path, dtype=np.float64)
    if raw.size % 3 != 0:
        raise ValueError(f"Invalid points file shape at {points_path}")
    return raw.reshape(-1, 3)


def _downsample_indices(total_points: int, max_points: int):
    max_points = max(1, int(max_points))
    if total_points <= max_points:
        return np.arange(total_points)
    step = max(1, total_points // max_points)
    return np.arange(0, total_points, step)


def _extract_values(frame_df, variable: str):
    if variable == "velocity":
        ux = frame_df["x-velocity"] if "x-velocity" in frame_df else frame_df.get("u")
        vy = frame_df["y-velocity"] if "y-velocity" in frame_df else frame_df.get("v")
        wz = frame_df["z-velocity"] if "z-velocity" in frame_df else frame_df.get("w")
        if ux is None or vy is None or wz is None:
            raise ValueError("Velocity components are missing in prediction output.")
        return np.sqrt(np.asarray(ux) ** 2 + np.asarray(vy) ** 2 + np.asarray(wz) ** 2)

    mapping = {
        "T": "temperature",
        "u": "x-velocity",
        "v": "y-velocity",
        "w": "z-velocity",
    }
    col = mapping.get(variable, variable)
    if col in frame_df.columns:
        return np.asarray(frame_df[col])
    if variable in frame_df.columns:
        return np.asarray(frame_df[variable])
    raise ValueError(f"Variable '{variable}' not found in prediction output.")


def _camera_position(center: np.ndarray, radius: float, z_value: float, angle: float):
    pos = (
        float(center[0] + radius * np.cos(angle)),
        float(center[1] + radius * np.sin(angle)),
        float(z_value),
    )
    focal = (float(center[0]), float(center[1]), float(center[2]))
    viewup = (0.0, 0.0, 1.0)
    return [pos, focal, viewup]


@dataclass
class ViewerConfig:
    models_dir: Path
    processed_dir: Path
    mode_name: str
    trainer_name: str
    variable: str
    start_time: float
    end_time: float
    frames: int
    max_points: int
    point_size: float
    loop_cycles: int
    frame_ms: int
    rotate_camera: bool
    camera_radius_scale: float
    camera_height_scale: float
    backend: str
    dry_run: bool


class FrameProvider:
    def __init__(self, cfg: ViewerConfig):
        if cfg.mode_name not in {"pod", "dmd"}:
            raise NotImplementedError("Native viewer currently supports `mode=pod` and `mode=dmd`.")

        self.cfg = cfg
        self.runner = OnlinePredictionRunner(
            models_dir=cfg.models_dir,
            mode_name=cfg.mode_name,
            trainer_name=cfg.trainer_name,
        )

        points_path = cfg.processed_dir / "points.bin"
        if not points_path.exists():
            raise FileNotFoundError(f"points.bin not found: {points_path}")

        coords = _load_points(points_path)
        self.keep = _downsample_indices(len(coords), cfg.max_points)
        self.coords = coords[self.keep].astype(np.float32, copy=False)
        self.times = np.linspace(cfg.start_time, cfg.end_time, cfg.frames).astype(np.float64)
        self.scalar_name = f"{cfg.variable}_value"

        self.value_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_max_items = 24

        first_vals = self.get_values(0)
        vmin = float(np.min(first_vals))
        vmax = float(np.max(first_vals))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        self.scalar_range = (vmin, vmax)

        self.center = np.mean(self.coords, axis=0)
        span = np.ptp(self.coords, axis=0)
        diag = float(np.linalg.norm(span))
        self.scene_scale = diag if diag > 1e-6 else 1.0

    def get_values(self, frame_idx: int):
        frame_idx = max(0, min(int(frame_idx), len(self.times) - 1))
        if frame_idx in self.value_cache:
            vals = self.value_cache.pop(frame_idx)
            self.value_cache[frame_idx] = vals
            return vals

        t_val = float(self.times[frame_idx])
        frame_df = self.runner.step(t_val)
        vals = _extract_values(frame_df, self.cfg.variable)[self.keep].astype(np.float32, copy=False)

        self.value_cache[frame_idx] = vals
        while len(self.value_cache) > self.cache_max_items:
            self.value_cache.popitem(last=False)
        return vals

    def summary(self):
        return {
            "points_used": int(len(self.coords)),
            "frames": int(len(self.times)),
            "time_range": [float(self.times[0]), float(self.times[-1])],
            "variable": self.cfg.variable,
            "mode": self.cfg.mode_name,
            "trainer": self.cfg.trainer_name,
            "backend": self.cfg.backend,
        }


class PyVistaViewer:
    def __init__(self, provider: FrameProvider):
        self.provider = provider
        self.cfg = provider.cfg
        self.current_idx = 0
        self.playing = False
        self.completed_loops = 0
        self.max_cycles = max(1, int(self.cfg.loop_cycles))
        self.camera_radius = self.provider.scene_scale * float(self.cfg.camera_radius_scale)
        self.camera_height = float(self.provider.center[2] + self.provider.scene_scale * float(self.cfg.camera_height_scale))

        self.plotter = None
        self.cloud = None
        self.slider_widget = None
        self._syncing_slider = False

    def _set_status_text(self, text: str):
        if self.plotter is None:
            return
        self.plotter.add_text(text, name="play_state", position="lower_left", font_size=10)

    def _set_slider_value(self, frame_idx: int):
        if self.slider_widget is None:
            return
        rep = self.slider_widget.GetRepresentation()
        if rep is None:
            return
        self._syncing_slider = True
        try:
            rep.SetValue(float(frame_idx))
        finally:
            self._syncing_slider = False

    def _set_frame(self, frame_idx: int, sync_slider: bool = True):
        frame_idx = max(0, min(int(frame_idx), len(self.provider.times) - 1))
        self.current_idx = frame_idx
        self.cloud[self.provider.scalar_name] = self.provider.get_values(frame_idx)
        self.cloud.Modified()

        if self.cfg.rotate_camera:
            angle = (2.0 * np.pi * frame_idx) / max(1, len(self.provider.times))
            self.plotter.camera_position = _camera_position(
                center=self.provider.center,
                radius=self.camera_radius,
                z_value=self.camera_height,
                angle=angle,
            )

        t_val = float(self.provider.times[frame_idx])
        self.plotter.add_text(
            f"t = {t_val:.6f} | frame {frame_idx + 1}/{len(self.provider.times)}",
            name="time_info",
            position="upper_left",
            font_size=11,
        )
        if sync_slider:
            self._set_slider_value(frame_idx)
        self.plotter.render()

    def _on_slider(self, slider_value: float):
        if self._syncing_slider:
            return
        self._set_frame(int(round(slider_value)), sync_slider=False)

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.completed_loops = 0
        state = "Playing" if self.playing else "Paused"
        self._set_status_text(state)

    def _on_play_checkbox(self, checked: bool):
        self.playing = bool(checked)
        if self.playing:
            self.completed_loops = 0
        self._set_status_text("Playing" if self.playing else "Paused")

    def _on_keypress_observer(self, _obj=None, _event=None):
        if self.plotter is None or self.plotter.iren is None:
            return
        try:
            key_sym = self.plotter.iren.interactor.GetKeySym()
        except Exception:
            return
        key = str(key_sym).lower()
        if key in {"space", "p"}:
            self._toggle_play()
        elif key in {"right", "n", "period"}:
            self._step_forward()
        elif key in {"left", "b", "comma"}:
            self._step_backward()
        elif key in {"r", "home"}:
            self._restart()

    def _timer_tick(self, _step):
        if not self.playing:
            return
        try:
            next_idx = self.current_idx + 1
            if next_idx >= len(self.provider.times):
                self.completed_loops += 1
                if self.completed_loops >= self.max_cycles:
                    self.playing = False
                    self._set_status_text("Finished loops")
                    return
                next_idx = 0
            self._set_frame(next_idx)
        except Exception as exc:
            self.playing = False
            self._set_status_text(f"Playback error: {exc}")
            print(f"[PyVistaViewer] timer callback error: {exc}")

    def _step_forward(self):
        self._set_frame((self.current_idx + 1) % len(self.provider.times))

    def _step_backward(self):
        self._set_frame((self.current_idx - 1) % len(self.provider.times))

    def _restart(self):
        self.playing = False
        self.completed_loops = 0
        self._set_frame(0)
        self._set_status_text("Paused")

    def run(self):
        self.plotter = pv.Plotter(window_size=(1400, 900))
        self.plotter.set_background("#0f1b2a")
        self.cloud = pv.PolyData(self.provider.coords)
        self.cloud[self.provider.scalar_name] = self.provider.get_values(0)

        self.plotter.add_mesh(
            self.cloud,
            scalars=self.provider.scalar_name,
            cmap="turbo",
            clim=list(self.provider.scalar_range),
            point_size=float(self.cfg.point_size),
            render_points_as_spheres=True,
            lighting=False,
        )
        if self.cfg.rotate_camera:
            self.plotter.camera_position = _camera_position(
                center=self.provider.center,
                radius=self.camera_radius,
                z_value=self.camera_height,
                angle=0.0,
            )

        self.plotter.add_axes()
        self.plotter.add_text(
            "Play: checkbox or Space/P | Step: Left/Right or B/N | Restart: R/Home",
            name="help_text",
            position="upper_right",
            font_size=10,
        )
        self.slider_widget = self.plotter.add_slider_widget(
            self._on_slider,
            rng=[0, len(self.provider.times) - 1],
            value=0,
            title="Frame",
            pointa=(0.025, 0.1),
            pointb=(0.40, 0.1),
            interaction_event="always",
        )
        self.plotter.add_checkbox_button_widget(
            self._on_play_checkbox,
            value=False,
            position=(10.0, 12.0),
            size=28,
            border_size=2,
            color_on="#2eb872",
            color_off="#6b7280",
            background_color="#0f1b2a",
        )
        self.plotter.add_text("Play", name="play_label", position=(44, 16), font_size=10)
        self.plotter.iren.add_observer("KeyPressEvent", self._on_keypress_observer)
        self.plotter.add_timer_event(max_steps=10_000_000, duration=int(self.cfg.frame_ms), callback=self._timer_tick)
        self._set_frame(0)
        self._set_status_text("Paused")
        self.plotter.show(title="ROM Native Viewer (PyVista)")


class MatplotlibViewer:
    def __init__(self, provider: FrameProvider):
        self.provider = provider
        self.cfg = provider.cfg
        self.current_idx = 0
        self.playing = False
        self.completed_loops = 0
        self.max_cycles = max(1, int(self.cfg.loop_cycles))
        self.base_azim = 25.0

        self.fig = None
        self.ax = None
        self.scatter = None
        self.slider = None
        self.timer = None
        self.title = None

    def _set_frame(self, frame_idx: int):
        frame_idx = max(0, min(int(frame_idx), len(self.provider.times) - 1))
        self.current_idx = frame_idx
        vals = self.provider.get_values(frame_idx)

        self.scatter.set_array(vals)
        self.scatter.set_clim(*self.provider.scalar_range)

        if self.cfg.rotate_camera:
            azim = self.base_azim + (360.0 * frame_idx / max(1, len(self.provider.times)))
            self.ax.view_init(elev=22.0, azim=azim)

        t_val = float(self.provider.times[frame_idx])
        self.title.set_text(f"ROM Native Viewer (Matplotlib) | t={t_val:.6f} | frame {frame_idx + 1}/{len(self.provider.times)}")
        self.fig.canvas.draw_idle()

    def _on_slider(self, value):
        self._set_frame(int(round(value)))

    def _toggle_play(self, _event=None):
        self.playing = not self.playing
        if self.playing:
            self.completed_loops = 0

    def _step_forward(self, _event=None):
        self._set_frame((self.current_idx + 1) % len(self.provider.times))
        if self.slider is not None:
            self.slider.set_val(self.current_idx)

    def _step_backward(self, _event=None):
        self._set_frame((self.current_idx - 1) % len(self.provider.times))
        if self.slider is not None:
            self.slider.set_val(self.current_idx)

    def _restart(self, _event=None):
        self.completed_loops = 0
        self._set_frame(0)
        if self.slider is not None:
            self.slider.set_val(0)

    def _on_key(self, event):
        if event.key == " ":
            self._toggle_play()
        elif event.key == "right":
            self._step_forward()
        elif event.key == "left":
            self._step_backward()
        elif event.key == "r":
            self._restart()

    def _on_timer(self):
        if not self.playing:
            return
        next_idx = self.current_idx + 1
        if next_idx >= len(self.provider.times):
            self.completed_loops += 1
            if self.completed_loops >= self.max_cycles:
                self.playing = False
                return
            next_idx = 0
        self._set_frame(next_idx)
        if self.slider is not None:
            self.slider.set_val(next_idx)

    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        coords = self.provider.coords
        vals0 = self.provider.get_values(0)

        self.fig = plt.figure(figsize=(13, 9))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.scatter = self.ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=vals0,
            cmap="turbo",
            s=float(self.cfg.point_size),
            vmin=self.provider.scalar_range[0],
            vmax=self.provider.scalar_range[1],
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        try:
            span = np.ptp(coords, axis=0)
            self.ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2] if span[2] > 0 else 1.0)))
        except Exception:
            pass
        self.title = self.ax.set_title("")
        self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.7, label=self.cfg.variable)

        self.fig.subplots_adjust(bottom=0.22)
        slider_ax = self.fig.add_axes([0.10, 0.10, 0.46, 0.03])
        self.slider = Slider(
            slider_ax,
            "Frame",
            0,
            len(self.provider.times) - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        btn_prev = Button(self.fig.add_axes([0.60, 0.08, 0.08, 0.05]), "Prev")
        btn_play = Button(self.fig.add_axes([0.70, 0.08, 0.08, 0.05]), "Play/Pause")
        btn_next = Button(self.fig.add_axes([0.80, 0.08, 0.08, 0.05]), "Next")
        btn_reset = Button(self.fig.add_axes([0.90, 0.08, 0.08, 0.05]), "Restart")
        btn_prev.on_clicked(self._step_backward)
        btn_play.on_clicked(self._toggle_play)
        btn_next.on_clicked(self._step_forward)
        btn_reset.on_clicked(self._restart)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(self.cfg.frame_ms))
        self.timer.add_callback(self._on_timer)
        self.timer.start()

        self._set_frame(0)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Native desktop viewer for ROM point-cloud playback")
    parser.add_argument("--models-dir", type=Path, default=ROOT_DIR / "models")
    parser.add_argument("--processed-dir", type=Path, default=ROOT_DIR / "data" / "processed")
    parser.add_argument("--mode", dest="mode_name", default="pod")
    parser.add_argument("--trainer", dest="trainer_name", default="rbf")
    parser.add_argument("--variable", choices=VARIABLE_CHOICES, default="velocity")
    parser.add_argument("--start-time", type=float, default=0.005)
    parser.add_argument("--end-time", type=float, default=0.050)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--max-points", type=int, default=150000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--loop-cycles", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=70)
    parser.add_argument("--rotate-camera", action="store_true")
    parser.add_argument("--camera-radius-scale", type=float, default=1.8)
    parser.add_argument("--camera-height-scale", type=float, default=0.6)
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and data loading without opening GUI.")
    return parser.parse_args()


def _resolve_backend(requested: str):
    if requested == "auto":
        return "pyvista" if pv is not None else "matplotlib"
    return requested


def main():
    args = parse_args()
    if args.end_time <= args.start_time:
        raise SystemExit("Invalid range: end-time must be larger than start-time.")
    if args.frames < 2:
        raise SystemExit("Invalid value: frames must be >= 2.")

    backend = _resolve_backend(args.backend)
    if backend == "pyvista" and pv is None:
        raise SystemExit(
            "PyVista backend requested but pyvista is not installed. "
            "Install with: python -m pip install pyvista "
            "or run with --backend matplotlib"
        )

    cfg = ViewerConfig(
        models_dir=Path(args.models_dir),
        processed_dir=Path(args.processed_dir),
        mode_name=args.mode_name,
        trainer_name=args.trainer_name,
        variable=args.variable,
        start_time=float(args.start_time),
        end_time=float(args.end_time),
        frames=int(args.frames),
        max_points=int(args.max_points),
        point_size=float(args.point_size),
        loop_cycles=int(args.loop_cycles),
        frame_ms=int(args.frame_ms),
        rotate_camera=bool(args.rotate_camera),
        camera_radius_scale=float(args.camera_radius_scale),
        camera_height_scale=float(args.camera_height_scale),
        backend=backend,
        dry_run=bool(args.dry_run),
    )
    provider = FrameProvider(cfg)
    print("Native viewer config:", provider.summary())

    if cfg.dry_run:
        # Validate that frames are computable without launching UI.
        provider.get_values(0)
        provider.get_values(len(provider.times) - 1)
        print("Dry run success.")
        return

    if backend == "pyvista":
        PyVistaViewer(provider).run()
    else:
        MatplotlibViewer(provider).run()


if __name__ == "__main__":
    main()
