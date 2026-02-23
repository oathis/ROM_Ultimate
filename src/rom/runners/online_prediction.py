import json
from pathlib import Path

import numpy as np
import pandas as pd

from rom.core.factory import load_trainer
from rom.interfaces.online_runner import OnlineRunner


class OnlinePredictionRunner(OnlineRunner):
    def __init__(self, models_dir: Path, mode_name: str = "pod", trainer_name: str = "rbf"):
        self.models_dir = Path(models_dir)
        self.mode_name = mode_name
        self.trainer_name = trainer_name
        self.models = {}
        self.reconstructors = {}
        self.variables = []

        self.load_artifacts()

    def _load_pod_artifacts(self, var_dir: Path):
        modes_path = var_dir / "pod_modes.npy"
        mean_path = var_dir / "pod_mean.npy"
        if not modes_path.exists() or not mean_path.exists():
            return None
        return {
            "type": "pod",
            "modes": np.load(modes_path),
            "mean": np.load(mean_path),
        }

    def _load_dmd_artifacts(self, var_dir: Path):
        modes_path = var_dir / "dmd_modes.npy"
        eigs_path = var_dir / "dmd_eigs.npy"
        omega_path = var_dir / "dmd_omega.npy"
        amps_path = var_dir / "dmd_amplitudes.npy"
        meta_path = var_dir / "dmd_meta.json"

        required = [modes_path, eigs_path, amps_path, meta_path]
        if not all(path.exists() for path in required):
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dt = float(meta.get("dt", 0.0))
        t0 = float(meta.get("t0", 0.0))
        if dt <= 0:
            raise ValueError(f"Invalid DMD dt in metadata: {meta_path}")

        eigs = np.asarray(np.load(eigs_path), dtype=np.complex128)
        if omega_path.exists():
            omega = np.asarray(np.load(omega_path), dtype=np.complex128)
        else:
            omega = np.log(eigs) / dt

        return {
            "type": "dmd",
            "modes": np.asarray(np.load(modes_path), dtype=np.complex128),
            "eigs": eigs,
            "omega": omega,
            "amplitudes": np.asarray(np.load(amps_path), dtype=np.complex128),
            "dt": dt,
            "t0": t0,
        }

    def load_artifacts(self, mode_path=None, model_path=None):
        """
        Load mode artifacts and optional offline models for all variables.
        Arguments are ignored; directories are discovered from models_dir.
        """
        mode_root = self.models_dir / self.mode_name
        scoped_trainer_root = self.models_dir / self.trainer_name / self.mode_name
        legacy_trainer_root = self.models_dir / self.trainer_name
        if scoped_trainer_root.exists():
            trainer_root = scoped_trainer_root
        else:
            trainer_root = legacy_trainer_root
        trainer_available = trainer_root.exists()

        if not mode_root.exists():
            raise FileNotFoundError(f"Mode directory not found: {mode_root}")
        if self.mode_name == "pod" and not trainer_available:
            raise FileNotFoundError(
                f"Trainer directory not found for POD reconstruction: {trainer_root}"
            )

        self.variables = [d.name for d in mode_root.iterdir() if d.is_dir()]
        print(f"Loading models for variables: {self.variables}")

        for var in self.variables:
            var_mode_dir = mode_root / var
            if self.mode_name == "pod":
                payload = self._load_pod_artifacts(var_mode_dir)
            elif self.mode_name == "dmd":
                payload = self._load_dmd_artifacts(var_mode_dir)
            else:
                raise NotImplementedError(f"Unsupported reconstruction mode: {self.mode_name}")

            if payload is None:
                print(f"Warning: mode artifacts missing for {var}. Skipping.")
                continue
            self.reconstructors[var] = payload

            model_file = trainer_root / var / f"{self.trainer_name}_model.pkl"
            if trainer_available and model_file.exists():
                self.models[var] = load_trainer(self.trainer_name, str(model_file))
            elif self.mode_name == "pod":
                print(f"Warning: offline model missing for POD variable {var}. Skipping variable.")
                self.reconstructors.pop(var, None)

    def _reconstruct_pod(self, var: str, t_input: np.ndarray):
        model = self.models.get(var)
        if model is None:
            return None
        payload = self.reconstructors[var]
        coeffs = np.asarray(model.predict(t_input))
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        field = payload["mean"] + payload["modes"] @ coeffs.T
        return np.asarray(field).reshape(-1)

    def _reconstruct_dmd(self, var: str, t: float):
        payload = self.reconstructors[var]
        tau = float(t) - float(payload["t0"])
        dynamics = payload["amplitudes"] * np.exp(payload["omega"] * tau)
        field = payload["modes"] @ dynamics
        field_real = np.real_if_close(field, tol=1e5)
        if np.iscomplexobj(field_real):
            field_real = field_real.real
        return np.asarray(field_real, dtype=np.float64).reshape(-1)

    def step(self, t: float) -> pd.DataFrame:
        """
        Predict full field at time t.
        """
        results = {}
        t_input = np.array([[float(t)]], dtype=np.float64)
        n_nodes = 0

        for var in self.variables:
            if var not in self.reconstructors:
                continue

            payload = self.reconstructors[var]
            if payload["type"] == "pod":
                field = self._reconstruct_pod(var, t_input)
            elif payload["type"] == "dmd":
                field = self._reconstruct_dmd(var, float(t))
            else:
                continue

            if field is None:
                continue
            if n_nodes == 0:
                n_nodes = int(field.shape[0])
            results[var] = field

        if n_nodes == 0:
            return pd.DataFrame({"nodenumber": np.array([], dtype=np.int64)})

        df = pd.DataFrame({"nodenumber": np.arange(n_nodes, dtype=np.int64)})
        for var in self.variables:
            if var not in results:
                continue

            col_name = var
            if var == "T":
                col_name = "temperature"
            elif var == "u":
                col_name = "x-velocity"
            elif var == "v":
                col_name = "y-velocity"
            elif var == "w":
                col_name = "z-velocity"
            df[col_name] = results[var]
        return df

    def add_coordinates(self, points_path: Path, df: pd.DataFrame) -> pd.DataFrame:
        if not points_path.exists() or df.empty:
            return df

        coords = np.fromfile(points_path, dtype=np.float64)
        if coords.size % 3 != 0:
            print(f"Warning: Invalid points.bin format: {points_path}")
            return df
        coords = coords.reshape(-1, 3)

        if len(coords) != len(df):
            print(f"Warning: Coordinate count {len(coords)} != Node count {len(df)}")
            return df

        df["x-coordinate"] = coords[:, 0]
        df["y-coordinate"] = coords[:, 1]
        df["z-coordinate"] = coords[:, 2]

        cols = [
            "nodenumber",
            "x-coordinate",
            "y-coordinate",
            "z-coordinate",
            "x-velocity",
            "y-velocity",
            "z-velocity",
            "temperature",
        ]
        final_cols = [c for c in cols if c in df.columns]
        return df[final_cols]
