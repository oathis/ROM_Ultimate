import json
from pathlib import Path

import numpy as np

from rom.interfaces.mode_builder import ModeBuilder


class DMDBuilder(ModeBuilder):
    """
    Exact DMD (Dynamic Mode Decomposition).

    Snapshot matrix convention:
    - Input snapshots X has shape (n_features, n_snapshots)
    - Time evolves along the 2nd axis
    """

    def __init__(self, rank=32, dt=None, time_values=None):
        self.rank = rank
        self.dt = dt
        self.time_values = time_values

        self._modes = None
        self._eigvals = None
        self._omega = None
        self._amplitudes = None
        self._t0 = None
        self._dt = None
        self._train_times = None
        self._rank_effective = None

    def _resolve_time_grid(self, n_snapshots: int):
        if self.time_values is not None:
            times = np.asarray(self.time_values, dtype=np.float64).reshape(-1)
            if times.size != n_snapshots:
                raise ValueError(
                    f"time_values length ({times.size}) must match snapshot count ({n_snapshots})."
                )
            if times.size < 2:
                raise ValueError("DMD requires at least two time points.")

            diffs = np.diff(times)
            if np.any(diffs <= 0):
                raise ValueError("time_values must be strictly increasing.")

            dt_from_data = float(np.median(diffs))
            tol = max(1e-12, abs(dt_from_data) * 1e-6)
            if float(np.max(np.abs(diffs - dt_from_data))) > tol:
                raise ValueError(
                    "Non-uniform time grid detected. Current DMD implementation expects constant dt."
                )

            if self.dt is None or float(self.dt) <= 0.0:
                dt_eff = dt_from_data
            else:
                dt_eff = float(self.dt)

            return float(times[0]), dt_eff, times

        if self.dt is None or float(self.dt) <= 0.0:
            raise ValueError("DMDBuilder requires positive dt when time_values is not provided.")

        dt_eff = float(self.dt)
        times = np.arange(n_snapshots, dtype=np.float64) * dt_eff
        return 0.0, dt_eff, times

    def fit(self, snapshots):
        X = np.asarray(snapshots, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"DMDBuilder expects 2D snapshots, got shape={X.shape}")
        if X.shape[1] < 2:
            raise ValueError("DMD requires at least 2 snapshots.")

        n_features, n_snapshots = X.shape
        t0, dt_eff, times = self._resolve_time_grid(n_snapshots)

        X1 = X[:, :-1]
        X2 = X[:, 1:]

        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        if s.size == 0:
            raise ValueError("SVD failed: no singular values.")

        max_rank = int(s.size)
        requested_rank = max_rank if self.rank is None else max(1, min(int(self.rank), max_rank))

        svd_tol = np.finfo(np.float64).eps * max(X1.shape) * float(s[0])
        valid = np.where(s[:requested_rank] > svd_tol)[0]
        if valid.size == 0:
            raise ValueError(
                "All retained singular values are numerically zero. "
                "Try reducing rank or checking input data."
            )
        rank_eff = int(valid[-1]) + 1

        U_r = U[:, :rank_eff]
        s_r = s[:rank_eff]
        V_r = Vh.conj().T[:, :rank_eff]
        s_r_inv = np.diag(1.0 / s_r)

        A_tilde = U_r.conj().T @ X2 @ V_r @ s_r_inv
        eigvals, W = np.linalg.eig(A_tilde)
        modes = X2 @ V_r @ s_r_inv @ W
        amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
        omega = np.log(eigvals) / dt_eff

        self._modes = np.asarray(modes, dtype=np.complex128)
        self._eigvals = np.asarray(eigvals, dtype=np.complex128)
        self._omega = np.asarray(omega, dtype=np.complex128)
        self._amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        self._t0 = float(t0)
        self._dt = float(dt_eff)
        self._train_times = np.asarray(times, dtype=np.float64)
        self._rank_effective = int(rank_eff)
        return self

    def transform(self, snapshots):
        """
        Return real-valued latent features for offline trainers.

        We project snapshots onto complex DMD modes and concatenate
        real/imag parts: shape (n_snapshots, 2 * rank_effective).
        """
        if self._modes is None:
            raise RuntimeError("DMDBuilder is not fitted yet.")

        X = np.asarray(snapshots, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"DMDBuilder expects 2D snapshots, got shape={X.shape}")

        coeffs = np.linalg.lstsq(self._modes, X, rcond=None)[0]  # (rank, n_snapshots), complex
        feats = np.hstack([coeffs.real.T, coeffs.imag.T])  # (n_snapshots, 2*rank)
        return np.asarray(feats, dtype=np.float64)

    def reconstruct_at_time(self, t: float):
        if self._modes is None:
            raise RuntimeError("DMDBuilder is not fitted yet.")
        tau = float(t) - float(self._t0)
        dynamics = self._amplitudes * np.exp(self._omega * tau)
        field = self._modes @ dynamics
        return np.real_if_close(field, tol=1e5)

    def save(self, path: str):
        if self._modes is None:
            raise RuntimeError("DMDBuilder is not fitted yet.")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "dmd_modes.npy", self._modes)
        np.save(out_dir / "dmd_eigs.npy", self._eigvals)
        np.save(out_dir / "dmd_omega.npy", self._omega)
        np.save(out_dir / "dmd_amplitudes.npy", self._amplitudes)
        np.save(out_dir / "dmd_train_times.npy", self._train_times)

        metadata = {
            "rank_requested": None if self.rank is None else int(self.rank),
            "rank_effective": int(self._rank_effective),
            "dt": float(self._dt),
            "t0": float(self._t0),
            "n_train_times": int(self._train_times.size),
        }
        (out_dir / "dmd_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
