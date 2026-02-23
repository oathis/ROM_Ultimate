from pathlib import Path

import numpy as np

from rom.interfaces.mode_builder import ModeBuilder
from rom.modes.pod import POD


class PODBuilder(ModeBuilder):
    def __init__(self, rank=None, energy_threshold=0.999):
        self.rank = rank
        self.energy_threshold = energy_threshold
        self._pod = None

    def fit(self, snapshots):
        X = np.asarray(snapshots, dtype=np.float64)
        self._pod = POD(n_modes=self.rank, energy_threshold=self.energy_threshold)
        self._pod.fit(X)
        return self

    def transform(self, snapshots):
        if self._pod is None:
            raise RuntimeError("PODBuilder is not fitted yet.")
        X = np.asarray(snapshots, dtype=np.float64)
        centered = X - self._pod.mean
        # Return shape (n_samples, n_modes) so offline trainers can consume it directly.
        return (self._pod.modes.T @ centered).T

    def save(self, path: str):
        if self._pod is None:
            raise RuntimeError("PODBuilder is not fitted yet.")
        self._pod.save(Path(path), prefix="pod")
