import pickle

import numpy as np

from rom.interfaces.offline_trainer import OfflineTrainer


class ProjectionTrainer(OfflineTrainer):
    def __init__(self, solver="galerkin", stabilization=False, ridge=1e-8):
        self.solver = solver
        self.stabilization = stabilization
        self.ridge = ridge
        self._weights = None

    def fit(self, x_train, y_train):
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
        gram = x_aug.T @ x_aug + self.ridge * np.eye(x_aug.shape[1])
        self._weights = np.linalg.solve(gram, x_aug.T @ y)
        return self

    def predict(self, x):
        if self._weights is None:
            raise RuntimeError("Model not fitted yet.")
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        x_aug = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
        return x_aug @ self._weights

    def save(self, path: str):
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            return pickle.load(fp)
