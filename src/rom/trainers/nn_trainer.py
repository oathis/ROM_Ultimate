import pickle

import numpy as np

from rom.interfaces.offline_trainer import OfflineTrainer


class NNTrainer(OfflineTrainer):
    """
    Baseline dense-model surrogate implemented as linear regression.
    This keeps trainer swapping functional without a torch dependency.
    """

    def __init__(self, hidden_dims=None, activation="relu", epochs=200):
        self.hidden_dims = hidden_dims or [128, 128]
        self.activation = activation
        self.epochs = epochs
        self._weights = None

    def fit(self, x_train, y_train):
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
        self._weights = np.linalg.pinv(x_aug) @ y
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
