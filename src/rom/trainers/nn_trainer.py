import pickle
from typing import Iterable

import numpy as np
from scipy.special import erf

from rom.interfaces.offline_trainer import OfflineTrainer


class NNTrainer(OfflineTrainer):
    """
    Multi-layer perceptron (MLP) regressor for latent trajectory modeling.
    Input:  x (n_samples, n_features)
    Output: y (n_samples, n_targets)
    """

    def __init__(
        self,
        hidden_dims=None,
        activation="tanh",
        epochs=600,
        learning_rate=1e-3,
        batch_size=256,
        weight_decay=1e-6,
        patience=80,
        tol=1e-8,
        random_state=42,
    ):
        self.hidden_dims = self._normalize_hidden_dims(hidden_dims or [128, 128])
        self.activation = str(activation).strip().lower()
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.tol = float(tol)
        self.random_state = int(random_state)

        # Backward-compatibility for models saved by the old linear baseline.
        self._weights = None

        self._layer_weights = None
        self._layer_biases = None
        self._x_mean = None
        self._x_scale = None
        self._y_mean = None
        self._y_scale = None
        self._train_loss = None
        self._epochs_trained = 0

    @staticmethod
    def _normalize_hidden_dims(hidden_dims: Iterable[int] | None):
        if hidden_dims is None:
            return [128, 128]
        if isinstance(hidden_dims, str):
            tokens = [tok.strip() for tok in hidden_dims.split(",") if tok.strip()]
            hidden_dims = [int(tok) for tok in tokens]
        dims = []
        for item in hidden_dims:
            value = int(item)
            if value <= 0:
                raise ValueError(f"hidden_dims entries must be positive, got {value}")
            dims.append(value)
        return dims

    def _activation_forward(self, z: np.ndarray):
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        if self.activation == "linear":
            return z
        if self.activation == "gelu":
            return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activation_grad(self, z: np.ndarray):
        if self.activation == "relu":
            return (z > 0.0).astype(np.float64)
        if self.activation == "tanh":
            t = np.tanh(z)
            return 1.0 - t * t
        if self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            return s * (1.0 - s)
        if self.activation == "linear":
            return np.ones_like(z, dtype=np.float64)
        if self.activation == "gelu":
            return 0.5 * (1.0 + erf(z / np.sqrt(2.0))) + (z * np.exp(-0.5 * z * z)) / np.sqrt(2.0 * np.pi)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _init_layers(self, n_in: int, n_out: int):
        rng = np.random.default_rng(self.random_state)
        dims = [int(n_in), *self.hidden_dims, int(n_out)]
        weights = []
        biases = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            is_hidden = i < len(dims) - 2

            if is_hidden and self.activation in {"relu", "gelu"}:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            w = rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros((fan_out,), dtype=np.float64)
            weights.append(w)
            biases.append(b)
        return weights, biases

    def _forward(self, x_norm: np.ndarray):
        a_values = [x_norm]
        z_values = []
        a = x_norm
        n_layers = len(self._layer_weights)
        for i in range(n_layers):
            z = a @ self._layer_weights[i] + self._layer_biases[i]
            z_values.append(z)
            if i < n_layers - 1:
                a = self._activation_forward(z)
            else:
                a = z
            a_values.append(a)
        return a_values, z_values

    def fit(self, x_train, y_train):
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x/y sample mismatch: x={x.shape}, y={y.shape}")

        n_samples, n_in = x.shape
        n_out = y.shape[1]
        if n_samples < 2:
            raise ValueError("NNTrainer requires at least 2 samples.")

        self._x_mean = np.mean(x, axis=0, keepdims=True)
        self._x_scale = np.std(x, axis=0, keepdims=True)
        self._x_scale[self._x_scale < 1e-12] = 1.0
        self._y_mean = np.mean(y, axis=0, keepdims=True)
        self._y_scale = np.std(y, axis=0, keepdims=True)
        self._y_scale[self._y_scale < 1e-12] = 1.0

        x_norm = (x - self._x_mean) / self._x_scale
        y_norm = (y - self._y_mean) / self._y_scale

        self._layer_weights, self._layer_biases = self._init_layers(n_in=n_in, n_out=n_out)
        self._weights = None

        m_w = [np.zeros_like(w) for w in self._layer_weights]
        v_w = [np.zeros_like(w) for w in self._layer_weights]
        m_b = [np.zeros_like(b) for b in self._layer_biases]
        v_b = [np.zeros_like(b) for b in self._layer_biases]

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = 0

        batch_size = max(1, min(int(self.batch_size), n_samples))
        rng = np.random.default_rng(self.random_state)

        best_loss = float("inf")
        best_params = None
        no_improve = 0

        n_layers = len(self._layer_weights)
        out_dim = max(1, int(n_out))

        for epoch in range(max(1, int(self.epochs))):
            order = rng.permutation(n_samples)
            epoch_loss_sum = 0.0

            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                xb = x_norm[idx]
                yb = y_norm[idx]
                bs = xb.shape[0]

                a_values, z_values = self._forward(xb)
                pred = a_values[-1]
                err = pred - yb

                data_loss = float(np.mean(err * err))
                reg_loss = float(self.weight_decay * sum(np.sum(w * w) for w in self._layer_weights))
                batch_loss = data_loss + reg_loss
                epoch_loss_sum += batch_loss * bs

                dA = (2.0 / (bs * out_dim)) * err
                grad_w = [None] * n_layers
                grad_b = [None] * n_layers

                for layer in range(n_layers - 1, -1, -1):
                    if layer == n_layers - 1:
                        dZ = dA
                    else:
                        dZ = dA * self._activation_grad(z_values[layer])

                    a_prev = a_values[layer]
                    grad_w[layer] = a_prev.T @ dZ + (2.0 * self.weight_decay * self._layer_weights[layer])
                    grad_b[layer] = np.sum(dZ, axis=0)
                    dA = dZ @ self._layer_weights[layer].T

                step += 1
                for i in range(n_layers):
                    m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * grad_w[i]
                    v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (grad_w[i] * grad_w[i])
                    m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grad_b[i]
                    v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (grad_b[i] * grad_b[i])

                    m_w_hat = m_w[i] / (1.0 - beta1**step)
                    v_w_hat = v_w[i] / (1.0 - beta2**step)
                    m_b_hat = m_b[i] / (1.0 - beta1**step)
                    v_b_hat = v_b[i] / (1.0 - beta2**step)

                    self._layer_weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                    self._layer_biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            epoch_loss = epoch_loss_sum / n_samples
            if epoch_loss + self.tol < best_loss:
                best_loss = epoch_loss
                best_params = (
                    [w.copy() for w in self._layer_weights],
                    [b.copy() for b in self._layer_biases],
                )
                no_improve = 0
            else:
                no_improve += 1
                if self.patience > 0 and no_improve >= self.patience:
                    self._epochs_trained = epoch + 1
                    break

            self._epochs_trained = epoch + 1

        if best_params is not None:
            self._layer_weights = best_params[0]
            self._layer_biases = best_params[1]
        self._train_loss = float(best_loss)
        return self

    def _predict_linear_legacy(self, x_arr: np.ndarray):
        x_aug = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
        return x_aug @ self._weights

    def predict(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        # Backward compatibility: support models trained by the old linear baseline.
        if getattr(self, "_layer_weights", None) is None:
            if getattr(self, "_weights", None) is None:
                raise RuntimeError("Model not fitted yet.")
            return self._predict_linear_legacy(x_arr)

        x_norm = (x_arr - self._x_mean) / self._x_scale
        a = x_norm
        n_layers = len(self._layer_weights)
        for i in range(n_layers):
            z = a @ self._layer_weights[i] + self._layer_biases[i]
            if i < n_layers - 1:
                a = self._activation_forward(z)
            else:
                a = z
        y_norm = a
        return y_norm * self._y_scale + self._y_mean

    def save(self, path: str):
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
        # Ensure legacy artifacts remain usable after class upgrades.
        if not hasattr(obj, "_layer_weights"):
            obj._layer_weights = None
        if not hasattr(obj, "_layer_biases"):
            obj._layer_biases = None
        if not hasattr(obj, "_x_mean"):
            obj._x_mean = None
        if not hasattr(obj, "_x_scale"):
            obj._x_scale = None
        if not hasattr(obj, "_y_mean"):
            obj._y_mean = None
        if not hasattr(obj, "_y_scale"):
            obj._y_scale = None
        return obj
