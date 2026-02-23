
import pickle
import numpy as np
from scipy.interpolate import RBFInterpolator
from rom.interfaces.offline_trainer import OfflineTrainer

class RBFTrainer(OfflineTrainer):
    def __init__(self, kernel='linear', epsilon=1.0):
        self.kernel = kernel
        self.epsilon = epsilon
        self.model = None
        self._input_shape = None
        self._output_shape = None

    def fit(self, x_train, y_train):
        """
        Train RBF network.
        
        Args:
            x_train: (n_samples, n_features) - e.g. time steps
            y_train: (n_samples, n_targets) - e.g. POD coefficients
        """
        self._input_shape = x_train.shape
        self._output_shape = y_train.shape
        
        # Scipy RBFInterpolator handles multi-dimensional y automatically
        # Kernel options: 'linear', 'thin_plate_spline', 'cubic', 'quintic', 'gaussian', etc.
        self.model = RBFInterpolator(x_train, y_train, kernel=self.kernel, epsilon=self.epsilon)
        print(f"RBFTrainer fitted with kernel={self.kernel}, epsilon={self.epsilon}")
        return self

    def predict(self, x):
        """
        Predict outputs for new inputs.
        
        Args:
            x: (n_samples, n_features)
        Returns:
            y_pred: (n_samples, n_targets)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        
        return self.model(x)

    def save(self, path: str):
        """Save the entire object using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"RBFTrainer saved to {path}")

    @staticmethod
    def load(path: str):
        """Load a saved RBFTrainer object."""
        with open(path, "rb") as f:
            return pickle.load(f)
