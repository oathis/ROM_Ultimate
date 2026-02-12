from abc import ABC, abstractmethod


class OfflineTrainer(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        """Train mapping between inputs and latent state."""

    @abstractmethod
    def predict(self, x):
        """Inference in latent space."""

    @abstractmethod
    def save(self, path: str):
        """Persist trained model artifacts."""
