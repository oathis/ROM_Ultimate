from abc import ABC, abstractmethod


class ModeBuilder(ABC):
    @abstractmethod
    def fit(self, snapshots):
        """Fit reduced-order modes from high-dimensional snapshots."""

    @abstractmethod
    def transform(self, snapshots):
        """Project snapshots into latent coordinates."""

    @abstractmethod
    def save(self, path: str):
        """Persist trained mode artifacts."""
