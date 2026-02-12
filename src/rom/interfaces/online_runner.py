from abc import ABC, abstractmethod


class OnlineRunner(ABC):
    @abstractmethod
    def load_artifacts(self, mode_path: str, model_path: str):
        """Load mode/model artifacts required for online inference."""

    @abstractmethod
    def step(self, online_input):
        """Run a single online step and return reconstructed output."""
