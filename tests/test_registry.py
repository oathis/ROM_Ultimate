from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.runner_registry import REGISTRY as RUNNER_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


def test_registry_keys_exist():
    assert "pod" in MODE_REGISTRY
    assert "rbf" in TRAINER_REGISTRY
    assert "static" in RUNNER_REGISTRY
