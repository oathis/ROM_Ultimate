from importlib import import_module
import inspect

from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.runner_registry import REGISTRY as RUNNER_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


def _resolve_class(path_spec: str):
    module_name, class_name = path_spec.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def _instantiate(cls, kwargs: dict):
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if accepts_kwargs:
        return cls(**kwargs)

    allowed = {
        name
        for name, p in params.items()
        if name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def _build(name: str, registry: dict, kwargs: dict):
    if name not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown component '{name}'. Supported: {supported}")
    cls = _resolve_class(registry[name])
    return _instantiate(cls, kwargs)


def build_mode(name: str, **kwargs):
    return _build(name, MODE_REGISTRY, kwargs)


def build_trainer(name: str, **kwargs):
    return _build(name, TRAINER_REGISTRY, kwargs)


def build_runner(name: str, **kwargs):
    return _build(name, RUNNER_REGISTRY, kwargs)


def load_trainer(name: str, path: str):
    if name not in TRAINER_REGISTRY:
        supported = ", ".join(sorted(TRAINER_REGISTRY.keys()))
        raise ValueError(f"Unknown trainer '{name}'. Supported: {supported}")
    cls = _resolve_class(TRAINER_REGISTRY[name])
    if not hasattr(cls, "load"):
        raise NotImplementedError(f"Trainer '{name}' does not support artifact loading.")
    return cls.load(path)
