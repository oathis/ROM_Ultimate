from pathlib import Path


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
