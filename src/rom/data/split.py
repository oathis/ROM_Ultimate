from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np


RAW_CSV_PATTERN = re.compile(r"^xresult-(.+)\.csv$", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_split(data, ratio=0.8):
    """
    Generic sequence split retained for backward compatibility.
    """
    cutoff = int(len(data) * ratio)
    return data[:cutoff], data[cutoff:]


def parse_time_from_filename(filename: str) -> float:
    match = RAW_CSV_PATTERN.match(str(filename).strip())
    if not match:
        raise ValueError(f"Unsupported raw filename format: {filename}")
    return float(match.group(1))


def discover_raw_csv_series(raw_dir: Path):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    series = []
    for csv_path in raw_dir.glob("xresult-*.csv"):
        try:
            t = parse_time_from_filename(csv_path.name)
        except ValueError:
            continue
        series.append((t, csv_path))

    if not series:
        raise FileNotFoundError(f"No valid xresult-*.csv files found in {raw_dir}")

    series.sort(key=lambda item: item[0])
    return series


def build_time_split_indices(
    n_samples: int,
    mode: str = "extrapolation",
    test_ratio: float = 0.2,
    min_train_samples: int = 8,
):
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for split, got {n_samples}.")
    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"interpolation", "extrapolation"}:
        raise ValueError(f"Unsupported split mode: {mode}. Use interpolation or extrapolation.")

    min_train = max(1, int(min_train_samples))
    if n_samples <= min_train:
        raise ValueError(f"Not enough samples ({n_samples}) for min_train_samples={min_train}.")

    test_count = int(round(n_samples * float(test_ratio)))
    test_count = max(1, test_count)
    test_count = min(test_count, n_samples - min_train)

    all_idx = np.arange(n_samples, dtype=np.int64)
    if normalized_mode == "extrapolation":
        split_at = n_samples - test_count
        train_idx = all_idx[:split_at]
        test_idx = all_idx[split_at:]
        return train_idx, test_idx

    if n_samples > 2:
        candidates = np.arange(1, n_samples - 1, dtype=np.int64)
    else:
        candidates = all_idx

    if candidates.size == 0:
        candidates = all_idx

    if candidates.size <= test_count:
        test_idx = candidates.copy()
    else:
        pick_positions = np.linspace(0, candidates.size - 1, num=test_count, dtype=np.int64)
        test_idx = np.unique(candidates[pick_positions])
        if test_idx.size < test_count:
            remain = np.setdiff1d(candidates, test_idx, assume_unique=False)
            need = test_count - test_idx.size
            if remain.size > 0:
                test_idx = np.sort(np.concatenate([test_idx, remain[:need]]))

    train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
    if train_idx.size < min_train:
        raise ValueError(
            f"Interpolation split failed to keep enough train samples: "
            f"{train_idx.size} < min_train_samples={min_train}"
        )
    return train_idx, test_idx


def build_raw_split_manifest(
    raw_dir: Path,
    mode: str = "extrapolation",
    test_ratio: float = 0.2,
    min_train_samples: int = 8,
    split_id: str | None = None,
):
    series = discover_raw_csv_series(raw_dir)
    times = np.asarray([t for t, _ in series], dtype=np.float64)
    files = [p.name for _, p in series]

    train_idx, test_idx = build_time_split_indices(
        n_samples=times.size,
        mode=mode,
        test_ratio=test_ratio,
        min_train_samples=min_train_samples,
    )

    manifest = {
        "schema_version": 1,
        "split_id": str(split_id).strip() if split_id else f"split-{_now_iso().replace(':', '').replace('+00:00', 'Z')}",
        "created_at": _now_iso(),
        "raw_dir": str(Path(raw_dir)),
        "split_mode": str(mode).strip().lower(),
        "test_ratio": float(test_ratio),
        "min_train_samples": int(min_train_samples),
        "n_total": int(times.size),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "all_files": files,
        "all_times": [float(v) for v in times.tolist()],
        "train_indices": [int(i) for i in train_idx.tolist()],
        "test_indices": [int(i) for i in test_idx.tolist()],
        "train_files": [files[int(i)] for i in train_idx.tolist()],
        "test_files": [files[int(i)] for i in test_idx.tolist()],
        "train_times": [float(times[int(i)]) for i in train_idx.tolist()],
        "test_times": [float(times[int(i)]) for i in test_idx.tolist()],
    }
    return manifest


def save_split_manifest(path: Path, manifest: dict):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_split_manifest(path: Path):
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_keys = {"train_files", "test_files"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Invalid split manifest, missing keys: {', '.join(missing)}")
    return payload


def subset_files_from_manifest(manifest: dict, subset: str):
    normalized = str(subset or "all").strip().lower()
    if normalized == "all":
        all_files = list(manifest.get("all_files") or [])
        if all_files:
            return all_files
        seen = set()
        merged = []
        for name in list(manifest.get("train_files") or []) + list(manifest.get("test_files") or []):
            if name in seen:
                continue
            seen.add(name)
            merged.append(name)
        return merged
    if normalized == "train":
        return list(manifest.get("train_files") or [])
    if normalized == "test":
        return list(manifest.get("test_files") or [])
    raise ValueError(f"Unsupported subset '{subset}'. Use all, train, or test.")
