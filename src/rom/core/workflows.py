from pathlib import Path
import json

import numpy as np
import pandas as pd

from rom.core.factory import build_mode, build_runner, build_trainer
from rom.data.preprocess import process_transient_data
from rom.runners.online_prediction import OnlinePredictionRunner


def _discover_snapshots(processed_dir: Path):
    return sorted(processed_dir.glob("Snapshot_*.npy"))


def _resolve_eval_modes(eval_mode: str):
    normalized = str(eval_mode or "none").strip().lower()
    if normalized == "none":
        return []
    if normalized == "both":
        return ["interpolation", "extrapolation"]
    if normalized in {"interpolation", "extrapolation"}:
        return [normalized]
    raise ValueError(f"Unsupported eval_mode='{eval_mode}'. Use one of: none, interpolation, extrapolation, both")


def _build_eval_split_indices(n_samples: int, mode: str, val_ratio: float, min_train_samples: int):
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for validation split, got {n_samples}.")
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    min_train = max(1, int(min_train_samples))
    if n_samples <= min_train:
        raise ValueError(f"Not enough samples ({n_samples}) for min_train_samples={min_train}.")

    val_count = int(round(n_samples * float(val_ratio)))
    val_count = max(1, val_count)
    val_count = min(val_count, n_samples - min_train)

    all_idx = np.arange(n_samples, dtype=np.int64)
    if mode == "extrapolation":
        split_at = n_samples - val_count
        train_idx = all_idx[:split_at]
        val_idx = all_idx[split_at:]
        return train_idx, val_idx

    if mode == "interpolation":
        if n_samples > 2:
            candidates = np.arange(1, n_samples - 1, dtype=np.int64)
        else:
            candidates = all_idx

        if candidates.size == 0:
            candidates = all_idx

        if candidates.size <= val_count:
            val_idx = candidates.copy()
        else:
            pick_positions = np.linspace(0, candidates.size - 1, num=val_count, dtype=np.int64)
            val_idx = np.unique(candidates[pick_positions])
            if val_idx.size < val_count:
                remain = np.setdiff1d(candidates, val_idx, assume_unique=False)
                need = val_count - val_idx.size
                if remain.size > 0:
                    val_idx = np.sort(np.concatenate([val_idx, remain[:need]]))

        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        if train_idx.size < min_train:
            raise ValueError(
                f"Interpolation split failed to keep enough train samples: "
                f"{train_idx.size} < min_train_samples={min_train}"
            )
        return train_idx, val_idx

    raise ValueError(f"Unsupported split mode: {mode}")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    if true_arr.ndim == 1:
        true_arr = true_arr.reshape(-1, 1)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)

    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"Prediction shape mismatch: true={true_arr.shape}, pred={pred_arr.shape}")

    err = pred_arr - true_arr
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_abs_error = float(np.max(np.abs(err)))
    mean_abs_true = float(np.mean(np.abs(true_arr)))
    rel_mae = float(mae / (mean_abs_true + 1e-12))
    ss_res = float(np.sum((true_arr - pred_arr) ** 2))
    ss_tot = float(np.sum((true_arr - np.mean(true_arr)) ** 2))
    r2 = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "rel_mae": rel_mae,
        "r2": r2,
    }


def run_preprocess(raw_dir: Path, processed_dir: Path):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    process_transient_data(raw_dir, processed_dir)
    return {
        "processed_dir": str(processed_dir),
        "snapshots": [p.name for p in _discover_snapshots(processed_dir)],
    }


def run_mode_training(processed_dir: Path, models_dir: Path, mode_name: str, mode_params: dict | None = None):
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    mode_params = dict(mode_params or {})

    if mode_name == "dmd" and mode_params.get("time_values") is None:
        doe_path = processed_dir / "doe.csv"
        if doe_path.exists():
            doe = pd.read_csv(doe_path)
            time_col = str(mode_params.get("time_column", "time"))
            if time_col in doe.columns:
                mode_params["time_values"] = doe[time_col].to_numpy(dtype=np.float64)

    snapshots = _discover_snapshots(processed_dir)
    if not snapshots:
        raise FileNotFoundError(f"No Snapshot_*.npy found in {processed_dir}")

    mode_root = models_dir / mode_name
    summary = {}

    for snap_path in snapshots:
        variable = snap_path.stem.replace("Snapshot_", "")
        X = np.load(snap_path)

        mode_builder = build_mode(mode_name, **mode_params)
        mode_builder.fit(X)
        latent = np.asarray(mode_builder.transform(X))
        if latent.ndim == 1:
            latent = latent.reshape(-1, 1)

        if latent.shape[0] != X.shape[1]:
            if latent.shape[1] == X.shape[1]:
                latent = latent.T
            else:
                raise ValueError(
                    f"Latent shape {latent.shape} is incompatible with snapshots {X.shape} for '{variable}'"
                )

        out_dir = mode_root / variable
        out_dir.mkdir(parents=True, exist_ok=True)
        mode_builder.save(str(out_dir))
        np.save(out_dir / "latent.npy", latent)

        summary[variable] = {
            "snapshot_shape": tuple(X.shape),
            "latent_shape": tuple(latent.shape),
            "artifact_dir": str(out_dir),
        }
        if mode_name == "dmd":
            summary[variable]["dmd_rank_effective"] = int(getattr(mode_builder, "_rank_effective", 0) or 0)
            summary[variable]["dmd_dt"] = float(getattr(mode_builder, "_dt", 0.0) or 0.0)
            summary[variable]["dmd_t0"] = float(getattr(mode_builder, "_t0", 0.0) or 0.0)

    return summary


def run_offline_training(
    processed_dir: Path,
    models_dir: Path,
    mode_name: str,
    trainer_name: str,
    trainer_params: dict | None = None,
    input_column: str = "time",
    eval_mode: str = "none",
    val_ratio: float = 0.2,
    min_train_samples: int = 8,
):
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    trainer_params = trainer_params or {}

    doe_path = processed_dir / "doe.csv"
    if not doe_path.exists():
        raise FileNotFoundError(f"Missing DOE file: {doe_path}")

    doe = pd.read_csv(doe_path)
    if input_column not in doe.columns:
        raise ValueError(f"Column '{input_column}' is missing in {doe_path}")
    x_train = doe[input_column].to_numpy().reshape(-1, 1)
    eval_modes = _resolve_eval_modes(eval_mode)

    mode_root = models_dir / mode_name
    if not mode_root.exists():
        raise FileNotFoundError(f"Mode artifact directory not found: {mode_root}")

    summary = {}
    for var_dir in sorted([d for d in mode_root.iterdir() if d.is_dir()]):
        latent_path = var_dir / "latent.npy"
        if not latent_path.exists():
            continue

        y_train = np.load(latent_path)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_train.shape[0] != x_train.shape[0]:
            if y_train.shape[1] == x_train.shape[0]:
                y_train = y_train.T
            else:
                raise ValueError(
                    f"Sample count mismatch for '{var_dir.name}': latent {y_train.shape}, inputs {x_train.shape}"
                )

        validation_summary = {}
        for mode in eval_modes:
            train_idx, val_idx = _build_eval_split_indices(
                n_samples=x_train.shape[0],
                mode=mode,
                val_ratio=float(val_ratio),
                min_train_samples=int(min_train_samples),
            )
            x_fit = x_train[train_idx]
            y_fit = y_train[train_idx]
            x_val = x_train[val_idx]
            y_val = y_train[val_idx]

            eval_trainer = build_trainer(trainer_name, **trainer_params)
            eval_trainer.fit(x_fit, y_fit)
            y_pred = np.asarray(eval_trainer.predict(x_val))
            metrics = _regression_metrics(y_val, y_pred)
            validation_summary[mode] = {
                "train_shape": tuple(x_fit.shape),
                "val_shape": tuple(x_val.shape),
                "train_idx_range": [int(train_idx.min()), int(train_idx.max())],
                "val_idx_range": [int(val_idx.min()), int(val_idx.max())],
                "metrics": metrics,
            }

        trainer = build_trainer(trainer_name, **trainer_params)
        trainer.fit(x_train, y_train)

        out_dir = models_dir / trainer_name / mode_name / var_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"{trainer_name}_model.pkl"
        trainer.save(str(model_path))

        summary[var_dir.name] = {
            "input_shape": tuple(x_train.shape),
            "target_shape": tuple(y_train.shape),
            "model_path": str(model_path),
            "model_namespace": f"{trainer_name}/{mode_name}/{var_dir.name}",
            "validation": validation_summary,
        }

    if not summary:
        raise RuntimeError(f"No offline models were trained from {mode_root}")
    return summary


def run_online_prediction(
    time_value: float,
    models_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    mode_name: str,
    trainer_name: str,
    runner_name: str = "reconstruction",
):
    models_dir = Path(models_dir)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    if runner_name == "reconstruction":
        runner = OnlinePredictionRunner(models_dir=models_dir, mode_name=mode_name, trainer_name=trainer_name)
        df = runner.step(float(time_value))
        df = runner.add_coordinates(processed_dir / "points.bin", df)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"predicted_result-{time_value}.csv"
        df.to_csv(output_path, index=False)
        return {
            "runner": runner_name,
            "output_path": str(output_path),
            "rows": int(len(df)),
            "output_type": "csv",
        }

    mode_path = models_dir / mode_name
    mode_scoped_model_path = models_dir / trainer_name / mode_name
    legacy_model_path = models_dir / trainer_name
    model_path = mode_scoped_model_path if mode_scoped_model_path.exists() else legacy_model_path
    runner = build_runner(runner_name)
    runner.load_artifacts(str(mode_path), str(model_path))
    payload = runner.step(float(time_value))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predicted_result-{time_value}.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)

    rows = len(payload) if hasattr(payload, "__len__") else None
    return {
        "runner": runner_name,
        "output_path": str(output_path),
        "rows": rows,
        "output_type": "json",
    }


def run_full_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    predictions_dir: Path,
    mode_name: str = "pod",
    trainer_name: str = "rbf",
    runner_name: str = "reconstruction",
    mode_params: dict | None = None,
    trainer_params: dict | None = None,
    predict_time: float | None = None,
    offline_eval_mode: str = "none",
    offline_val_ratio: float = 0.2,
    offline_min_train_samples: int = 8,
):
    preprocess_summary = run_preprocess(raw_dir, processed_dir)
    mode_summary = run_mode_training(processed_dir, models_dir, mode_name, mode_params)
    offline_summary = run_offline_training(
        processed_dir=processed_dir,
        models_dir=models_dir,
        mode_name=mode_name,
        trainer_name=trainer_name,
        trainer_params=trainer_params,
        eval_mode=offline_eval_mode,
        val_ratio=offline_val_ratio,
        min_train_samples=offline_min_train_samples,
    )

    online_summary = None
    if predict_time is not None:
        online_summary = run_online_prediction(
            predict_time,
            models_dir=models_dir,
            processed_dir=processed_dir,
            output_dir=predictions_dir,
            mode_name=mode_name,
            trainer_name=trainer_name,
            runner_name=runner_name,
        )

    return {
        "preprocess": preprocess_summary,
        "mode_training": mode_summary,
        "offline_training": offline_summary,
        "online_prediction": online_summary,
    }
