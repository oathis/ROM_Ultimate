from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from rom.core.factory import build_mode, build_runner, build_trainer
from rom.data.preprocess import process_transient_data
from rom.data.split import (
    build_raw_split_manifest,
    load_split_manifest,
    save_split_manifest,
    subset_files_from_manifest,
)
from rom.runners.online_prediction import OnlinePredictionRunner


NO_TRAINER_NAME = "none"


def _discover_snapshots(processed_dir: Path):
    return sorted(processed_dir.glob("Snapshot_*.npy"))


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key == "time_values":
                continue
            out[str(key)] = _to_jsonable(val)
        return out
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mode_uses_offline_trainer(mode_name: str):
    return str(mode_name or "").strip().lower() != "dmd"


def _is_none_trainer_name(trainer_name: str):
    return str(trainer_name or "").strip().lower() in {"", NO_TRAINER_NAME}


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


def _field_column_for_variable(variable: str):
    mapping = {
        "T": "temperature",
        "u": "x-velocity",
        "v": "y-velocity",
        "w": "z-velocity",
    }
    return mapping.get(variable, variable)


def run_dataset_split(
    raw_dir: Path,
    output_path: Path,
    split_mode: str = "extrapolation",
    test_ratio: float = 0.2,
    min_train_samples: int = 8,
    split_id: str | None = None,
):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    manifest = build_raw_split_manifest(
        raw_dir=raw_dir,
        mode=split_mode,
        test_ratio=float(test_ratio),
        min_train_samples=int(min_train_samples),
        split_id=split_id,
    )
    saved_path = save_split_manifest(output_path, manifest)
    manifest["manifest_path"] = str(saved_path)
    return manifest


def run_preprocess(
    raw_dir: Path,
    processed_dir: Path,
    split_manifest_path: Path | None = None,
    subset: str = "all",
):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    selected_files = None
    split_manifest = None
    subset_norm = str(subset or "all").strip().lower()

    def _build_dataset_manifest(target_dir: Path, target_subset: str, target_selected_files):
        return {
            "updated_at": _now_iso(),
            "raw_dir": str(raw_dir),
            "processed_dir": str(target_dir),
            "subset": target_subset,
            "split_manifest_path": None if split_manifest_path is None else str(Path(split_manifest_path)),
            "split_mode": None if split_manifest is None else split_manifest.get("split_mode"),
            "n_train": None if split_manifest is None else split_manifest.get("n_train"),
            "n_test": None if split_manifest is None else split_manifest.get("n_test"),
            "selected_files": None if target_selected_files is None else int(len(target_selected_files)),
            "snapshots": [p.name for p in _discover_snapshots(target_dir)],
        }

    def _derive_split_output_dir(base_dir: Path, target_subset: str):
        base_name = base_dir.name
        lowered = base_name.lower()
        for suffix in ("-train", "-test", "_train", "_test"):
            if lowered.endswith(suffix):
                root = base_name[: -len(suffix)]
                if root:
                    return base_dir.with_name(f"{root}-{target_subset}")
        return base_dir.with_name(f"{base_name}-{target_subset}")

    if split_manifest_path is not None:
        split_manifest = load_split_manifest(split_manifest_path)
        if subset_norm == "all":
            generated = {}
            for split_subset in ("train", "test"):
                subset_files = subset_files_from_manifest(split_manifest, split_subset)
                if not subset_files:
                    raise ValueError(f"Split manifest contains no files for subset='{split_subset}'")

                target_dir = _derive_split_output_dir(processed_dir, split_subset)
                process_transient_data(raw_dir, target_dir, selected_filenames=subset_files)
                target_summary = {
                    "processed_dir": str(target_dir),
                    "subset": split_subset,
                    "selected_files": int(len(subset_files)),
                    "snapshots": [p.name for p in _discover_snapshots(target_dir)],
                }
                _write_json(target_dir / "dataset_manifest.json", _build_dataset_manifest(target_dir, split_subset, subset_files))
                generated[split_subset] = target_summary

            return {
                "processed_dir": str(processed_dir),
                "split_manifest_path": str(Path(split_manifest_path)),
                "subset": "all",
                "split_mode": split_manifest.get("split_mode"),
                "n_train": split_manifest.get("n_train"),
                "n_test": split_manifest.get("n_test"),
                "generated": generated,
            }

        selected_files = subset_files_from_manifest(split_manifest, subset_norm)
        if subset_norm in {"train", "test"} and not selected_files:
            raise ValueError(f"Split manifest contains no files for subset='{subset_norm}'")

    process_transient_data(raw_dir, processed_dir, selected_filenames=selected_files)

    summary = {
        "processed_dir": str(processed_dir),
        "snapshots": [p.name for p in _discover_snapshots(processed_dir)],
    }
    if split_manifest is not None:
        summary.update(
            {
                "split_manifest_path": str(Path(split_manifest_path)),
                "subset": subset_norm,
                "selected_files": int(len(selected_files)),
                "split_mode": split_manifest.get("split_mode"),
                "n_train": split_manifest.get("n_train"),
                "n_test": split_manifest.get("n_test"),
            }
        )

    _write_json(processed_dir / "dataset_manifest.json", _build_dataset_manifest(processed_dir, subset_norm, selected_files))

    return summary


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

    mode_manifest = {
        "updated_at": _now_iso(),
        "mode_name": mode_name,
        "processed_dir": str(processed_dir),
        "mode_params": _to_jsonable(mode_params),
        "variables": sorted(summary.keys()),
        "variable_summary": _to_jsonable(summary),
    }
    _write_json(mode_root / "mode_manifest.json", mode_manifest)

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

    if not _mode_uses_offline_trainer(mode_name):
        if _is_none_trainer_name(trainer_name):
            summary = {}
            for var_dir in sorted([d for d in mode_root.iterdir() if d.is_dir()]):
                latent_path = var_dir / "latent.npy"
                if not latent_path.exists():
                    continue
                latent_shape = tuple(np.load(latent_path, mmap_mode="r").shape)
                summary[var_dir.name] = {
                    "input_shape": tuple(x_train.shape),
                    "target_shape": latent_shape,
                    "model_path": None,
                    "model_namespace": None,
                    "validation": {},
                    "skipped": True,
                    "reason": "dmd_direct_reconstruction",
                }
            if not summary:
                raise RuntimeError(f"No DMD latent artifacts found under {mode_root}")
            return summary
    elif _is_none_trainer_name(trainer_name):
        raise ValueError("trainer='none' is only valid when mode='dmd'.")

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

    trainer_manifest = {
        "updated_at": _now_iso(),
        "trainer_name": trainer_name,
        "mode_name": mode_name,
        "input_column": input_column,
        "trainer_params": _to_jsonable(trainer_params),
        "eval_mode": eval_mode,
        "val_ratio": float(val_ratio),
        "min_train_samples": int(min_train_samples),
        "variables": sorted(summary.keys()),
        "variable_summary": _to_jsonable(summary),
    }
    _write_json(models_dir / trainer_name / mode_name / "trainer_manifest.json", trainer_manifest)

    return summary


def run_test_evaluation(
    processed_test_dir: Path,
    models_dir: Path,
    mode_name: str,
    trainer_name: str,
    output_dir: Path,
    input_column: str = "time",
):
    processed_test_dir = Path(processed_test_dir)
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)

    doe_path = processed_test_dir / "doe.csv"
    if not doe_path.exists():
        raise FileNotFoundError(f"Missing DOE file for test set: {doe_path}")

    doe = pd.read_csv(doe_path)
    if input_column not in doe.columns:
        raise ValueError(f"Column '{input_column}' is missing in {doe_path}")
    time_values = doe[input_column].to_numpy(dtype=np.float64)

    snapshots = {}
    for snap_path in _discover_snapshots(processed_test_dir):
        variable = snap_path.stem.replace("Snapshot_", "")
        arr = np.asarray(np.load(snap_path), dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Snapshot must be 2D for '{variable}', got {arr.shape}")

        if arr.shape[1] != time_values.size:
            if arr.shape[0] == time_values.size:
                arr = arr.T
            else:
                raise ValueError(
                    f"Snapshot/time mismatch for '{variable}': snapshot={arr.shape}, times={time_values.shape}"
                )
        snapshots[variable] = arr

    if not snapshots:
        raise RuntimeError(f"No Snapshot_*.npy files found in {processed_test_dir}")

    runner = OnlinePredictionRunner(models_dir=models_dir, mode_name=mode_name, trainer_name=trainer_name)
    rows = []

    for t_idx, t in enumerate(time_values):
        pred_df = runner.step(float(t))
        for variable, y_true_full in snapshots.items():
            field_column = _field_column_for_variable(variable)
            if field_column not in pred_df.columns:
                continue

            y_true = np.asarray(y_true_full[:, t_idx], dtype=np.float64).reshape(-1, 1)
            y_pred = np.asarray(pred_df[field_column].to_numpy(dtype=np.float64), dtype=np.float64).reshape(-1, 1)
            if y_true.shape != y_pred.shape:
                raise ValueError(
                    f"Prediction node count mismatch for '{variable}' at index {t_idx}: "
                    f"true={y_true.shape}, pred={y_pred.shape}"
                )

            metrics = _regression_metrics(y_true, y_pred)
            r2 = metrics.get("r2")
            l2_error = float(np.linalg.norm(y_pred - y_true, ord=2))
            l2_true_norm = float(np.linalg.norm(y_true, ord=2))
            if l2_true_norm <= 1e-12:
                # Degenerate baseline: if truth is effectively zero, report 0% only when prediction matches.
                l2_error_pct = 0.0 if l2_error <= 1e-12 else float("nan")
            else:
                l2_error_pct = float((l2_error / l2_true_norm) * 100.0)
            rows.append(
                {
                    "time": float(t),
                    "time_index": int(t_idx),
                    "variable": variable,
                    "field_column": field_column,
                    "r2": r2,
                    "l2_error": l2_error,
                    "l2_error_pct": l2_error_pct,
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                }
            )

    if not rows:
        raise RuntimeError("No evaluation rows were created. Check variable/column mapping and model artifacts.")

    eval_df = pd.DataFrame(rows)
    summary_rows = []
    for variable, group in eval_df.groupby("variable"):
        valid_r2 = group["r2"].dropna()
        summary_rows.append(
            {
                "variable": variable,
                "samples": int(group.shape[0]),
                "r2_mean": None if valid_r2.empty else float(valid_r2.mean()),
                "r2_min": None if valid_r2.empty else float(valid_r2.min()),
                "r2_max": None if valid_r2.empty else float(valid_r2.max()),
                "l2_mean": float(group["l2_error"].mean()),
                "l2_min": float(group["l2_error"].min()),
                "l2_max": float(group["l2_error"].max()),
                "rmse_mean": float(group["rmse"].mean()),
                "mae_mean": float(group["mae"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("variable")

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_by_time_path = output_dir / "evaluation_by_time.csv"
    evaluation_summary_path = output_dir / "evaluation_summary.csv"
    plot_path = output_dir / "evaluation_plot.png"
    eval_df.to_csv(evaluation_by_time_path, index=False)
    summary_df.to_csv(evaluation_summary_path, index=False)

    plot_error = None
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax_r2, ax_err = axes

        for variable in sorted(eval_df["variable"].unique()):
            var_df = eval_df[eval_df["variable"] == variable].sort_values("time")
            ax_r2.plot(var_df["time"], var_df["r2"], label=variable)
            if "l2_error_pct" in var_df.columns:
                ax_err.plot(var_df["time"], var_df["l2_error_pct"], label=variable)
            else:
                ax_err.plot(var_df["time"], var_df["l2_error"], label=variable)

        ax_r2.set_title("R2 on Test Set")
        ax_r2.set_ylabel("R2")
        ax_r2.grid(True, alpha=0.3)
        ax_r2.legend(loc="best")

        ax_err.set_title("L2 Error on Test Set")
        ax_err.set_xlabel(input_column)
        ax_err.set_ylabel("L2 norm (%)")
        ax_err.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        plot_error = str(exc)

    report = {
        "updated_at": _now_iso(),
        "processed_test_dir": str(processed_test_dir),
        "models_dir": str(models_dir),
        "mode_name": mode_name,
        "trainer_name": trainer_name,
        "input_column": input_column,
        "n_times": int(time_values.size),
        "n_variables": int(summary_df.shape[0]),
        "evaluation_by_time_path": str(evaluation_by_time_path),
        "evaluation_summary_path": str(evaluation_summary_path),
        "evaluation_plot_path": str(plot_path),
        # Backward-compatible aliases for older UI readers.
        "r2_by_time_path": str(evaluation_by_time_path),
        "r2_summary_path": str(evaluation_summary_path),
        "r2_plot_path": str(plot_path),
        "plot_error": plot_error,
        "variable_summary": _to_jsonable(summary_rows),
    }
    _write_json(output_dir / "evaluation_manifest.json", report)
    return report


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
