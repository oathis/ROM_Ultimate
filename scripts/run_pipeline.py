import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_full_pipeline
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.runner_registry import REGISTRY as RUNNER_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


def _parse_hidden_dims(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run full ROM pipeline: preprocess -> mode -> offline -> online.")
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data/raw/Dataset"), help="Raw CSV directory")
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data/processed"), help="Processed output directory")
    parser.add_argument("--models-dir", default=str(ROOT_DIR / "models"), help="Root model directory")
    parser.add_argument("--predictions-dir", default=str(ROOT_DIR / "predictions"), help="Prediction output directory")
    parser.add_argument("--mode", default="pod", choices=sorted(MODE_REGISTRY.keys()), help="Mode builder name")
    parser.add_argument("--trainer", default="rbf", choices=sorted(TRAINER_REGISTRY.keys()), help="Offline trainer name")
    parser.add_argument("--runner", default="reconstruction", choices=sorted(RUNNER_REGISTRY.keys()), help="Online runner name")
    parser.add_argument("--rank", type=int, default=32, help="Target rank/modes")
    parser.add_argument("--auto-rank", action="store_true", help="Use mode default rank behavior (pass rank=None)")
    parser.add_argument("--energy-threshold", type=float, default=0.999, help="POD energy threshold")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="DMD time-step. Use <=0 to auto-infer from processed doe.csv time column.",
    )
    parser.add_argument("--time-column", default="time", help="Column in doe.csv used for DMD time grid inference")
    parser.add_argument("--kernel", default="cubic", help="RBF kernel")
    parser.add_argument("--epsilon", type=float, default=1.0, help="RBF epsilon")
    parser.add_argument("--hidden-dims", default="128,128", help="NN hidden dims as comma-separated ints")
    parser.add_argument("--activation", default="relu", help="NN activation label")
    parser.add_argument("--epochs", type=int, default=200, help="NN epochs")
    parser.add_argument("--solver", default="galerkin", help="Projection solver name")
    parser.add_argument("--stabilization", action="store_true", help="Projection stabilization toggle")
    parser.add_argument("--ridge", type=float, default=1e-8, help="Projection ridge regularization")
    parser.add_argument(
        "--eval-mode",
        default="none",
        choices=["none", "interpolation", "extrapolation", "both"],
        help="Optional validation profile for legacy offline diagnostics",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio used for split profiles")
    parser.add_argument("--min-train-samples", type=int, default=8, help="Minimum samples kept for train split")
    parser.add_argument("--predict-time", type=float, default=None, help="Optional time for immediate online prediction")
    args = parser.parse_args()

    mode_params = {
        "rank": None if args.auto_rank else args.rank,
        "energy_threshold": args.energy_threshold,
        "dt": args.dt,
        "time_column": args.time_column,
    }
    trainer_params = {
        "kernel": args.kernel,
        "epsilon": args.epsilon,
        "hidden_dims": _parse_hidden_dims(args.hidden_dims),
        "activation": args.activation,
        "epochs": args.epochs,
        "solver": args.solver,
        "stabilization": args.stabilization,
        "ridge": args.ridge,
    }

    summary = run_full_pipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        predictions_dir=Path(args.predictions_dir),
        mode_name=args.mode,
        trainer_name=args.trainer,
        runner_name=args.runner,
        mode_params=mode_params,
        trainer_params=trainer_params,
        predict_time=args.predict_time,
        offline_eval_mode=args.eval_mode,
        offline_val_ratio=args.val_ratio,
        offline_min_train_samples=args.min_train_samples,
    )

    print("Pipeline completed.")
    print(f"Preprocess snapshots: {summary['preprocess']['snapshots']}")
    print(f"Mode variables: {list(summary['mode_training'].keys())}")
    print(f"Offline models: {list(summary['offline_training'].keys())}")
    if summary["online_prediction"] is not None:
        print(f"Prediction: {summary['online_prediction']['output_path']}")


if __name__ == "__main__":
    main()
