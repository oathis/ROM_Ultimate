import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_offline_training
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


NO_TRAINER_NAME = "none"


def _parse_hidden_dims(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def main():
    trainer_choices = sorted(TRAINER_REGISTRY.keys()) + [NO_TRAINER_NAME]
    parser = argparse.ArgumentParser(description="Train offline surrogate models from latent trajectories.")
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data/processed"), help="Processed directory (contains doe.csv)")
    parser.add_argument("--models-dir", default=str(ROOT_DIR / "models"), help="Root model directory")
    parser.add_argument("--mode", default="pod", choices=sorted(MODE_REGISTRY.keys()), help="Mode artifact namespace to consume")
    parser.add_argument(
        "--trainer",
        default="rbf",
        choices=trainer_choices,
        help="Offline trainer name (registry key). Use 'none' for DMD direct reconstruction mode.",
    )
    parser.add_argument("--kernel", default="cubic", help="RBF kernel")
    parser.add_argument("--epsilon", type=float, default=1.0, help="RBF epsilon")
    parser.add_argument("--hidden-dims", default="128,128", help="NN hidden dims as comma-separated ints")
    parser.add_argument("--activation", default="tanh", help="NN activation label")
    parser.add_argument("--epochs", type=int, default=600, help="NN epochs")
    parser.add_argument("--solver", default="galerkin", help="Projection solver name")
    parser.add_argument("--stabilization", action="store_true", help="Projection stabilization toggle")
    parser.add_argument("--ridge", type=float, default=1e-8, help="Projection ridge regularization")
    parser.add_argument("--input-column", default="time", help="Column in doe.csv used as model input")
    parser.add_argument(
        "--eval-mode",
        default="none",
        choices=["none", "interpolation", "extrapolation", "both"],
        help="Optional validation profile for legacy offline diagnostics",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio used for split profiles")
    parser.add_argument("--min-train-samples", type=int, default=8, help="Minimum samples kept for train split")
    args = parser.parse_args()

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
    summary = run_offline_training(
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        mode_name=args.mode,
        trainer_name=args.trainer,
        trainer_params=trainer_params,
        input_column=args.input_column,
        eval_mode=args.eval_mode,
        val_ratio=args.val_ratio,
        min_train_samples=args.min_train_samples,
    )

    if args.mode == "dmd" and args.trainer == NO_TRAINER_NAME:
        print(f"DMD direct reconstruction selected. Offline trainer stage skipped for {len(summary)} variables.")
    else:
        print(f"Trainer '{args.trainer}' fitted for {len(summary)} variables.")
    for var, info in summary.items():
        print(f"- {var}: target={info['target_shape']} -> {info['model_path']}")
        val_block = info.get("validation", {})
        for split_name, split_info in val_block.items():
            metrics = split_info.get("metrics", {})
            print(
                f"  [{split_name}] rmse={metrics.get('rmse')} "
                f"mae={metrics.get('mae')} r2={metrics.get('r2')}"
            )


if __name__ == "__main__":
    main()
