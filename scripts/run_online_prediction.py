import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_online_prediction
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.runner_registry import REGISTRY as RUNNER_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


NO_TRAINER_NAME = "none"


def main():
    trainer_choices = sorted(TRAINER_REGISTRY.keys()) + [NO_TRAINER_NAME]
    parser = argparse.ArgumentParser(description="Run online prediction and export runner output.")
    parser.add_argument("time", type=float, help="Time value for prediction")
    parser.add_argument("--models-dir", default=str(ROOT_DIR / "models"), help="Root model directory")
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data/processed"), help="Processed directory (points.bin)")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "predictions"), help="Prediction output directory")
    parser.add_argument("--mode", default="pod", choices=sorted(MODE_REGISTRY.keys()), help="Mode namespace to use")
    parser.add_argument(
        "--trainer",
        default="rbf",
        choices=trainer_choices,
        help="Trainer namespace to use. Use 'none' for DMD direct reconstruction mode.",
    )
    parser.add_argument(
        "--runner",
        default="reconstruction",
        choices=sorted(RUNNER_REGISTRY.keys()),
        help="Online runner name (registry key)",
    )
    args = parser.parse_args()

    summary = run_online_prediction(
        time_value=args.time,
        models_dir=Path(args.models_dir),
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        mode_name=args.mode,
        trainer_name=args.trainer,
        runner_name=args.runner,
    )
    print(
        f"Saved {summary['output_type']} output using runner='{summary['runner']}' "
        f"({summary['rows']} rows) to {summary['output_path']}"
    )


if __name__ == "__main__":
    main()
