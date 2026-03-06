import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_test_evaluation
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


NO_TRAINER_NAME = "none"


def main():
    trainer_choices = sorted(TRAINER_REGISTRY.keys()) + [NO_TRAINER_NAME]
    parser = argparse.ArgumentParser(
        description="Evaluate trained ROM on processed test set and export evaluation diagnostics (R2 + L2)."
    )
    parser.add_argument(
        "--processed-test-dir",
        default=str(ROOT_DIR / "data/processed/test"),
        help="Processed test directory (contains doe.csv and Snapshot_*.npy)",
    )
    parser.add_argument("--models-dir", default=str(ROOT_DIR / "models"), help="Root model directory")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "predictions/eval"),
        help="Evaluation output directory for csv/png reports",
    )
    parser.add_argument("--mode", default="pod", choices=sorted(MODE_REGISTRY.keys()), help="Mode namespace")
    parser.add_argument(
        "--trainer",
        default="rbf",
        choices=trainer_choices,
        help="Trainer namespace (used for POD reconstruction). Use 'none' for DMD direct reconstruction mode.",
    )
    parser.add_argument("--input-column", default="time", help="Time/input column in test doe.csv")
    args = parser.parse_args()

    summary = run_test_evaluation(
        processed_test_dir=Path(args.processed_test_dir),
        models_dir=Path(args.models_dir),
        mode_name=args.mode,
        trainer_name=args.trainer,
        output_dir=Path(args.output_dir),
        input_column=args.input_column,
    )

    print("Test evaluation completed.")
    print(f"Evaluation by time: {summary['evaluation_by_time_path']}")
    print(f"Evaluation summary: {summary['evaluation_summary_path']}")
    print(f"Evaluation plot: {summary['evaluation_plot_path']}")
    if summary.get("plot_error"):
        print(f"Plot warning: {summary['plot_error']}")


if __name__ == "__main__":
    main()
