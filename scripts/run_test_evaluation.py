import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_test_evaluation
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY
from rom.registry.trainer_registry import REGISTRY as TRAINER_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained ROM on processed test set and export R2 diagnostics."
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
        choices=sorted(TRAINER_REGISTRY.keys()),
        help="Trainer namespace (used for POD reconstruction)",
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
    print(f"R2 by time: {summary['r2_by_time_path']}")
    print(f"R2 summary: {summary['r2_summary_path']}")
    print(f"R2 plot: {summary['r2_plot_path']}")
    if summary.get("plot_error"):
        print(f"Plot warning: {summary['plot_error']}")


if __name__ == "__main__":
    main()
