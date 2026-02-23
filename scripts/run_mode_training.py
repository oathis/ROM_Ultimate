import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_mode_training
from rom.registry.mode_registry import REGISTRY as MODE_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Train mode decomposition artifacts from processed snapshots.")
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data/processed"), help="Processed snapshot directory")
    parser.add_argument("--models-dir", default=str(ROOT_DIR / "models"), help="Root model directory")
    parser.add_argument("--mode", default="pod", choices=sorted(MODE_REGISTRY.keys()), help="Mode builder name (registry key)")
    parser.add_argument("--rank", type=int, default=32, help="Target rank/modes")
    parser.add_argument("--auto-rank", action="store_true", help="Use mode default rank behavior (pass rank=None)")
    parser.add_argument("--energy-threshold", type=float, default=0.999, help="POD energy threshold")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="DMD time-step. Use <=0 to auto-infer from processed doe.csv time column.",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Column in doe.csv used for DMD time grid inference.",
    )
    args = parser.parse_args()

    mode_params = {
        "rank": None if args.auto_rank else args.rank,
        "energy_threshold": args.energy_threshold,
        "dt": args.dt,
        "time_column": args.time_column,
    }
    summary = run_mode_training(
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        mode_name=args.mode,
        mode_params=mode_params,
    )

    print(f"Mode '{args.mode}' trained for {len(summary)} variables.")
    for var, info in summary.items():
        print(f"- {var}: latent={info['latent_shape']} -> {info['artifact_dir']}")


if __name__ == "__main__":
    main()
