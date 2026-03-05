import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_dataset_split


def main():
    parser = argparse.ArgumentParser(description="Create train/test split manifest from raw xresult-*.csv files.")
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data/raw/Dataset"), help="Raw CSV directory")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output split manifest path (json). Default: artifacts/splits/split_manifest.json",
    )
    parser.add_argument(
        "--mode",
        default="extrapolation",
        choices=["interpolation", "extrapolation"],
        help="Split policy: interpolation=mid-time sampling, extrapolation=tail holdout",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of samples reserved for test")
    parser.add_argument("--min-train-samples", type=int, default=8, help="Minimum samples kept in train")
    parser.add_argument("--split-id", default=None, help="Optional split identifier")
    args = parser.parse_args()

    output_path = Path(args.output_path) if args.output_path else ROOT_DIR / "artifacts/splits/split_manifest.json"
    summary = run_dataset_split(
        raw_dir=Path(args.raw_dir),
        output_path=output_path,
        split_mode=args.mode,
        test_ratio=args.test_ratio,
        min_train_samples=args.min_train_samples,
        split_id=args.split_id,
    )

    print(f"Split manifest saved: {summary['manifest_path']}")
    print(
        f"mode={summary['split_mode']} total={summary['n_total']} "
        f"train={summary['n_train']} test={summary['n_test']}"
    )


if __name__ == "__main__":
    main()
