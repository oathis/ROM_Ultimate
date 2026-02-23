import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_preprocess


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing (raw CSV -> processed snapshots).")
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data/raw/Dataset"), help="Raw CSV directory")
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data/processed"), help="Processed output directory")
    args = parser.parse_args()

    summary = run_preprocess(Path(args.raw_dir), Path(args.processed_dir))
    print(f"Processed directory: {summary['processed_dir']}")
    print(f"Snapshots: {summary['snapshots']}")


if __name__ == "__main__":
    main()
