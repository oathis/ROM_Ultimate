import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR
from rom.core.workflows import run_preprocess


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing (raw CSV -> processed snapshots).")
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data/raw/Dataset"), help="Raw CSV directory")
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data/processed"), help="Processed output directory")
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional split manifest json path created by run_dataset_split.py",
    )
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "train", "test"],
        help="Subset to preprocess when split manifest is provided. all=generate both -train/-test in one run.",
    )
    args = parser.parse_args()

    summary = run_preprocess(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
        subset=args.subset,
    )
    if summary.get("generated"):
        print(f"Requested base directory: {summary['processed_dir']}")
        for split_subset, item in summary["generated"].items():
            print(f"[{split_subset}] Processed directory: {item['processed_dir']}")
            print(f"[{split_subset}] Snapshots: {item['snapshots']}")
            print(f"[{split_subset}] selected_files={item.get('selected_files')}")
    else:
        print(f"Processed directory: {summary['processed_dir']}")
        print(f"Snapshots: {summary['snapshots']}")
    if summary.get("split_manifest_path") and not summary.get("generated"):
        print(
            f"Split subset: {summary.get('subset')} "
            f"(selected_files={summary.get('selected_files')}, mode={summary.get('split_mode')})"
        )
    elif summary.get("split_manifest_path") and summary.get("generated"):
        print(
            f"Split subset: all (mode={summary.get('split_mode')}, "
            f"n_train={summary.get('n_train')}, n_test={summary.get('n_test')})"
        )


if __name__ == "__main__":
    main()
