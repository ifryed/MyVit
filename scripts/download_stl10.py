#!/usr/bin/env python3
"""Download STL-10 into data/stl-10."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "stl-10")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=DEFAULT_ROOT,
        help="Directory for the STL-10 binaries. Created if missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from pl_datamodule import ensure_stl10, stl10_is_ready

    if stl10_is_ready(args.data_root):
        print(f"STL-10 already present at {args.data_root}")
        return
    ensure_stl10(args.data_root)
    print(f"STL-10 ready at {args.data_root}")


if __name__ == "__main__":
    main()
