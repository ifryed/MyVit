#!/usr/bin/env python3
"""Download CIFAR-10 and write the PNG layout CIFARDataset expects."""

import argparse
import csv
import os

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "cifar-10")
TRAIN_COUNT = 50000
TEST_COUNT = 10000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=DEFAULT_ROOT,
        help="Directory for train/, test/, and trainLabels.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the PNGs and labels even if they are already complete.",
    )
    return parser.parse_args()


def png_count(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(name.endswith(".png") for name in os.listdir(directory))


def label_count(path):
    if not os.path.isfile(path):
        return 0
    with open(path, newline="") as handle:
        rows = sum(1 for _ in handle)
    return max(0, rows - 1)


def split_ready(image_dir, count, labels_path=None):
    if png_count(image_dir) != count:
        return False
    if labels_path is None:
        return True
    return label_count(labels_path) == count


def export_split(dataset, image_dir, labels_path=None):
    os.makedirs(image_dir, exist_ok=True)
    labels = []
    total = len(dataset)
    for index in range(1, total + 1):
        image, label = dataset[index - 1]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.convert("RGB").save(os.path.join(image_dir, f"{index}.png"))
        if labels_path is not None:
            labels.append((index, dataset.classes[label]))
        if index % 5000 == 0 or index == total:
            print(f"  wrote {index}/{total}")
    if labels_path is not None:
        with open(labels_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "label"])
            writer.writerows(labels)


def main():
    args = parse_args()
    try:
        from torchvision.datasets import CIFAR10
    except ImportError:
        raise SystemExit("torchvision is required. pip install torchvision")

    labels_path = os.path.join(args.data_root, "trainLabels.csv")
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    if (
        not args.force
        and split_ready(train_dir, TRAIN_COUNT, labels_path)
        and split_ready(test_dir, TEST_COUNT)
    ):
        print(f"CIFAR-10 already present at {args.data_root}")
        return

    print(f"Downloading CIFAR-10 into {args.data_root}")
    train_set = CIFAR10(args.data_root, train=True, download=True)
    test_set = CIFAR10(args.data_root, train=False, download=True)

    if args.force or not split_ready(train_dir, TRAIN_COUNT, labels_path):
        print("Writing training PNGs")
        export_split(train_set, train_dir, labels_path)
    if args.force or not split_ready(test_dir, TEST_COUNT):
        print("Writing test PNGs")
        export_split(test_set, test_dir)
    print(f"CIFAR-10 ready at {args.data_root}")


if __name__ == "__main__":
    main()
