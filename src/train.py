import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pl_datamodule import (
    CIFAR_MEAN,
    CIFAR_STD,
    CIFARDataModule,
    STL10_CLASSES,
    STL10_MEAN,
    STL10_STD,
    STL10DataModule,
)
from cifarDataset import CIFAR10_CLASSES
from pl_model import ViTClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "lightning_logs")
DATASETS = {
    "cifar": {
        "image_size": 32,
        "num_classes": 10,
        "patch_size": 4,
        "batch_size": 128,
        "data_root": os.path.join(PROJECT_ROOT, "data", "cifar-10"),
        "classes": CIFAR10_CLASSES,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
        "datamodule": CIFARDataModule,
    },
    "stl10": {
        "image_size": 96,
        "num_classes": 10,
        "patch_size": 8,
        "batch_size": 64,
        "data_root": os.path.join(PROJECT_ROOT, "data", "stl-10"),
        "classes": STL10_CLASSES,
        "mean": STL10_MEAN,
        "std": STL10_STD,
        "datamodule": STL10DataModule,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier on CIFAR-10 or STL-10")
    parser.add_argument(
        "--model",
        choices=("vit", "swin"),
        default="vit",
        help="Architecture to train. --embedding-dim, --num-heads, --num-layers, and --dropout apply to vit. --embed-dim, --depths, --swin-heads, and --window-size apply to swin.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="cifar",
        help="cifar is 32x32. stl10 is 96x96 and is downloaded into data/stl-10 if missing.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset directory. Defaults to data/cifar-10 or data/stl-10.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to 128 for cifar and 64 for stl10.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Patch side length in pixels. Defaults to 4 for cifar, 8 for stl10 with vit, and 4 for stl10 with swin. Must divide the image side.",
    )
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Swin window. Defaults to 4 on cifar (8x8 grid) and 6 on stl10 (24x24 grid).",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 2, 6, 2])
    parser.add_argument("--swin-heads", type=int, nargs="+", default=[3, 6, 12, 24])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="TensorBoard run name. Defaults to <model>_<dataset>",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = DATASETS[args.dataset]
    if args.data_root is None:
        args.data_root = spec["data_root"]
    if args.batch_size is None:
        args.batch_size = spec["batch_size"]
    if args.patch_size is None:
        # Four Swin stages downsample three times, so a 96px image needs patch size 4
        # (24 -> 12 -> 6 -> 3). Patch size 8 leaves an odd grid and the merge asserts.
        if args.dataset == "stl10" and args.model == "swin":
            args.patch_size = 4
        else:
            args.patch_size = spec["patch_size"]
    if args.window_size is None:
        # 7 is the ImageNet window. It does not divide an 8 or 24 token grid,
        # so the extra cells are padding and the later stages collapse to one window.
        args.window_size = 6 if args.dataset == "stl10" else 4
    if spec["image_size"] % args.patch_size != 0:
        raise SystemExit(
            f"patch size {args.patch_size} must divide image size {spec['image_size']} for {args.dataset}"
        )
    if args.model == "swin":
        grid = spec["image_size"] // args.patch_size
        merges = len(args.depths) - 1
        if grid % (2 ** merges) != 0:
            raise SystemExit(
                f"Swin with {len(args.depths)} stages downsamples {merges} times, so the patch grid "
                f"({grid}) must be divisible by {2 ** merges}. "
                f"Use a smaller --patch-size or fewer --depths."
            )
    pl.seed_everything(42, workers=True)

    datamodule = spec["datamodule"](
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = ViTClassifier(
        model_name=args.model,
        image_size=spec["image_size"],
        num_classes=spec["num_classes"],
        class_names=spec["classes"],
        mean=spec["mean"],
        std=spec["std"],
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        depths=args.depths,
        swin_heads=args.swin_heads,
    )
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name or f"{args.model}_{args.dataset}",
        log_graph=True,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
        default_root_dir=args.log_dir,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"TensorBoard logs: {logger.log_dir}")
    print(f"View with: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
