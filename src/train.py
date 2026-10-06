import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pl_datamodule import CIFARDataModule
from pl_model import ViTClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cifar-10")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "lightning_logs")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT on CIFAR-10")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment-name", type=str, default="vit_cifar")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    datamodule = CIFARDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = ViTClassifier(
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
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
