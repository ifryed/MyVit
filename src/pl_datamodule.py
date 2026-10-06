import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from cifarDataset import CIFARDataset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=128, num_workers=4, val_split=0.1):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        eval_transform = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

        labeled = CIFARDataset(self.data_root, train=True, transform=train_transform)
        labeled_eval = CIFARDataset(self.data_root, train=True, transform=eval_transform)

        val_size = int(len(labeled) * self.val_split)
        train_size = len(labeled) - val_size
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, _ = random_split(
            labeled, [train_size, val_size], generator=generator
        )
        generator = torch.Generator().manual_seed(42)
        _, self.val_dataset = random_split(
            labeled_eval, [train_size, val_size], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
