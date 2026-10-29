import os
import urllib.request

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import STL10
from torchvision.datasets.utils import check_integrity, extract_archive
import pytorch_lightning as pl
from cifarDataset import CIFAR10_CLASSES, CIFARDataset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Channel stats of the STL-10 labeled training set, pixels in [0, 1].
STL10_MEAN = (0.44671062, 0.43980984, 0.40664645)
STL10_STD = (0.26034098, 0.25657727, 0.27126738)
STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]
STL10_ARCHIVE = "stl10_binary.tar.gz"
STL10_MD5 = "91f7769df0f17e558f3565bffb0c7dfb"
# Stanford's host often refuses connections. The first URL is an unmodified copy
# of the official archive and uses the checksum torchvision expects.
STL10_URLS = (
    "https://huggingface.co/datasets/DATAML200/stl10/resolve/main/stl10_binary.tar.gz",
    "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
)
STL10_FILES = (
    ("train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"),
    ("train_y.bin", "5a34089d4802c674881badbb80307741"),
    ("unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"),
    ("test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"),
    ("test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"),
)


def stl10_is_ready(root):
    base = os.path.join(root, "stl10_binary")
    return all(
        check_integrity(os.path.join(base, name), md5) for name, md5 in STL10_FILES
    )


def ensure_stl10(root):
    if stl10_is_ready(root):
        return
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, STL10_ARCHIVE)
    if not check_integrity(archive, STL10_MD5):
        errors = []
        for url in STL10_URLS:
            try:
                print(f"Downloading STL-10 from {url}")
                _download(url, archive)
                if check_integrity(archive, STL10_MD5):
                    break
                errors.append(f"{url}: checksum mismatch")
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("Could not download STL-10. " + "; ".join(errors))
    extract_archive(archive, root)
    if not stl10_is_ready(root):
        raise RuntimeError(f"STL-10 files in {root} failed the checksum")


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(destination, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 100 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                if total:
                    print(f"STL-10 {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB")
                else:
                    print(f"STL-10 {downloaded / 1e9:.2f} GB")
                next_report += 100 * 1024 * 1024

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


class STL10DataModule(pl.LightningDataModule):
    """STL-10 at its native 96×96. Train is the 5k labeled split; val is the 8k test split."""

    def __init__(self, data_root, batch_size=64, num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(STL10_MEAN, STL10_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(STL10_MEAN, STL10_STD),
            ]
        )
        ensure_stl10(self.data_root)
        self.train_dataset = STL10(
            self.data_root, split="train", download=False, transform=train_transform
        )
        self.val_dataset = STL10(
            self.data_root, split="test", download=False, transform=eval_transform
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
