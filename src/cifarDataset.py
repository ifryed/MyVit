import csv
import os

import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_image(path):
    return skimage.io.imread(path).transpose(2, 0, 1)


def split_image_to_patches(im, patch_size=4):
    h, w = im.shape[1:]
    patches = []
    shape = (h // patch_size, w // patch_size)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = im[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches, shape


def from_patches_to_image(patches, shape):
    patch_size = patches[0].shape[1]
    h, w = shape
    im = np.zeros((3, h * patch_size, w * patch_size))
    for i in range(h):
        for j in range(w):
            im[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size] = (
                patches[i * w + j]
            )
    return im


class CIFARDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.root = os.path.join(root, "train" if train else "test")
        self.transform = transform
        self.classes = CIFAR10_CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self.data = sorted(
            [name for name in os.listdir(self.root) if name.endswith(".png")],
            key=lambda name: int(os.path.splitext(name)[0]),
        )

        tmp_image = load_image(os.path.join(self.root, self.data[0]))
        self.patch_size = 4
        self.patch_shape = split_image_to_patches(tmp_image, self.patch_size)[1]
        self.patches_per_image = self.patch_shape[0] * self.patch_shape[1]
        self.image_size = tmp_image.shape[1]

        self.labels = None
        if train:
            labels_path = os.path.join(root, "trainLabels.csv")
            id_to_label = {}
            with open(labels_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    id_to_label[int(row["id"])] = self.class_to_idx[row["label"]]
            self.labels = torch.tensor(
                [id_to_label[int(os.path.splitext(name)[0])] for name in self.data],
                dtype=torch.long,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.data[idx])
        image = torch.from_numpy(np.ascontiguousarray(load_image(path))).float() / 255.0
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, self.labels[idx]
