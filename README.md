# MyViT

A Vision Transformer built from scratch in PyTorch and trained on CIFAR-10 or STL-10 with PyTorch Lightning.

The model turns each 32×32 image into a sequence of patch tokens, adds a learnable class token and positional embeddings, runs the sequence through transformer encoder blocks, and classifies from the class token.

README and [`docs/architecture.md`](docs/architecture.md) describe the project as it is now.

## Layout

```
MyViT/
├── src/
│   ├── train.py           # training entry point
│   ├── ViT.py             # full model: patches, CLS, positions, stack, head
│   ├── transBlock.py      # patch embedding, attention, MLP, encoder block
│   ├── pl_model.py        # Lightning module, loss, optimizer, TensorBoard images
│   ├── pl_datamodule.py   # CIFAR-10 and STL-10 dataloaders
│   ├── cifarDataset.py    # CIFAR-10 PNG loader and patch helpers
├── scripts/
│   ├── download_cifar10.py
│   └── download_stl10.py
├── data/cifar-10/         # written by scripts/download_cifar10.py
├── data/stl-10/           # written by scripts/download_stl10.py
├── docs/architecture.md
```

## Requirements

- Python 3
- PyTorch
- torchvision
- PyTorch Lightning
- TensorBoard
- NumPy
- scikit-image
- matplotlib

```bash
pip install torch torchvision pytorch-lightning tensorboard numpy scikit-image matplotlib
```

## Data

From the repository root:

```bash
python scripts/download_cifar10.py
python scripts/download_stl10.py
```

`download_cifar10.py` fetches the official CIFAR-10 archive and writes the PNG layout the loader expects:

```
data/cifar-10/
├── trainLabels.csv    # columns: id, label
├── train/             # 1.png, 2.png, ...
└── test/              # optional; training does not use it
```

`trainLabels.csv` maps each image id to one of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Images are read as RGB, converted to channel-first tensors, and scaled to `[0, 1]`. Training applies a random horizontal flip, then CIFAR-10 normalization. Validation uses normalization only.

The labeled training set is split 90% / 10% with a fixed seed (`42`). The same indices are used for both copies of the data so train and val do not share images, while val images are not flipped.

STL-10 is 96×96 RGB with the same 10-class setup (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). `download_stl10.py` stores the binaries in `data/stl-10`. Training with `--dataset stl10` downloads the same files if they are missing. Training uses the 5,000 labeled images. Validation uses the official 8,000-image test split. The ViT default patch size is 8 (144 tokens). Swin keeps patch size 4 so its four stages can keep halving the 24×24 grid.

## Train

From the repository root:

```bash
python src/train.py
```

Python adds `src/` to the import path because that is the script directory. Defaults:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `cifar` | `cifar` (32×32) or `stl10` (96×96) |
| `--data-root` | `data/cifar-10` or `data/stl-10` | Dataset directory |
| `--batch-size` | `128` for cifar, `64` for stl10 | Batch size |
| `--num-workers` | `4` | DataLoader workers |
| `--max-epochs` | `20` | Training epochs |
| `--lr` | `1e-3` | AdamW learning rate |
| `--weight-decay` | `0.05` | AdamW weight decay |
| `--embedding-dim` | `128` | Token width |
| `--num-heads` | `4` | Attention heads |
| `--num-layers` | `6` | Encoder blocks |
| `--patch-size` | `4` for cifar, `8` for stl10 ViT, `4` for stl10 Swin | Patch side length in pixels |
| `--dropout` | `0.1` | Dropout after positional embeddings |
| `--log-dir` | `lightning_logs` | TensorBoard root |
| `--experiment-name` | `<model>_<dataset>` | Run name under the log dir |
| `--fast-dev-run` | off | One train and one val batch, then stop |

Image size is 32 for CIFAR-10 and 96 for STL-10. The number of classes is 10. The MLP inside each block expands to `4 × embedding_dim`. `embedding_dim` must be divisible by `num_heads`. `patch_size` must divide the image side.

```bash
python src/train.py --dataset stl10
```

A short smoke test:

```bash
python src/train.py --fast-dev-run
```

Optimizer is AdamW. The learning rate follows cosine annealing over `max_epochs`. Loss is cross-entropy. The global seed is 42.

Logs go to `lightning_logs/vit_cifar/version_N/`. View them with:

```bash
tensorboard --logdir lightning_logs
```

Logged scalars: `train_loss`, `train_acc`, `val_loss`, `val_acc`, and learning rate. Logged images: a grid of training samples from the first batch, a validation attention overlay (class token attending to patches, averaged over layers and heads), and a text panel of predicted vs true class names for that validation batch.

VS Code launch config `train` in `.vscode/launch.json` runs `src/train.py`.

## Model in one pass

For a batch of shape `(B, 3, 32, 32)` and the default patch size of 4:

1. **Patch embedding.** Non-overlapping 4×4 patches (64 of them) are flattened from 48 values to vectors of size `embedding_dim`.
2. **Class token.** A learnable vector is prepended, so the sequence length is 65.
3. **Positions.** A learnable vector is added at each sequence index.
4. **Encoder.** Six blocks of layer norm, multi-head attention, residual, MLP, residual. Each block also returns its attention map.
5. **Head.** A linear layer maps the class-token output to 10 logits.

Details, tensor shapes, and how the attention overlay is built are in [`docs/architecture.md`](docs/architecture.md).

## Learning notes

| File | Topic |
| --- | --- |
| `stage1.md` | Why patches, positions, attention, and the class token exist |
| `stage2.md` | Load CIFAR-10, split patches, reconstruct the image |
| `stage3.md` | Patch embedding |
| `stage4.md` | Positional embeddings |
| `stage5.md` | Single-head self-attention |
| `stage6.md` | Multi-head attention |
| `stage7.md` | Encoder block |

`src/sandbox.ipynb` is the patch visualization notebook from the early stages.
