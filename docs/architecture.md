# Architecture

MyViT follows the Vision Transformer encoder used for classification: patch tokens, a class token, positional embeddings, a stack of encoder blocks, and a linear head on the class token. It is sized for CIFAR-10 (32×32 RGB, 10 classes), not ImageNet.

Implementation lives in `src/ViT.py` and `src/transBlock.py`. Training, loss, and logging live in `src/pl_model.py`.

## Forward pass

`ViT.forward` expects `(B, 3, H, W)` with `H == W == image_size`. With the defaults (`image_size=32`, `patch_size=4`, `embedding_dim=128`, `num_heads=4`, `num_layers=6`):

| Step | Shape | Where |
| --- | --- | --- |
| Input image | `(B, 3, 32, 32)` | |
| Patch tokens | `(B, 64, 128)` | `PatchEmbedding` |
| Prepend CLS | `(B, 65, 128)` | `ViT.cls_token` |
| Add positions, dropout | `(B, 65, 128)` | `ViT.pos_embed` |
| Each encoder block | `(B, 65, 128)` plus attention `(B, 4, 65, 65)` | `TrasnformerBlock` |
| Class token vector | `(B, 128)` | index `0` of the sequence |
| Logits | `(B, 10)` | `ViT.classifier` |

The forward return value is `(logits, attentions)`, where `attentions` is a list of one map per layer.

`num_patches = (image_size // patch_size) ** 2`. The positional embedding has length `num_patches + 1` so it covers the class token and every patch.

## Patch embedding

`PatchEmbedding` in `src/transBlock.py` cuts the image into a grid of non-overlapping patches and projects each one with a linear layer.

For patch side `P` and `C` channels, each patch is `P × P × C` numbers. On CIFAR with `P=4` that is 48. The linear layer maps 48 → `embedding_dim`. A convolution with kernel and stride equal to `P` is the same map; this code does the reshape explicitly:

1. `(B, C, H, W)` → `(B, C, n_h, P, n_w, P)`
2. Permute so each patch’s pixels sit together
3. Flatten to `(B, n_h * n_w, C * P * P)`
4. `nn.Linear` to `(B, n_patches, embedding_dim)`

The class stores a `position_embedding` parameter, but `ViT` does not pass a `position` index into it. Sequence position is added later by `ViT.pos_embed`, which also has a slot for the class token. Patch embedding only produces content tokens.

## Class token

`cls_token` is a learnable parameter of shape `(1, 1, embedding_dim)`, expanded to the batch and concatenated in front of the patch tokens. It has no pixels of its own. Every block lets it attend to the patches (and to itself). The classification head reads only the class-token position after the last block: `x[:, 0]`.

## Positional embeddings

Self-attention does not use token order. Two patches with the same pixels would be interchangeable if nothing marked where they sat in the grid. `pos_embed` is a learnable tensor of shape `(1, num_patches + 1, embedding_dim)`, added to the token sequence before dropout. Index `0` belongs to the class token; the rest follow patch order (row-major, matching the reshape in `PatchEmbedding`).

## Multi-head self-attention

`MultiHeadAttention` is what the encoder uses.

1. One linear layer produces Q, K, and V: `(B, N, D)` → `(B, N, 3D)`.
2. Split into `num_heads` heads of width `head_dim = D / num_heads`.
3. Scores are `Q Kᵀ / sqrt(head_dim)`, then softmax over the key axis.
4. The weighted values are concatenated across heads and projected back to `D`.

The module returns both the mixed tokens and the attention weights `(B, heads, N, N)`. `N` is `num_patches + 1`.

`Attention` in the same file is the single-head version from the earlier exercise. The encoder does not call it. It computes `Q @ K.T` without batch dimensions, so it is not a drop-in replacement for the multi-head module.

## Encoder block

The class name in code is `TrasnformerBlock`.

```
x = LayerNorm(input)
x = x + MultiHeadAttention(x)
x = x + MLP(x)
```

The MLP is two linear layers with GELU in between. Hidden width is `4 * embedding_dim`. There is a single layer norm, before attention. The MLP residual is applied to the post-attention tensor and does not have its own layer norm. Sequence length and embedding width stay the same through every block.

## Training wrapper

`ViTClassifier` (`src/pl_model.py`) holds a `ViT`, cross-entropy loss, and the optimizer configuration.

- Optimizer: AdamW on all parameters.
- Schedule: `CosineAnnealingLR` with `T_max` equal to `max_epochs`.
- Logged metrics: loss and accuracy for `train` and `val`. Step-level logging is on for training loss only.

`configure_optimizers` returns the optimizer and the scheduler together so Lightning steps the cosine schedule once per epoch.

## Attention overlay

At the end of each validation epoch the module logs an image called `val/attention_map`.

1. Keep the first validation batch (up to 16 images), their labels, predictions, and per-layer attention.
2. Undo CIFAR normalization so the photos are viewable.
3. Average attention over layers: `(layers, B, heads, N, N)` → `(B, heads, N, N)`.
4. Take the class token’s weights on the patch tokens only: index `0` along the query axis, drop the class-token key (`[:, :, 0, 1:]`).
5. Average over heads and reshape to the patch grid `(B, 1, grid, grid)`.
6. Divide by the per-image maximum, bilinear-upsample to 32×32, and blend with a jet colormap (45% photo, 55% color).

`val/predictions` is a text log of `pred` and `label` class names for those same images. `train/samples` is a denormalized grid from the first training batch of epoch 0.

## Data path

`CIFARDataset` lists `*.png` files under `data/cifar-10/train` (or `test`), sorted by numeric filename. For the train split it reads `trainLabels.csv` and returns `(image, class_index)`. Pixel values are `float32` in `[0, 1]`, shape `(3, 32, 32)`.

`split_image_to_patches` and `from_patches_to_image` in `src/cifarDataset.py` are the NumPy helpers used while learning the patch grid. The model does not call them; `PatchEmbedding` does the same split inside the forward pass.

`CIFARDataModule` builds two dataset objects over the same files: one with `RandomHorizontalFlip` plus `Normalize`, one with `Normalize` only. A 10% `random_split` with seed 42 assigns the same indices, so the validation images are held out and are not augmented.

Normalization constants:

```
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```
