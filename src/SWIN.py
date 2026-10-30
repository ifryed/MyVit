import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    x: [B, H, W, C]

    return:
        windows: [num_windows * B,
                  window_size,
                  window_size,
                  C]
    """
    B, H, W, C = x.shape
    h_num = H // window_size
    w_num = W // window_size
    windows = x.view(B, h_num, window_size, w_num, window_size, C)
    windows_permute = windows.permute(0, 1, 3, 2, 4, 5)
    windows_reshape = windows_permute.reshape(
        B * h_num * w_num, window_size, window_size, C
    )
    return windows_reshape


def attention_mask(height, width, window_size, shift_size, device):
    """Mask for shifted windows and for padding.

    A cyclic shift only connects neighboring windows. When the feature map
    fits in one window, the shift mask instead cuts attention inside that
    window, so the shift is dropped. Padded cells get their own id so real
    tokens do not attend to them.
    """
    if min(height, width) <= window_size:
        shift_size = 0

    pad_r = (window_size - width % window_size) % window_size
    pad_b = (window_size - height % window_size) % window_size
    padded_h = height + pad_b
    padded_w = width + pad_r
    if shift_size == 0 and pad_r == 0 and pad_b == 0:
        return None, padded_h, padded_w, shift_size

    img_mask = torch.zeros((1, padded_h, padded_w, 1), device=device)
    if shift_size > 0:
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        region = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = region
                region += 1
    if pad_b > 0:
        img_mask[:, height:, :, :] = -1
    if pad_r > 0:
        img_mask[:, :, width:, :] = -1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.reshape(-1, window_size * window_size)
    attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask, padded_h, padded_w, shift_size


def window_reverse(windows, window_size, H, W):
    """
    Reverse window_partition.

    windows:
        [num_windows * B,
         window_size,
         window_size,
         C]

    return:
        x: [B, H, W, C]
    """
    h_num = H // window_size
    w_num = W // window_size
    B, C = windows.shape[0] / h_num / w_num, windows.shape[-1]
    B = int(B)

    x = windows.reshape(B, h_num, w_num, window_size, window_size, C)
    x_permute = x.permute(0, 1, 3, 2, 4, 5)
    x_reshape = x_permute.reshape(B, H, W, C)
    return x_reshape


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, atten_drop=0.0):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads

        self.atten_drop = nn.Dropout(atten_drop)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        Xs = torch.arange(window_size)
        Ys = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([Xs, Ys], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords += window_size - 1

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Convert (row, col) -> one integer
        relative_position_index = (
            relative_coords[0] * (2 * window_size - 1) + relative_coords[1]
        )

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, attn_mask=None):
        """
        x: [B_windows, N, C]

        N = window_size ** 2
        """

        B_, N, C = x.shape

        qkv = self.qkv(x)  # qkv.shape = [B_,N,3*C]
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        atten = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)

        # add bias
        relative_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        atten = atten + relative_bias.unsqueeze(0)
        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            atten = atten.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            atten = atten + attn_mask.unsqueeze(1).unsqueeze(0)
            atten = atten.view(B_, self.num_heads, N, N)

        # softmax
        atten = F.softmax(atten, dim=-1)
        atten = self.atten_drop(atten)
        x = atten @ V
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B_, N, C)
        x = self.proj(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """

        B, H, W, C = x.shape

        assert H % 2 == 0
        assert W % 2 == 0

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinStage(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        downsample=True,
    ):
        super().__init__()

        shift_size = window_size // 2

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(i % 2) * shift_size,
                )
                for i in range(depth)
            ]
        )

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, C_in, H, W]

        return:
            [B, H/patch_size, W/patch_size, embed_dim]
        """

        x = self.proj(x)

        # currently:
        # [B, embed_dim, H', W']

        x = x.permute(0, 2, 3, 1)

        # now:
        # [B, H', W', embed_dim]

        x = self.norm(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0, attn_drop=0.0):
        assert (
            0 <= shift_size < window_size
        ), "shift_size must be in the range of [0, window_size)"
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_drop = attn_drop
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, H, W, C = x.shape

        short_cut = x

        # 1. Normalization
        x = self.norm1(x)

        attn_mask, Hp, Wp, shift_size = attention_mask(
            H, W, self.window_size, self.shift_size, x.device
        )
        if Hp != H or Wp != W:
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))

        # 2. Attention
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        x_window = window_partition(x, self.window_size)

        # [B*num_windows, window_size, window_size, C]
        # 3. Flaten each window to tokens
        x_window = x_window.reshape(-1, self.window_size * self.window_size, C)

        # 4. Apply attention
        x_window = self.attn(x_window, attn_mask=attn_mask)
        x_window = x_window.reshape(-1, self.window_size, self.window_size, C)

        # 5. reverse window_partition
        x = window_reverse(x_window, self.window_size, Hp, Wp)
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        if Hp != H or Wp != W:
            x = x[:, :H, :W, :].contiguous()

        # 6. Add & Norm
        x = short_cut + x
        short_cut = x
        x = self.norm2(x)

        # 7. MLP
        x = short_cut + self.mlp(x)

        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
    ):
        super().__init__()

        if len(depths) != len(num_heads):
            raise ValueError(
                f"depths and num_heads must have the same length, got {len(depths)} and {len(num_heads)}"
            )
        for i, heads in enumerate(num_heads):
            dim = embed_dim * 2**i
            if dim % heads != 0:
                raise ValueError(
                    f"Stage {i} width {dim} is not divisible by {heads} heads"
                )

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.embed_dim = embed_dim

        self.stages = nn.ModuleList()

        for i in range(self.num_layers):
            dim = embed_dim * 2**i

            stage = SwinStage(
                dim=dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=(i < self.num_layers - 1),
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(embed_dim * 2 ** (self.num_layers - 1))
        self.head = nn.Linear(embed_dim * 2 ** (self.num_layers - 1), num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        x = x.mean(dim=[1, 2])
        x = self.head(x)
        return x
