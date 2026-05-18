import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_sam.config import HieraConfig


def _to_2tuple(x: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return (int(x[0]), int(x[1]))


def window_partition(x: mx.array, window_size: int) -> tuple[mx.array, tuple[int, int]]:
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
    hp, wp = h + pad_h, w + pad_w
    x = x.reshape(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition(
    windows: mx.array,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> mx.array:
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.reshape(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :]
    return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = [
            nn.Linear(dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[1](nn.gelu(self.layers[0](x)))


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=7, stride=4, padding=3)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 1)
        return self.proj(x)


class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_stride: tuple[int, int] | None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_stride = q_stride
        self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride) if q_stride else None
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        if self.pool is not None:
            q = q.reshape(b, h, w, -1)
            q = self.pool(q)
            h, w = q.shape[1:3]
            q = q.reshape(b, h * w, self.num_heads, -1)

        x = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=1.0 / math.sqrt(q.shape[-1]),
        )
        x = x.transpose(0, 2, 1, 3).reshape(b, h, w, -1)
        return self.proj(x)


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float,
        q_stride: tuple[int, int] | None,
        window_size: int,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.q_stride = q_stride
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_stride)
        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm1(x)

        if self.dim != self.dim_out:
            shortcut = self.pool(self.proj(x))

        window_size = self.window_size
        if window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            h, w = shortcut.shape[1:3]
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            pad_hw = (h + pad_h, w + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (h, w))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Hiera(nn.Module):
    def __init__(self, config: HieraConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.embed_dim)
        self.pos_embed_full = mx.zeros((1, *config.pos_embed_hw, config.embed_dim))

        depth = sum(config.stages)
        self.stage_ends = [sum(config.stages[:i]) - 1 for i in range(1, len(config.stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][: config.q_pool]
        self.global_att_blocks = set(config.global_att_blocks)

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        cur_stage = 1
        blocks = []
        for i in range(depth):
            dim_out = embed_dim
            window_size = config.window_spec[cur_stage - 1]
            if i in self.global_att_blocks:
                window_size = 0
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * config.dim_mul)
                num_heads = int(num_heads * config.head_mul)
                cur_stage += 1
            blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=config.mlp_ratio,
                    q_stride=config.q_stride if i in self.q_pool_blocks else None,
                    window_size=window_size,
                )
            )
            embed_dim = dim_out
        self.blocks = blocks
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]]

    def __call__(self, x: mx.array) -> list[mx.array]:
        x = self.patch_embed(x)
        x = x + self.pos_embed_full[:, : x.shape[1], : x.shape[2], :]
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stage_ends:
                outputs.append(x.transpose(0, 3, 1, 2))
        return outputs
