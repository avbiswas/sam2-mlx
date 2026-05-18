import math

import mlx.core as mx
import mlx.nn as nn

from sam_mlx.models.image_encoder import PositionEmbeddingSine
from sam_mlx.models.sam_heads import Attention, LayerNorm2d


def _resize_1d(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = x.shape[axis]
    if in_size == out_size:
        return x
    scale = in_size / out_size
    src = (mx.arange(out_size, dtype=mx.float32) + 0.5) * scale - 0.5
    lo_f = mx.floor(src)
    lo = mx.clip(lo_f, 0, in_size - 1).astype(mx.int32)
    hi = mx.clip(lo_f + 1, 0, in_size - 1).astype(mx.int32)
    w = (src - lo_f).astype(x.dtype)

    lo_vals = mx.take(x, lo, axis=axis)
    hi_vals = mx.take(x, hi, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_size
    w = w.reshape(shape)
    return lo_vals * (1.0 - w) + hi_vals * w


def upsample_mask(mask: mx.array, size: tuple[int, int]) -> mx.array:
    x = _resize_1d(mask, size[0], axis=2)
    return _resize_1d(x, size[1], axis=3)


class MaskDownSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.norm0 = LayerNorm2d(4)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = LayerNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = LayerNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv0(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = nn.gelu(self.norm0(x))
        x = self.conv1(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = nn.gelu(self.norm1(x))
        x = self.conv2(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = nn.gelu(self.norm2(x))
        x = self.conv3(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = nn.gelu(self.norm3(x))
        return self.conv4(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


class CXBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = mx.ones((256,)) * 1e-6
        self.dwconv = nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256)
        self.norm = LayerNorm2d(256)
        self.pwconv1 = nn.Linear(256, 1024)
        self.pwconv2 = nn.Linear(1024, 256)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = self.norm(x).transpose(0, 2, 3, 1)
        x = self.pwconv2(nn.gelu(self.pwconv1(x)))
        x = self.gamma * x
        return residual + x.transpose(0, 3, 1, 2)


class MemoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_downsampler = MaskDownSampler()
        self.pix_feat_proj = nn.Conv2d(256, 256, kernel_size=1)
        self.fuser = [CXBlock(), CXBlock()]
        self.out_proj = nn.Conv2d(256, 64, kernel_size=1)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)

    def __call__(self, pix_feat: mx.array, masks_high_res: mx.array, skip_mask_sigmoid: bool = False) -> dict:
        masks = masks_high_res if skip_mask_sigmoid else mx.sigmoid(masks_high_res)
        masks = self.mask_downsampler(masks)
        x = self.pix_feat_proj(pix_feat.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = x + masks
        for layer in self.fuser:
            x = layer(x)
        x = self.out_proj(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return {"vision_features": x, "vision_pos_enc": [self.position_encoding(x).astype(x.dtype)]}


def compute_axial_rope(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 4, dtype=mx.float32)[: dim // 4] / dim))
    t = mx.arange(end_x * end_y, dtype=mx.float32)
    tx = t % end_x
    ty = mx.floor(t / end_x)
    phase = mx.concatenate([tx[:, None] * freqs[None, :], ty[:, None] * freqs[None, :]], axis=-1)
    return mx.cos(phase), mx.sin(phase)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array, repeat: int = 1) -> mx.array:
    if repeat != 1:
        cos = mx.repeat(cos[:, None, :], repeat, axis=1).reshape(-1, cos.shape[-1])
        sin = mx.repeat(sin[:, None, :], repeat, axis=1).reshape(-1, sin.shape[-1])
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos
    return mx.stack([out_even, out_odd], axis=-1).reshape(x.shape)


class RoPEAttention(Attention):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 1, downsample_rate: int = 1, kv_in_dim: int | None = None, rope_k_repeat: bool = False):
        super().__init__(embedding_dim, num_heads, downsample_rate=downsample_rate, kv_in_dim=kv_in_dim)
        self.rope_k_repeat = rope_k_repeat
        self._rope_cache = None

    def __call__(self, q: mx.array, k: mx.array, v: mx.array, num_k_exclude_rope: int = 0) -> mx.array:
        if self._rope_cache is None:
            self._rope_cache = compute_axial_rope(self.internal_dim // self.num_heads, 64, 64)
            mx.eval(*self._rope_cache)
        cos, sin = self._rope_cache
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        repeat = k.shape[-2] // q.shape[-2] if self.rope_k_repeat and q.shape[-2] != 0 else 1
        q = apply_rope(q, cos, sin)
        rope_len = k.shape[-2] - num_k_exclude_rope
        if rope_len > 0:
            k_rope = apply_rope(k[:, :, :rope_len, :], cos, sin, repeat=repeat)
            if num_k_exclude_rope > 0:
                k = mx.concatenate([k_rope, k[:, :, rope_len:, :]], axis=2)
            else:
                k = k_rope
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]))
        return self.out_proj(self._recombine_heads(out))


class MemoryAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = RoPEAttention(256, 1)
        self.cross_attn_image = RoPEAttention(256, 1, kv_in_dim=64, rope_k_repeat=True)
        self.linear1 = nn.Linear(256, 2048)
        self.linear2 = nn.Linear(2048, 256)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

    def __call__(self, tgt: mx.array, memory: mx.array, pos: mx.array, query_pos: mx.array, num_k_exclude_rope: int = 0) -> mx.array:
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, tgt2)
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.cross_attn_image(tgt2, memory + pos, memory, num_k_exclude_rope=num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        return tgt + self.linear2(nn.relu(self.linear1(tgt2)))


class MemoryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [MemoryAttentionLayer() for _ in range(4)]
        self.norm = nn.LayerNorm(256)

    def __call__(self, curr: mx.array, curr_pos: mx.array, memory: mx.array, memory_pos: mx.array, num_obj_ptr_tokens: int = 0) -> mx.array:
        output = curr + 0.1 * curr_pos
        output = output.transpose(1, 0, 2)
        curr_pos = curr_pos.transpose(1, 0, 2)
        memory = memory.transpose(1, 0, 2)
        memory_pos = memory_pos.transpose(1, 0, 2)
        for layer in self.layers:
            output = layer(output, memory, memory_pos, curr_pos, num_k_exclude_rope=num_obj_ptr_tokens)
        return self.norm(output).transpose(1, 0, 2)
