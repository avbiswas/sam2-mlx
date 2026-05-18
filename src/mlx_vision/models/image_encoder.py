import math

import mlx.core as mx
import mlx.nn as nn

from mlx_vision.config import SAM2_1_HIERA_SMALL_IMAGE_ENCODER, Sam2ImageEncoderConfig
from mlx_vision.models.hiera import Hiera


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 256, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi
        self._cache = {}

    def __call__(self, x: mx.array) -> mx.array:
        b, _, h, w = x.shape
        key = (b, h, w, str(x.dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        y = mx.arange(1, h + 1, dtype=mx.float32).reshape(1, h, 1)
        x_pos = mx.arange(1, w + 1, dtype=mx.float32).reshape(1, 1, w)
        y = mx.broadcast_to(y, (b, h, w))
        x_pos = mx.broadcast_to(x_pos, (b, h, w))
        if self.normalize:
            eps = 1e-6
            y = y / (y[:, -1:, :] + eps) * self.scale
            x_pos = x_pos / (x_pos[:, :, -1:] + eps) * self.scale
        dim_t = mx.arange(self.num_pos_feats, dtype=mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_pos[..., None] / dim_t
        pos_y = y[..., None] / dim_t
        pos_x = mx.stack([mx.sin(pos_x[..., 0::2]), mx.cos(pos_x[..., 1::2])], axis=4).reshape(b, h, w, -1)
        pos_y = mx.stack([mx.sin(pos_y[..., 0::2]), mx.cos(pos_y[..., 1::2])], axis=4).reshape(b, h, w, -1)
        out = mx.concatenate([pos_y, pos_x], axis=3).transpose(0, 3, 1, 2).astype(x.dtype)
        mx.eval(out)
        self._cache[key] = out
        return out


def upsample_nearest_2x(x: mx.array) -> mx.array:
    return mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)


class FpnNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.backbone_channel_list = config.backbone_channel_list
        self.d_model = config.d_model
        self.fpn_top_down_levels = set(config.fpn_top_down_levels)
        self.convs = [nn.Conv2d(dim, config.d_model, kernel_size=1) for dim in config.backbone_channel_list]

    def __call__(self, xs: list[mx.array]) -> tuple[list[mx.array], list[mx.array]]:
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        prev = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i].transpose(0, 2, 3, 1)
            lateral = self.convs[n - i](x).transpose(0, 3, 1, 2)
            if i in self.fpn_top_down_levels and prev is not None:
                prev = lateral + upsample_nearest_2x(prev).astype(mx.float32)
            else:
                prev = lateral
            out[i] = prev
            pos[i] = self.position_encoding(prev).astype(prev.dtype)
        return out, pos


class Sam2ImageEncoder(nn.Module):
    def __init__(self, config: Sam2ImageEncoderConfig = SAM2_1_HIERA_SMALL_IMAGE_ENCODER):
        super().__init__()
        self.config = config
        self.trunk = Hiera(config.hiera)
        self.neck = FpnNeck(config.fpn)
        self.scalp = config.fpn.scalp

    def __call__(self, sample: mx.array) -> dict[str, mx.array | list[mx.array]]:
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features = features[: -self.scalp]
            pos = pos[: -self.scalp]
        return {
            "vision_features": features[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
