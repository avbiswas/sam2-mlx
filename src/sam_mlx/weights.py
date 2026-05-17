from pathlib import Path

import mlx.core as mx

from sam_mlx.models import Sam2ImageEncoder
from sam_mlx.models import Sam2ImageSegmenter


def load_image_encoder(path: str | Path) -> Sam2ImageEncoder:
    model = Sam2ImageEncoder()
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_image_segmenter(path: str | Path) -> Sam2ImageSegmenter:
    model = Sam2ImageSegmenter()
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
