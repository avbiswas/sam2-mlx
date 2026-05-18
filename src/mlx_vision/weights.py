from pathlib import Path

import mlx.core as mx

from mlx_vision.config import model_config_for_name
from mlx_vision.models import Sam2ImageEncoder
from mlx_vision.models import Sam2ImageSegmenter


def load_image_encoder(path: str | Path, model_id: str | None = None) -> Sam2ImageEncoder:
    path = Path(path)
    config = model_config_for_name(model_id or path.name)
    model = Sam2ImageEncoder(config=config)
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_image_segmenter(path: str | Path, model_id: str | None = None) -> Sam2ImageSegmenter:
    path = Path(path)
    config = model_config_for_name(model_id or path.name)
    model = Sam2ImageSegmenter(config=config)
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
