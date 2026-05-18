from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn

from mlx_sam.config import model_config_for_name
from mlx_sam.models import Sam2ImageEncoder
from mlx_sam.models import Sam2ImageSegmenter


def _quantization_config(path: Path) -> dict | None:
    config_path = path.with_suffix(path.suffix + ".json")
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    return config.get("quantization")


def sam2_quant_predicate(group_size: int = 64, recipe: str = "all", bits: int = 4):
    """Quantize SAM2 linear layers that have group-aligned weight dimensions."""

    skip_names = (
        "point_embeddings",
        "not_a_point_embed",
        "no_mask_embed",
        "iou_token",
        "mask_tokens",
        "obj_score_token",
        "position_encoding",
    )

    def predicate(path: str, module):
        if not isinstance(module, nn.Linear):
            return False
        if any(name in path for name in skip_names):
            return False
        if not hasattr(module, "weight"):
            return False
        shape = module.weight.shape
        if len(shape) != 2 or any(dim % group_size != 0 for dim in shape):
            return False
        if recipe == "q8_trunk_mask_q4_memory":
            if path.startswith("trunk.blocks") or path.startswith("sam_mask_decoder"):
                return {"group_size": group_size, "bits": 8, "mode": "affine"}
            if path.startswith("memory_attention") or path.startswith("memory_encoder.fuser") or path.startswith("obj_ptr"):
                return {"group_size": group_size, "bits": 4, "mode": "affine"}
            return False
        if recipe == "trunk":
            return path.startswith("trunk.blocks")
        if recipe == "trunk_memory":
            return path.startswith("trunk.blocks") or path.startswith("memory_attention") or path.startswith("memory_encoder.fuser")
        if recipe == "trunk_memory_obj":
            return (
                path.startswith("trunk.blocks")
                or path.startswith("memory_attention")
                or path.startswith("memory_encoder.fuser")
                or path.startswith("obj_ptr")
            )
        if recipe == "no_mask_decoder":
            return not path.startswith("sam_mask_decoder") and not path.startswith("sam_prompt_encoder")
        if recipe != "all":
            raise ValueError(f"Unknown SAM2 quantization recipe: {recipe}")
        return {"group_size": group_size, "bits": bits, "mode": "affine"}

    return predicate


def apply_quantization(model, quantization: dict | None):
    if not quantization:
        return model
    nn.quantize(
        model,
        group_size=int(quantization.get("group_size", 64)),
        bits=int(quantization["bits"]),
        mode=quantization.get("mode", "affine"),
        class_predicate=sam2_quant_predicate(
            int(quantization.get("group_size", 64)),
            recipe=quantization.get("recipe", "all"),
            bits=int(quantization["bits"]),
        ),
    )
    return model


def load_image_encoder(path: str | Path, model_id: str | None = None) -> Sam2ImageEncoder:
    path = Path(path)
    config = model_config_for_name(model_id or path.name)
    model = Sam2ImageEncoder(config=config)
    apply_quantization(model, _quantization_config(path))
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_image_segmenter(path: str | Path, model_id: str | None = None) -> Sam2ImageSegmenter:
    path = Path(path)
    config = model_config_for_name(model_id or path.name)
    model = Sam2ImageSegmenter(config=config)
    apply_quantization(model, _quantization_config(path))
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
