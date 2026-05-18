from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_sam.config import model_config_for_name


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2.1-hiera-tiny": "sam2.1_hiera_tiny.pt",
    "facebook/sam2.1-hiera-small": "sam2.1_hiera_small.pt",
    "facebook/sam2.1-hiera-base-plus": "sam2.1_hiera_base_plus.pt",
    "facebook/sam2.1-hiera-large": "sam2.1_hiera_large.pt",
}


def tensor(sd: dict, key: str) -> np.ndarray:
    return sd[key].detach().cpu().numpy()


def conv_weight(sd: dict, key: str) -> np.ndarray:
    # Torch Conv2d: OIHW. MLX Conv2d: OHWI.
    return np.transpose(tensor(sd, key), (0, 2, 3, 1))


def conv_transpose_weight(sd: dict, key: str) -> np.ndarray:
    # Torch ConvTranspose2d: IOHW. MLX ConvTranspose2d: OHWI.
    return np.transpose(tensor(sd, key), (1, 2, 3, 0))


def tiled_window_pos(window: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    repeats = [1, 1, math.ceil(target_hw[0] / window.shape[2]), math.ceil(target_hw[1] / window.shape[3])]
    tiled = np.tile(window, repeats)
    return tiled[:, :, : target_hw[0], : target_hw[1]]


def load_torch_state_dict(checkpoint: str | Path) -> dict:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Conversion requires PyTorch. Run with `uv run --extra torch-parity ...`.") from exc

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    return sd, F


def convert_state_dict(sd: dict, model_id: str) -> dict[str, np.ndarray]:
    config = model_config_for_name(model_id)
    weights: dict[str, np.ndarray] = {}
    _, F = load_torch_state_dict.__globals__.get("_loaded_checkpoint", (None, None)) or (None, None)
    if F is None:
        import torch.nn.functional as F

    pe = sd["image_encoder.trunk.pos_embed"]
    pew = sd["image_encoder.trunk.pos_embed_window"]
    pe = F.interpolate(pe, size=config.hiera.pos_embed_hw, mode="bicubic")
    pe_np = pe.detach().cpu().numpy()
    pew_np = pew.detach().cpu().numpy()
    weights["trunk.pos_embed_full"] = np.transpose(pe_np + tiled_window_pos(pew_np, config.hiera.pos_embed_hw), (0, 2, 3, 1))

    weights["trunk.patch_embed.proj.weight"] = conv_weight(sd, "image_encoder.trunk.patch_embed.proj.weight")
    weights["trunk.patch_embed.proj.bias"] = tensor(sd, "image_encoder.trunk.patch_embed.proj.bias")

    for i in range(sum(config.hiera.stages)):
        prefix = f"image_encoder.trunk.blocks.{i}"
        out = f"trunk.blocks.{i}"
        for name in ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias"):
            weights[f"{out}.{name}"] = tensor(sd, f"{prefix}.{name}")
        weights[f"{out}.mlp.layers.0.weight"] = tensor(sd, f"{prefix}.mlp.layers.0.weight")
        weights[f"{out}.mlp.layers.0.bias"] = tensor(sd, f"{prefix}.mlp.layers.0.bias")
        weights[f"{out}.mlp.layers.1.weight"] = tensor(sd, f"{prefix}.mlp.layers.1.weight")
        weights[f"{out}.mlp.layers.1.bias"] = tensor(sd, f"{prefix}.mlp.layers.1.bias")
        proj_w = f"{prefix}.proj.weight"
        if proj_w in sd:
            weights[f"{out}.proj.weight"] = tensor(sd, proj_w)
            weights[f"{out}.proj.bias"] = tensor(sd, f"{prefix}.proj.bias")

    for i in range(4):
        weights[f"neck.convs.{i}.weight"] = conv_weight(sd, f"image_encoder.neck.convs.{i}.conv.weight")
        weights[f"neck.convs.{i}.bias"] = tensor(sd, f"image_encoder.neck.convs.{i}.conv.bias")

    prefix = "sam_prompt_encoder"
    out = "sam_prompt_encoder"
    weights[f"{out}.pe_layer.positional_encoding_gaussian_matrix"] = tensor(sd, f"{prefix}.pe_layer.positional_encoding_gaussian_matrix")
    for i in range(4):
        weights[f"{out}.point_embeddings.{i}.weight"] = tensor(sd, f"{prefix}.point_embeddings.{i}.weight")
    weights[f"{out}.not_a_point_embed.weight"] = tensor(sd, f"{prefix}.not_a_point_embed.weight")
    weights[f"{out}.no_mask_embed.weight"] = tensor(sd, f"{prefix}.no_mask_embed.weight")
    weights[f"{out}.mask_downscaling_0.weight"] = conv_weight(sd, f"{prefix}.mask_downscaling.0.weight")
    weights[f"{out}.mask_downscaling_0.bias"] = tensor(sd, f"{prefix}.mask_downscaling.0.bias")
    weights[f"{out}.mask_downscaling_1.weight"] = tensor(sd, f"{prefix}.mask_downscaling.1.weight")
    weights[f"{out}.mask_downscaling_1.bias"] = tensor(sd, f"{prefix}.mask_downscaling.1.bias")
    weights[f"{out}.mask_downscaling_3.weight"] = conv_weight(sd, f"{prefix}.mask_downscaling.3.weight")
    weights[f"{out}.mask_downscaling_3.bias"] = tensor(sd, f"{prefix}.mask_downscaling.3.bias")
    weights[f"{out}.mask_downscaling_4.weight"] = tensor(sd, f"{prefix}.mask_downscaling.4.weight")
    weights[f"{out}.mask_downscaling_4.bias"] = tensor(sd, f"{prefix}.mask_downscaling.4.bias")
    weights[f"{out}.mask_downscaling_6.weight"] = conv_weight(sd, f"{prefix}.mask_downscaling.6.weight")
    weights[f"{out}.mask_downscaling_6.bias"] = tensor(sd, f"{prefix}.mask_downscaling.6.bias")

    prefix = "sam_mask_decoder"
    out = "sam_mask_decoder"
    weights[f"{out}.iou_token.weight"] = tensor(sd, f"{prefix}.iou_token.weight")
    weights[f"{out}.mask_tokens.weight"] = tensor(sd, f"{prefix}.mask_tokens.weight")
    weights[f"{out}.obj_score_token.weight"] = tensor(sd, f"{prefix}.obj_score_token.weight")
    weights[f"{out}.output_upscaling_0.weight"] = conv_transpose_weight(sd, f"{prefix}.output_upscaling.0.weight")
    weights[f"{out}.output_upscaling_0.bias"] = tensor(sd, f"{prefix}.output_upscaling.0.bias")
    weights[f"{out}.output_upscaling_1.weight"] = tensor(sd, f"{prefix}.output_upscaling.1.weight")
    weights[f"{out}.output_upscaling_1.bias"] = tensor(sd, f"{prefix}.output_upscaling.1.bias")
    weights[f"{out}.output_upscaling_3.weight"] = conv_transpose_weight(sd, f"{prefix}.output_upscaling.3.weight")
    weights[f"{out}.output_upscaling_3.bias"] = tensor(sd, f"{prefix}.output_upscaling.3.bias")
    weights[f"{out}.conv_s0.weight"] = conv_weight(sd, f"{prefix}.conv_s0.weight")
    weights[f"{out}.conv_s0.bias"] = tensor(sd, f"{prefix}.conv_s0.bias")
    weights[f"{out}.conv_s1.weight"] = conv_weight(sd, f"{prefix}.conv_s1.weight")
    weights[f"{out}.conv_s1.bias"] = tensor(sd, f"{prefix}.conv_s1.bias")

    for layer in range(2):
        for attn_name in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                base = f"{prefix}.transformer.layers.{layer}.{attn_name}.{proj_name}"
                dest = f"{out}.transformer.layers.{layer}.{attn_name}.{proj_name}"
                weights[f"{dest}.weight"] = tensor(sd, f"{base}.weight")
                weights[f"{dest}.bias"] = tensor(sd, f"{base}.bias")
        for norm in ("norm1", "norm2", "norm3", "norm4"):
            weights[f"{out}.transformer.layers.{layer}.{norm}.weight"] = tensor(sd, f"{prefix}.transformer.layers.{layer}.{norm}.weight")
            weights[f"{out}.transformer.layers.{layer}.{norm}.bias"] = tensor(sd, f"{prefix}.transformer.layers.{layer}.{norm}.bias")
        for mlp_layer in range(2):
            weights[f"{out}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.weight"] = tensor(sd, f"{prefix}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.weight")
            weights[f"{out}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.bias"] = tensor(sd, f"{prefix}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.bias")

    for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        weights[f"{out}.transformer.final_attn_token_to_image.{proj_name}.weight"] = tensor(sd, f"{prefix}.transformer.final_attn_token_to_image.{proj_name}.weight")
        weights[f"{out}.transformer.final_attn_token_to_image.{proj_name}.bias"] = tensor(sd, f"{prefix}.transformer.final_attn_token_to_image.{proj_name}.bias")
    weights[f"{out}.transformer.norm_final_attn.weight"] = tensor(sd, f"{prefix}.transformer.norm_final_attn.weight")
    weights[f"{out}.transformer.norm_final_attn.bias"] = tensor(sd, f"{prefix}.transformer.norm_final_attn.bias")

    for i in range(4):
        for layer in range(3):
            weights[f"{out}.output_hypernetworks_mlps.{i}.layers.{layer}.weight"] = tensor(sd, f"{prefix}.output_hypernetworks_mlps.{i}.layers.{layer}.weight")
            weights[f"{out}.output_hypernetworks_mlps.{i}.layers.{layer}.bias"] = tensor(sd, f"{prefix}.output_hypernetworks_mlps.{i}.layers.{layer}.bias")
    for head in ("iou_prediction_head", "pred_obj_score_head"):
        for layer in range(3):
            weights[f"{out}.{head}.layers.{layer}.weight"] = tensor(sd, f"{prefix}.{head}.layers.{layer}.weight")
            weights[f"{out}.{head}.layers.{layer}.bias"] = tensor(sd, f"{prefix}.{head}.layers.{layer}.bias")

    weights["no_obj_ptr"] = tensor(sd, "no_obj_ptr")
    weights["no_mem_embed"] = tensor(sd, "no_mem_embed")
    weights["no_mem_pos_enc"] = tensor(sd, "no_mem_pos_enc")
    weights["maskmem_tpos_enc"] = tensor(sd, "maskmem_tpos_enc")
    weights["no_obj_embed_spatial"] = tensor(sd, "no_obj_embed_spatial")
    for layer in range(3):
        weights[f"obj_ptr_proj.layers.{layer}.weight"] = tensor(sd, f"obj_ptr_proj.layers.{layer}.weight")
        weights[f"obj_ptr_proj.layers.{layer}.bias"] = tensor(sd, f"obj_ptr_proj.layers.{layer}.bias")
    weights["obj_ptr_tpos_proj.weight"] = tensor(sd, "obj_ptr_tpos_proj.weight")
    weights["obj_ptr_tpos_proj.bias"] = tensor(sd, "obj_ptr_tpos_proj.bias")

    me = "memory_encoder"
    out = "memory_encoder"
    conv_map = [(0, "conv0"), (3, "conv1"), (6, "conv2"), (9, "conv3"), (12, "conv4")]
    norm_map = [(1, "norm0"), (4, "norm1"), (7, "norm2"), (10, "norm3")]
    for idx, name in conv_map:
        weights[f"{out}.mask_downsampler.{name}.weight"] = conv_weight(sd, f"{me}.mask_downsampler.encoder.{idx}.weight")
        weights[f"{out}.mask_downsampler.{name}.bias"] = tensor(sd, f"{me}.mask_downsampler.encoder.{idx}.bias")
    for idx, name in norm_map:
        weights[f"{out}.mask_downsampler.{name}.weight"] = tensor(sd, f"{me}.mask_downsampler.encoder.{idx}.weight")
        weights[f"{out}.mask_downsampler.{name}.bias"] = tensor(sd, f"{me}.mask_downsampler.encoder.{idx}.bias")
    weights[f"{out}.pix_feat_proj.weight"] = conv_weight(sd, f"{me}.pix_feat_proj.weight")
    weights[f"{out}.pix_feat_proj.bias"] = tensor(sd, f"{me}.pix_feat_proj.bias")
    weights[f"{out}.out_proj.weight"] = conv_weight(sd, f"{me}.out_proj.weight")
    weights[f"{out}.out_proj.bias"] = tensor(sd, f"{me}.out_proj.bias")
    for i in range(2):
        src = f"{me}.fuser.layers.{i}"
        dst = f"{out}.fuser.{i}"
        weights[f"{dst}.gamma"] = tensor(sd, f"{src}.gamma")
        weights[f"{dst}.dwconv.weight"] = conv_weight(sd, f"{src}.dwconv.weight")
        weights[f"{dst}.dwconv.bias"] = tensor(sd, f"{src}.dwconv.bias")
        weights[f"{dst}.norm.weight"] = tensor(sd, f"{src}.norm.weight")
        weights[f"{dst}.norm.bias"] = tensor(sd, f"{src}.norm.bias")
        weights[f"{dst}.pwconv1.weight"] = tensor(sd, f"{src}.pwconv1.weight")
        weights[f"{dst}.pwconv1.bias"] = tensor(sd, f"{src}.pwconv1.bias")
        weights[f"{dst}.pwconv2.weight"] = tensor(sd, f"{src}.pwconv2.weight")
        weights[f"{dst}.pwconv2.bias"] = tensor(sd, f"{src}.pwconv2.bias")

    ma = "memory_attention"
    out = "memory_attention"
    for i in range(4):
        for attn_name in ("self_attn", "cross_attn_image"):
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                weights[f"{out}.layers.{i}.{attn_name}.{proj_name}.weight"] = tensor(sd, f"{ma}.layers.{i}.{attn_name}.{proj_name}.weight")
                weights[f"{out}.layers.{i}.{attn_name}.{proj_name}.bias"] = tensor(sd, f"{ma}.layers.{i}.{attn_name}.{proj_name}.bias")
        for lin in ("linear1", "linear2"):
            weights[f"{out}.layers.{i}.{lin}.weight"] = tensor(sd, f"{ma}.layers.{i}.{lin}.weight")
            weights[f"{out}.layers.{i}.{lin}.bias"] = tensor(sd, f"{ma}.layers.{i}.{lin}.bias")
        for norm in ("norm1", "norm2", "norm3"):
            weights[f"{out}.layers.{i}.{norm}.weight"] = tensor(sd, f"{ma}.layers.{i}.{norm}.weight")
            weights[f"{out}.layers.{i}.{norm}.bias"] = tensor(sd, f"{ma}.layers.{i}.{norm}.bias")
    weights[f"{out}.norm.weight"] = tensor(sd, f"{ma}.norm.weight")
    weights[f"{out}.norm.bias"] = tensor(sd, f"{ma}.norm.bias")
    return weights


def download_hf_checkpoint(model_id: str, cache_dir: str | Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    if model_id not in HF_MODEL_ID_TO_FILENAMES:
        raise ValueError(f"Unsupported HF SAM2.1 model id: {model_id}. Supported: {', '.join(HF_MODEL_ID_TO_FILENAMES)}")
    return Path(hf_hub_download(repo_id=model_id, filename=HF_MODEL_ID_TO_FILENAMES[model_id], cache_dir=cache_dir))


def default_output_path(checkpoint: Path, output_dir: Path) -> Path:
    return output_dir / f"{checkpoint.stem}_image_segmenter.safetensors"


def convert_checkpoint(checkpoint: str | Path, output: str | Path, model_id: str) -> Path:
    sd, F = load_torch_state_dict(checkpoint)
    load_torch_state_dict.__globals__["_loaded_checkpoint"] = (sd, F)
    weights = convert_state_dict(sd, model_id)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output),
        {key: mx.array(value.astype(np.float32)) for key, value in weights.items()},
        metadata={"format": "mlx", "model_id": model_id},
    )
    print(f"saved {len(weights)} tensors to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SAM2.1 Torch checkpoints to MLX safetensors.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-id", choices=HF_MODEL_ID_TO_FILENAMES.keys(), help="Hugging Face SAM2.1 model id to download and convert.")
    source.add_argument("--checkpoint", type=Path, help="Local SAM2.1 .pt checkpoint to convert.")
    parser.add_argument("--model-id", help="Model id/config name for local checkpoints, e.g. facebook/sam2.1-hiera-small.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    if args.hf_id:
        checkpoint = download_hf_checkpoint(args.hf_id, cache_dir=args.cache_dir)
        model_id = args.hf_id
    else:
        checkpoint = args.checkpoint
        model_id = args.model_id or checkpoint.stem

    output = args.output or default_output_path(checkpoint, args.output_dir)
    convert_checkpoint(checkpoint, output, model_id)


if __name__ == "__main__":
    main()
