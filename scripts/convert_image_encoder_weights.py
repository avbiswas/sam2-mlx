import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def tensor(sd, key):
    return sd[key].detach().cpu().numpy()


def conv_weight(sd, key):
    # Torch Conv2d: OIHW. MLX Conv2d: OHWI.
    return np.transpose(tensor(sd, key), (0, 2, 3, 1))


def conv_transpose_weight(sd, key):
    # Torch ConvTranspose2d: IOHW. MLX ConvTranspose2d: OHWI.
    return np.transpose(tensor(sd, key), (1, 2, 3, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    args = parser.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    weights = {}

    pe = sd["image_encoder.trunk.pos_embed"]
    pew = sd["image_encoder.trunk.pos_embed_window"]
    pe = F.interpolate(pe, size=(256, 256), mode="bicubic")
    tiled = pew.tile([x // y for x, y in zip(pe.shape, pew.shape)])
    weights["trunk.pos_embed_full"] = np.transpose((pe + tiled).numpy(), (0, 2, 3, 1))

    weights["trunk.patch_embed.proj.weight"] = conv_weight(sd, "image_encoder.trunk.patch_embed.proj.weight")
    weights["trunk.patch_embed.proj.bias"] = tensor(sd, "image_encoder.trunk.patch_embed.proj.bias")

    for i in range(16):
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), {k: mx.array(v.astype(np.float32)) for k, v in weights.items()}, metadata={"format": "mlx"})
    print(f"saved {len(weights)} tensors to {args.output}")


if __name__ == "__main__":
    main()
