import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO = ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_REPO))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mlx_sam.preprocess import preprocess_video


MODEL_ID_TO_CONFIG = {
    "facebook/sam2.1-hiera-tiny": "sam2.1/sam2.1_hiera_t.yaml",
    "facebook/sam2.1-hiera-small": "sam2.1/sam2.1_hiera_s.yaml",
    "facebook/sam2.1-hiera-base-plus": "sam2.1/sam2.1_hiera_b+.yaml",
    "facebook/sam2.1-hiera-large": "sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_tiny": "sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1/sam2.1_hiera_l.yaml",
}


def infer_model_id(checkpoint: Path) -> str:
    stem = checkpoint.stem
    for key in MODEL_ID_TO_CONFIG:
        if key in stem:
            return key
    if "tiny" in stem:
        return "sam2.1_hiera_tiny"
    if "small" in stem:
        return "sam2.1_hiera_small"
    if "base_plus" in stem or "b+" in stem:
        return "sam2.1_hiera_base_plus"
    if "large" in stem:
        return "sam2.1_hiera_large"
    raise ValueError(f"Could not infer model id from checkpoint name: {checkpoint}")


def build_model(checkpoint: Path, model_id: str):
    config_dir = SAM2_REPO / "sam2" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=MODEL_ID_TO_CONFIG[model_id])
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path, default=ROOT / "data/torch_prompt_mask.npz")
    parser.add_argument("--point", nargs=2, type=float, default=(588.0, 626.0))
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--multimask", action="store_true", default=True)
    args = parser.parse_args()

    pixels = preprocess_video(args.video, limit=1)
    point_coords = np.array([[args.point]], dtype=np.float32)
    point_labels = np.array([[args.label]], dtype=np.int64)
    model_id = args.model_id or infer_model_id(args.checkpoint)
    model = build_model(args.checkpoint, model_id)

    with torch.inference_mode():
        backbone_out = model.forward_image(torch.from_numpy(pixels))
        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(1, 256, 64, 64)
        pix_feat = pix_feat + model.no_mem_embed.permute(1, 2, 0).view(1, 256, 1, 1)
        out = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs={
                "point_coords": torch.from_numpy(point_coords),
                "point_labels": torch.from_numpy(point_labels).to(torch.int32),
            },
            high_res_features=high_res_features,
            multimask_output=args.multimask,
        )

    low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits = out
    arrays = {
        "pixel_values": pixels,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "low_res_multimasks": low_res_multimasks.cpu().numpy(),
        "high_res_multimasks": high_res_multimasks.cpu().numpy(),
        "ious": ious.cpu().numpy(),
        "low_res_masks": low_res_masks.cpu().numpy(),
        "high_res_masks": high_res_masks.cpu().numpy(),
        "obj_ptr": obj_ptr.cpu().numpy(),
        "object_score_logits": object_score_logits.cpu().numpy(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    meta = {"video": str(args.video), "checkpoint": str(args.checkpoint), "model_id": model_id, "point": list(args.point), "keys": {k: list(v.shape) for k, v in arrays.items()}}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
