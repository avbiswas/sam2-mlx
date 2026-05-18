import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO = ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_REPO))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mlx_vision.weights import load_image_segmenter


MODEL_ID_TO_CONFIG = {
    "facebook/sam2.1-hiera-tiny": "sam2.1/sam2.1_hiera_t.yaml",
    "facebook/sam2.1-hiera-small": "sam2.1/sam2.1_hiera_s.yaml",
    "facebook/sam2.1-hiera-base-plus": "sam2.1/sam2.1_hiera_b+.yaml",
    "facebook/sam2.1-hiera-large": "sam2.1/sam2.1_hiera_l.yaml",
}


def sync_torch(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def summary(values: list[float]) -> dict:
    return {
        "runs": len(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def build_torch_model(checkpoint: Path, model_id: str, device: torch.device):
    config_dir = SAM2_REPO / "sam2" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=MODEL_ID_TO_CONFIG[model_id])
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    model.eval()
    return model.to(device)


def torch_encoded_prompt(model, pixels: torch.Tensor, point_coords: torch.Tensor, point_labels: torch.Tensor):
    backbone_out = model.forward_image(pixels)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    high_res_features = [
        x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
        for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
    ]
    pix_feat = vision_feats[-1].permute(1, 2, 0).view(1, 256, 64, 64)
    pix_feat = pix_feat + model.no_mem_embed.permute(1, 2, 0).view(1, 256, 1, 1)
    return model._forward_sam_heads(
        backbone_features=pix_feat,
        point_inputs={"point_coords": point_coords, "point_labels": point_labels},
        high_res_features=high_res_features,
        multimask_output=True,
    )


def torch_encode_once(model, pixels: torch.Tensor):
    backbone_out = model.forward_image(pixels)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    high_res_features = [
        x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
        for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
    ]
    pix_feat = vision_feats[-1].permute(1, 2, 0).view(1, 256, 64, 64)
    pix_feat = pix_feat + model.no_mem_embed.permute(1, 2, 0).view(1, 256, 1, 1)
    return pix_feat, high_res_features


def torch_prompt_decode(model, pix_feat, high_res_features, point_coords, point_labels):
    return model._forward_sam_heads(
        backbone_features=pix_feat,
        point_inputs={"point_coords": point_coords, "point_labels": point_labels},
        high_res_features=high_res_features,
        multimask_output=True,
    )


def benchmark_torch(ref: np.lib.npyio.NpzFile, checkpoint: Path, model_id: str, warmup: int, runs: int) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_torch_model(checkpoint, model_id, device)
    pixels = torch.from_numpy(ref["pixel_values"]).to(device)
    point_coords = torch.from_numpy(ref["point_coords"]).to(device)
    point_labels = torch.from_numpy(ref["point_labels"]).to(device).to(torch.int32)

    with torch.inference_mode():
        for _ in range(warmup):
            torch_encoded_prompt(model, pixels, point_coords, point_labels)
            sync_torch(device)
        full = []
        for _ in range(runs):
            start = time.perf_counter()
            torch_encoded_prompt(model, pixels, point_coords, point_labels)
            sync_torch(device)
            full.append((time.perf_counter() - start) * 1000.0)

        pix_feat, high_res_features = torch_encode_once(model, pixels)
        sync_torch(device)
        for _ in range(warmup):
            torch_prompt_decode(model, pix_feat, high_res_features, point_coords, point_labels)
            sync_torch(device)
        prompt = []
        for _ in range(runs):
            start = time.perf_counter()
            torch_prompt_decode(model, pix_feat, high_res_features, point_coords, point_labels)
            sync_torch(device)
            prompt.append((time.perf_counter() - start) * 1000.0)
    return {"device": str(device), "full_image_plus_prompt": summary(full), "prompt_decode_with_cached_image": summary(prompt)}


def benchmark_mlx(ref: np.lib.npyio.NpzFile, weights: Path, model_id: str, warmup: int, runs: int) -> dict:
    model = load_image_segmenter(weights, model_id=model_id)
    pixels = mx.array(ref["pixel_values"])
    coords = mx.array(ref["point_coords"])
    labels = mx.array(ref["point_labels"])

    for _ in range(warmup):
        out = model(pixels, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])
    full = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model(pixels, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])
        full.append((time.perf_counter() - start) * 1000.0)

    encoded = model.encode_image(pixels)
    mx.eval(encoded["vision_features"], encoded["high_res_features"])
    for _ in range(warmup):
        out = model.predict_from_encoded(encoded, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])
    prompt = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model.predict_from_encoded(encoded, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])
        prompt.append((time.perf_counter() - start) * 1000.0)
    return {"device": "mlx-default", "full_image_plus_prompt": summary(full), "prompt_decode_with_cached_image": summary(prompt)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=MODEL_ID_TO_CONFIG.keys(), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    ref = np.load(args.reference)
    torch_report = benchmark_torch(ref, args.checkpoint, args.model_id, args.warmup, args.runs)
    mlx_report = benchmark_mlx(ref, args.weights, args.model_id, args.warmup, args.runs)
    report = {
        "model_id": args.model_id,
        "input_shape": list(ref["pixel_values"].shape),
        "torch": torch_report,
        "mlx": mlx_report,
        "speedup_full_image_plus_prompt": torch_report["full_image_plus_prompt"]["mean_ms"] / mlx_report["full_image_plus_prompt"]["mean_ms"],
        "speedup_prompt_decode_with_cached_image": torch_report["prompt_decode_with_cached_image"]["mean_ms"] / mlx_report["prompt_decode_with_cached_image"]["mean_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
