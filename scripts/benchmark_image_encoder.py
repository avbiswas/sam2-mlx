import argparse
import json
import statistics
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

from mlx_vision.weights import load_image_encoder


def build_torch_model(checkpoint: Path, device: torch.device):
    config_dir = SAM2_REPO / "sam2" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="sam2.1/sam2.1_hiera_s.yaml")
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd)
    return model.image_encoder.eval().to(device)


def sync_torch(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(times_ms: list[float]) -> dict:
    return {
        "runs": len(times_ms),
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "stdev_ms": float(statistics.stdev(times_ms)) if len(times_ms) > 1 else 0.0,
    }


def benchmark_torch(pixels: np.ndarray, checkpoint: Path, warmup: int, runs: int) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_torch_model(checkpoint, device)
    x = torch.from_numpy(pixels).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            out = model(x)
            _ = out["vision_features"]
            sync_torch(device)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            out = model(x)
            _ = out["vision_features"]
            sync_torch(device)
            times.append((time.perf_counter() - start) * 1000.0)
    return {"device": str(device), **summarize(times)}


def benchmark_mlx(pixels: np.ndarray, weights: Path, warmup: int, runs: int) -> dict:
    model = load_image_encoder(weights)
    x = mx.array(pixels)
    for _ in range(warmup):
        out = model(x)
        mx.eval(out["vision_features"], out["backbone_fpn"], out["vision_pos_enc"])
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model(x)
        mx.eval(out["vision_features"], out["backbone_fpn"], out["vision_pos_enc"])
        times.append((time.perf_counter() - start) * 1000.0)
    return {"device": "mlx-default", **summarize(times)}


def count_params_safetensors(weights: Path) -> int:
    arrays = mx.load(str(weights))
    return int(sum(array.size for array in arrays.values()))


def estimate_image_encoder_macs() -> dict:
    h = w = 256
    embed = 96
    heads = 1
    stages = (1, 2, 11, 2)
    stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
    q_pool_blocks = [x + 1 for x in stage_ends[:-1]][:3]
    global_blocks = {7, 10, 13}
    window_spec = (8, 4, 14, 7)
    cur_stage = 1
    macs = 0
    breakdown = {}

    patch = h * w * embed * 3 * 7 * 7
    macs += patch
    breakdown["patch_embed"] = patch

    attn_total = 0
    mlp_total = 0
    linear_total = 0
    for i in range(sum(stages)):
        dim = embed
        dim_out = embed
        window = window_spec[cur_stage - 1]
        if i in global_blocks:
            window = 0
        if i - 1 in stage_ends:
            dim_out = embed * 2
            heads *= 2
            cur_stage += 1
        q_pool = i in q_pool_blocks
        n = h * w
        qh, qw = (h // 2, w // 2) if q_pool else (h, w)
        nq = qh * qw

        qkv = n * dim * dim_out * 3
        proj = nq * dim_out * dim_out
        skip = n * dim * dim_out if dim != dim_out else 0
        hidden = int(dim_out * 4)
        mlp = nq * (dim_out * hidden + hidden * dim_out)
        if window > 0:
            num_windows = (h // window) * (w // window)
            q_tokens = (window // 2) * (window // 2) if q_pool else window * window
            kv_tokens = window * window
            attn = 2 * num_windows * q_tokens * kv_tokens * dim_out
        else:
            attn = 2 * nq * n * dim_out

        linear_total += qkv + proj + skip
        attn_total += attn
        mlp_total += mlp
        macs += qkv + proj + skip + attn + mlp
        if q_pool:
            h, w = qh, qw
        embed = dim_out

    fpn = (
        64 * 64 * 768 * 256
        + 128 * 128 * 384 * 256
        + 256 * 256 * 192 * 256
        + 256 * 256 * 96 * 256
    )
    macs += fpn
    breakdown.update({
        "attention": attn_total,
        "linear_qkv_proj_skip": linear_total,
        "mlp": mlp_total,
        "fpn_1x1": fpn,
        "total": macs,
    })
    return {k: int(v) for k, v in breakdown.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/torch_image_embeddings.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_encoder.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks/image_encoder_latency.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    ref = np.load(args.reference)
    pixels = ref["pixel_values"]
    batch = int(pixels.shape[0])
    macs = estimate_image_encoder_macs()
    report = {
        "input_shape": list(pixels.shape),
        "batch_size": batch,
        "params": count_params_safetensors(args.weights),
        "approx_macs_per_frame": macs["total"],
        "approx_gmacs_per_frame": macs["total"] / 1e9,
        "approx_macs_breakdown": macs,
        "torch": benchmark_torch(pixels, args.checkpoint, args.warmup, args.runs),
        "mlx": benchmark_mlx(pixels, args.weights, args.warmup, args.runs),
    }
    for key in ("torch", "mlx"):
        report[key]["mean_ms_per_frame"] = report[key]["mean_ms"] / batch
        report[key]["median_ms_per_frame"] = report[key]["median_ms"] / batch
    report["speedup_mlx_vs_torch_mean"] = report["torch"]["mean_ms"] / report["mlx"]["mean_ms"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
