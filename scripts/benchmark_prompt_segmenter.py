import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_sam.weights import load_image_segmenter

ROOT = Path(__file__).resolve().parents[1]


def summary(values):
    return {
        "runs": len(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/torch_prompt_mask.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks/prompt_segmenter_latency.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    ref = np.load(args.reference)
    pixels = mx.array(ref["pixel_values"])
    coords = mx.array(ref["point_coords"])
    labels = mx.array(ref["point_labels"])
    model = load_image_segmenter(args.weights)

    for _ in range(args.warmup):
        encoded = model.encode_image(pixels)
        out = model.predict_from_encoded(encoded, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])

    full = []
    for _ in range(args.runs):
        start = time.perf_counter()
        out = model(pixels, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])
        full.append((time.perf_counter() - start) * 1000.0)

    encoded = model.encode_image(pixels)
    mx.eval(encoded["vision_features"], encoded["high_res_features"])
    prompt = []
    for _ in range(args.runs):
        start = time.perf_counter()
        out = model.predict_from_encoded(encoded, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"])
        prompt.append((time.perf_counter() - start) * 1000.0)

    report = {
        "input_shape": list(ref["pixel_values"].shape),
        "full_image_plus_prompt": summary(full),
        "prompt_decode_with_cached_image": summary(prompt),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
