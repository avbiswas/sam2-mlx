import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from sam_mlx.weights import load_image_encoder

ROOT = Path(__file__).resolve().parents[1]


def stats(name: str, ref: np.ndarray, got: np.ndarray) -> dict:
    diff = got.astype(np.float64) - ref.astype(np.float64)
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    return {
        "name": name,
        "shape": list(ref.shape),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_rel": float(np.max(np.abs(diff) / denom)),
        "ref_mean": float(np.mean(ref)),
        "got_mean": float(np.mean(got)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/torch_image_embeddings.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_encoder.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parity/image_embeddings_parity.json")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    ref = np.load(args.reference)
    model = load_image_encoder(args.weights)
    pixels = mx.array(ref["pixel_values"])
    out = model(pixels)
    mx.eval(out["vision_features"], out["backbone_fpn"], out["vision_pos_enc"])

    rows = []
    rows.append(stats("vision_features", ref["vision_features"], np.array(out["vision_features"])))
    for i, x in enumerate(out["backbone_fpn"]):
        rows.append(stats(f"backbone_fpn_{i}", ref[f"backbone_fpn_{i}"], np.array(x)))
    for i, x in enumerate(out["vision_pos_enc"]):
        rows.append(stats(f"vision_pos_enc_{i}", ref[f"vision_pos_enc_{i}"], np.array(x)))

    print(json.dumps(rows, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    failures = [
        row for row in rows
        if row["max_abs"] > args.atol and row["max_rel"] > args.rtol
    ]
    if failures:
        raise SystemExit(f"parity failed for {[x['name'] for x in failures]}")


if __name__ == "__main__":
    main()
