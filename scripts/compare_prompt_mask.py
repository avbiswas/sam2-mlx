import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_sam.weights import load_image_segmenter

ROOT = Path(__file__).resolve().parents[1]


def row(name: str, ref: np.ndarray, got: np.ndarray) -> dict:
    diff = got.astype(np.float64) - ref.astype(np.float64)
    return {
        "name": name,
        "shape": list(ref.shape),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "ref_mean": float(np.mean(ref)),
        "got_mean": float(np.mean(got)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/torch_prompt_mask.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parity/prompt_mask_parity.json")
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    ref = np.load(args.reference)
    if args.model_id is None and args.reference.with_suffix(".json").exists():
        meta = json.loads(args.reference.with_suffix(".json").read_text())
        args.model_id = meta.get("model_id")
    model = load_image_segmenter(args.weights, model_id=args.model_id)
    out = model(
        mx.array(ref["pixel_values"]),
        mx.array(ref["point_coords"]),
        mx.array(ref["point_labels"]),
        multimask_output=True,
    )
    mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])
    rows = [
        row("low_res_multimasks", ref["low_res_multimasks"], np.array(out["low_res_masks"])),
        row("ious", ref["ious"], np.array(out["ious"])),
        row("object_score_logits", ref["object_score_logits"], np.array(out["object_score_logits"])),
    ]
    print(json.dumps(rows, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    failures = [x for x in rows if x["max_abs"] > args.atol]
    if failures:
        raise SystemExit(f"mask parity failed for {[x['name'] for x in failures]}")


if __name__ == "__main__":
    main()
