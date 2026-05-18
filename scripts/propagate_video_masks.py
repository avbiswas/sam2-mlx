import argparse
import json
import time
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from mlx_sam.overlay import write_mask_overlay_video
from mlx_sam.preprocess import preprocess_video
from mlx_sam.weights import load_image_segmenter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--point", nargs=2, type=float, default=(588.0, 626.0))
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/video_prompt_masks.npy")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/video_prompt_overlay.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/benchmarks/video_mask_prompt_latency.json")
    args = parser.parse_args()

    pixels = preprocess_video(args.video, limit=args.frames)
    model = load_image_segmenter(args.weights)

    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    low_prompt = None
    masks = []
    ious = []
    times = []
    for i in range(pixels.shape[0]):
        frame = mx.array(pixels[i : i + 1])
        start = time.perf_counter()
        if i == 0:
            out = model(
                frame,
                mx.array(np.array([[args.point]], dtype=np.float32)),
                mx.array(np.array([[1]], dtype=np.int32)),
                multimask_output=True,
            )
            mx.eval(out["low_res_masks"], out["ious"])
            low = np.array(out["low_res_masks"])
            score = np.array(out["ious"])
            best = int(np.argmax(score[0]))
            low_prompt = low[:, best : best + 1]
            ious.append(score[0].tolist())
        else:
            out = model(
                frame,
                None,
                None,
                mask_input=mx.array(low_prompt.astype(np.float32)),
                multimask_output=False,
            )
            mx.eval(out["low_res_masks"], out["ious"])
            low_prompt = np.array(out["low_res_masks"])
            ious.append(np.array(out["ious"])[0].tolist())
        times.append((time.perf_counter() - start) * 1000.0)
        mask = cv2.resize(low_prompt[0, 0], (width, height), interpolation=cv2.INTER_LINEAR) > 0
        masks.append(mask.astype(np.float32))

    masks_np = np.stack(masks, axis=0)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, masks_np)
    overlay = write_mask_overlay_video(args.video, masks_np, args.output_video, limit=masks_np.shape[0])
    report = {
        **overlay,
        "mask_file": str(args.output_mask),
        "latency_ms": {
            "frames": len(times),
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        },
        "ious": ious,
        "method": "mlx_mask_prompt_feedback",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
