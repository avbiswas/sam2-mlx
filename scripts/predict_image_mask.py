import argparse
import json
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
    parser.add_argument("--point", nargs=2, type=float, default=(588.0, 626.0))
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/image_prompt_mask.npy")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/image_prompt_overlay.mp4")
    args = parser.parse_args()

    pixels = preprocess_video(args.video, limit=1)
    model = load_image_segmenter(args.weights)
    out = model(
        mx.array(pixels),
        mx.array(np.array([[args.point]], dtype=np.float32)),
        mx.array(np.array([[1]], dtype=np.int64)),
        multimask_output=True,
    )
    mx.eval(out["low_res_masks"], out["ious"])
    low = np.array(out["low_res_masks"])
    ious = np.array(out["ious"])
    best = low[0, int(np.argmax(ious[0]))]

    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {args.video}")
    h, w = frame.shape[:2]
    mask = cv2.resize(best, (w, h), interpolation=cv2.INTER_LINEAR) > 0
    masks = mask[None].astype(np.float32)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, masks)
    report = write_mask_overlay_video(args.video, masks, args.output_video, limit=1)
    report["ious"] = ious.tolist()
    report["selected_mask"] = int(np.argmax(ious[0]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
