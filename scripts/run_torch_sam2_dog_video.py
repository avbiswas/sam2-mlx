import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO = ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2_video_predictor
from mlx_sam.overlay import write_mask_overlay_video


def extract_frames(video: Path, output_dir: Path, limit: int | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.jpg"))
    if existing and (limit is None or len(existing) >= limit):
        return existing[:limit]

    for path in existing:
        path.unlink()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    frames: list[Path] = []
    try:
        idx = 0
        while limit is None or idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            path = output_dir / f"{idx:05d}.jpg"
            cv2.imwrite(str(path), frame)
            frames.append(path)
            idx += 1
    finally:
        cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "outputs/torch_sam2_dog_frames")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/torch_sam2_dog_overlay_full.mp4")
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/torch_sam2_dog_masks_full.npy")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/benchmarks/torch_sam2_dog_full.json")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--point", nargs=2, type=float, default=(625.0, 429.0), help="Original video pixel coordinates")
    args = parser.parse_args()

    frame_paths = extract_frames(args.video, args.frames_dir, args.frames)
    if not frame_paths:
        raise RuntimeError("No frames extracted")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(args.checkpoint),
        device=device,
        apply_postprocessing=True,
    )

    start = time.perf_counter()
    state = predictor.init_state(
        video_path=str(args.frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    init_ms = (time.perf_counter() - start) * 1000

    points = np.array([args.point], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
        normalize_coords=True,
    )

    masks_by_frame: dict[int, np.ndarray] = {}
    prop_start = time.perf_counter()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        # Single object. Logits are already at original video resolution.
        masks_by_frame[int(out_frame_idx)] = (out_mask_logits[0, 0] > 0).detach().cpu().numpy().astype(np.float32)
    prop_ms = (time.perf_counter() - prop_start) * 1000

    masks = np.stack([masks_by_frame[i] for i in range(len(masks_by_frame))], axis=0)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, masks)
    overlay = write_mask_overlay_video(args.video, masks, args.output_video, limit=masks.shape[0])

    report = {
        **overlay,
        "method": "official_torch_sam2_video_predictor",
        "device": str(device),
        "point_original_xy": list(args.point),
        "frames_dir": str(args.frames_dir),
        "mask_file": str(args.output_mask),
        "init_ms": init_ms,
        "propagate_ms": prop_ms,
        "mean_propagate_ms_per_frame": prop_ms / masks.shape[0],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
