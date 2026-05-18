import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mlx_sam.overlay import write_mask_overlay_video
from mlx_sam.video_predictor import SAM2VideoPredictor

ROOT = Path(__file__).resolve().parents[1]


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


def parse_points(values: list[float]) -> np.ndarray:
    if len(values) % 2:
        raise ValueError("--points expects an even number of x y values")
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX SAM2 video propagation with memory.")
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "outputs/video_memory_multiclick/frames")
    parser.add_argument("--frames", type=int, help="Optional frame limit. Omit to use the full video.")
    parser.add_argument("--points", nargs="+", type=float, default=[625.0, 429.0, 700.0, 470.0, 300.0, 250.0, 950.0, 610.0])
    parser.add_argument("--labels", nargs="+", type=int, default=[1, 1, 0, 0])
    parser.add_argument("--precompute-image-features", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--memory-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--memory-attention-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--num-maskmem", type=int, help="Temporal memory slots to use. Default keeps SAM2's 7-slot eval memory.")
    parser.add_argument("--max-obj-ptrs", type=int, help="Object pointer tokens to use in memory attention. Default keeps SAM2's 16.")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--raw-propagation", action="store_true", help="Skip public video-resolution mask materialization during propagation.")
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/video_memory_multiclick/mlx_masks.npy")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/video_memory_multiclick/mlx_overlay.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/benchmarks/video_memory_multiclick_mlx.json")
    args = parser.parse_args()

    points = parse_points(args.points)
    labels = np.array(args.labels, dtype=np.int32)
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Got {points.shape[0]} points but {labels.shape[0]} labels")

    frame_paths = extract_frames(args.video, args.frames_dir, args.frames)
    if not frame_paths:
        raise RuntimeError("No frames extracted")

    memory_dtype = None if args.memory_dtype == "float32" else args.memory_dtype
    memory_attention_dtype = None if args.memory_attention_dtype == "float32" else args.memory_attention_dtype
    predictor = SAM2VideoPredictor(
        checkpoint=args.weights,
        model_id=args.model_id,
        memory_dtype=memory_dtype,
        memory_attention_dtype=memory_attention_dtype,
        num_maskmem=args.num_maskmem,
        max_obj_ptrs_in_encoder=args.max_obj_ptrs,
    )

    init_start = time.perf_counter()
    state = predictor.init_state(
        str(args.frames_dir),
        precompute_image_features=args.precompute_image_features,
        feature_batch_size=args.feature_batch_size,
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0

    prompt_start = time.perf_counter()
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
        normalize_coords=True,
    )
    prompt_ms = (time.perf_counter() - prompt_start) * 1000.0

    masks_by_frame: dict[int, np.ndarray] = {}
    frame_latencies = []
    prop_start = time.perf_counter()
    last = prop_start
    for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, return_masks=not args.raw_propagation):
        now = time.perf_counter()
        frame_latencies.append((now - last) * 1000.0)
        last = now
        if out_mask_logits is not None and not args.skip_save:
            masks_by_frame[int(out_frame_idx)] = (out_mask_logits[0, 0] > 0).astype(np.float32)
    propagate_ms = (time.perf_counter() - prop_start) * 1000.0

    overlay = {}
    mask_file = None
    output_frames = state["num_frames"]
    if masks_by_frame:
        masks = np.stack([masks_by_frame[i] for i in sorted(masks_by_frame)], axis=0)
        output_frames = masks.shape[0]
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_mask, masks)
        mask_file = str(args.output_mask)
        if not args.skip_overlay:
            overlay = write_mask_overlay_video(args.video, masks, args.output_video, limit=masks.shape[0])

    report = {
        **overlay,
        "method": "mlx_sam2_video_predictor",
        "model_id": args.model_id,
        "weights": str(args.weights),
        "points_original_xy": points.tolist(),
        "labels": labels.tolist(),
        "memory_dtype": args.memory_dtype,
        "memory_attention_dtype": args.memory_attention_dtype,
        "num_maskmem": args.num_maskmem or 7,
        "max_obj_ptrs": args.max_obj_ptrs or 16,
        "precompute_image_features": args.precompute_image_features,
        "feature_batch_size": args.feature_batch_size,
        "skip_save": args.skip_save,
        "skip_overlay": args.skip_overlay,
        "raw_propagation": args.raw_propagation,
        "frames_dir": str(args.frames_dir),
        "frames": int(output_frames),
        "mask_file": mask_file,
        "init_ms": init_ms,
        "prompt_ms": prompt_ms,
        "propagate_ms": propagate_ms,
        "mean_propagate_ms_per_frame": propagate_ms / output_frames,
        "frame_latency_ms": {
            "mean": float(np.mean(frame_latencies)),
            "median": float(np.median(frame_latencies)),
            "min": float(np.min(frame_latencies)),
            "max": float(np.max(frame_latencies)),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
