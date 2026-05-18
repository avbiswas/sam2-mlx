import argparse
import json
import time
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from mlx_sam.overlay import write_mask_overlay_video
from mlx_sam.models.memory import upsample_mask
from mlx_sam.video_predictor import SAM2VideoPredictor
from benchmark_video_memory_mlx import extract_frames, parse_points

ROOT = Path(__file__).resolve().parents[1]


def binary_iou_report(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = np.asarray(reference) > 0.5
    cand = np.asarray(candidate) > 0.5
    frames = min(ref.shape[0], cand.shape[0])
    ref = ref[:frames]
    cand = cand[:frames]
    inter = np.logical_and(ref, cand).sum(axis=(1, 2))
    union = np.logical_or(ref, cand).sum(axis=(1, 2))
    iou = np.ones(frames, dtype=np.float64)
    nonempty_union = union > 0
    iou[nonempty_union] = inter[nonempty_union] / union[nonempty_union]
    ref_present = ref.sum(axis=(1, 2)) > 0
    cand_present = cand.sum(axis=(1, 2)) > 0
    nonempty_ref_iou = iou[ref_present]
    return {
        "frames": int(frames),
        "mean_iou_all": float(iou.mean()),
        "mean_iou_nonempty_reference": float(nonempty_ref_iou.mean()) if nonempty_ref_iou.size else 1.0,
        "median_iou_nonempty_reference": float(np.median(nonempty_ref_iou)) if nonempty_ref_iou.size else 1.0,
        "min_iou_nonempty_reference": float(nonempty_ref_iou.min()) if nonempty_ref_iou.size else 1.0,
        "presence_match_frames": int((ref_present == cand_present).sum()),
        "presence_total_frames": int(frames),
    }


def sampled_frame_indices(num_frames: int, step: int) -> list[int]:
    if step < 1:
        raise ValueError("--skip-step must be >= 1")
    indices = list(range(0, num_frames, step))
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return indices


def nearest_sampled_frame(frame_idx: int, sampled: list[int]) -> tuple[int, int]:
    if not sampled:
        raise ValueError("sampled frame list is empty")
    nearest_logical_idx = min(range(len(sampled)), key=lambda idx: (abs(sampled[idx] - frame_idx), sampled[idx]))
    return nearest_logical_idx, sampled[nearest_logical_idx]


def interpolate_lowres_logits_mlx(sampled_logits: dict[int, mx.array], num_frames: int, mode: str) -> mx.array:
    sampled = sorted(sampled_logits)
    if not sampled or sampled[0] != 0:
        raise ValueError("sampled logits must include frame 0")

    chunks = []
    for left, right in zip(sampled, sampled[1:]):
        left_logits = sampled_logits[left]
        right_logits = sampled_logits[right]
        span = max(right - left, 1)
        if mode == "hold":
            chunk = mx.broadcast_to(left_logits, (span, *left_logits.shape[1:]))
        elif mode == "nearest":
            offsets = mx.arange(span, dtype=mx.float32)
            use_right = (offsets > (span - offsets)).reshape(span, 1, 1, 1)
            left_chunk = mx.broadcast_to(left_logits, (span, *left_logits.shape[1:]))
            right_chunk = mx.broadcast_to(right_logits, (span, *right_logits.shape[1:]))
            chunk = mx.where(use_right, right_chunk, left_chunk)
        else:
            alpha = (mx.arange(span, dtype=mx.float32) / float(span)).reshape(span, 1, 1, 1)
            chunk = (1.0 - alpha) * left_logits + alpha * right_logits
        chunks.append(chunk)
    chunks.append(sampled_logits[sampled[-1]])
    out = mx.concatenate(chunks, axis=0)
    if out.shape[0] != num_frames:
        raise RuntimeError(f"Expected {num_frames} interpolated frames, got {out.shape[0]}")
    return out


def upsample_binary_masks_mlx(lowres_logits: mx.array, height: int, width: int, batch_size: int) -> np.ndarray:
    masks = []
    batch_size = max(1, int(batch_size))
    for start in range(0, lowres_logits.shape[0], batch_size):
        logits = upsample_mask(lowres_logits[start : start + batch_size], (height, width))
        binary = (logits[:, 0] > 0).astype(mx.float32)
        mx.eval(binary)
        masks.append(np.array(binary))
    return np.concatenate(masks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX SAM2 frame skipping plus mask interpolation.")
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "outputs/video_frame_skip/frames")
    parser.add_argument("--frames", type=int, help="Optional frame limit. Omit to use the full video.")
    parser.add_argument("--skip-step", type=int, default=2, help="Run SAM2 on every k-th frame, then interpolate skipped frames.")
    parser.add_argument("--interpolation", choices=["linear", "nearest", "hold"], default="linear")
    parser.add_argument("--points", nargs="+", type=float, default=[625.0, 429.0, 700.0, 470.0, 300.0, 250.0, 950.0, 610.0])
    parser.add_argument("--labels", nargs="+", type=int, default=[1, 1, 0, 0])
    parser.add_argument("--prompt-frame", type=int, default=0, help="Frame that received the user click. Snapped to nearest sampled frame for skip-step > 1.")
    parser.add_argument("--precompute-image-features", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--memory-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--memory-attention-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--baseline-mask", type=Path, help="Optional full-frame baseline .npy/.npz mask to compare against.")
    parser.add_argument("--postprocess-batch-size", type=int, default=16, help="Frames per MLX batch for final low-res-to-video-res upsample.")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/video_frame_skip/mlx_masks_skip.npy")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/video_frame_skip/mlx_overlay_skip.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/benchmarks/video_frame_skip_mlx.json")
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
        image_size=args.image_size,
        memory_dtype=memory_dtype,
        memory_attention_dtype=memory_attention_dtype,
    )

    total_start = time.perf_counter()
    init_start = time.perf_counter()
    state = predictor.init_state(
        str(args.frames_dir),
        precompute_image_features=args.precompute_image_features,
        feature_batch_size=args.feature_batch_size,
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0

    sampled_indices = sampled_frame_indices(state["num_frames"], args.skip_step)
    requested_prompt_frame = int(np.clip(args.prompt_frame, 0, state["num_frames"] - 1))
    prompt_logical_idx, prompt_frame = nearest_sampled_frame(requested_prompt_frame, sampled_indices)

    prompt_start = time.perf_counter()
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=prompt_frame,
        obj_id=1,
        points=points,
        labels=labels,
        normalize_coords=True,
    )
    prompt_ms = (time.perf_counter() - prompt_start) * 1000.0

    sampled_logits: dict[int, mx.array] = {}
    frame_latencies = []

    propagate_start = time.perf_counter()
    for obj_idx in range(len(state["obj_ids"])):
        cond = state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"][prompt_frame]
        predictor._ensure_memory_encoded(
            state,
            obj_idx,
            prompt_frame,
            cond,
            is_mask_from_points=True,
            memory_frame_idx=prompt_logical_idx,
        )

    propagation_order = [
        *range(prompt_logical_idx, len(sampled_indices)),
        *range(prompt_logical_idx - 1, -1, -1),
    ]
    for logical_idx in propagation_order:
        physical_idx = sampled_indices[logical_idx]
        frame_start = time.perf_counter()
        outputs = []
        for obj_idx in range(len(state["obj_ids"])):
            obj_output_dict = state["output_dict_per_obj"][obj_idx]
            if physical_idx in obj_output_dict["cond_frame_outputs"]:
                out = obj_output_dict["cond_frame_outputs"][physical_idx]
            else:
                out = predictor._run_single_frame_inference(
                    state,
                    obj_idx,
                    physical_idx,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    reverse=logical_idx < prompt_logical_idx,
                    run_mem_encoder=True,
                    memory_frame_idx=logical_idx,
                )
                obj_output_dict["non_cond_frame_outputs"][physical_idx] = out
            state["frames_tracked_per_obj"][obj_idx][physical_idx] = {"reverse": logical_idx < prompt_logical_idx}
            outputs.append(out)
        sampled_logits[physical_idx] = outputs[0]["pred_masks"]
        frame_latencies.append((time.perf_counter() - frame_start) * 1000.0)
    model_propagate_ms = (time.perf_counter() - propagate_start) * 1000.0

    interp_start = time.perf_counter()
    lowres_logits = interpolate_lowres_logits_mlx(sampled_logits, state["num_frames"], args.interpolation)
    mx.eval(lowres_logits)
    lowres_interpolation_ms = (time.perf_counter() - interp_start) * 1000.0

    upsample_start = time.perf_counter()
    masks = upsample_binary_masks_mlx(lowres_logits, state["video_height"], state["video_width"], args.postprocess_batch_size)
    upsample_ms = (time.perf_counter() - upsample_start) * 1000.0
    interpolation_ms = (time.perf_counter() - interp_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, masks)

    overlay = {}
    if not args.skip_overlay:
        overlay = write_mask_overlay_video(args.video, masks, args.output_video, limit=masks.shape[0])

    accuracy = None
    if args.baseline_mask is not None:
        baseline = np.load(args.baseline_mask)
        if isinstance(baseline, np.lib.npyio.NpzFile):
            key = "masks" if "masks" in baseline else baseline.files[0]
            baseline = baseline[key]
        if baseline.ndim == 4:
            baseline = baseline[:, 0]
        accuracy = binary_iou_report(baseline, masks)

    computed_frames = len(sampled_indices)
    output_frames = state["num_frames"]
    report = {
        **overlay,
        "method": "mlx_sam2_frame_skip_interpolation",
        "model_id": args.model_id,
        "weights": str(args.weights),
        "image_size": args.image_size,
        "frames_dir": str(args.frames_dir),
        "output_mask": str(args.output_mask),
        "baseline_mask": str(args.baseline_mask) if args.baseline_mask is not None else None,
        "points_original_xy": points.tolist(),
        "labels": labels.tolist(),
        "requested_prompt_frame": requested_prompt_frame,
        "snapped_prompt_frame": prompt_frame,
        "snapped_prompt_logical_idx": prompt_logical_idx,
        "skip_step": args.skip_step,
        "interpolation": args.interpolation,
        "precompute_image_features": args.precompute_image_features,
        "feature_batch_size": args.feature_batch_size,
        "memory_dtype": args.memory_dtype,
        "memory_attention_dtype": args.memory_attention_dtype,
        "postprocess": "mlx_lowres_interpolate_batched_upsample",
        "postprocess_batch_size": args.postprocess_batch_size,
        "output_frames": int(output_frames),
        "computed_frames": int(computed_frames),
        "computed_frame_fraction": float(computed_frames / output_frames),
        "sampled_indices": sampled_indices,
        "init_ms": init_ms,
        "prompt_ms": prompt_ms,
        "model_propagate_ms": model_propagate_ms,
        "lowres_interpolation_ms": lowres_interpolation_ms,
        "upsample_ms": upsample_ms,
        "interpolation_ms": interpolation_ms,
        "total_ms": total_ms,
        "mean_model_ms_per_computed_frame": model_propagate_ms / computed_frames,
        "mean_model_ms_per_output_frame": model_propagate_ms / output_frames,
        "mean_total_ms_per_output_frame": total_ms / output_frames,
        "frame_latency_ms": {
            "mean": float(np.mean(frame_latencies)),
            "median": float(np.median(frame_latencies)),
            "min": float(np.min(frame_latencies)),
            "max": float(np.max(frame_latencies)),
        },
        "accuracy_vs_baseline": accuracy,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
