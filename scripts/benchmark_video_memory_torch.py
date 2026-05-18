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
from mlx_vision.overlay import write_mask_overlay_video


MODEL_CONFIGS = {
    "facebook/sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "facebook/sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "facebook/sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "facebook/sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def model_config_for_name(name: str) -> str:
    if name in MODEL_CONFIGS:
        return MODEL_CONFIGS[name]
    lowered = name.lower()
    if "hiera_tiny" in lowered or "hiera-tiny" in lowered:
        return MODEL_CONFIGS["sam2.1_hiera_tiny"]
    if "hiera_small" in lowered or "hiera-small" in lowered:
        return MODEL_CONFIGS["sam2.1_hiera_small"]
    if "hiera_base_plus" in lowered or "hiera-base-plus" in lowered or "hiera_b+" in lowered:
        return MODEL_CONFIGS["sam2.1_hiera_base_plus"]
    if "hiera_large" in lowered or "hiera-large" in lowered:
        return MODEL_CONFIGS["sam2.1_hiera_large"]
    raise ValueError(f"Could not infer official SAM2 config from: {name}")


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


def device_for_run() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def propagate_torch_raw(predictor, state, start_frame_idx=None, max_frame_num_to_track=None, reverse: bool = False):
    """Official SAM2 propagation loop without final video-resolution mask resize."""
    predictor.propagate_in_video_preflight(state)
    obj_ids = state["obj_ids"]
    num_frames = state["num_frames"]
    batch_size = predictor._get_obj_num(state)

    if start_frame_idx is None:
        start_frame_idx = min(
            t
            for obj_output_dict in state["output_dict_per_obj"].values()
            for t in obj_output_dict["cond_frame_outputs"]
        )
    if max_frame_num_to_track is None:
        max_frame_num_to_track = num_frames
    if reverse:
        end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
        processing_order = range(start_frame_idx, end_frame_idx - 1, -1) if start_frame_idx > 0 else []
    else:
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

    for frame_idx in processing_order:
        pred_masks_per_obj = [None] * batch_size
        for obj_idx in range(batch_size):
            obj_output_dict = state["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                pred_masks = current_out["pred_masks"].to(state["device"], non_blocking=True)
                if predictor.clear_non_cond_mem_around_input:
                    predictor._clear_obj_non_cond_mem_around_input(state, frame_idx, obj_idx)
            else:
                current_out, pred_masks = predictor._run_single_frame_inference(
                    inference_state=state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
            pred_masks_per_obj[obj_idx] = pred_masks
        if len(pred_masks_per_obj) > 1:
            yield frame_idx, obj_ids, torch.cat(pred_masks_per_obj, dim=0)
        else:
            yield frame_idx, obj_ids, pred_masks_per_obj[0]


def main():
    parser = argparse.ArgumentParser(description="Benchmark official Torch SAM2 video propagation with memory.")
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "outputs/video_memory_multiclick/frames")
    parser.add_argument("--frames", type=int, help="Optional frame limit. Omit to use the full video.")
    parser.add_argument("--points", nargs="+", type=float, default=[625.0, 429.0, 700.0, 470.0, 300.0, 250.0, 950.0, 610.0])
    parser.add_argument("--labels", nargs="+", type=int, default=[1, 1, 0, 0])
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/video_memory_multiclick/torch_masks.npy")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/video_memory_multiclick/torch_overlay.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/benchmarks/video_memory_multiclick_torch.json")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--raw-propagation", action="store_true", help="Skip official final video-resolution mask resize during propagation.")
    args = parser.parse_args()

    points = parse_points(args.points)
    labels = np.array(args.labels, dtype=np.int32)
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Got {points.shape[0]} points but {labels.shape[0]} labels")

    frame_paths = extract_frames(args.video, args.frames_dir, args.frames)
    if not frame_paths:
        raise RuntimeError("No frames extracted")

    device = device_for_run()
    config = model_config_for_name(args.model_id or str(args.checkpoint))
    predictor = build_sam2_video_predictor(config, str(args.checkpoint), device=device, apply_postprocessing=True)

    init_start = time.perf_counter()
    state = predictor.init_state(
        video_path=str(args.frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
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
    iterator = propagate_torch_raw(predictor, state) if args.raw_propagation else predictor.propagate_in_video(state)
    for out_frame_idx, _out_obj_ids, out_mask_logits in iterator:
        now = time.perf_counter()
        frame_latencies.append((now - last) * 1000.0)
        last = now
        if not args.skip_save:
            masks_by_frame[int(out_frame_idx)] = (out_mask_logits[0, 0] > 0).detach().cpu().numpy().astype(np.float32)
    propagate_ms = (time.perf_counter() - prop_start) * 1000.0

    overlay = {}
    mask_file = None
    output_frames = len(frame_paths)
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
        "method": "official_torch_sam2_video_predictor",
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "config": config,
        "device": str(device),
        "points_original_xy": points.tolist(),
        "labels": labels.tolist(),
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
