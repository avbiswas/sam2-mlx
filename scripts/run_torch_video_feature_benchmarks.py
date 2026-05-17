import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO = ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2_video_predictor


@dataclass(frozen=True)
class PromptEvent:
    frame_idx: int
    obj_id: int
    kind: str
    points: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()
    box: tuple[float, float, float, float] | None = None
    clear_old_points: bool = True


def extract_frames(video: Path, output_dir: Path, limit: int | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.jpg"))
    if existing and ((limit is None) or len(existing) == limit):
        return existing

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


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_prompt(predictor, state: dict, event: PromptEvent) -> dict:
    points = np.array(event.points, dtype=np.float32) if event.points else None
    labels = np.array(event.labels, dtype=np.int32) if event.labels else None
    frame_idx, obj_ids, logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=event.frame_idx,
        obj_id=event.obj_id,
        points=points,
        labels=labels,
        box=np.array(event.box, dtype=np.float32) if event.box is not None else None,
        clear_old_points=event.clear_old_points,
        normalize_coords=True,
    )
    return {
        "frame_idx": int(frame_idx),
        "obj_ids": [int(x) for x in obj_ids],
        "mask_areas": [int((logits[i, 0] > 0).detach().cpu().numpy().sum()) for i in range(logits.shape[0])],
    }


def blank_masks(num_frames: int, obj_ids: list[int], height: int, width: int) -> dict[int, np.ndarray]:
    return {obj_id: np.zeros((num_frames, height, width), dtype=np.uint8) for obj_id in obj_ids}


def collect_propagation(
    predictor,
    state: dict,
    obj_ids: list[int],
    num_frames: int,
    height: int,
    width: int,
    start_frame_idx: int | None = None,
    max_frame_num_to_track: int | None = None,
    reverse: bool = False,
) -> tuple[dict[int, np.ndarray], float, list[int]]:
    masks_by_obj = blank_masks(num_frames, obj_ids, height, width)
    obj_id_to_col = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
    visited = []
    start = time.perf_counter()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    ):
        visited.append(int(out_frame_idx))
        logits = out_mask_logits.detach().cpu().numpy()
        for row_idx, obj_id in enumerate(out_obj_ids):
            col = obj_id_to_col[int(obj_id)]
            masks_by_obj[obj_ids[col]][int(out_frame_idx)] = (logits[row_idx, 0] > 0).astype(np.uint8)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return masks_by_obj, elapsed_ms, visited


def stack_masks(masks_by_obj: dict[int, np.ndarray], obj_ids: list[int]) -> np.ndarray:
    return np.stack([masks_by_obj[obj_id] for obj_id in obj_ids], axis=1).astype(np.uint8)


def write_fixture(
    output_dir: Path,
    scenario: str,
    video: Path,
    masks: np.ndarray,
    obj_ids: list[int],
    events: list[PromptEvent],
    prompt_reports: list[dict],
    timings: dict,
    visited_frames: list[int],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_file = output_dir / f"{scenario}_torch_masks.npy"
    report_file = output_dir / f"{scenario}_torch_report.json"
    np.save(mask_file, masks)
    areas = masks.sum(axis=(2, 3))
    report = {
        "scenario": scenario,
        "video": str(video),
        "mask_file": str(mask_file),
        "shape": list(masks.shape),
        "obj_ids": obj_ids,
        "events": [
            {
                "frame_idx": event.frame_idx,
                "obj_id": event.obj_id,
                "kind": event.kind,
                "points": [list(p) for p in event.points],
                "labels": list(event.labels),
                "box": list(event.box) if event.box is not None else None,
                "clear_old_points": event.clear_old_points,
            }
            for event in events
        ],
        "prompt_reports": prompt_reports,
        "timings_ms": timings,
        "visited_frames": visited_frames,
        "area_by_object": {
            str(obj_id): {
                "min": int(areas[:, idx].min()),
                "max": int(areas[:, idx].max()),
                "nonzero_frames": int((areas[:, idx] > 0).sum()),
                "spot": {str(frame): int(areas[frame, idx]) for frame in [0, 30, 60, 90, 120, 149, 180, 240, masks.shape[0] - 1] if frame < masks.shape[0]},
            }
            for idx, obj_id in enumerate(obj_ids)
        },
    }
    report_file.write_text(json.dumps(report, indent=2))
    return report


def scenario_multi_object(ctx: dict) -> tuple[np.ndarray, list[int], list[PromptEvent], list[dict], dict, list[int]]:
    predictor, state = ctx["predictor"], ctx["state"]
    events = [
        PromptEvent(0, 1, "positive_point", ((625.0, 429.0),), (1,)),
        PromptEvent(0, 2, "positive_point", ((300.0, 255.0),), (1,)),
    ]
    prompt_reports = [add_prompt(predictor, state, event) for event in events]
    obj_ids = [1, 2]
    masks_by_obj, prop_ms, visited = collect_propagation(predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"])
    return stack_masks(masks_by_obj, obj_ids), obj_ids, events, prompt_reports, {"propagate_ms": prop_ms}, visited


def scenario_box_prompt(ctx: dict) -> tuple[np.ndarray, list[int], list[PromptEvent], list[dict], dict, list[int]]:
    predictor, state = ctx["predictor"], ctx["state"]
    events = [PromptEvent(0, 1, "box_prompt", box=(480.0, 315.0, 820.0, 585.0))]
    prompt_reports = [add_prompt(predictor, state, events[0])]
    obj_ids = [1]
    masks_by_obj, prop_ms, visited = collect_propagation(predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"])
    return stack_masks(masks_by_obj, obj_ids), obj_ids, events, prompt_reports, {"propagate_ms": prop_ms}, visited


def scenario_negative_clicks(ctx: dict) -> tuple[np.ndarray, list[int], list[PromptEvent], list[dict], dict, list[int]]:
    predictor, state = ctx["predictor"], ctx["state"]
    # Person-focused correction fixture: start from the woman, then add a negative
    # click on the nearby dog/leash region after propagation has already begun.
    events = [PromptEvent(0, 1, "person_positive_point", ((300.0, 250.0),), (1,))]
    prompt_reports = [add_prompt(predictor, state, events[0])]
    obj_ids = [1]
    masks_by_obj, first_ms, first_visited = collect_propagation(
        predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"], start_frame_idx=0, max_frame_num_to_track=min(ctx["frames"], 90)
    )
    correction_frame = min(30, ctx["frames"] - 1)
    events.append(PromptEvent(correction_frame, 1, "person_negative_click_correction", ((330.0, 270.0), (620.0, 430.0)), (1, 0), clear_old_points=False))
    prompt_reports.append(add_prompt(predictor, state, events[-1]))
    masks_by_obj2, second_ms, second_visited = collect_propagation(
        predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"], start_frame_idx=correction_frame
    )
    for obj_id in obj_ids:
        masks_by_obj[obj_id][correction_frame:] = masks_by_obj2[obj_id][correction_frame:]
    return stack_masks(masks_by_obj, obj_ids), obj_ids, events, prompt_reports, {"first_propagate_ms": first_ms, "second_propagate_ms": second_ms}, first_visited + second_visited


def scenario_cross_frame_corrections(ctx: dict) -> tuple[np.ndarray, list[int], list[PromptEvent], list[dict], dict, list[int]]:
    predictor, state = ctx["predictor"], ctx["state"]
    events = [PromptEvent(0, 1, "positive_point", ((625.0, 429.0),), (1,))]
    prompt_reports = [add_prompt(predictor, state, events[0])]
    obj_ids = [1]
    masks_by_obj, first_ms, first_visited = collect_propagation(predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"])
    correction_frame = min(120, ctx["frames"] - 1)
    events.append(PromptEvent(correction_frame, 1, "positive_correction", ((1053.0, 448.0),), (1,), clear_old_points=False))
    prompt_reports.append(add_prompt(predictor, state, events[-1]))
    masks_by_obj2, forward_ms, forward_visited = collect_propagation(
        predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"], start_frame_idx=correction_frame
    )
    for obj_id in obj_ids:
        masks_by_obj[obj_id][correction_frame:] = masks_by_obj2[obj_id][correction_frame:]
    return stack_masks(masks_by_obj, obj_ids), obj_ids, events, prompt_reports, {"initial_propagate_ms": first_ms, "corrected_forward_ms": forward_ms}, first_visited + forward_visited


def scenario_bidirectional_middle(ctx: dict) -> tuple[np.ndarray, list[int], list[PromptEvent], list[dict], dict, list[int]]:
    predictor, state = ctx["predictor"], ctx["state"]
    middle = min(ctx.get("bidirectional_frame", max(120, ctx["frames"] // 2)), ctx["frames"] - 1)
    point = ctx.get("bidirectional_point", (1053.0, 448.0))
    events = [PromptEvent(middle, 1, "middle_positive_point", (point,), (1,))]
    prompt_reports = [add_prompt(predictor, state, events[0])]
    obj_ids = [1]
    forward, forward_ms, forward_visited = collect_propagation(
        predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"], start_frame_idx=middle, reverse=False
    )
    backward, backward_ms, backward_visited = collect_propagation(
        predictor, state, obj_ids, ctx["frames"], ctx["height"], ctx["width"], start_frame_idx=middle, reverse=True
    )
    for obj_id in obj_ids:
        forward[obj_id][:middle] = backward[obj_id][:middle]
    return stack_masks(forward, obj_ids), obj_ids, events, prompt_reports, {"forward_ms": forward_ms, "backward_ms": backward_ms}, forward_visited + backward_visited


SCENARIOS: dict[str, Callable[[dict], tuple[np.ndarray, list[int], list[PromptEvent], list[dict], dict, list[int]]]] = {
    "multi_object": scenario_multi_object,
    "box_prompt": scenario_box_prompt,
    "negative_clicks": scenario_negative_clicks,
    "cross_frame_corrections": scenario_cross_frame_corrections,
    "bidirectional_middle": scenario_bidirectional_middle,
}


def build_context(args, frames_dir: Path, num_frames: int, height: int, width: int) -> dict:
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(args.checkpoint),
        device=device(),
        apply_postprocessing=True,
    )
    init_start = time.perf_counter()
    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0
    return {
        "predictor": predictor,
        "state": state,
        "frames": num_frames,
        "height": height,
        "width": width,
        "init_ms": init_ms,
        "bidirectional_frame": args.bidirectional_frame,
        "bidirectional_point": tuple(args.bidirectional_point) if args.bidirectional_point else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "outputs/torch_feature_benchmark_frames")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/feature_benchmarks")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS.keys()], default="all")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--bidirectional-frame", type=int)
    parser.add_argument("--bidirectional-point", nargs=2, type=float)
    args = parser.parse_args()

    frame_paths = extract_frames(args.video, args.frames_dir, args.frames)
    if not frame_paths:
        raise RuntimeError("No frames extracted")
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"Could not read extracted frame: {frame_paths[0]}")
    height, width = sample.shape[:2]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    reports = []
    for scenario in scenarios:
        ctx = build_context(args, args.frames_dir, len(frame_paths), height, width)
        masks, obj_ids, events, prompt_reports, timings, visited = SCENARIOS[scenario](ctx)
        timings = {"init_ms": ctx["init_ms"], **timings}
        reports.append(write_fixture(args.output_dir, scenario, args.video, masks, obj_ids, events, prompt_reports, timings, visited))

    summary = {
        "video": str(args.video),
        "frames": len(frame_paths),
        "height": height,
        "width": width,
        "scenarios": [{k: report[k] for k in ["scenario", "mask_file", "shape", "obj_ids", "timings_ms"]} for report in reports],
    }
    summary_path = args.output_dir / "torch_feature_benchmarks_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
