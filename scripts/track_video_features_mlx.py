import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from mlx_sam.overlay import write_mask_overlay_video
from mlx_sam.preprocess import preprocess_video
from mlx_sam.weights import load_image_segmenter

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromptEvent:
    frame_idx: int
    obj_id: int
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


@dataclass
class ObjectTrackState:
    obj_id: int
    cond_memories: list[dict] = field(default_factory=list)
    memories: list[dict] = field(default_factory=list)
    low_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    tracked_frames: dict[int, bool] = field(default_factory=dict)


class VideoTrackState:
    def __init__(self, model, pixels: np.ndarray, width: int, height: int):
        self.model = model
        self.pixels = pixels
        self.width = width
        self.height = height
        self.objects: dict[int, ObjectTrackState] = {}

    def object(self, obj_id: int) -> ObjectTrackState:
        if obj_id not in self.objects:
            self.objects[obj_id] = ObjectTrackState(obj_id=obj_id)
        return self.objects[obj_id]

    def encode_frame(self, frame_idx: int) -> dict:
        return self.model.encode_image(mx.array(self.pixels[frame_idx : frame_idx + 1]))

    def run_object_on_frame(
        self,
        obj: ObjectTrackState,
        encoded: dict,
        frame_idx: int,
        prompt: PromptEvent | None,
        track_in_reverse: bool,
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        if prompt is None and any(int(memory["frame_idx"]) == frame_idx for memory in obj.cond_memories):
            low = obj.low_by_frame[frame_idx]
            obj.tracked_frames[frame_idx] = track_in_reverse
            return low_to_video_mask(low, self.width, self.height), (time.perf_counter() - start) * 1000.0

        if prompt is not None and not obj.cond_memories:
            out = predict_initial(
                self.model,
                encoded,
                original_to_sam(prompt.points, self.width, self.height),
                np.array(prompt.labels, dtype=np.int32),
                multimask=True,
            )
        elif prompt is not None:
            mask_input = obj.low_by_frame.get(frame_idx)
            if mask_input is None:
                prev = predict_tracked(
                    self.model,
                    encoded,
                    obj.memories,
                    obj.cond_memories,
                    None,
                    self.width,
                    self.height,
                    current_frame_idx=frame_idx,
                    track_in_reverse=track_in_reverse,
                )
                mask_input, _, _ = best_low_mask(prev)
            out = predict_tracked(
                self.model,
                encoded,
                obj.memories,
                obj.cond_memories,
                prompt,
                self.width,
                self.height,
                mask_input,
                current_frame_idx=frame_idx,
                track_in_reverse=track_in_reverse,
            )
        else:
            out = predict_tracked(
                self.model,
                encoded,
                obj.memories,
                obj.cond_memories,
                None,
                self.width,
                self.height,
                current_frame_idx=frame_idx,
                track_in_reverse=track_in_reverse,
            )

        low, _, _ = best_low_mask(out)
        memory = encode_memory_item(self.model, encoded, low, out, frame_idx, is_mask_from_points=(prompt is not None))
        if prompt is not None:
            obj.cond_memories = [m for m in obj.cond_memories if m.get("frame_idx") != frame_idx]
            obj.cond_memories.append(memory)
            obj.memories = [m for m in obj.memories if m.get("frame_idx") != frame_idx]
        else:
            obj.memories = [m for m in obj.memories if m.get("frame_idx") != frame_idx]
            obj.memories.append(memory)
        obj.low_by_frame[frame_idx] = low
        obj.tracked_frames[frame_idx] = track_in_reverse
        return low_to_video_mask(low, self.width, self.height), (time.perf_counter() - start) * 1000.0

    def propagate(self, obj_ids: list[int], order: list[int], prompt_events: dict[int, list[PromptEvent]]) -> tuple[np.ndarray, list[float]]:
        masks = np.zeros((self.pixels.shape[0], len(obj_ids), self.height, self.width), dtype=np.uint8)
        timings: list[float] = []
        track_in_reverse = len(order) > 1 and order[0] > order[-1]
        for frame_idx in order:
            encoded = self.encode_frame(frame_idx)
            events_by_obj = {event.obj_id: event for event in prompt_events.get(frame_idx, [])}
            for obj_col, obj_id in enumerate(obj_ids):
                obj = self.object(obj_id)
                if not obj.cond_memories and obj_id not in events_by_obj:
                    continue
                mask, elapsed = self.run_object_on_frame(obj, encoded, frame_idx, events_by_obj.get(obj_id), track_in_reverse)
                masks[frame_idx, obj_col] = mask
                timings.append(elapsed)
        return masks, timings


def original_to_sam(points: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    arr = np.array(points, dtype=np.float32)
    arr[:, 0] *= 1024.0 / width
    arr[:, 1] *= 1024.0 / height
    return arr


def box_to_sam_points(box: tuple[float, float, float, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    points = original_to_sam(((x0, y0), (x1, y1)), width, height)
    labels = np.array([2, 3], dtype=np.int32)
    return points, labels


def best_low_mask(out: dict) -> tuple[np.ndarray, np.ndarray, int]:
    low = np.array(out["low_res_masks"])
    ious = np.array(out["ious"])
    idx = int(np.argmax(ious[0])) if ious.shape[1] > 1 else 0
    return low[:, idx : idx + 1], ious, idx


def encode_memory_item(model, encoded: dict, low: np.ndarray, out: dict, frame_idx: int, is_mask_from_points: bool = False) -> dict:
    mem = model.encode_memory(encoded["vision_features"], mx.array(low), out["object_score_logits"], is_mask_from_points=is_mask_from_points)
    mx.eval(mem["vision_features"], mem["vision_pos_enc"])
    return {
        "maskmem_features": mem["vision_features"],
        "maskmem_pos_enc": mem["vision_pos_enc"][0],
        "obj_ptr": out["obj_ptr"],
        "frame_idx": frame_idx,
    }


def predict_initial(model, encoded: dict, points_1024: np.ndarray, labels: np.ndarray, multimask: bool = True) -> dict:
    out = model.predict_from_encoded(
        encoded,
        mx.array(points_1024[None, ...].astype(np.float32)),
        mx.array(labels[None, ...].astype(np.int32)),
        multimask_output=multimask,
    )
    mx.eval(out["low_res_masks"], out["ious"], out["obj_ptr"], out["object_score_logits"])
    return out


def predict_tracked(
    model,
    encoded: dict,
    memories: list[dict],
    cond_memories: list[dict],
    prompt: PromptEvent | None = None,
    width: int = 1280,
    height: int = 720,
    mask_input: np.ndarray | None = None,
    current_frame_idx: int | None = None,
    track_in_reverse: bool = False,
) -> dict:
    conditioned = model.condition_with_memories(
        encoded,
        memories,
        cond_memories=cond_memories,
        current_frame_idx=current_frame_idx,
        track_in_reverse=track_in_reverse,
    )
    conditioned_encoded = dict(encoded)
    conditioned_encoded["vision_features"] = conditioned
    if prompt is None:
        point_coords = None
        point_labels = None
    else:
        point_coords = mx.array(original_to_sam(prompt.points, width, height)[None, ...].astype(np.float32))
        point_labels = mx.array(np.array(prompt.labels, dtype=np.int32)[None, ...])
    out = model.predict_from_encoded(
        conditioned_encoded,
        point_coords,
        point_labels,
        mask_input=mx.array(mask_input.astype(np.float32)) if mask_input is not None else None,
        multimask_output=False,
        add_no_mem_embed=False,
    )
    mx.eval(out["low_res_masks"], out["ious"], out["obj_ptr"], out["object_score_logits"])
    return out


def low_to_video_mask(low: np.ndarray, width: int, height: int) -> np.ndarray:
    return (cv2.resize(low[0, 0], (width, height), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)


def run_one_direction(model, pixels: np.ndarray, order: list[int], init_event: PromptEvent, width: int, height: int, correction_events: dict[int, PromptEvent] | None = None) -> tuple[np.ndarray, list[float]]:
    state = VideoTrackState(model, pixels, width, height)
    correction_events = correction_events or {}
    events_by_frame: dict[int, list[PromptEvent]] = {init_event.frame_idx: [init_event]}
    for event in correction_events.values():
        events_by_frame.setdefault(event.frame_idx, []).append(event)
    masks, timings = state.propagate([init_event.obj_id], order, events_by_frame)
    return masks[:, 0], timings


def apply_forward_correction(
    model,
    pixels: np.ndarray,
    init_event: PromptEvent,
    correction: PromptEvent,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[float]]:
    state = VideoTrackState(model, pixels, width, height)
    obj_ids = [init_event.obj_id]
    full_masks, timings = state.propagate(obj_ids, list(range(pixels.shape[0])), {init_event.frame_idx: [init_event]})
    obj = state.object(init_event.obj_id)
    obj.memories = [m for m in obj.memories if int(m["frame_idx"]) < correction.frame_idx]
    corrected, correction_timings = state.propagate(
        obj_ids,
        list(range(correction.frame_idx, pixels.shape[0])),
        {correction.frame_idx: [correction]},
    )
    full_masks[correction.frame_idx :, 0] = corrected[correction.frame_idx :, 0]
    timings.extend(correction_timings)
    return full_masks[:, 0], timings


def apply_bidirectional_correction(
    model,
    pixels: np.ndarray,
    init_event: PromptEvent,
    correction: PromptEvent,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[float]]:
    base_state = VideoTrackState(model, pixels, width, height)
    obj_ids = [init_event.obj_id]
    masks, timings = base_state.propagate(obj_ids, list(range(pixels.shape[0])), {init_event.frame_idx: [init_event]})

    forward_state = base_state
    obj = forward_state.object(init_event.obj_id)
    obj.memories = [m for m in obj.memories if int(m["frame_idx"]) < correction.frame_idx]
    forward, forward_timings = forward_state.propagate(obj_ids, list(range(correction.frame_idx, pixels.shape[0])), {correction.frame_idx: [correction]})

    # Match Meta's video predictor behavior: reverse propagation reuses the
    # same object state, including future-side non-conditioning memories already
    # produced by the corrected forward pass.
    backward, backward_timings = forward_state.propagate(obj_ids, list(range(correction.frame_idx, -1, -1)), {})

    masks[: correction.frame_idx, 0] = backward[: correction.frame_idx, 0]
    masks[correction.frame_idx :, 0] = forward[correction.frame_idx :, 0]
    timings.extend(forward_timings)
    timings.extend(backward_timings)
    return masks[:, 0], timings


def run_multi_object(model, pixels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    events = [
        PromptEvent(0, 1, ((625.0, 429.0),), (1,)),
        PromptEvent(0, 2, ((300.0, 255.0),), (1,)),
    ]
    state = VideoTrackState(model, pixels, width, height)
    masks, timings = state.propagate([1, 2], list(range(pixels.shape[0])), {0: events})
    return masks, {"objects": len(events), "latency_ms": summarize_timings(timings)}


def run_box_prompt(model, pixels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    points, labels = box_to_sam_points((480.0, 315.0, 820.0, 585.0), width, height)
    event = PromptEvent(0, 1, tuple(map(tuple, points * np.array([width / 1024.0, height / 1024.0], dtype=np.float32))), tuple(int(x) for x in labels))
    masks, timings = run_one_direction(model, pixels, list(range(pixels.shape[0])), event, width, height)
    return masks[:, None], {"latency_ms": summarize_timings(timings)}


def run_negative_clicks(model, pixels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    init = PromptEvent(0, 1, ((300.0, 250.0),), (1,))
    correction = PromptEvent(min(30, pixels.shape[0] - 1), 1, ((330.0, 270.0), (620.0, 430.0)), (1, 0))
    masks, timings = apply_forward_correction(model, pixels, init, correction, width, height)
    return masks[:, None], {"latency_ms": summarize_timings(timings), "correction_frame": correction.frame_idx}


def run_cross_frame_corrections(model, pixels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    init = PromptEvent(0, 1, ((625.0, 429.0),), (1,))
    correction = PromptEvent(min(120, pixels.shape[0] - 1), 1, ((1053.0, 448.0),), (1,))
    masks, timings = apply_forward_correction(model, pixels, init, correction, width, height)
    return masks[:, None], {"latency_ms": summarize_timings(timings), "correction_frame": correction.frame_idx}


def run_nle_bidirectional_correction(model, pixels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    init = PromptEvent(0, 1, ((300.0, 250.0),), (1,))
    middle = min(max(60, pixels.shape[0] // 2), pixels.shape[0] - 1)
    correction = PromptEvent(middle, 1, ((370.0, 255.0), (625.0, 429.0)), (1, 0))
    masks, timings = apply_bidirectional_correction(model, pixels, init, correction, width, height)
    return masks[:, None], {"latency_ms": summarize_timings(timings), "correction_frame": correction.frame_idx, "propagation": "bidirectional"}


def run_bidirectional_middle(model, pixels: np.ndarray, width: int, height: int, frame_idx: int | None = None, point: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    middle = min(frame_idx if frame_idx is not None else max(120, pixels.shape[0] // 2), pixels.shape[0] - 1)
    init = PromptEvent(middle, 1, (point or (1053.0, 448.0),), (1,))
    state = VideoTrackState(model, pixels, width, height)
    obj_ids = [init.obj_id]
    forward, forward_timings = state.propagate(obj_ids, list(range(middle, pixels.shape[0])), {init.frame_idx: [init]})
    # Official SAM2 keeps the same inference state when propagating backward
    # from the middle, so reverse memory selection can use the future-side
    # memories generated by the forward pass.
    backward, backward_timings = state.propagate(obj_ids, list(range(middle, -1, -1)), {})
    forward = forward[:, 0]
    backward = backward[:, 0]
    masks = forward
    masks[:middle] = backward[:middle]
    return masks[:, None], {"latency_ms": summarize_timings(forward_timings + backward_timings), "middle_frame": middle}


def summarize_timings(timings: list[float]) -> dict:
    return {
        "frames": len(timings),
        "mean": float(np.mean(timings)),
        "median": float(np.median(timings)),
        "min": float(np.min(timings)),
        "max": float(np.max(timings)),
    }


SCENARIOS = {
    "multi_object": run_multi_object,
    "box_prompt": run_box_prompt,
    "negative_clicks": run_negative_clicks,
    "cross_frame_corrections": run_cross_frame_corrections,
    "bidirectional_middle": run_bidirectional_middle,
    "nle_bidirectional_correction": run_nle_bidirectional_correction,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), required=True)
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--frames", type=int, default=130)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/feature_benchmarks_130f")
    parser.add_argument("--bidirectional-frame", type=int)
    parser.add_argument("--bidirectional-point", nargs=2, type=float)
    args = parser.parse_args()

    pixels = preprocess_video(args.video, limit=args.frames)
    model = load_image_segmenter(args.weights)

    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if args.scenario == "bidirectional_middle" and (args.bidirectional_frame is not None or args.bidirectional_point is not None):
        masks, report = run_bidirectional_middle(
            model,
            pixels,
            width,
            height,
            frame_idx=args.bidirectional_frame,
            point=tuple(args.bidirectional_point) if args.bidirectional_point else None,
        )
    else:
        masks, report = SCENARIOS[args.scenario](model, pixels, width, height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / f"{args.scenario}_mlx_masks.npy"
    report_path = args.output_dir / f"{args.scenario}_mlx_report.json"
    overlay_path = args.output_dir / f"{args.scenario}_mlx_overlay.mp4"
    np.save(mask_path, masks.astype(np.uint8))
    write_mask_overlay_video(args.video, masks[:, 0], overlay_path, limit=masks.shape[0])
    report = {
        "scenario": args.scenario,
        "video": str(args.video),
        "mask_file": str(mask_path),
        "overlay_file": str(overlay_path),
        "shape": list(masks.shape),
        **report,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
