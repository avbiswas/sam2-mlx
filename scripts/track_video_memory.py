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


def best_low_mask(out: dict) -> tuple[np.ndarray, np.ndarray, int]:
    low = np.array(out["low_res_masks"])
    ious = np.array(out["ious"])
    idx = int(np.argmax(ious[0])) if ious.shape[1] > 1 else 0
    return low[:, idx : idx + 1], ious, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--point", nargs=2, type=float, default=(588.0, 626.0))
    parser.add_argument("--output-mask", type=Path, default=ROOT / "outputs/video_memory_masks.npy")
    parser.add_argument("--output-video", type=Path, default=ROOT / "outputs/video_memory_overlay.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/benchmarks/video_memory_latency.json")
    args = parser.parse_args()

    pixels = preprocess_video(args.video, limit=args.frames)
    model = load_image_segmenter(args.weights)

    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    cond_memories = []
    memories = []
    masks = []
    timings = []
    scores = []

    for frame_idx in range(pixels.shape[0]):
        frame = mx.array(pixels[frame_idx : frame_idx + 1])
        start = time.perf_counter()
        encoded = model.encode_image(frame)
        if frame_idx == 0:
            out = model.predict_from_encoded(
                encoded,
                mx.array(np.array([[args.point]], dtype=np.float32)),
                mx.array(np.array([[1]], dtype=np.int32)),
                multimask_output=True,
            )
        else:
            conditioned = model.condition_with_memories(encoded, memories, cond_memories=cond_memories, current_frame_idx=frame_idx)
            conditioned_encoded = dict(encoded)
            conditioned_encoded["vision_features"] = conditioned
            out = model.predict_from_encoded(conditioned_encoded, None, None, multimask_output=False, add_no_mem_embed=False)

        mx.eval(out["low_res_masks"], out["ious"], out["obj_ptr"], out["object_score_logits"])
        low, ious, best = best_low_mask(out)
        mem = model.encode_memory(encoded["vision_features"], mx.array(low), out["object_score_logits"], is_mask_from_points=(frame_idx == 0))
        mx.eval(mem["vision_features"], mem["vision_pos_enc"])
        memory = {
            "maskmem_features": mem["vision_features"],
            "maskmem_pos_enc": mem["vision_pos_enc"][0],
            "obj_ptr": out["obj_ptr"],
            "frame_idx": frame_idx,
        }
        if frame_idx == 0:
            cond_memories.append(memory)
        else:
            memories.append(memory)

        mask = cv2.resize(low[0, 0], (width, height), interpolation=cv2.INTER_LINEAR) > 0
        masks.append(mask.astype(np.float32))
        scores.append({"ious": ious[0].tolist(), "selected": best})
        timings.append((time.perf_counter() - start) * 1000.0)

    masks_np = np.stack(masks, axis=0)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, masks_np)
    overlay = write_mask_overlay_video(args.video, masks_np, args.output_video, limit=masks_np.shape[0])
    report = {
        **overlay,
        "mask_file": str(args.output_mask),
        "method": "mlx_sam2_memory_cond_plus_last_6_frames",
        "latency_ms": {
            "frames": len(timings),
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
        },
        "scores": scores,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
