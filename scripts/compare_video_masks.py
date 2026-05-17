import argparse
import json
from pathlib import Path

import numpy as np


def load_binary_masks(path: Path) -> np.ndarray:
    masks = np.load(path)
    if isinstance(masks, np.lib.npyio.NpzFile):
        first_key = masks.files[0]
        masks = masks[first_key]
    masks = np.asarray(masks)
    if masks.ndim not in {3, 4}:
        raise ValueError(f"Expected masks shaped T,H,W or T,O,H,W, got {masks.shape}")
    return masks > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spot-frames", nargs="*", type=int, default=[0, 30, 60, 80, 90, 100, 120, 149, 180, 240, 288])
    args = parser.parse_args()

    ref = load_binary_masks(args.reference)
    cand = load_binary_masks(args.candidate)
    frames = min(ref.shape[0], cand.shape[0])
    ref = ref[:frames]
    cand = cand[:frames]
    if ref.ndim != cand.ndim:
        if ref.ndim == 3:
            ref = ref[:, None]
        if cand.ndim == 3:
            cand = cand[:, None]
    if ref.ndim == 4 or cand.ndim == 4:
        if ref.ndim != 4 or cand.ndim != 4:
            raise ValueError(f"Could not align mask ranks: reference {ref.shape}, candidate {cand.shape}")
        objects = min(ref.shape[1], cand.shape[1])
        ref = ref[:, :objects]
        cand = cand[:, :objects]

    spatial_axes = tuple(range(2, ref.ndim)) if ref.ndim == 4 else (1, 2)
    inter = np.logical_and(ref, cand).sum(axis=spatial_axes)
    union = np.logical_or(ref, cand).sum(axis=spatial_axes)
    iou = np.ones(union.shape, dtype=np.float64)
    nonempty_union = union > 0
    iou[nonempty_union] = inter[nonempty_union] / union[nonempty_union]

    ref_area = ref.sum(axis=spatial_axes)
    cand_area = cand.sum(axis=spatial_axes)
    ref_present = ref_area > 0
    cand_present = cand_area > 0
    nonempty_ref_iou = iou[ref_present]

    spot_checks = {}
    for frame in args.spot_frames:
        if 0 <= frame < frames:
            if ref.ndim == 4:
                spot_checks[str(frame)] = [
                    {
                        "object_index": obj_idx,
                        "reference_area": int(ref_area[frame, obj_idx]),
                        "candidate_area": int(cand_area[frame, obj_idx]),
                        "iou": float(iou[frame, obj_idx]),
                    }
                    for obj_idx in range(ref.shape[1])
                ]
            else:
                spot_checks[str(frame)] = {
                    "reference_area": int(ref_area[frame]),
                    "candidate_area": int(cand_area[frame]),
                    "iou": float(iou[frame]),
                }

    per_object = None
    if ref.ndim == 4:
        per_object = []
        for obj_idx in range(ref.shape[1]):
            obj_ref_present = ref_present[:, obj_idx]
            obj_iou = iou[:, obj_idx]
            obj_nonempty_ref_iou = obj_iou[obj_ref_present]
            per_object.append(
                {
                    "object_index": obj_idx,
                    "mean_iou_all": float(obj_iou.mean()),
                    "mean_iou_nonempty_reference": float(obj_nonempty_ref_iou.mean()) if obj_nonempty_ref_iou.size else 1.0,
                    "median_iou_nonempty_reference": float(np.median(obj_nonempty_ref_iou)) if obj_nonempty_ref_iou.size else 1.0,
                    "presence_match_frames": int((obj_ref_present == cand_present[:, obj_idx]).sum()),
                    "presence_total_frames": int(frames),
                }
            )

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "frames": int(frames),
        "objects": int(ref.shape[1]) if ref.ndim == 4 else 1,
        "mean_iou_all": float(iou.mean()),
        "mean_iou_nonempty_reference": float(nonempty_ref_iou.mean()) if nonempty_ref_iou.size else 1.0,
        "median_iou_nonempty_reference": float(np.median(nonempty_ref_iou)) if nonempty_ref_iou.size else 1.0,
        "min_iou_nonempty_reference": float(nonempty_ref_iou.min()) if nonempty_ref_iou.size else 1.0,
        "presence_match_frames": int((ref_present == cand_present).sum()),
        "presence_total_frames": int(ref_present.size),
        "false_positive_empty_reference_indices": np.argwhere(~ref_present & cand_present).tolist(),
        "false_negative_nonempty_reference_indices": np.argwhere(ref_present & ~cand_present).tolist(),
        "spot_checks": spot_checks,
    }
    if per_object is not None:
        report["per_object"] = per_object
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
