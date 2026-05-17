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
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped T,H,W or T,1,H,W, got {masks.shape}")
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

    inter = np.logical_and(ref, cand).sum(axis=(1, 2))
    union = np.logical_or(ref, cand).sum(axis=(1, 2))
    iou = np.ones(frames, dtype=np.float64)
    nonempty_union = union > 0
    iou[nonempty_union] = inter[nonempty_union] / union[nonempty_union]

    ref_area = ref.sum(axis=(1, 2))
    cand_area = cand.sum(axis=(1, 2))
    ref_present = ref_area > 0
    cand_present = cand_area > 0
    nonempty_ref_iou = iou[ref_present]

    spot_checks = {}
    for frame in args.spot_frames:
        if 0 <= frame < frames:
            spot_checks[str(frame)] = {
                "reference_area": int(ref_area[frame]),
                "candidate_area": int(cand_area[frame]),
                "iou": float(iou[frame]),
            }

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "frames": int(frames),
        "mean_iou_all": float(iou.mean()),
        "mean_iou_nonempty_reference": float(nonempty_ref_iou.mean()) if nonempty_ref_iou.size else 1.0,
        "median_iou_nonempty_reference": float(np.median(nonempty_ref_iou)) if nonempty_ref_iou.size else 1.0,
        "min_iou_nonempty_reference": float(nonempty_ref_iou.min()) if nonempty_ref_iou.size else 1.0,
        "presence_match_frames": int((ref_present == cand_present).sum()),
        "presence_total_frames": int(frames),
        "false_positive_empty_reference_frames": np.where(~ref_present & cand_present)[0].tolist(),
        "false_negative_nonempty_reference_frames": np.where(ref_present & ~cand_present)[0].tolist(),
        "spot_checks": spot_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
