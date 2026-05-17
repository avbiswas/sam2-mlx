from pathlib import Path

import cv2
import numpy as np


def load_masks(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        masks = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = "masks" if "masks" in data else data.files[0]
        masks = data[key]
    else:
        raise ValueError(f"Expected .npy or .npz masks, got: {path}")
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape T,H,W or T,1,H,W, got {masks.shape}")
    return masks


def synthetic_masks_for_video(video_path: str | Path, limit: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    masks = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        while limit is None or len(masks) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            t = len(masks)
            denom = max(min(total, limit or total) - 1, 1)
            cx = int(w * (0.25 + 0.5 * (t / denom)))
            cy = int(h * 0.52)
            radius = max(8, min(h, w) // 7)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, thickness=-1)
            masks.append(mask)
    finally:
        cap.release()
    if not masks:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return np.stack(masks, axis=0)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32)


def write_mask_overlay_video(
    video_path: str | Path,
    masks: np.ndarray,
    output_path: str | Path,
    color: tuple[int, int, int] = (30, 144, 255),
    alpha: float = 0.55,
    threshold: float = 0.5,
    draw_contour: bool = True,
    limit: int | None = None,
) -> dict:
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_path}")

    frames = 0
    try:
        while frames < len(masks) and (limit is None or frames < limit):
            ok, frame = cap.read()
            if not ok:
                break
            mask = _resize_mask(masks[frames], width, height)
            binary = mask > threshold
            overlay = frame.copy()
            overlay[binary] = color
            blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

            if draw_contour:
                contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(blended, contours, -1, color, 2)

            writer.write(blended)
            frames += 1
    finally:
        cap.release()
        writer.release()

    return {
        "video": str(video_path),
        "output": str(output_path),
        "frames": frames,
        "fps": float(fps),
        "width": width,
        "height": height,
    }
