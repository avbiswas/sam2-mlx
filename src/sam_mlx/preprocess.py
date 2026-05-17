from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image, image_size: int = 1024) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))[None, ...]


def read_video_frames(path: str | Path, limit: int | None = None) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames: list[Image.Image] = []
    try:
        while limit is None or len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        cap.release()
    return frames


def preprocess_video(path: str | Path, image_size: int = 1024, limit: int | None = None) -> np.ndarray:
    frames = read_video_frames(path, limit=limit)
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.concatenate([preprocess_image(frame, image_size=image_size) for frame in frames], axis=0)
