from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from mlx_sam.video_predictor import SAM2VideoPredictor


APP_ROOT = Path("outputs/manual_app")
DEFAULT_MODEL_ID = "avbiswas/sam2.1-hiera-base-plus-mlx-8bit"
SESSIONS: dict[str, dict] = {}


def _frame_data_url(frame: np.ndarray, quality: int = 86) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _jpeg_bytes(frame: np.ndarray, quality: int = 86) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode frame")
    return encoded.tobytes()


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_idx}")
        return frame
    finally:
        cap.release()


def _video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1,
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        }
    finally:
        cap.release()


def _parse_color(value) -> tuple[int, int, int]:
    if value is None:
        return (194, 26, 255)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 6:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
            return (b, g, r)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        r, g, b = [max(0, min(255, int(v))) for v in value]
        return (b, g, r)
    raise ValueError(f"Expected color as #rrggbb or [r,g,b], got {value!r}")


def _object_color(index: int, base_color: tuple[int, int, int] = (194, 26, 255)) -> tuple[int, int, int]:
    palette = [
        base_color,
        (125, 195, 25),
        (255, 128, 47),
        (102, 209, 255),
        (246, 92, 139),
        (255, 212, 0),
    ]
    return palette[index % len(palette)]


def _overlay_frame(frame: np.ndarray, masks: np.ndarray, color: tuple[int, int, int] = (194, 26, 255)) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, None]
    elif masks.ndim == 3:
        masks = masks[:, None]
    if masks.ndim != 4:
        raise ValueError(f"Expected masks shaped O,1,H,W, O,H,W, or H,W; got {masks.shape}")
    overlay = frame.copy()
    blended = frame.copy()
    for obj_idx in range(masks.shape[0]):
        mask = masks[obj_idx, 0] > 0
        if not np.any(mask):
            continue
        obj_color = _object_color(obj_idx, color)
        color_arr = np.array(obj_color, dtype=np.uint8)
        overlay[mask] = color_arr
        blended = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, obj_color, 2)
    return blended


def _write_sparse_overlay_video(
    video_path: Path,
    frame_indices: list[int],
    masks: np.ndarray,
    output_path: Path,
    color: tuple[int, int, int],
) -> None:
    if not frame_indices:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = _video_info(video_path)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        for idx, mask in zip(frame_indices, masks):
            frame = _read_frame(video_path, idx)
            writer.write(_overlay_frame(frame, mask, color=color))
    finally:
        writer.release()


def _load_saved_mask_for_frame(session: dict, frame_idx: int) -> np.ndarray:
    path = session["session_dir"] / "masks.npz"
    if not path.exists():
        raise FileNotFoundError("No final masks are available yet")
    data = np.load(path)
    masks = data["masks"]
    frame_indices = data["frame_indices"].astype(np.int32)
    if frame_indices.size == 0:
        raise ValueError("Final mask artifact contains no frames")
    frame_idx = int(frame_idx)
    right = int(np.searchsorted(frame_indices, frame_idx, side="left"))
    if right < frame_indices.size and int(frame_indices[right]) == frame_idx:
        mask = masks[right]
    elif right <= 0:
        mask = masks[0]
    elif right >= frame_indices.size:
        mask = masks[-1]
    else:
        left = right - 1
        left_idx = int(frame_indices[left])
        right_idx = int(frame_indices[right])
        alpha = (frame_idx - left_idx) / max(right_idx - left_idx, 1)
        mask = (1.0 - alpha) * masks[left] + alpha * masks[right]
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4:
        raise ValueError(f"Unexpected saved mask shape: {mask.shape}")
    return mask.astype(np.float32)


def _json_line(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def _group_points_by_object(points: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[dict]] = {}
    for point in points:
        obj_id = max(1, int(point.get("obj_id", point.get("object_id", 1))))
        grouped.setdefault(obj_id, []).append(point)
    parsed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for obj_id, obj_points in grouped.items():
        coords = np.array([[p["x"], p["y"]] for p in obj_points], dtype=np.float32)
        labels = np.array([p["label"] for p in obj_points], dtype=np.int32)
        if not np.any(labels == 1):
            raise ValueError(f"object {obj_id} needs at least one positive point")
        parsed[obj_id] = (coords, labels)
    return parsed


def _model_cache_key(request: dict) -> tuple:
    return (
        request.get("model_id") or DEFAULT_MODEL_ID,
        int(request.get("image_size") or 768),
        request.get("memory_dtype"),
        request.get("memory_attention_dtype"),
    )


def _get_preview_context(session: dict, request: dict) -> tuple[SAM2VideoPredictor, dict]:
    cache = session.setdefault("preview_cache", {})
    key = _model_cache_key(request)
    cached = cache.get(key)
    if cached is None:
        predictor = SAM2VideoPredictor.from_pretrained(
            key[0],
            image_size=key[1],
            memory_dtype=key[2],
            memory_attention_dtype=key[3],
        )
        state = predictor.init_state(session["video_path"], precompute_image_features=False)
        cached = {"predictor": predictor, "state": state}
        cache[key] = cached
    else:
        predictor = cached["predictor"]
        state = cached["state"]
        predictor.reset_state(state)
    return predictor, state


def _run_one_direction(
    predictor: SAM2VideoPredictor,
    state: dict,
    video_path: Path,
    start_frame_idx: int,
    reverse: bool,
    yield_every: int,
    frame_step: int,
    color: tuple[int, int, int],
    phase: str,
) -> Iterable[tuple[dict, np.ndarray]]:
    total = state["num_frames"]
    max_frames = start_frame_idx + 1 if reverse else total - start_frame_idx
    for event in predictor.stream_in_video(
        state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frames,
        reverse=reverse,
        yield_every=yield_every,
        return_full=True,
        frame_step=frame_step,
    ):
        if event["type"] == "final":
            if event["masks"] is None:
                continue
            yield {
                "type": "direction_final",
                "phase": phase,
                "frame_indices": event["frame_indices"],
            }, event["masks"]
            continue
        if event["type"] != "frame":
            continue
        frame_idx = int(event["frame_idx"])
        frame = _read_frame(video_path, frame_idx)
        overlay = _overlay_frame(frame, event["masks"], color=color)
        yield {
            "type": "frame",
            "phase": phase,
            "frame_idx": frame_idx,
            "obj_ids": event["obj_ids"],
            "overlay": _frame_data_url(overlay),
            "progress": 0.0,
        }, event["masks"]


def create_app():
    app = FastAPI(title="mlx-sam lab")
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.post("/api/upload")
    async def upload(video: UploadFile = File(...)):
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        session_dir = APP_ROOT / session_id
        session_dir.mkdir(parents=True)
        suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
        video_path = session_dir / f"input{suffix}"
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
        info = _video_info(video_path)
        frame = _read_frame(video_path, 0)
        SESSIONS[session_id] = {"video_path": video_path, "session_dir": session_dir, "preview_cache": {}, **info}
        return JSONResponse({"session_id": session_id, "video_url": f"/api/video/{session_id}", "frame": _frame_data_url(frame), **info})

    @app.get("/api/frame/{session_id}/{frame_idx}")
    def frame(session_id: str, frame_idx: int):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        frame_idx = max(0, min(int(frame_idx), int(session["frames"]) - 1))
        frame_bgr = _read_frame(session["video_path"], frame_idx)
        return JSONResponse({"frame_idx": frame_idx, "frame": _frame_data_url(frame_bgr)})

    @app.get("/api/video/{session_id}")
    def video(session_id: str):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        return FileResponse(session["video_path"], media_type="video/mp4", filename="input.mp4")

    @app.get("/api/artifact/{session_id}/{name}")
    def artifact(session_id: str, name: str):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        if name not in {"masks.npy", "masks.npz", "overlay.mp4"}:
            return JSONResponse({"error": "unknown artifact"}, status_code=404)
        path = session["session_dir"] / name
        if not path.exists():
            return JSONResponse({"error": "artifact not ready"}, status_code=404)
        media_type = "video/mp4" if name.endswith(".mp4") else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/api/overlay_frame/{session_id}/{frame_idx}")
    def overlay_frame(session_id: str, frame_idx: int):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        try:
            frame_idx = max(0, min(int(frame_idx), int(session["frames"]) - 1))
            frame_bgr = _read_frame(session["video_path"], frame_idx)
            mask = _load_saved_mask_for_frame(session, frame_idx)
            color = session.get("last_overlay_color", (194, 26, 255))
            overlay = _overlay_frame(frame_bgr, mask, color=color)
            return Response(
                content=_jpeg_bytes(overlay),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=404)

    @app.post("/api/prompt")
    async def prompt(request: dict):
        started = time.perf_counter()
        try:
            session = SESSIONS[request["session_id"]]
            points = request.get("points") or []
            if not points:
                return JSONResponse({"error": "at least one point is required"}, status_code=400)
            grouped_points = _group_points_by_object(points)
            color = _parse_color(request.get("color"))
            frame_idx = max(0, min(int(request.get("frame_idx", 0)), int(session["frames"]) - 1))
            predictor, state = _get_preview_context(session, request)
            obj_ids = []
            masks = None
            out_frame_idx = frame_idx
            for obj_id, (coords, labels) in grouped_points.items():
                out_frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=coords,
                    labels=labels,
                    normalize_coords=True,
                )
            if masks is None:
                raise ValueError("No prompt masks were produced")
            frame_bgr = _read_frame(session["video_path"], out_frame_idx)
            overlay = _overlay_frame(frame_bgr, masks, color=color)
            return JSONResponse(
                {
                    "type": "prompt",
                    "frame_idx": int(out_frame_idx),
                    "obj_ids": list(obj_ids),
                    "overlay": _frame_data_url(overlay),
                    "ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.post("/api/segment")
    async def segment(request: dict):
        def generate():
            started = time.perf_counter()
            try:
                session = SESSIONS[request["session_id"]]
                video_path = session["video_path"]
                session_dir = session["session_dir"]
                frame_idx = int(request.get("frame_idx", 0))
                points = request.get("points") or []
                if not points:
                    raise ValueError("At least one positive or negative point is required")
                grouped_points = _group_points_by_object(points)
                predictor = SAM2VideoPredictor.from_pretrained(
                    request.get("model_id") or DEFAULT_MODEL_ID,
                    image_size=int(request.get("image_size") or 768),
                    memory_dtype=request.get("memory_dtype"),
                    memory_attention_dtype=request.get("memory_attention_dtype"),
                )
                precompute = bool(request.get("precompute_image_features"))
                feature_batch_size = max(1, int(request.get("feature_batch_size") or 4))
                if precompute:
                    yield _json_line(
                        {
                            "type": "status",
                            "message": f"Precomputing video features with batch size {feature_batch_size}...",
                            "progress": 0.0,
                            "ms": (time.perf_counter() - started) * 1000.0,
                        }
                    )
                state = predictor.init_state(
                    video_path,
                    precompute_image_features=precompute,
                    feature_batch_size=feature_batch_size,
                )
                for obj_id, (coords, labels) in grouped_points.items():
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=coords,
                        labels=labels,
                        normalize_coords=True,
                    )
                direction = request.get("direction") or "forward"
                yield_every = max(1, int(request.get("yield_every") or 30))
                frame_step = max(1, int(request.get("frame_step") or 1))
                color = _parse_color(request.get("color"))
                collected: dict[int, np.ndarray] = {}
                phases = []
                if direction in {"backward", "both"}:
                    phases.append(("backward", True))
                if direction in {"forward", "both"}:
                    phases.append(("forward", False))
                if not phases:
                    raise ValueError(f"Unknown direction: {direction}")
                total_steps = sum(frame_idx + 1 if reverse else state["num_frames"] - frame_idx for _, reverse in phases)
                seen = 0
                for phase, reverse in phases:
                    for payload, masks in _run_one_direction(
                        predictor,
                        state,
                        video_path,
                        frame_idx,
                        reverse=reverse,
                        yield_every=yield_every,
                        frame_step=frame_step,
                        color=color,
                        phase=phase,
                    ):
                        if payload["type"] == "direction_final":
                            for out_frame_idx, mask in zip(payload["frame_indices"].tolist(), masks):
                                collected[int(out_frame_idx)] = mask
                            continue
                        collected[int(payload["frame_idx"])] = masks
                        seen += yield_every
                        payload["progress"] = min(0.98, seen / max(total_steps, 1))
                        payload["ms"] = (time.perf_counter() - started) * 1000.0
                        yield _json_line(payload)
                if collected:
                    ordered = sorted(collected)
                    masks = np.stack([collected[idx][:, 0] for idx in ordered], axis=0)
                    mask_path = session_dir / "masks.npz"
                    overlay_path = session_dir / "overlay.mp4"
                    session["last_overlay_color"] = color
                    np.savez(mask_path, masks=masks, frame_indices=np.asarray(ordered, dtype=np.int32))
                    _write_sparse_overlay_video(video_path, ordered, masks, overlay_path, color=color)
                else:
                    ordered = []
                    mask_path = None
                    overlay_path = None
                yield _json_line(
                    {
                        "type": "final",
                        "is_final": True,
                        "frame_indices": ordered,
                        "mask_path": None if mask_path is None else str(mask_path),
                        "overlay_path": None if overlay_path is None else str(overlay_path),
                        "mask_url": f"/api/artifact/{request['session_id']}/masks.npz" if mask_path else None,
                        "overlay_url": f"/api/artifact/{request['session_id']}/overlay.mp4" if overlay_path else None,
                        "ms": (time.perf_counter() - started) * 1000.0,
                    }
                )
            except Exception as exc:
                yield _json_line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install app dependencies with: uv sync --extra app") from exc
    uvicorn.run("mlx_sam.server:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
