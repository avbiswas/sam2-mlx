# API Server

`mlx-sam` includes a small local FastAPI server for manual video segmentation.
It is meant for development, demos, and API experiments around the core
`SAM2VideoPredictor` runtime.

The browser frontend served at `/` is intentionally a simple demo app. Treat it
as a reference client for the API, not as the product UI.

## Start

Install the optional app dependencies:

```bash
uv sync --extra app
```

Run the server:

```bash
uv run mlx-sam-app --host 127.0.0.1 --port 7861
```

Open:

```text
http://127.0.0.1:7861
```

## Endpoints

### `POST /api/upload`

Uploads a video and creates an in-memory session.

Request: multipart form with a `video` file.

Response includes:

```json
{
  "session_id": "...",
  "video_url": "/api/video/...",
  "width": 1280,
  "height": 720,
  "frames": 289,
  "fps": 29.97
}
```

### `GET /api/video/{session_id}`

Returns the uploaded source video for browser playback.

### `GET /api/frame/{session_id}/{frame_idx}`

Returns a JPEG data URL for a single source frame. This is mainly for fallback
or debugging; the demo frontend uses native video playback for smooth playback.

### `POST /api/prompt`

Runs image-level prompt preview on the selected frame.

Request body:

```json
{
  "session_id": "...",
  "model_id": "avbiswas/sam2.1-hiera-small-mlx-4bit",
  "frame_idx": 42,
  "points": [{"x": 640.0, "y": 360.0, "label": 1}],
  "image_size": 768,
  "color": "#ff1ac2",
  "memory_dtype": "bfloat16",
  "memory_attention_dtype": "bfloat16"
}
```

The prompt preview cache is keyed by video session, model id, image size, and
memory dtypes. It reuses the loaded predictor/state for repeated point edits.

### `POST /api/segment`

Runs video propagation and streams newline-delimited JSON events.

Request body:

```json
{
  "session_id": "...",
  "model_id": "avbiswas/sam2.1-hiera-small-mlx-4bit",
  "frame_idx": 42,
  "points": [{"x": 640.0, "y": 360.0, "label": 1}],
  "direction": "both",
  "yield_every": 10,
  "frame_step": 2,
  "precompute_image_features": false,
  "feature_batch_size": 4,
  "image_size": 768,
  "color": "#ff1ac2",
  "memory_dtype": "bfloat16",
  "memory_attention_dtype": "bfloat16"
}
```

Stream event types:

- `status`: server-side status, for example precompute start.
- `frame`: live overlay preview for a processed frame.
- `final`: final artifact locations.
- `error`: error message.

The final mask artifact is saved as:

```text
outputs/manual_app/{session_id}/masks.npz
```

It contains:

- `masks`: mask logits for processed frames.
- `frame_indices`: original video frame indexes for each mask.

### `GET /api/overlay_frame/{session_id}/{frame_idx}`

Returns a JPEG overlay for a source frame using the saved final masks. If
`frame_step > 1`, the server linearly interpolates between neighboring saved
mask logits.

### `GET /api/artifact/{session_id}/{name}`

Returns generated artifacts such as:

- `masks.npz`
- `overlay.mp4`

## Notes

Sessions are process-local and stored in memory. Uploaded files and artifacts
are written under `outputs/manual_app/`. Restarting the server clears the
session registry, but existing files remain on disk.

Full video feature precompute is currently performed per `/api/segment` call.
Prompt preview has its own cache, but video propagation state is not reused
between separate segmentation runs yet.
