# mlx-vision

MLX-native vision models for Apple Silicon. The first supported model family is
Meta SAM 2.1 for interactive image segmentation and video object tracking.

The goal of this repo is practical local video segmentation: load an MLX SAM2
checkpoint, click objects in a video, add positive/negative corrections, track
forward or backward from the edited frame, and render masks back to reviewable
overlay videos. The default runtime is Python 3.14 + MLX and does not install
PyTorch.

## What You Can Do

- Segment an image from points or boxes.
- Track one or more objects through a video with SAM2 memory.
- Add positive and negative correction clicks across frames.
- Start from the middle of a clip and propagate forward, backward, or both.
- Use box prompts in the video flow.
- Render `.npy` or `.npz` masks as overlay videos for visual inspection.
- Convert SAM2.1 Hugging Face checkpoints into MLX `.safetensors`.
- Load converted checkpoints from local disk or Hugging Face.

The public video predictor mirrors the official SAM2 method names where the
implemented behavior matches closely:

- `SAM2VideoPredictor.from_pretrained(...)`
- `init_state(...)`
- `add_new_points_or_box(...)`
- `add_new_points(...)`
- `add_new_mask(...)`
- `propagate_in_video(...)`
- `clear_all_prompts_in_frame(...)`
- `reset_state(...)`

## Benchmarks

Current indicative results on this machine with `facebook/sam2.1-hiera-small`.
Generated reports live under `outputs/benchmarks/`.

| Workload | Torch/MPS | MLX | Result |
| --- | ---: | ---: | --- |
| Image encoder | `104 ms/frame` | `81 ms/frame` | MLX `1.28x` faster |
| Cached prompt decode | n/a | `4 ms` | interactive mask decode |
| Full image + prompt | n/a | `85 ms` | image embedding plus prompt |
| Full dog video, post-prompt propagation | `331 ms/frame` | `189 ms/frame` | MLX `1.75x` faster with feature precompute |
| Full dog video, total run | `100.5 s` | `94.8 s` | MLX faster end to end |
| Raw propagation, no save/overlay/final resize | `407 ms/frame` | `287 ms/frame` | MLX `1.42x` faster |

The fastest video path uses batched image-feature precompute and bfloat16 memory
attention:

```bash
uv run python scripts/benchmark_video_memory_mlx.py \
  --model-id facebook/sam2.1-hiera-small \
  --weights checkpoints/sam2.1_hiera_small_image_segmenter.safetensors \
  --frames-dir outputs/video_memory_multiclick/small_frames_full \
  --precompute-image-features \
  --feature-batch-size 4 \
  --memory-dtype bfloat16 \
  --memory-attention-dtype bfloat16 \
  --output-mask outputs/video_memory_multiclick/small_mlx_masks_full_precompute_bf16_attn.npy \
  --output-video outputs/video_memory_multiclick/small_mlx_overlay_full_precompute_bf16_attn.mp4 \
  --report outputs/benchmarks/video_memory_multiclick_small_mlx_full_precompute_bf16_attn.json \
  --points 625 429 700 470 300 250 950 610 \
  --labels 1 1 0 0
```

Torch comparison command:

```bash
uv run --extra torch-parity python scripts/benchmark_video_memory_torch.py \
  --model-id facebook/sam2.1-hiera-small \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --frames-dir outputs/video_memory_multiclick/small_frames_full \
  --output-mask outputs/video_memory_multiclick/small_torch_masks_full.npy \
  --output-video outputs/video_memory_multiclick/small_torch_overlay_full.mp4 \
  --report outputs/benchmarks/video_memory_multiclick_small_torch_full.json \
  --points 625 429 700 470 300 250 950 610 \
  --labels 1 1 0 0
```

Other benchmark entry points:

```bash
uv run --extra torch-parity python scripts/benchmark_image_encoder.py --warmup 3 --runs 10
uv run python scripts/benchmark_prompt_segmenter.py --warmup 3 --runs 20
uv run python scripts/track_video_memory.py --frames 150 \
  --report outputs/benchmarks/video_memory_latency_150f.json
```

## Install

```bash
uv sync --python 3.14
```

Torch is only used for conversion and comparison fixtures:

```bash
uv sync --python 3.14 --extra torch-parity
```

Reference repositories may exist locally for development, but they are not
runtime dependencies:

```text
third_party/sam2
references/mlx-vlm
```

## Quick API

```python
import numpy as np

from mlx_vision import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained(
    "avbiswas/sam2.1-hiera-small-mlx-fp32"
)
state = predictor.init_state("third_party/sam2/demo/data/gallery/01_dog.mp4")

frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
    state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[625.0, 429.0]], dtype=np.float32),
    labels=np.array([1], dtype=np.int32),
)

for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    # masks is a NumPy float32 array shaped O,1,H,W in original video resolution.
    pass
```

Local checkpoint loading:

```python
from mlx_vision import SAM2VideoPredictor

predictor = SAM2VideoPredictor(
    checkpoint="checkpoints/sam2.1_hiera_small_image_segmenter.safetensors"
)
```

## Image Segmentation

Run one prompted frame and write an overlay:

```bash
uv run python scripts/predict_image_mask.py \
  --point 500 610 \
  --output-video outputs/image_prompt_overlay.mp4 \
  --output-mask outputs/image_prompt_mask.npy
```

Coordinates are in the resized `1024x1024` SAM input space.

## Video Tracking

SAM2 memory tracker:

```bash
uv run python scripts/track_video_memory.py --frames 289 \
  --point 500 610 \
  --output-video outputs/dog_memory_overlay_full_v3.mp4 \
  --output-mask outputs/dog_memory_masks_full_v3.npy \
  --report outputs/benchmarks/dog_memory_latency_full_v3.json
```

The current memory tracker uses:

- first-frame point, box, or mask prompts
- SAM2 memory encoder and memory attention
- object pointers
- dynamic multimask fallback on unstable single-mask tracking outputs
- conditioning-frame memory plus SAM2-style frame-indexed temporal memory
  selection
- click-frame masks binarized before memory encoding, matching the official
  postprocessing path
- shared image features for multi-object tracking
- forward, backward, and bidirectional correction replay

## Overlay Utility

Render masks onto a video:

```bash
uv run python scripts/overlay_masks.py \
  --masks outputs/dog_memory_masks_full_v3.npy \
  --output outputs/dog_memory_overlay_from_masks.mp4
```

The overlay script accepts `.npy` or `.npz` masks shaped `T,H,W` or `T,1,H,W`.
Synthetic overlays are only for writer smoke tests and require:

```bash
uv run python scripts/overlay_masks.py --synthetic-smoke-test
```

## Convert Weights

Convert from Hugging Face:

```bash
uv run --extra torch-parity mlx-vision-convert \
  --hf-id facebook/sam2.1-hiera-small \
  --output-dir checkpoints
```

Supported source ids:

```text
facebook/sam2.1-hiera-tiny
facebook/sam2.1-hiera-small
facebook/sam2.1-hiera-base-plus
facebook/sam2.1-hiera-large
```

Convert a local Torch checkpoint:

```bash
uv run --extra torch-parity mlx-vision-convert \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --model-id facebook/sam2.1-hiera-small \
  --output checkpoints/sam2.1_hiera_small_image_segmenter.safetensors
```

The converted checkpoint includes the Hiera image encoder, FPN neck, prompt
encoder, mask decoder, object pointer projection, memory encoder, and memory
attention. Generated checkpoints are ignored by git.

The old script path remains as a compatibility wrapper:

```bash
uv run --extra torch-parity python scripts/convert_image_encoder_weights.py \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --model-id facebook/sam2.1-hiera-small
```

## Feature Regression

Run MLX feature scenarios and compare against Torch fixtures:

```bash
uv run python scripts/run_feature_regression.py --frames 130
```

Regenerate official Torch fixtures first:

```bash
uv run python scripts/run_feature_regression.py --refresh-torch --frames 130
```

Compare existing outputs without rerunning MLX:

```bash
uv run python scripts/run_feature_regression.py --skip-mlx --frames 130
```

Covered scenarios:

- `multi_object`
- `box_prompt`
- `negative_clicks`
- `cross_frame_corrections`
- `bidirectional_middle`

Current MLX-vs-Torch feature benchmark results on the 130-frame dog-gallery
fixture:

| Scenario | Mean IoU | Presence |
| --- | ---: | ---: |
| `multi_object` | `0.973` | `260 / 260` |
| `box_prompt` | `0.953` | `129 / 130` |
| `negative_clicks` | `0.972` | `130 / 130` |
| `cross_frame_corrections` | `0.974` | `130 / 130` |
| `bidirectional_middle` | `0.924` | `128 / 130` |

On `02_cups.mp4`, the same bidirectional benchmark with a center-cup prompt at
frame 120 is tighter:

- `bidirectional_middle` mean IoU: `0.979`
- Presence: `130 / 130`
- Report: `outputs/cups_feature_benchmarks_130f/bidirectional_middle_mlx_vs_torch.json`
- Overlay: `outputs/cups_feature_benchmarks_130f/bidirectional_middle_mlx_overlay.mp4`

Feature reports and inspection overlays are written under:

```text
outputs/feature_benchmarks_130f/
```

## Parity Validation

Parity is used as a guardrail, not as the main product surface. The current
full-video dog run compares MLX against the official Torch `SAM2VideoPredictor`
on all 289 frames:

- Mean mask IoU over all frames: about `0.977`
- Median mask IoU on non-empty Torch frames: about `0.979`
- Presence match: `289 / 289` frames
- Official Torch overlay: `outputs/torch_sam2_dog_overlay_full_twitter.mp4`
- MLX overlay: `outputs/dog_memory_overlay_full_v3_twitter.mp4`
- Comparison report: `outputs/benchmarks/dog_memory_mlx_vs_torch_full_v3.json`

Image and prompt parity fixtures:

```bash
uv run --extra torch-parity python scripts/export_torch_image_embeddings.py --frames 2
uv run python scripts/compare_image_embeddings.py

uv run --extra torch-parity python scripts/export_torch_prompt_mask.py
uv run python scripts/compare_prompt_mask.py
```

Current low-level parity results:

- Image `vision_features` max abs error: about `1.63e-05`
- Prompted low-res masks max abs error: about `4.67e-05`
- Prompted IoU max abs error: about `4.77e-07`

Reports are written under:

```text
outputs/parity/
```

## Runtime Dependency Boundary

Default runtime should not include Torch:

```bash
uv sync --python 3.14
uv run python - <<'PY'
import importlib.util as u
print({m: bool(u.find_spec(m)) for m in ["torch", "torchvision", "hydra", "iopath", "mlx", "cv2"]})
PY
```

Expected:

```text
torch=False, torchvision=False, hydra=False, iopath=False, mlx=True, cv2=True
```
