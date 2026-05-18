# sam-mlx

MLX inference port of Meta's SAM 2.1. The most tested checkpoint is
`facebook/sam2.1-hiera-small`; the converter understands the SAM2.1 tiny,
small, base-plus, and large Hugging Face checkpoints.

The runtime package is Python 3.14 + MLX and does not install PyTorch. PyTorch is
only used through the optional `torch-parity` extra for checkpoint conversion and
parity fixtures.

## Current Checkpoint

Expected local source checkpoint:

```text
checkpoints/sam2.1_hiera_small.pt
```

Converted MLX checkpoint:

```text
checkpoints/sam2.1_hiera_small_image_segmenter.safetensors
```

This converted checkpoint includes:

- Hiera image encoder
- FPN neck
- prompt encoder
- mask decoder
- object pointer projection
- memory encoder
- memory attention

The older image-encoder-only conversion may also exist locally:

```text
checkpoints/sam2.1_hiera_small_image_encoder.safetensors
```

Generated checkpoints are ignored by git.

## Setup

```bash
uv sync --python 3.14
```

For Torch parity and conversion scripts:

```bash
uv sync --python 3.14 --extra torch-parity
```

Reference repositories are expected locally but are not runtime dependencies:

```text
third_party/sam2
references/mlx-vlm
```

## Convert Weights

```bash
uv run --extra torch-parity sam-mlx-convert \
  --hf-id facebook/sam2.1-hiera-small \
  --output-dir checkpoints
```

This writes:

```text
checkpoints/sam2.1_hiera_small_image_segmenter.safetensors
```

Supported Hugging Face source ids:

```text
facebook/sam2.1-hiera-tiny
facebook/sam2.1-hiera-small
facebook/sam2.1-hiera-base-plus
facebook/sam2.1-hiera-large
```

Convert a local Torch checkpoint:

```bash
uv run --extra torch-parity sam-mlx-convert \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --model-id facebook/sam2.1-hiera-small \
  --output checkpoints/sam2.1_hiera_small_image_segmenter.safetensors
```

The old script path remains as a compatibility wrapper:

```bash
uv run --extra torch-parity python scripts/convert_image_encoder_weights.py \
  --checkpoint checkpoints/sam2.1_hiera_small.pt \
  --model-id facebook/sam2.1-hiera-small
```

## Parity Fixtures

Generate Torch image-embedding fixtures:

```bash
uv run --extra torch-parity python scripts/export_torch_image_embeddings.py --frames 2
uv run python scripts/compare_image_embeddings.py
```

Generate Torch prompted-mask fixtures:

```bash
uv run --extra torch-parity python scripts/export_torch_prompt_mask.py
uv run python scripts/compare_prompt_mask.py
```

Current parity results:

- Image `vision_features` max abs error: about `1.63e-05`
- Prompted low-res masks max abs error: about `4.67e-05`
- Prompted IoU max abs error: about `4.77e-07`

Reports are written under:

```text
outputs/parity/
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

Mask-prompt feedback baseline:

```bash
uv run python scripts/propagate_video_masks.py --frames 30
```

SAM2 memory tracker:

```bash
uv run python scripts/track_video_memory.py --frames 289 \
  --point 500 610 \
  --output-video outputs/dog_memory_overlay_full_v3.mp4 \
  --output-mask outputs/dog_memory_masks_full_v3.npy \
  --report outputs/benchmarks/dog_memory_latency_full_v3.json
```

Package API, matching the official SAM2 video predictor method names:

```python
import numpy as np

from sam_mlx import SAM2VideoPredictor

predictor = SAM2VideoPredictor(
    checkpoint="checkpoints/sam2.1_hiera_small_image_segmenter.safetensors"
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

Implemented API methods:

- `SAM2VideoPredictor.from_pretrained(...)`
- `init_state(...)`
- `add_new_points_or_box(...)`
- `add_new_points(...)`
- `add_new_mask(...)`
- `propagate_in_video(...)`
- `clear_all_prompts_in_frame(...)`
- `reset_state(...)`

The current memory tracker uses:

- first-frame point prompt
- SAM2 memory encoder
- SAM2 memory attention
- object pointers
- dynamic multimask fallback on unstable single-mask tracking outputs
- conditioning-frame memory plus SAM2-style frame-indexed temporal memory
  selection
- click-frame masks binarized before memory encoding, matching the official
  postprocessing path

The dog-gallery parity run compares MLX against the official Torch
`SAM2VideoPredictor` output on all 289 frames:

- Mean mask IoU over all frames: about `0.977`
- Median mask IoU on non-empty Torch frames: about `0.979`
- Presence match: `289 / 289` frames
- Official Torch overlay: `outputs/torch_sam2_dog_overlay_full_twitter.mp4`
- MLX overlay: `outputs/dog_memory_overlay_full_v3_twitter.mp4`
- Comparison report: `outputs/benchmarks/dog_memory_mlx_vs_torch_full_v3.json`

To regenerate the official Torch dog fixture:

```bash
uv run --extra torch-parity python scripts/run_torch_sam2_dog_video.py
```

To regenerate the MLX-vs-Torch dog comparison report:

```bash
uv run python scripts/compare_video_masks.py \
  --reference outputs/torch_sam2_dog_masks_full.npy \
  --candidate outputs/dog_memory_masks_full_v3.npy \
  --output outputs/benchmarks/dog_memory_mlx_vs_torch_full_v3.json
```

## Feature Parity Benchmarks

Generate official Torch fixtures for the video UX/state features we still need
to replicate:

```bash
uv run --extra torch-parity python scripts/run_torch_video_feature_benchmarks.py \
  --scenario all \
  --frames 130 \
  --frames-dir outputs/torch_feature_benchmark_frames_130f \
  --output-dir outputs/feature_benchmarks_130f
```

This writes `T,O,H,W` mask fixtures and per-scenario reports for:

- `multi_object`
- `box_prompt`
- `negative_clicks`
- `cross_frame_corrections`
- `bidirectional_middle`

The current generated fixture summary is:

```text
outputs/feature_benchmarks_130f/torch_feature_benchmarks_summary.json
```

Run MLX and compare every tracked feature against the Torch fixtures:

```bash
uv run python scripts/run_feature_regression.py --frames 130
```

To regenerate Torch fixtures first, use:

```bash
uv run python scripts/run_feature_regression.py --refresh-torch --frames 130
```

During inner-loop work, compare existing outputs without rerunning MLX:

```bash
uv run python scripts/run_feature_regression.py --skip-mlx --frames 130
```

Use the same comparator directly for a single scenario:

```bash
uv run python scripts/track_video_features_mlx.py \
  --scenario multi_object \
  --frames 130 \
  --output-dir outputs/feature_benchmarks_130f

uv run python scripts/compare_video_masks.py \
  --reference outputs/feature_benchmarks_130f/multi_object_torch_masks.npy \
  --candidate outputs/feature_benchmarks_130f/multi_object_mlx_masks.npy \
  --output outputs/feature_benchmarks_130f/multi_object_mlx_vs_torch.json
```

Current MLX-vs-Torch feature benchmark results on the 130-frame dog-gallery
fixture:

- `multi_object`: mean IoU `0.973`, presence `260 / 260`
- `box_prompt`: mean IoU `0.953`, presence `129 / 130`
- `negative_clicks`: mean IoU `0.972`, presence `130 / 130`
- `cross_frame_corrections`: mean IoU `0.974`, presence `130 / 130`
- `bidirectional_middle`: mean IoU `0.924`, presence `128 / 130`

The bidirectional case is currently functionally correct but still below the
other scenarios. The remaining mismatch is a two-frame object-presence boundary
near occlusion: Torch keeps tiny masks on frames 76-77, while MLX gates those
frames as no-object.

On another gallery video, `02_cups.mp4`, the same bidirectional benchmark with a
center-cup prompt at frame 120 is substantially tighter:

- `bidirectional_middle` on cups: mean IoU `0.979`, presence `130 / 130`
- Report: `outputs/cups_feature_benchmarks_130f/bidirectional_middle_mlx_vs_torch.json`
- Overlay: `outputs/cups_feature_benchmarks_130f/bidirectional_middle_mlx_overlay.mp4`

The MLX feature runner now uses a shared video state for multiple objects. It
encodes each frame once, keeps per-object conditioning/non-conditioning memory
banks, and supports NLE-style correction replay:

- forward replay from a correction frame for positive/negative click edits
- bidirectional replay from a middle-frame correction
- preserved masks outside the edited replay range

An MLX-only inspection scenario for middle-frame editor correction is available:

```bash
uv run python scripts/track_video_features_mlx.py \
  --scenario nle_bidirectional_correction \
  --frames 90 \
  --output-dir outputs/feature_benchmarks_130f
```

Feature reports and inspection overlays are written under:

```text
outputs/feature_benchmarks_130f/
```

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

## Benchmarks

Image encoder:

```bash
uv run --extra torch-parity python scripts/benchmark_image_encoder.py --warmup 3 --runs 10
```

Prompt segmentation:

```bash
uv run python scripts/benchmark_prompt_segmenter.py --warmup 3 --runs 20
```

Video memory tracking:

```bash
uv run python scripts/track_video_memory.py --frames 150 \
  --report outputs/benchmarks/video_memory_latency_150f.json
```

Official Torch-vs-MLX full-video memory benchmark with multiple positive and
negative clicks:

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

Current indicative numbers on this machine:

- Image encoder MLX: about `81 ms/frame`
- Image encoder Torch/MPS: about `104 ms/frame`
- MLX image encoder speedup: about `1.28x`
- Cached prompt decode: about `4 ms`
- Full image + prompt: about `85 ms`
- Dog full-video memory tracker: about `269 ms/frame` on the 289-frame run
- Dog full-video multi-click memory tracker, post-prompt propagation:
  - Official Torch: about `331 ms/frame`
  - MLX on-demand with bfloat memory attention: about `351 ms/frame`
  - MLX with batched image-feature precompute and bfloat memory attention:
    about `189 ms/frame`
- Dog full-video total for the same multi-click run:
  - Official Torch: about `100.5 s`
  - MLX with batched image-feature precompute and bfloat memory attention:
    about `94.8 s`
- Dog full-video raw propagation, excluding mask saving, overlay, and final
  video-resolution output resize:
  - Official Torch raw: about `407 ms/frame`
  - MLX raw with bfloat memory attention: about `287 ms/frame`

Benchmark reports are written under:

```text
outputs/benchmarks/
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
