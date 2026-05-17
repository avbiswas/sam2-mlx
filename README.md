# sam-mlx

MLX inference port of Meta's SAM 2.1, currently targeting
`facebook/sam2.1-hiera-small`.

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
uv run --extra torch-parity python scripts/convert_image_encoder_weights.py
```

This writes:

```text
checkpoints/sam2.1_hiera_small_image_segmenter.safetensors
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

The current memory tracker uses:

- first-frame point prompt
- SAM2 memory encoder
- SAM2 memory attention
- object pointers
- the prompted conditioning frame plus up to the last six recent memory frames

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

Current indicative numbers on this machine:

- Image encoder MLX: about `81 ms/frame`
- Image encoder Torch/MPS: about `104 ms/frame`
- MLX image encoder speedup: about `1.28x`
- Cached prompt decode: about `4 ms`
- Full image + prompt: about `85 ms`
- Dog full-video memory tracker: about `269 ms/frame` on the 289-frame run

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
