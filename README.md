# mlx-sam

MLX-native SAM models for Apple Silicon. The first supported model family is
Meta SAM 2.1 for interactive image segmentation and video object tracking.

The goal of this repo is practical local video segmentation: load an MLX SAM2
checkpoint, click objects in a video, add positive/negative corrections, track
forward or backward from the edited frame, and render masks back to reviewable
overlay videos. The default runtime is Python 3.14 + MLX and does not install
PyTorch.

https://github.com/user-attachments/assets/0946cad0-8af8-4efc-b504-f7416083d64c

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


## Quick API

You can pip install this library

```bash
pip install mlx-sam
```

Load a Hugging Face checkpoint, add a prompt, and stream masks as they are
generated:

```python
import numpy as np

from mlx_sam import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained(
    "avbiswas/sam2.1-hiera-small-mlx"
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

For UI or worker streaming, use `stream_in_video(...)`. It returns dictionary
events, throttles intermediate frame events with `yield_every`, and can emit a
final stacked mask tensor:

```python
for event in predictor.stream_in_video(state, yield_every=30, return_full=True):
    if event["type"] == "frame":
        frame_idx = event["frame_idx"]
        masks = event["masks"]  # O,1,H,W for this frame
    elif event["type"] == "final":
        frame_indices = event["frame_indices"]  # T
        masks = event["masks"]  # T,O,1,H,W for every processed frame
```

Local checkpoint loading works the same way:

```python
from mlx_sam import SAM2VideoPredictor

predictor = SAM2VideoPredictor(
    checkpoint="checkpoints/sam2.1_hiera_small_image_segmenter.safetensors"
)
```

Spatial downsampling is opt-in through `image_size`. The default is `1024`,
matching SAM2. Lower values trade mask quality for latency and memory:

```python
predictor = SAM2VideoPredictor.from_pretrained(
    "avbiswas/sam2.1-hiera-small-mlx",
    image_size=768,
    memory_dtype="float16",
    memory_attention_dtype="float16",
)
```

For editor-style use, precompute image features once during `init_state`.
This is a parity-preserving speed path for repeated propagation and correction
passes, at the cost of higher upfront work and cached memory:

```python
state = predictor.init_state(
    "clip.mp4",
    precompute_image_features=True,
    feature_batch_size=4,
)
```

Track from the middle of a clip in either direction by choosing
`start_frame_idx` and `reverse`. Run both directions to build bidirectional
results around an edit frame:

```python
edit_frame = 120

predictor.add_new_points_or_box(
    state,
    frame_idx=edit_frame,
    obj_id=1,
    points=np.array([[640.0, 360.0]], dtype=np.float32),
    labels=np.array([1], dtype=np.int32),
)

for frame_idx, obj_ids, masks in predictor.propagate_in_video(
    state, start_frame_idx=edit_frame, reverse=True
):
    pass

for frame_idx, obj_ids, masks in predictor.propagate_in_video(
    state, start_frame_idx=edit_frame, reverse=False
):
    pass
```

Use positive and negative clicks together by setting labels to `1` and `0`.
Pass `clear_old_points=False` to accumulate correction clicks on a frame:

```python
predictor.add_new_points_or_box(
    state,
    frame_idx=120,
    obj_id=1,
    points=np.array([[640.0, 360.0], [710.0, 370.0]], dtype=np.float32),
    labels=np.array([1, 0], dtype=np.int32),
    clear_old_points=False,
)
```

Temporal downsampling is available today as an explicit preview experiment in
`scripts/benchmark_video_frame_skip_mlx.py`: it runs SAM2 on every `k`-th frame,
snaps arbitrary prompt frames to the nearest sampled frame, propagates forward
and backward over the sampled frames, and interpolates skipped masks. Normal
`propagate_in_video(...)` still evaluates every frame.

See [scripts/README.md](scripts/README.md) for benchmark commands, temporal
downsampling experiments, quantization, conversion, parity checks, and upload
helpers.

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

## Convert Weights

Convert from Hugging Face:

```bash
uv run --extra torch-parity mlx-sam-convert \
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
uv run --extra torch-parity mlx-sam-convert \
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


Current low-level parity results:

- Image `vision_features` max abs error: about `1.63e-05`
- Prompted low-res masks max abs error: about `4.67e-05`
- Prompted IoU max abs error: about `4.77e-07`



## Model Catalog

Benchmarks below were run on an Apple M2 Max with 32 GB unified memory. The
source media is `third_party/sam2/demo/data/gallery/01_dog.mp4`, a
`1280x720`, 289-frame clip at 29.97 FPS (`9.64 s`). The fp32 speed and parity
rows use the prompted first-frame fixture at `1024x1024` internal resolution;
speedup is MLX full-image-plus-prompt latency versus the original Torch/MPS
model of the same SAM2.1 family.

| FP32 model | Size | Torch/MPS | MLX | Speedup | Parity vs Torch main |
| --- | ---: | ---: | ---: | ---: | --- |
| `avbiswas/sam2.1-hiera-tiny-mlx` | `172.6 MiB` | `96.6 ms` | `71.3 ms` | `1.36x` | mask mean abs `1.17e-05`, IoU max abs `1.43e-06` |
| `avbiswas/sam2.1-hiera-small-mlx` | `199.7 MiB` | `112.5 ms` | `84.5 ms` | `1.33x` | mask mean abs `8.14e-06`, IoU max abs `4.77e-07` |
| `avbiswas/sam2.1-hiera-base-plus-mlx` | `336.4 MiB` | `203.5 ms` | `144.7 ms` | `1.41x` | mask mean abs `5.04e-06`, IoU max abs `3.49e-06` |
| `avbiswas/sam2.1-hiera-large-mlx` | `892.2 MiB` | `433.0 ms` | `341.1 ms` | `1.27x` | mask mean abs `7.84e-06`, IoU max abs `2.50e-06` |

Quantized checkpoints reduce memory footprint and distribution size. On current
MLX kernels they should not be assumed to speed up video tracking; in our tests
quantization primarily helps memory, not latency.

| Quantized model | Size | Variant | Parity vs fp32 MLX |
| --- | ---: | --- | --- |
| `avbiswas/sam2.1-hiera-tiny-mlx-16bit` | `86.3 MiB` | fp16 | mask mean abs `5.43e-03`, IoU max abs `9.36e-04` |
| `avbiswas/sam2.1-hiera-tiny-mlx-8bit` | `69.0 MiB` | int8 | mask mean abs `6.19e-02`, IoU max abs `2.80e-03` |
| `avbiswas/sam2.1-hiera-tiny-mlx-4bit` | `49.2 MiB` | mixed-q4 | mask mean abs `6.29e-02`, IoU max abs `2.58e-03` |
| `avbiswas/sam2.1-hiera-small-mlx-16bit` | `99.9 MiB` | fp16 | mask mean abs `8.24e-03`, IoU max abs `1.10e-03` |
| `avbiswas/sam2.1-hiera-small-mlx-8bit` | `76.7 MiB` | int8 | mask mean abs `2.99e-02`, IoU max abs `1.90e-03` |
| `avbiswas/sam2.1-hiera-small-mlx-4bit` | `56.4 MiB` | mixed-q4 | mask mean abs `2.87e-02`, IoU max abs `8.80e-04` |
| `avbiswas/sam2.1-hiera-base-plus-mlx-16bit` | `168.2 MiB` | fp16 | mask mean abs `1.58e-03`, IoU max abs `8.83e-04` |
| `avbiswas/sam2.1-hiera-base-plus-mlx-8bit` | `124.6 MiB` | int8 | mask mean abs `2.24e-02`, IoU max abs `8.98e-03` |
| `avbiswas/sam2.1-hiera-base-plus-mlx-4bit` | `95.8 MiB` | mixed-q4 | mask mean abs `2.70e-02`, IoU max abs `6.11e-03` |
| `avbiswas/sam2.1-hiera-large-mlx-16bit` | `446.2 MiB` | fp16 | mask mean abs `2.11e-03`, IoU max abs `8.34e-05` |
| `avbiswas/sam2.1-hiera-large-mlx-8bit` | `300.2 MiB` | int8 | mask mean abs `1.57e-02`, IoU max abs `2.71e-03` |
| `avbiswas/sam2.1-hiera-large-mlx-4bit` | `249.7 MiB` | mixed-q4 | mask mean abs `1.56e-02`, IoU max abs `2.61e-03` |


All models can be found here: https://huggingface.co/collections/avbiswas/sam2-mlx
