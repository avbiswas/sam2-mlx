# mlx-sam

MLX-native SAM models for Apple Silicon. The first supported model family is
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

Quantized checkpoints are intended to reduce memory footprint and distribution
size. On current MLX kernels they should not be assumed to speed up video
tracking; in our tests quantization primarily helps memory, not latency.

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

For editor-style use, precomputing image features is the main parity-preserving
speed lever. On the 80-frame dog benchmark, feature precompute reduced
post-prompt propagation from about `437 ms/frame` to about `269 ms/frame` with
exact mask parity (`1.0` mean IoU against the non-precomputed run):

This mode is opt-in and has no effect on the default predictor behavior. By
default, `init_state(...)` computes frame features on demand. Passing
`precompute_image_features=True` computes and caches all frame image features
upfront, which increases init time and memory use but makes later propagation
and interactive correction passes faster. The masks should be unchanged; the
benchmark above verifies exact parity for the tested clip.

```bash
uv run python scripts/benchmark_video_memory_mlx.py \
  --frames 80 \
  --memory-dtype bfloat16 \
  --memory-attention-dtype bfloat16 \
  --precompute-image-features \
  --feature-batch-size 4 \
  --skip-overlay \
  --report outputs/benchmarks/speed_small_precompute_b4_80.json
```

There are also explicit speed/quality knobs for lower-latency previews:

```bash
uv run python scripts/benchmark_video_memory_mlx.py \
  --frames 80 \
  --memory-dtype bfloat16 \
  --memory-attention-dtype bfloat16 \
  --num-maskmem 2 \
  --max-obj-ptrs 4
```

On the same 80-frame run this reached about `272 ms/frame`, but mean IoU against
the full-memory default dropped to about `0.970`, so it should be treated as a
preview mode rather than the parity default.

### Lower-resolution experiment

The video predictor also supports an experimental lower internal input
resolution:

```bash
uv run python scripts/benchmark_video_memory_mlx.py \
  --frames 80 \
  --image-size 768 \
  --precompute-image-features \
  --feature-batch-size 4 \
  --memory-dtype float16 \
  --memory-attention-dtype float16 \
  --skip-overlay \
  --output-mask outputs/video_res_768/masks_80_fp16mem.npy \
  --report outputs/benchmarks/video_res_768_80_fp16mem.json
```

On the 80-frame dog benchmark, `768x768` internal resolution reduced propagation
from the `1024x1024` precompute baseline of about `269 ms/frame` to about
`53 ms/frame`. Compared against the `1024x1024` masks, mean IoU was `0.949`,
median non-empty IoU was `0.961`, and presence matched on `80 / 80` frames.
This is a preview-quality setting, not the parity default.

### Frame-skip interpolation experiment

To measure temporal subsampling as an explicit speed/quality tradeoff, run SAM2
on every `k`-th frame and interpolate logits for skipped frames:

```bash
uv run python scripts/benchmark_video_frame_skip_mlx.py \
  --frames 30 \
  --skip-step 2 \
  --prompt-frame 14 \
  --interpolation linear \
  --precompute-image-features \
  --feature-batch-size 4 \
  --memory-attention-dtype bfloat16 \
  --baseline-mask outputs/video_frame_skip/full_masks_30.npy \
  --skip-overlay
```

This is an experiment path only; normal propagation still evaluates every frame.
When `--prompt-frame` is not exactly on the sampled grid, the click is assigned
to the nearest sampled frame. Tracking then runs forward and backward from that
sampled prompt frame before interpolation fills the skipped frames. The
experiment keeps sampled masks as MLX low-res logits, interpolates them on
device, then runs one batched MLX upsample to video resolution for saving or
overlay. On a 30-frame dog-video smoke run against full-frame MLX masks:

| Skip step | Computed frames | Model ms/output frame | Total ms/output frame | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| 1, full baseline | 30 / 30 | `373 ms` | n/a | `1.000` |
| 2 | 16 / 30 | `201 ms` | `325 ms` | `0.967` |
| 3 | 11 / 30 | `115 ms` | `229 ms` | `0.945` |
| 4 | 9 / 30 | `88 ms` | `208 ms` | `0.923` |

For `k=3`, low-resolution interpolation itself was about `5 ms` total for all
30 frames; the remaining postprocess cost was mostly video-resolution upsample
and device-to-host transfer.

## Quantization

Generate and evaluate reduced-size variants for all converted checkpoints:

```bash
scripts/quantize_all_models.sh
```

This writes ignored local checkpoints under:

```text
checkpoints/quantized/
```

and reports under:

```text
outputs/benchmarks/quantization_sam2.1_hiera_*.json
```

Current small-checkpoint results against the prompted-mask fixture:

| Variant | Checkpoint | Active weights | Mask mean abs | IoU max abs | Full prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | `199.7 MiB` | `199.7 MiB` | `8.1e-06` | `4.8e-07` | `84.5 ms` |
| fp16 | `99.9 MiB` | `99.9 MiB` | `8.2e-03` | `1.1e-03` | `85.1 ms` |
| int8, all group-64 linears | `76.7 MiB` | `77.6 MiB` | `3.0e-02` | `1.9e-03` | `93.6 ms` |
| int4, all group-64 linears | `55.3 MiB` | `56.6 MiB` | `6.5e-01` | `7.2e-02` | `91.9 ms` |
| mixed-q4: int8 image/prompt, int4 memory | `56.4 MiB` | `57.3 MiB` | `2.9e-02` | `8.8e-04` | `93.8 ms` |

The useful 4-bit path is mixed precision, not pure int4. The current best
recipe is `q8_trunk_mask_q4_memory`: Hiera image trunk and SAM mask decoder
linears use 8-bit, SAM2 memory/object-pointer linears use 4-bit, and the
remaining floating weights are fp16. On a 40-frame dog video smoke run, this
mixed-q4 checkpoint matched fp32 masks with mean IoU `0.997` and `40 / 40`
presence matches.

Upload the local model files to Hugging Face with:

```bash
scripts/upload_hf_models.sh
```

Use `DRY_RUN=1` to print the `hf upload` commands without uploading.

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

from mlx_sam import SAM2VideoPredictor

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
from mlx_sam import SAM2VideoPredictor

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
