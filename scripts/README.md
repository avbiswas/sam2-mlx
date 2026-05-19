# Scripts

This directory contains development, benchmark, conversion, parity, and upload
helpers for `mlx-sam`. These scripts are intentionally separate from the public
runtime API so the main README can stay focused on using the package.

## User Utilities

Run one prompted frame and write an overlay:

```bash
uv run python scripts/predict_image_mask.py \
  --point 500 610 \
  --output-video outputs/image_prompt_overlay.mp4 \
  --output-mask outputs/image_prompt_mask.npy
```

Track a video with SAM2 memory:

```bash
uv run python scripts/track_video_memory.py --frames 289 \
  --point 500 610 \
  --output-video outputs/dog_memory_overlay_full_v3.mp4 \
  --output-mask outputs/dog_memory_masks_full_v3.npy \
  --report outputs/benchmarks/dog_memory_latency_full_v3.json
```

Render saved masks onto a video:

```bash
uv run python scripts/overlay_masks.py \
  --masks outputs/dog_memory_masks_full_v3.npy \
  --output outputs/dog_memory_overlay_from_masks.mp4
```

`overlay_masks.py` accepts `.npy` or `.npz` masks shaped `T,H,W` or `T,1,H,W`.
Synthetic overlays are only for writer smoke tests:

```bash
uv run python scripts/overlay_masks.py --synthetic-smoke-test
```

## Benchmarks

Current indicative results on this machine with `facebook/sam2.1-hiera-small`:

| Workload | Torch/MPS | MLX | Result |
| --- | ---: | ---: | --- |
| Image encoder | `104 ms/frame` | `81 ms/frame` | MLX `1.28x` faster |
| Cached prompt decode | n/a | `4 ms` | interactive mask decode |
| Full image + prompt | n/a | `85 ms` | image embedding plus prompt |
| Full dog video, post-prompt propagation | `331 ms/frame` | `189 ms/frame` | MLX `1.75x` faster with feature precompute |
| Full dog video, total run | `100.5 s` | `94.8 s` | MLX faster end to end |
| Raw propagation, no save/overlay/final resize | `407 ms/frame` | `287 ms/frame` | MLX `1.42x` faster |

Fast MLX video path with batched image-feature precompute and bfloat16 memory
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

## Spatial Downsampling

`SAM2VideoPredictor(image_size=768)` changes the internal square SAM input
resolution. Frames are resized before the image encoder and final masks are
upsampled back to the original video size.

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
`53 ms/frame`. Compared with the `1024x1024` masks, mean IoU was `0.949`,
median non-empty IoU was `0.961`, and presence matched on `80 / 80` frames.
This is a preview-quality setting, not the parity default.

## Temporal Downsampling

Temporal downsampling is an explicit experiment path. Normal
`propagate_in_video(...)` evaluates every frame.

`benchmark_video_frame_skip_mlx.py` runs SAM2 on every `k`-th frame and
interpolates logits for skipped frames:

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

When `--prompt-frame` is not exactly on the sampled grid, the click is assigned
to the nearest sampled frame. Tracking then runs forward and backward from that
sampled prompt frame before interpolation fills the skipped frames. The
experiment keeps sampled masks as MLX low-res logits, interpolates them on
device, then runs one batched MLX upsample to video resolution for saving or
overlay.

On a 30-frame dog-video smoke run against full-frame MLX masks:

| Skip step | Computed frames | Model ms/output frame | Total ms/output frame | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| 1, full baseline | 30 / 30 | `373 ms` | n/a | `1.000` |
| 2 | 16 / 30 | `201 ms` | `325 ms` | `0.967` |
| 3 | 11 / 30 | `115 ms` | `229 ms` | `0.945` |
| 4 | 9 / 30 | `88 ms` | `208 ms` | `0.923` |

For `k=3`, low-resolution interpolation itself was about `5 ms` total for all
30 frames; the remaining postprocess cost was mostly video-resolution upsample
and device-to-host transfer.

## Feature Precompute

Precomputing image features is the main parity-preserving speed lever for
editor-style use. By default, `init_state(...)` computes frame features on
demand. Passing `precompute_image_features=True` computes and caches all frame
image features upfront, which increases init time and memory use but makes
later propagation and interactive correction passes faster.

On the 80-frame dog benchmark, feature precompute reduced post-prompt
propagation from about `437 ms/frame` to about `269 ms/frame` with exact mask
parity (`1.0` mean IoU against the non-precomputed run):

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

## Lossy Preview Memory Settings

There are explicit speed/quality knobs for lower-latency previews:

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

