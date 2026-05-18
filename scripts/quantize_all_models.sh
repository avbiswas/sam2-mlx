#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export UV_CACHE_DIR

run_quantization() {
  local model_id="$1"
  local checkpoint="$2"
  local reference="$3"
  local runs="${4:-2}"
  local warmup="${5:-1}"

  uv run python scripts/quantize_sam2_model.py \
    --source "${checkpoint}" \
    --model-id "${model_id}" \
    --reference "${reference}" \
    --variants fp16 int8 mixed-q4 \
    --runs "${runs}" \
    --warmup "${warmup}"
}

run_quantization \
  "facebook/sam2.1-hiera-tiny" \
  "checkpoints/sam2.1_hiera_tiny_image_segmenter.safetensors" \
  "data/torch_prompt_mask_tiny.npz"

run_quantization \
  "facebook/sam2.1-hiera-small" \
  "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors" \
  "data/torch_prompt_mask.npz"

run_quantization \
  "facebook/sam2.1-hiera-base-plus" \
  "checkpoints/sam2.1_hiera_base_plus_image_segmenter.safetensors" \
  "data/torch_prompt_mask_base_plus.npz"

run_quantization \
  "facebook/sam2.1-hiera-large" \
  "checkpoints/sam2.1_hiera_large_image_segmenter.safetensors" \
  "data/torch_prompt_mask_large.npz" \
  1 \
  1

