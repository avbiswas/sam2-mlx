import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_sam.config import model_config_for_name
from mlx_sam.models import Sam2ImageSegmenter
from mlx_sam.weights import apply_quantization, load_image_segmenter, sam2_quant_predicate

ROOT = Path(__file__).resolve().parents[1]

MODEL_SLUGS = {
    "facebook/sam2.1-hiera-tiny": "sam2.1_hiera_tiny",
    "facebook/sam2.1-hiera-small": "sam2.1_hiera_small",
    "facebook/sam2.1-hiera-base-plus": "sam2.1_hiera_base_plus",
    "facebook/sam2.1-hiera-large": "sam2.1_hiera_large",
}


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def file_mib(path: Path) -> float:
    return bytes_to_mib(path.stat().st_size)


def diff_row(name: str, ref: np.ndarray, got: np.ndarray) -> dict:
    diff = got.astype(np.float64) - ref.astype(np.float64)
    return {
        "name": name,
        "shape": list(ref.shape),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "ref_mean": float(np.mean(ref)),
        "got_mean": float(np.mean(got)),
    }


def benchmark_prompt(model: Sam2ImageSegmenter, ref, warmup: int, runs: int) -> dict:
    pixels = mx.array(ref["pixel_values"])
    coords = mx.array(ref["point_coords"])
    labels = mx.array(ref["point_labels"])

    for _ in range(warmup):
        out = model(pixels, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])

    mx.reset_peak_memory()
    full = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model(pixels, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])
        full.append((time.perf_counter() - start) * 1000.0)
    full_peak = mx.get_peak_memory()

    encoded = model.encode_image(pixels)
    mx.eval(encoded["vision_features"], encoded["high_res_features"])

    mx.reset_peak_memory()
    prompt = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model.predict_from_encoded(encoded, coords, labels, multimask_output=True)
        mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])
        prompt.append((time.perf_counter() - start) * 1000.0)
    prompt_peak = mx.get_peak_memory()

    return {
        "full_image_plus_prompt_ms": {
            "runs": runs,
            "mean": float(np.mean(full)),
            "median": float(np.median(full)),
            "min": float(np.min(full)),
            "max": float(np.max(full)),
        },
        "cached_prompt_decode_ms": {
            "runs": runs,
            "mean": float(np.mean(prompt)),
            "median": float(np.median(prompt)),
            "min": float(np.min(prompt)),
            "max": float(np.max(prompt)),
        },
        "peak_memory_mib": {
            "full_image_plus_prompt": bytes_to_mib(full_peak),
            "cached_prompt_decode": bytes_to_mib(prompt_peak),
        },
    }


def prompt_accuracy(model: Sam2ImageSegmenter, ref) -> dict:
    out = model(
        mx.array(ref["pixel_values"]),
        mx.array(ref["point_coords"]),
        mx.array(ref["point_labels"]),
        multimask_output=True,
    )
    mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])
    return {
        "rows": [
            diff_row("low_res_multimasks", ref["low_res_multimasks"], np.array(out["low_res_masks"])),
            diff_row("ious", ref["ious"], np.array(out["ious"])),
            diff_row("object_score_logits", ref["object_score_logits"], np.array(out["object_score_logits"])),
        ]
    }


def cast_floating_params(params, dtype):
    if isinstance(params, dict):
        return {k: cast_floating_params(v, dtype) for k, v in params.items()}
    if isinstance(params, list):
        return [cast_floating_params(v, dtype) for v in params]
    if isinstance(params, tuple):
        return tuple(cast_floating_params(v, dtype) for v in params)
    if isinstance(params, mx.array) and mx.issubdtype(params.dtype, mx.floating):
        return params.astype(dtype)
    return params


def count_quantized_linears(model: Sam2ImageSegmenter) -> int:
    return sum(1 for _, module in model.named_modules() if isinstance(module, nn.QuantizedLinear))


def save_model(model: Sam2ImageSegmenter, output: Path, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output))
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2))


def build_variant(source: Path, output: Path, model_id: str, variant: str, group_size: int, recipe: str = "all") -> dict:
    start = time.perf_counter()
    model = load_image_segmenter(source, model_id=model_id)
    metadata = {
        "format": "mlx",
        "model_id": model_id,
        "source": str(source),
        "variant": variant,
    }

    if variant in {"fp16", "mixed-q4"}:
        model.update(cast_floating_params(model.parameters(), mx.float16))

    if variant == "fp16":
        pass
    elif variant in {"int8", "int4", "mixed-q4"}:
        bits = 8 if variant == "int8" else 4
        quantization = {"bits": bits, "group_size": group_size, "mode": "affine", "recipe": recipe}
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode="affine",
            class_predicate=sam2_quant_predicate(group_size, recipe=recipe, bits=bits),
        )
        metadata["quantization"] = quantization
        metadata["quantized_linear_count"] = count_quantized_linears(model)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    mx.eval(model.parameters())
    save_model(model, output, metadata)
    return {
        "variant": variant,
        "recipe": recipe if variant in {"int8", "int4", "mixed-q4"} else None,
        "path": str(output),
        "sidecar": str(output.with_suffix(output.suffix + ".json")),
        "size_mib": file_mib(output),
        "build_wall_time_s": time.perf_counter() - start,
        "metadata": metadata,
    }


def load_variant(path: Path, model_id: str) -> Sam2ImageSegmenter:
    config = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    model = Sam2ImageSegmenter(config=model_config_for_name(model_id))
    apply_quantization(model, config.get("quantization"))
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def default_report_path(model_id: str) -> Path:
    slug = MODEL_SLUGS.get(model_id, model_id.split("/")[-1].replace("-", "_").replace(".", "_"))
    return ROOT / f"outputs/benchmarks/quantization_{slug}.json"


def output_prefix(source: Path, model_id: str) -> str:
    stem = source.name.removesuffix(".safetensors")
    if stem.endswith("_image_segmenter"):
        return stem
    slug = MODEL_SLUGS.get(model_id, source.stem)
    return f"{slug}_image_segmenter"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize/evaluate a SAM2.1 MLX checkpoint.")
    parser.add_argument("--source", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/torch_prompt_mask.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/quantized")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--variants", nargs="+", default=["fp16", "int8", "int4", "mixed-q4"], choices=["fp16", "int8", "int4", "mixed-q4"])
    parser.add_argument("--int4-recipes", nargs="+", default=["all", "trunk_memory_obj"])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.report is None:
        args.report = default_report_path(args.model_id)

    ref = np.load(args.reference)
    report = {
        "source": str(args.source),
        "source_size_mib": file_mib(args.source),
        "model_id": args.model_id,
        "reference": str(args.reference),
        "variants": [],
    }

    baseline_model = load_image_segmenter(args.source, model_id=args.model_id)
    report["baseline_fp32"] = {
        "size_mib": file_mib(args.source),
        "accuracy": prompt_accuracy(baseline_model, ref),
        "benchmark": benchmark_prompt(baseline_model, ref, args.warmup, args.runs),
    }

    for variant in args.variants:
        recipes = args.int4_recipes if variant == "int4" else ["q8_trunk_mask_q4_memory"] if variant == "mixed-q4" else ["all"]
        for recipe in recipes:
            if variant == "fp16":
                suffix = "fp16"
            elif variant == "mixed-q4":
                suffix = recipe
            else:
                suffix = f"q{8 if variant == 'int8' else 4}_g{args.group_size}"
                if recipe != "all":
                    suffix = f"{suffix}_{recipe}"
            output = args.output_dir / f"{output_prefix(args.source, args.model_id)}_{suffix}.safetensors"
            variant_report = build_variant(args.source, output, args.model_id, variant, args.group_size, recipe=recipe)
            model = load_variant(output, args.model_id)
            variant_report["accuracy"] = prompt_accuracy(model, ref)
            variant_report["benchmark"] = benchmark_prompt(model, ref, args.warmup, args.runs)
            report["variants"].append(variant_report)
            print(json.dumps(variant_report, indent=2))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
