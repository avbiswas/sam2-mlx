import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = [
    "multi_object",
    "box_prompt",
    "negative_clicks",
    "cross_frame_corrections",
    "bidirectional_middle",
]
THRESHOLDS = {
    "multi_object": 0.95,
    "box_prompt": 0.94,
    "negative_clicks": 0.95,
    "cross_frame_corrections": 0.95,
    "bidirectional_middle": 0.92,
}


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--frames", type=int, default=130)
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "outputs/torch_feature_benchmark_frames_130f")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/feature_benchmarks_130f")
    parser.add_argument("--scenario", choices=["all", *DEFAULT_SCENARIOS], default="all")
    parser.add_argument("--refresh-torch", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    scenarios = DEFAULT_SCENARIOS if args.scenario == "all" else [args.scenario]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output_dir / "mlx_feature_regression_summary.json"

    if args.refresh_torch:
        run(
            [
                "uv",
                "run",
                "--extra",
                "torch-parity",
                "python",
                "scripts/run_torch_video_feature_benchmarks.py",
                "--scenario",
                args.scenario,
                "--video",
                str(args.video),
                "--frames",
                str(args.frames),
                "--frames-dir",
                str(args.frames_dir),
                "--output-dir",
                str(args.output_dir),
            ],
            ROOT,
        )

    results = []
    failures = []
    for scenario in scenarios:
        if not args.skip_mlx:
            run(
                [
                    sys.executable,
                    "scripts/track_video_features_mlx.py",
                    "--scenario",
                    scenario,
                    "--video",
                    str(args.video),
                    "--frames",
                    str(args.frames),
                    "--output-dir",
                    str(args.output_dir),
                ],
                ROOT,
            )

        report_path = args.output_dir / f"{scenario}_mlx_vs_torch.json"
        run(
            [
                sys.executable,
                "scripts/compare_video_masks.py",
                "--reference",
                str(args.output_dir / f"{scenario}_torch_masks.npy"),
                "--candidate",
                str(args.output_dir / f"{scenario}_mlx_masks.npy"),
                "--output",
                str(report_path),
            ],
            ROOT,
        )
        report = json.loads(report_path.read_text())
        threshold = THRESHOLDS[scenario]
        passed = report["mean_iou_all"] >= threshold
        result = {
            "scenario": scenario,
            "mean_iou_all": report["mean_iou_all"],
            "presence": f"{report['presence_match_frames']}/{report['presence_total_frames']}",
            "threshold": threshold,
            "passed": passed,
            "report": str(report_path),
        }
        results.append(result)
        if not passed:
            failures.append(result)

    summary = {"frames": args.frames, "output_dir": str(args.output_dir), "results": results, "passed": not failures}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
