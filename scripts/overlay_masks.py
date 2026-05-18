import argparse
import json
from pathlib import Path

from mlx_vision.overlay import load_masks, synthetic_masks_for_video, write_mask_overlay_video

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/overlay_demo.mp4")
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.masks:
        masks = load_masks(args.masks)
    elif args.synthetic_smoke_test:
        masks = synthetic_masks_for_video(args.video, limit=args.limit)
    else:
        raise SystemExit("Pass --masks for real overlays, or --synthetic-smoke-test for a writer smoke test.")
    report = write_mask_overlay_video(
        args.video,
        masks,
        args.output,
        alpha=args.alpha,
        threshold=args.threshold,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
