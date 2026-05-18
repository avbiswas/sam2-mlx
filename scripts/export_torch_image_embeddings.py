import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO = ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_REPO))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mlx_vision.preprocess import preprocess_video


def build_model(checkpoint: Path):
    config_dir = SAM2_REPO / "sam2" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="sam2.1/sam2.1_hiera_s.yaml")
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "third_party/sam2/demo/data/gallery/01_dog.mp4")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/torch_image_embeddings.npz")
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()

    pixels = preprocess_video(args.video, limit=args.frames)
    model = build_model(args.checkpoint)

    arrays = {"pixel_values": pixels}
    with torch.inference_mode():
        output = model.image_encoder(torch.from_numpy(pixels))
    arrays["vision_features"] = output["vision_features"].detach().cpu().numpy()
    for i, x in enumerate(output["backbone_fpn"]):
        arrays[f"backbone_fpn_{i}"] = x.detach().cpu().numpy()
    for i, x in enumerate(output["vision_pos_enc"]):
        arrays[f"vision_pos_enc_{i}"] = x.detach().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    meta = {
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "frames": int(pixels.shape[0]),
        "keys": {k: list(v.shape) for k, v in arrays.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
