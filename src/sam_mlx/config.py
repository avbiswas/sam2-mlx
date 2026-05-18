from dataclasses import dataclass


@dataclass(frozen=True)
class HieraConfig:
    image_size: int = 1024
    embed_dim: int = 96
    num_heads: int = 1
    stages: tuple[int, ...] = (1, 2, 11, 2)
    global_att_blocks: tuple[int, ...] = (7, 10, 13)
    window_spec: tuple[int, ...] = (8, 4, 14, 7)
    q_pool: int = 3
    q_stride: tuple[int, int] = (2, 2)
    mlp_ratio: float = 4.0
    dim_mul: float = 2.0
    head_mul: float = 2.0
    pos_embed_hw: tuple[int, int] = (256, 256)


@dataclass(frozen=True)
class FpnConfig:
    d_model: int = 256
    backbone_channel_list: tuple[int, ...] = (768, 384, 192, 96)
    fpn_top_down_levels: tuple[int, ...] = (2, 3)
    scalp: int = 1


@dataclass(frozen=True)
class Sam2ImageEncoderConfig:
    hiera: HieraConfig = HieraConfig()
    fpn: FpnConfig = FpnConfig()


SAM2_1_HIERA_SMALL_IMAGE_ENCODER = Sam2ImageEncoderConfig()


SAM2_1_HIERA_TINY_IMAGE_ENCODER = Sam2ImageEncoderConfig(
    hiera=HieraConfig(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_spec=(8, 4, 14, 7),
    ),
    fpn=FpnConfig(backbone_channel_list=(768, 384, 192, 96)),
)

SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER = Sam2ImageEncoderConfig(
    hiera=HieraConfig(
        embed_dim=112,
        num_heads=2,
        stages=(2, 3, 16, 3),
        global_att_blocks=(12, 16, 20),
        window_spec=(8, 4, 14, 7),
    ),
    fpn=FpnConfig(backbone_channel_list=(896, 448, 224, 112)),
)

SAM2_1_HIERA_LARGE_IMAGE_ENCODER = Sam2ImageEncoderConfig(
    hiera=HieraConfig(
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_spec=(8, 4, 16, 8),
    ),
    fpn=FpnConfig(backbone_channel_list=(1152, 576, 288, 144)),
)


SAM2_1_MODEL_CONFIGS = {
    "facebook/sam2.1-hiera-tiny": SAM2_1_HIERA_TINY_IMAGE_ENCODER,
    "facebook/sam2.1-hiera-small": SAM2_1_HIERA_SMALL_IMAGE_ENCODER,
    "facebook/sam2.1-hiera-base-plus": SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER,
    "facebook/sam2.1-hiera-large": SAM2_1_HIERA_LARGE_IMAGE_ENCODER,
    "sam2.1_hiera_tiny": SAM2_1_HIERA_TINY_IMAGE_ENCODER,
    "sam2.1_hiera_small": SAM2_1_HIERA_SMALL_IMAGE_ENCODER,
    "sam2.1_hiera_base_plus": SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER,
    "sam2.1_hiera_large": SAM2_1_HIERA_LARGE_IMAGE_ENCODER,
}


def model_config_for_name(name: str) -> Sam2ImageEncoderConfig:
    normalized = name.replace(".pt", "").replace("_image_segmenter.safetensors", "")
    if normalized in SAM2_1_MODEL_CONFIGS:
        return SAM2_1_MODEL_CONFIGS[normalized]
    lowered = normalized.lower()
    if "hiera_tiny" in lowered or "hiera-tiny" in lowered:
        return SAM2_1_HIERA_TINY_IMAGE_ENCODER
    if "hiera_small" in lowered or "hiera-small" in lowered:
        return SAM2_1_HIERA_SMALL_IMAGE_ENCODER
    if "hiera_base_plus" in lowered or "hiera-base-plus" in lowered or "hiera_b+" in lowered:
        return SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER
    if "hiera_large" in lowered or "hiera-large" in lowered:
        return SAM2_1_HIERA_LARGE_IMAGE_ENCODER
    raise ValueError(f"Could not infer SAM2.1 model config from name: {name}")
