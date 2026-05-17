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
