import mlx.core as mx
import numpy as np

from sam_mlx.models.image_encoder import Sam2ImageEncoder
from sam_mlx.models.memory import MemoryAttention, MemoryEncoder, upsample_mask_np
from sam_mlx.models.sam_heads import MaskDecoder, PromptEncoder, SamMLP


class Sam2ImageSegmenter(Sam2ImageEncoder):
    def __init__(self):
        super().__init__()
        self.sam_prompt_encoder = PromptEncoder()
        self.sam_mask_decoder = MaskDecoder()
        self.memory_encoder = MemoryEncoder()
        self.memory_attention = MemoryAttention()
        self.obj_ptr_proj = SamMLP(256, 256, 256, 3)
        self.no_mem_embed = mx.zeros((1, 1, 256))
        self.no_mem_pos_enc = mx.zeros((1, 1, 256))
        self.maskmem_tpos_enc = mx.zeros((7, 1, 1, 64))
        self.no_obj_ptr = mx.zeros((1, 256))
        self.no_obj_embed_spatial = mx.zeros((1, 64))

    def encode_image(self, pixels: mx.array) -> dict:
        out = super().__call__(pixels)
        projected = list(out["backbone_fpn"])
        high_res = self.sam_mask_decoder.project_high_res(projected)
        out["high_res_features"] = high_res
        return out

    def predict_from_encoded(
        self,
        encoded: dict,
        point_coords: mx.array | None,
        point_labels: mx.array | None,
        mask_input: mx.array | None = None,
        multimask_output: bool = True,
        add_no_mem_embed: bool = True,
    ) -> dict:
        if point_coords is None or point_labels is None:
            batch = encoded["vision_features"].shape[0]
            point_coords = mx.zeros((batch, 1, 2), dtype=mx.float32)
            point_labels = -mx.ones((batch, 1), dtype=mx.int32)
        sparse, dense = self.sam_prompt_encoder(point_coords, point_labels, masks=None)
        if mask_input is not None:
            sparse, dense = self.sam_prompt_encoder(point_coords, point_labels, masks=mask_input)
        image_embeddings = encoded["vision_features"]
        if add_no_mem_embed:
            image_embeddings = image_embeddings + self.no_mem_embed.transpose(0, 2, 1).reshape(1, 256, 1, 1)
        masks, ious, tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            high_res_features=encoded["high_res_features"],
        )
        return {
            "low_res_masks": masks.astype(mx.float32),
            "ious": ious,
            "sam_tokens": tokens,
            "object_score_logits": object_score_logits,
            "obj_ptr": self.project_object_pointer(tokens[:, 0, :], object_score_logits),
        }

    def project_object_pointer(self, token: mx.array, object_score_logits: mx.array) -> mx.array:
        obj_ptr = self.obj_ptr_proj(token)
        is_obj = (object_score_logits > 0).astype(obj_ptr.dtype)
        obj_ptr = is_obj * obj_ptr
        return obj_ptr + (1 - is_obj) * self.no_obj_ptr

    def encode_memory(self, vision_features: mx.array, low_res_mask: mx.array, object_score_logits: mx.array) -> dict:
        high_res = upsample_mask_np(low_res_mask.astype(mx.float32), (1024, 1024))
        mask_for_mem = mx.sigmoid(high_res) * 20.0 - 10.0
        mem = self.memory_encoder(vision_features, mask_for_mem, skip_mask_sigmoid=True)
        is_obj = (object_score_logits > 0).astype(mem["vision_features"].dtype)
        mem["vision_features"] = mem["vision_features"] + (1 - is_obj[..., None, None]) * self.no_obj_embed_spatial[..., None, None]
        return mem

    def condition_with_memory(self, encoded: dict, memory_features: mx.array, memory_pos: mx.array, obj_ptr: mx.array | None = None) -> mx.array:
        return self.condition_with_memories(
            encoded,
            cond_memories=[{"maskmem_features": memory_features, "maskmem_pos_enc": memory_pos, "obj_ptr": obj_ptr}],
        )

    def condition_with_memories(self, encoded: dict, memories: list[dict] | None = None, cond_memories: list[dict] | None = None) -> mx.array:
        feat = encoded["vision_features"]
        pos = encoded["vision_pos_enc"][-1]
        seq = feat.reshape(feat.shape[0], feat.shape[1], -1).transpose(2, 0, 1)
        seq_pos = pos.reshape(pos.shape[0], pos.shape[1], -1).transpose(2, 0, 1)
        mem_parts = []
        pos_parts = []
        ptr_parts = []

        def append_memory(memory: dict, t_pos: int):
            memory_features = memory["maskmem_features"]
            memory_pos = memory["maskmem_pos_enc"]
            mem = memory_features.reshape(memory_features.shape[0], memory_features.shape[1], -1).transpose(2, 0, 1)
            mem_pos = memory_pos.reshape(memory_pos.shape[0], memory_pos.shape[1], -1).transpose(2, 0, 1)
            mem_pos = mem_pos + self.maskmem_tpos_enc[7 - t_pos - 1]
            mem_parts.append(mem)
            pos_parts.append(mem_pos)
            if memory.get("obj_ptr") is not None:
                ptr_parts.append(memory["obj_ptr"])

        for memory in cond_memories or []:
            append_memory(memory, t_pos=0)

        for t_pos, memory in enumerate(reversed((memories or [])[-6:]), start=1):
            append_memory(memory, t_pos=t_pos)

        if not mem_parts:
            return feat + self.no_mem_embed.transpose(0, 2, 1).reshape(1, 256, 1, 1)

        mem = mx.concatenate(mem_parts, axis=0)
        mem_pos = mx.concatenate(pos_parts, axis=0)
        num_obj = 0
        if ptr_parts:
            ptr_src = mx.stack(ptr_parts[-16:], axis=0)
            ptr = ptr_src.reshape(-1, ptr_src.shape[1], 4, 64).transpose(0, 2, 1, 3).reshape(-1, ptr_src.shape[1], 64)
            ptr_pos = mx.zeros_like(ptr)
            mem = mx.concatenate([mem, ptr], axis=0)
            mem_pos = mx.concatenate([mem_pos, ptr_pos], axis=0)
            num_obj = ptr.shape[0]
        out = self.memory_attention(seq, seq_pos, mem, mem_pos, num_obj_ptr_tokens=num_obj)
        return out.transpose(1, 2, 0).reshape(feat.shape)

    def __call__(self, pixels: mx.array, point_coords: mx.array | None, point_labels: mx.array | None, mask_input: mx.array | None = None, multimask_output: bool = True) -> dict:
        encoded = self.encode_image(pixels)
        return self.predict_from_encoded(encoded, point_coords, point_labels, mask_input, multimask_output)


def select_best_mask(low_res_masks: np.ndarray, ious: np.ndarray) -> np.ndarray:
    idx = np.argmax(ious, axis=1)
    return low_res_masks[np.arange(low_res_masks.shape[0]), idx]
