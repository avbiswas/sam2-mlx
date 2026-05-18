import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vision.config import SAM2_1_HIERA_SMALL_IMAGE_ENCODER, Sam2ImageEncoderConfig
from mlx_vision.models.image_encoder import Sam2ImageEncoder
from mlx_vision.models.memory import MemoryAttention, MemoryEncoder, upsample_mask
from mlx_vision.models.sam_heads import MaskDecoder, PromptEncoder, SamMLP


class Sam2ImageSegmenter(Sam2ImageEncoder):
    def __init__(self, config: Sam2ImageEncoderConfig = SAM2_1_HIERA_SMALL_IMAGE_ENCODER):
        super().__init__(config=config)
        self.sam_prompt_encoder = PromptEncoder()
        self.sam_mask_decoder = MaskDecoder()
        self.memory_encoder = MemoryEncoder()
        self.memory_attention = MemoryAttention()
        self.obj_ptr_proj = SamMLP(256, 256, 256, 3)
        self.obj_ptr_tpos_proj = nn.Linear(256, 64)
        self.no_mem_embed = mx.zeros((1, 1, 256))
        self.no_mem_pos_enc = mx.zeros((1, 1, 256))
        self.maskmem_tpos_enc = mx.zeros((7, 1, 1, 64))
        self.no_obj_ptr = mx.zeros((1, 256))
        self.no_obj_embed_spatial = mx.zeros((1, 64))
        self.num_maskmem = 7
        self.max_obj_ptrs_in_encoder = 16
        self.memory_temporal_stride_for_eval = 1
        self.max_cond_frames_in_attn = -1
        self.memory_attention_dtype = None

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
            "obj_ptr": self.project_object_pointer(self._select_obj_ptr_token(tokens, ious, multimask_output), object_score_logits),
        }

    def _select_obj_ptr_token(self, tokens: mx.array, ious: mx.array, multimask_output: bool) -> mx.array:
        if not multimask_output or tokens.shape[1] == 1:
            return tokens[:, 0, :]
        best = mx.argmax(ious, axis=1)
        one_hot = mx.equal(mx.arange(tokens.shape[1])[None, :], best[:, None]).astype(tokens.dtype)
        return mx.sum(tokens * one_hot[:, :, None], axis=1)

    def project_object_pointer(self, token: mx.array, object_score_logits: mx.array) -> mx.array:
        obj_ptr = self.obj_ptr_proj(token)
        is_obj = (object_score_logits > 0).astype(obj_ptr.dtype)
        obj_ptr = is_obj * obj_ptr
        return obj_ptr + (1 - is_obj) * self.no_obj_ptr

    def encode_memory(self, vision_features: mx.array, low_res_mask: mx.array, object_score_logits: mx.array, is_mask_from_points: bool = False) -> dict:
        high_res = upsample_mask(low_res_mask.astype(mx.float32), (1024, 1024))
        if is_mask_from_points:
            mask_for_mem = (high_res > 0).astype(mx.float32)
        else:
            mask_for_mem = mx.sigmoid(high_res)
        mask_for_mem = mask_for_mem * 20.0 - 10.0
        mem = self.memory_encoder(vision_features, mask_for_mem, skip_mask_sigmoid=True)
        is_obj = (object_score_logits > 0).astype(mem["vision_features"].dtype)
        mem["vision_features"] = mem["vision_features"] + (1 - is_obj[..., None, None]) * self.no_obj_embed_spatial[..., None, None]
        return mem

    def condition_with_memory(self, encoded: dict, memory_features: mx.array, memory_pos: mx.array, obj_ptr: mx.array | None = None) -> mx.array:
        return self.condition_with_memories(
            encoded,
            cond_memories=[{"maskmem_features": memory_features, "maskmem_pos_enc": memory_pos, "obj_ptr": obj_ptr}],
        )

    def _obj_ptr_pos(self, distances: list[int], batch: int, max_distance: int = 15) -> mx.array:
        pos = mx.array(distances, dtype=mx.float32) / float(max(max_distance, 1))
        pe_dim = 128
        dim_t = mx.arange(pe_dim, dtype=mx.float32)
        dim_t = mx.power(10000.0, 2 * mx.floor(dim_t / 2) / pe_dim)
        enc = pos[:, None] / dim_t[None, :]
        enc = mx.concatenate([mx.sin(enc), mx.cos(enc)], axis=-1)
        enc = self.obj_ptr_tpos_proj(enc)
        return mx.broadcast_to(enc[:, None, :], (len(distances), batch, 64))

    def _select_closest_cond_frames(self, frame_idx: int, cond_by_frame: dict[int, dict]) -> tuple[dict[int, dict], dict[int, dict]]:
        max_cond = self.max_cond_frames_in_attn
        if max_cond == -1 or len(cond_by_frame) <= max_cond:
            return cond_by_frame, {}

        selected: dict[int, dict] = {}
        idx_before = max((t for t in cond_by_frame if t < frame_idx), default=None)
        if idx_before is not None:
            selected[idx_before] = cond_by_frame[idx_before]
        idx_after = min((t for t in cond_by_frame if t >= frame_idx), default=None)
        if idx_after is not None:
            selected[idx_after] = cond_by_frame[idx_after]

        remaining = max_cond - len(selected)
        extra = sorted((t for t in cond_by_frame if t not in selected), key=lambda t: abs(t - frame_idx))[:remaining]
        selected.update((t, cond_by_frame[t]) for t in extra)
        unselected = {t: out for t, out in cond_by_frame.items() if t not in selected}
        return selected, unselected

    def condition_with_memories(
        self,
        encoded: dict,
        memories: list[dict] | dict[int, dict] | None = None,
        cond_memories: list[dict] | dict[int, dict] | None = None,
        current_frame_idx: int | None = None,
        track_in_reverse: bool = False,
    ) -> mx.array:
        feat = encoded["vision_features"]
        pos = encoded["vision_pos_enc"][-1]
        seq = feat.reshape(feat.shape[0], feat.shape[1], -1).transpose(2, 0, 1)
        seq_pos = pos.reshape(pos.shape[0], pos.shape[1], -1).transpose(2, 0, 1)
        mem_parts: list[mx.array] = []
        pos_parts: list[mx.array] = []

        def as_frame_dict(items: list[dict] | dict[int, dict] | None) -> dict[int, dict]:
            if items is None:
                return {}
            if isinstance(items, dict):
                return {int(frame): memory for frame, memory in items.items()}
            return {int(memory["frame_idx"]): memory for memory in items if memory.get("frame_idx") is not None}

        def as_memory_list(items: list[dict] | dict[int, dict] | None) -> list[dict]:
            if items is None:
                return []
            if isinstance(items, dict):
                return list(items.values())
            return items

        def memory_frame(memory: dict) -> int | None:
            frame = memory.get("frame_idx")
            return int(frame) if frame is not None else None

        def append_memory(memory: dict, t_pos: int):
            memory_features = memory["maskmem_features"]
            memory_pos = memory["maskmem_pos_enc"]
            mem = memory_features.reshape(memory_features.shape[0], memory_features.shape[1], -1).transpose(2, 0, 1)
            mem_pos = memory_pos.reshape(memory_pos.shape[0], memory_pos.shape[1], -1).transpose(2, 0, 1)
            mem_pos = mem_pos + self.maskmem_tpos_enc[7 - t_pos - 1]
            mem_parts.append(mem)
            pos_parts.append(mem_pos)

        cond_by_frame = as_frame_dict(cond_memories)
        memory_by_frame = as_frame_dict(memories)

        if current_frame_idx is None or not cond_by_frame:
            for memory in as_memory_list(cond_memories):
                append_memory(memory, t_pos=0)
            # Fallback for direct callers without frame indices. Match SAM2's temporal
            # ordering: oldest remembered frame gets t_pos=1, latest gets t_pos=6.
            recent = as_memory_list(memories)[-(self.num_maskmem - 1) :]
            for t_pos, memory in enumerate(recent, start=self.num_maskmem - len(recent)):
                append_memory(memory, t_pos=t_pos)
            selected_cond: dict[int, dict] = {}
            unselected_cond: dict[int, dict] = {}
        else:
            selected_cond, unselected_cond = self._select_closest_cond_frames(int(current_frame_idx), cond_by_frame)
            for memory in selected_cond.values():
                append_memory(memory, t_pos=0)

            stride = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    prev_frame_idx = int(current_frame_idx) + t_rel if track_in_reverse else int(current_frame_idx) - t_rel
                elif track_in_reverse:
                    prev_frame_idx = -(-(int(current_frame_idx) + 2) // stride) * stride
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                else:
                    prev_frame_idx = ((int(current_frame_idx) - 2) // stride) * stride
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                memory = memory_by_frame.get(prev_frame_idx, unselected_cond.get(prev_frame_idx))
                if memory is not None:
                    append_memory(memory, t_pos=t_pos)

        if not mem_parts:
            return feat + self.no_mem_embed.transpose(0, 2, 1).reshape(1, 256, 1, 1)

        mem = mx.concatenate(mem_parts, axis=0)
        mem_pos = mx.concatenate(pos_parts, axis=0)
        num_obj = 0
        ptr_parts: list[tuple[int, mx.array]] = []
        if current_frame_idx is None:
            ptr_memories = as_memory_list(cond_memories) + as_memory_list(memories)[-(self.max_obj_ptrs_in_encoder - 1) :]
            for idx, memory in enumerate(ptr_memories):
                if memory.get("obj_ptr") is not None:
                    ptr_parts.append((idx, memory["obj_ptr"]))
            ptr_parts = ptr_parts[-self.max_obj_ptrs_in_encoder :]
        else:
            current = int(current_frame_idx)
            selected_for_ptr = selected_cond if "selected_cond" in locals() else {}
            for frame_idx, memory in selected_for_ptr.items():
                in_past = frame_idx >= current if track_in_reverse else frame_idx <= current
                if in_past and memory.get("obj_ptr") is not None:
                    signed = (current - frame_idx) * (-1 if track_in_reverse else 1)
                    ptr_parts.append((signed, memory["obj_ptr"]))
            for t_diff in range(1, self.max_obj_ptrs_in_encoder):
                frame_idx = current + t_diff if track_in_reverse else current - t_diff
                memory = memory_by_frame.get(frame_idx, unselected_cond.get(frame_idx) if "unselected_cond" in locals() else None)
                if memory is not None and memory.get("obj_ptr") is not None:
                    ptr_parts.append((t_diff, memory["obj_ptr"]))

        if ptr_parts:
            max_ptrs = self.max_obj_ptrs_in_encoder
            ptr_parts = ptr_parts[:max_ptrs]
            distances = [distance for distance, _ in ptr_parts]
            ptr_src = mx.stack([obj_ptr for _, obj_ptr in ptr_parts], axis=0)
            ptr = ptr_src.reshape(-1, ptr_src.shape[1], 4, 64).transpose(0, 2, 1, 3).reshape(-1, ptr_src.shape[1], 64)
            ptr_pos = self._obj_ptr_pos(distances, ptr_src.shape[1], max_distance=max_ptrs - 1)
            ptr_pos = mx.repeat(ptr_pos, 4, axis=0)
            mem = mx.concatenate([mem, ptr], axis=0)
            mem_pos = mx.concatenate([mem_pos, ptr_pos], axis=0)
            num_obj = ptr.shape[0]
        original_dtype = seq.dtype
        if self.memory_attention_dtype == "bfloat16":
            seq = seq.astype(mx.bfloat16)
            seq_pos = seq_pos.astype(mx.bfloat16)
            mem = mem.astype(mx.bfloat16)
            mem_pos = mem_pos.astype(mx.bfloat16)
        elif self.memory_attention_dtype == "float16":
            seq = seq.astype(mx.float16)
            seq_pos = seq_pos.astype(mx.float16)
            mem = mem.astype(mx.float16)
            mem_pos = mem_pos.astype(mx.float16)
        out = self.memory_attention(seq, seq_pos, mem, mem_pos, num_obj_ptr_tokens=num_obj)
        return out.astype(original_dtype).transpose(1, 2, 0).reshape(feat.shape)

    def __call__(self, pixels: mx.array, point_coords: mx.array | None, point_labels: mx.array | None, mask_input: mx.array | None = None, multimask_output: bool = True) -> dict:
        encoded = self.encode_image(pixels)
        return self.predict_from_encoded(encoded, point_coords, point_labels, mask_input, multimask_output)


def select_best_mask(low_res_masks: np.ndarray, ious: np.ndarray) -> np.ndarray:
    idx = np.argmax(ious, axis=1)
    return low_res_masks[np.arange(low_res_masks.shape[0]), idx]
