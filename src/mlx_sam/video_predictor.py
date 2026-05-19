from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_sam.models import Sam2ImageSegmenter
from mlx_sam.preprocess import preprocess_image
from mlx_sam.weights import load_image_segmenter


NO_OBJ_SCORE = -1024.0


def _as_points(points) -> np.ndarray:
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3:
        points = points[0]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape N,2 or 1,N,2; got {points.shape}")
    return points


def _as_labels(labels) -> np.ndarray:
    if labels is None:
        return np.zeros((0,), dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim == 2:
        labels = labels[0]
    if labels.ndim != 1:
        raise ValueError(f"Expected labels with shape N or 1,N; got {labels.shape}")
    return labels


def _concat_points(old: dict | None, points: np.ndarray, labels: np.ndarray) -> dict:
    if old is None:
        return {"point_coords": points.astype(np.float32), "point_labels": labels.astype(np.int32)}
    return {
        "point_coords": np.concatenate([old["point_coords"], points.astype(np.float32)], axis=0),
        "point_labels": np.concatenate([old["point_labels"], labels.astype(np.int32)], axis=0),
    }


def _load_video_frames(video_path: str | Path, image_size: int = 1024) -> tuple[np.ndarray, int, int]:
    path = Path(video_path)
    frames: list[Image.Image] = []
    if path.is_dir():
        files = sorted([*path.glob("*.jpg"), *path.glob("*.jpeg"), *path.glob("*.png")])
        if not files:
            raise ValueError(f"No image frames found in directory: {path}")
        for file in files:
            frames.append(Image.open(file).convert("RGB"))
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video path: {path}")
    width, height = frames[0].size
    pixels = np.concatenate([preprocess_image(frame, image_size=image_size) for frame in frames], axis=0)
    return pixels, height, width


def _slice_encoded(encoded: dict, index: int) -> dict:
    return {
        "vision_features": encoded["vision_features"][index : index + 1],
        "vision_pos_enc": [pos[index : index + 1] for pos in encoded["vision_pos_enc"]],
        "high_res_features": [feat[index : index + 1] for feat in encoded["high_res_features"]],
    }


def _original_to_sam(points: np.ndarray, width: int, height: int, image_size: int = 1024) -> np.ndarray:
    out = points.astype(np.float32).copy()
    out[:, 0] *= image_size / width
    out[:, 1] *= image_size / height
    return out


def _best_low_mask(out: dict) -> mx.array:
    low = out["low_res_masks"]
    ious = out["ious"]
    if ious.shape[1] <= 1:
        return low[:, :1]
    best = mx.argmax(ious, axis=1)
    gather_idx = best.reshape(-1, 1, 1, 1)
    gather_idx = mx.broadcast_to(gather_idx, (low.shape[0], 1, low.shape[2], low.shape[3]))
    return mx.take_along_axis(low, gather_idx, axis=1)


def _video_res_logits(low: np.ndarray, width: int, height: int) -> np.ndarray:
    low_np = np.array(low)
    return cv2.resize(low_np[0, 0].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)[None, None]


class SAM2VideoPredictor:
    """MLX video predictor with the same public method names as Meta's SAM2VideoPredictor.

    This class intentionally uses NumPy arrays for prompt inputs and returned mask logits
    so the runtime package remains PyTorch-free.
    """

    image_size = 1024

    def __init__(
        self,
        model: Sam2ImageSegmenter | None = None,
        checkpoint: str | Path | None = None,
        model_id: str | None = None,
        image_size: int | None = None,
        memory_dtype: str | None = None,
        memory_attention_dtype: str | None = None,
        num_maskmem: int | None = None,
        max_obj_ptrs_in_encoder: int | None = None,
        non_overlap_masks: bool = False,
        clear_non_cond_mem_around_input: bool = False,
        add_all_frames_to_correct_as_cond: bool = False,
        **_: object,
    ):
        if model is None:
            if checkpoint is None:
                checkpoint = Path("checkpoints/sam2.1_hiera_small_image_segmenter.safetensors")
            model = load_image_segmenter(checkpoint, model_id=model_id)
        self.model = model
        if image_size is not None:
            self.image_size = int(image_size)
        if hasattr(self.model, "set_image_size"):
            self.model.set_image_size(self.image_size)
        self.memory_dtype = memory_dtype
        self.model.memory_attention_dtype = memory_attention_dtype
        if num_maskmem is not None:
            self.model.num_maskmem = int(num_maskmem)
        if max_obj_ptrs_in_encoder is not None:
            self.model.max_obj_ptrs_in_encoder = int(max_obj_ptrs_in_encoder)
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        path = Path(model_id)
        if path.exists():
            return cls(checkpoint=path, model_id=kwargs.pop("model_id", None), **kwargs)
        from huggingface_hub import hf_hub_download, list_repo_files

        filename = kwargs.pop("filename", None)
        repo_files = list_repo_files(model_id)
        if filename is None:
            safetensors = [file for file in repo_files if file.endswith(".safetensors")]
            if len(safetensors) != 1:
                raise ValueError(f"Expected exactly one safetensors file in {model_id}; found {safetensors}")
            filename = safetensors[0]
        checkpoint = hf_hub_download(repo_id=model_id, filename=filename)
        sidecar = f"{filename}.json"
        if sidecar in repo_files:
            hf_hub_download(repo_id=model_id, filename=sidecar)
        return cls(checkpoint=checkpoint, model_id=model_id, **kwargs)

    def init_state(
        self,
        video_path,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        precompute_image_features: bool = False,
        feature_batch_size: int = 4,
    ) -> dict:
        pixels, video_height, video_width = _load_video_frames(video_path, image_size=self.image_size)
        state = {
            "images": pixels,
            "num_frames": int(pixels.shape[0]),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "async_loading_frames": async_loading_frames,
            "keep_image_features": precompute_image_features,
            "video_height": video_height,
            "video_width": video_width,
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        if precompute_image_features:
            self._precompute_image_features(state, feature_batch_size=feature_batch_size)
        else:
            self._get_image_feature(state, 0)
        return state

    def _precompute_image_features(self, inference_state: dict, feature_batch_size: int = 4) -> None:
        batch_size = max(1, int(feature_batch_size))
        images = inference_state["images"]
        cached = inference_state["cached_features"]
        for start in range(0, inference_state["num_frames"], batch_size):
            encoded = self.model.encode_image(mx.array(images[start : start + batch_size]))
            eval_targets = [
                encoded["vision_features"],
                *encoded["vision_pos_enc"],
                *encoded["high_res_features"],
            ]
            mx.eval(*eval_targets)
            batch = encoded["vision_features"].shape[0]
            for offset in range(batch):
                cached[start + offset] = _slice_encoded(encoded, offset)

    def _obj_id_to_idx(self, inference_state: dict, obj_id: int) -> int:
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is not None:
            return obj_idx
        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        inference_state["temp_output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        inference_state["frames_tracked_per_obj"][obj_idx] = {}
        return obj_idx

    def _get_image_feature(self, inference_state: dict, frame_idx: int) -> dict:
        cached = inference_state["cached_features"].get(frame_idx)
        if cached is not None:
            return cached
        encoded = self.model.encode_image(mx.array(inference_state["images"][frame_idx : frame_idx + 1]))
        if inference_state.get("keep_image_features"):
            eval_targets = [
                encoded["vision_features"],
                *encoded["vision_pos_enc"],
                *encoded["high_res_features"],
            ]
            mx.eval(*eval_targets)
            inference_state["cached_features"][frame_idx] = encoded
        else:
            inference_state["cached_features"] = {frame_idx: encoded}
        return encoded

    def _encode_memory(self, encoded: dict, low: mx.array, out: dict, frame_idx: int, is_mask_from_points: bool) -> dict:
        mem = self.model.encode_memory(encoded["vision_features"], low, out["object_score_logits"], is_mask_from_points=is_mask_from_points)
        vision_features = mem["vision_features"]
        vision_pos_enc = mem["vision_pos_enc"][0]
        if self.memory_dtype == "bfloat16":
            vision_features = vision_features.astype(mx.bfloat16)
            vision_pos_enc = vision_pos_enc.astype(mx.bfloat16)
        elif self.memory_dtype == "float16":
            vision_features = vision_features.astype(mx.float16)
            vision_pos_enc = vision_pos_enc.astype(mx.float16)
        return {
            "maskmem_features": vision_features,
            "maskmem_pos_enc": vision_pos_enc,
            "obj_ptr": out["obj_ptr"],
            "frame_idx": frame_idx,
        }

    def _ensure_memory_encoded(
        self,
        inference_state: dict,
        obj_idx: int,
        frame_idx: int,
        out: dict,
        is_mask_from_points: bool,
        memory_frame_idx: int | None = None,
    ):
        if out.get("maskmem_features") is not None:
            return
        encoded = self._get_image_feature(inference_state, frame_idx)
        memory = self._encode_memory(
            encoded,
            out["pred_masks"],
            out,
            frame_idx if memory_frame_idx is None else memory_frame_idx,
            is_mask_from_points=is_mask_from_points,
        )
        out.update(memory)
        mx.eval(out["maskmem_features"], out["maskmem_pos_enc"])

    def _propagate_preflight(self, inference_state: dict):
        if not inference_state["obj_ids"]:
            raise RuntimeError("No input points or masks are provided for any object")
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            for frame_idx, out in list(obj_output_dict["cond_frame_outputs"].items()):
                self._ensure_memory_encoded(inference_state, obj_idx, frame_idx, out, is_mask_from_points=True)
            for frame_idx, out in list(obj_output_dict["non_cond_frame_outputs"].items()):
                if frame_idx in inference_state["point_inputs_per_obj"][obj_idx] or frame_idx in inference_state["mask_inputs_per_obj"][obj_idx]:
                    self._ensure_memory_encoded(inference_state, obj_idx, frame_idx, out, is_mask_from_points=True)

    def _predict_initial(self, encoded: dict, points: np.ndarray, labels: np.ndarray) -> dict:
        out = self.model.predict_from_encoded(
            encoded,
            mx.array(points[None].astype(np.float32)),
            mx.array(labels[None].astype(np.int32)),
            multimask_output=True,
        )
        return out

    def _predict_tracked(
        self,
        encoded: dict,
        obj_output_dict: dict,
        frame_idx: int,
        point_inputs: dict | None = None,
        prev_low: np.ndarray | None = None,
        reverse: bool = False,
        memory_frame_idx: int | None = None,
    ) -> dict:
        cond = list(obj_output_dict["cond_frame_outputs"].values())
        mem = list(obj_output_dict["non_cond_frame_outputs"].values())
        current_frame_idx = frame_idx if memory_frame_idx is None else memory_frame_idx
        conditioned = self.model.condition_with_memories(encoded, mem, cond_memories=cond, current_frame_idx=current_frame_idx, track_in_reverse=reverse)
        conditioned_encoded = dict(encoded)
        conditioned_encoded["vision_features"] = conditioned
        if point_inputs is None:
            coords = labels = None
        else:
            coords = mx.array(point_inputs["point_coords"][None].astype(np.float32))
            labels = mx.array(point_inputs["point_labels"][None].astype(np.int32))
        out = self.model.predict_from_encoded(
            conditioned_encoded,
            coords,
            labels,
            mask_input=prev_low if prev_low is not None else None,
            multimask_output=False,
            add_no_mem_embed=False,
        )
        return out

    def _run_single_frame_inference(
        self,
        inference_state: dict,
        obj_idx: int,
        frame_idx: int,
        is_init_cond_frame: bool,
        point_inputs: dict | None,
        reverse: bool,
        prev_low: np.ndarray | None = None,
        run_mem_encoder: bool = True,
        memory_frame_idx: int | None = None,
    ) -> dict:
        encoded = self._get_image_feature(inference_state, frame_idx)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        if is_init_cond_frame and point_inputs is not None and prev_low is None:
            out = self._predict_initial(encoded, point_inputs["point_coords"], point_inputs["point_labels"])
        else:
            out = self._predict_tracked(
                encoded,
                obj_output_dict,
                frame_idx,
                point_inputs=point_inputs,
                prev_low=prev_low,
                reverse=reverse,
                memory_frame_idx=memory_frame_idx,
            )
        low = _best_low_mask(out)
        current_out = {
            "pred_masks": low,
            "obj_ptr": out["obj_ptr"],
            "object_score_logits": out["object_score_logits"],
            "maskmem_features": None,
            "maskmem_pos_enc": None,
        }
        if run_mem_encoder:
            memory = self._encode_memory(encoded, low, out, frame_idx if memory_frame_idx is None else memory_frame_idx, is_mask_from_points=(point_inputs is not None))
            current_out.update(memory)
        eval_targets = [current_out["pred_masks"], current_out["obj_ptr"], current_out["object_score_logits"]]
        if current_out["maskmem_features"] is not None:
            eval_targets.extend([current_out["maskmem_features"], current_out["maskmem_pos_enc"]])
        mx.eval(*eval_targets)
        return current_out

    def _output_masks(self, inference_state: dict, outputs_by_obj: list[dict | None]) -> np.ndarray:
        h = inference_state["video_height"]
        w = inference_state["video_width"]
        masks = np.full((len(outputs_by_obj), 1, h, w), NO_OBJ_SCORE, dtype=np.float32)
        for idx, out in enumerate(outputs_by_obj):
            if out is not None:
                masks[idx] = _video_res_logits(out["pred_masks"], w, h)
        if self.non_overlap_masks and masks.shape[0] > 1:
            winners = np.argmax(masks[:, 0], axis=0)
            for idx in range(masks.shape[0]):
                masks[idx, 0] = np.where(winners == idx, masks[idx, 0], np.minimum(masks[idx, 0], -10.0))
        return masks

    def add_new_points_or_box(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
        points=None,
        labels=None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box=None,
    ):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        points = _as_points(points)
        labels = _as_labels(labels)
        if box is not None:
            if not clear_old_points:
                raise ValueError("cannot add box without clearing old points")
            box_arr = np.asarray(box, dtype=np.float32).reshape(2, 2)
            points = np.concatenate([box_arr, points], axis=0)
            labels = np.concatenate([np.array([2, 3], dtype=np.int32), labels], axis=0)
        if points.shape[0] == 0:
            raise ValueError("at least one of points or box must be provided as input")
        if normalize_coords:
            points = _original_to_sam(points, inference_state["video_width"], inference_state["video_height"], self.image_size)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        old = None if clear_old_points else point_inputs_per_frame.get(frame_idx)
        point_inputs = _concat_points(old, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in tracked
        reverse = False if is_init_cond_frame else tracked[frame_idx]["reverse"]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx) or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        prev_low = mx.clip(prev_out["pred_masks"], -32.0, 32.0) if prev_out is not None else None
        # Match Meta's predictor: prompt edits skip memory encoding here. The
        # memory is encoded during propagation preflight after edits are finalized.
        out = self._run_single_frame_inference(inference_state, obj_idx, frame_idx, is_init_cond_frame, point_inputs, reverse, prev_low=prev_low, run_mem_encoder=False)
        obj_output_dict[storage_key][frame_idx] = out
        if storage_key == "cond_frame_outputs":
            obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        return frame_idx, inference_state["obj_ids"], self._output_masks(inference_state, [self._output_for_obj(inference_state, idx, frame_idx) for idx in range(len(inference_state["obj_ids"]))])

    def add_new_points(self, *args, **kwargs):
        return self.add_new_points_or_box(*args, **kwargs)

    def add_new_mask(self, inference_state: dict, frame_idx: int, obj_id: int, mask):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError(f"Expected mask with shape H,W; got {mask.shape}")
        low = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)[None, None]
        low = np.where(low >= 0.5, 20.0, -20.0).astype(np.float32)
        encoded = self._get_image_feature(inference_state, frame_idx)
        low_mx = mx.array(low)
        out = {"low_res_masks": low_mx, "obj_ptr": self.model.no_obj_ptr, "object_score_logits": mx.ones((1, 1))}
        current_out = {
            "pred_masks": low_mx,
            "object_score_logits": out["object_score_logits"],
            "obj_ptr": out["obj_ptr"],
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "frame_idx": frame_idx,
        }
        tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_cond = frame_idx not in tracked or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        inference_state["output_dict_per_obj"][obj_idx][storage_key][frame_idx] = current_out
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = low
        return frame_idx, inference_state["obj_ids"], self._output_masks(inference_state, [self._output_for_obj(inference_state, idx, frame_idx) for idx in range(len(inference_state["obj_ids"]))])

    def _output_for_obj(self, inference_state: dict, obj_idx: int, frame_idx: int) -> dict | None:
        out_dict = inference_state["output_dict_per_obj"][obj_idx]
        return out_dict["cond_frame_outputs"].get(frame_idx) or out_dict["non_cond_frame_outputs"].get(frame_idx)

    def propagate_in_video(
        self,
        inference_state: dict,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse: bool = False,
        return_masks: bool = True,
        frame_step: int = 1,
    ):
        frame_step = int(frame_step)
        if frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}")
        self._propagate_preflight(inference_state)
        if start_frame_idx is None:
            start_frame_idx = min(
                frame
                for out_dict in inference_state["output_dict_per_obj"].values()
                for frame in out_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = inference_state["num_frames"]
        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track, 0)
            order = range(start_frame_idx, end - 1, -frame_step) if start_frame_idx > 0 else []
        else:
            end = min(start_frame_idx + max_frame_num_to_track, inference_state["num_frames"] - 1)
            order = range(start_frame_idx, end + 1, frame_step)

        for frame_idx in order:
            outputs: list[dict | None] = []
            for obj_idx in range(len(inference_state["obj_ids"])):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    out = obj_output_dict["cond_frame_outputs"][frame_idx]
                else:
                    out = self._run_single_frame_inference(inference_state, obj_idx, frame_idx, False, None, reverse, run_mem_encoder=True)
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                outputs.append(out)
            masks = self._output_masks(inference_state, outputs) if return_masks else None
            yield frame_idx, inference_state["obj_ids"], masks

    def stream_in_video(
        self,
        inference_state: dict,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse: bool = False,
        yield_every: int | None = 30,
        return_full: bool = False,
        frame_step: int = 1,
    ):
        """Yield throttled video mask events for UI or worker streaming.

        Frame events are dictionaries with:
        - type: "frame"
        - is_final: False
        - frame_idx: source video frame index
        - step: zero-based processed-frame counter in this propagation call
        - obj_ids: current object ids
        - masks: NumPy float32 array shaped O,1,H,W

        If return_full is true, the last event has type "final", is_final true,
        frame_indices for every processed frame, and masks stacked as T,O,1,H,W.
        """
        if yield_every is not None and int(yield_every) < 1:
            raise ValueError(f"yield_every must be >= 1 or None, got {yield_every}")
        frame_step = int(frame_step)
        if frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}")
        throttle = None if yield_every is None else int(yield_every)
        frame_indices: list[int] = []
        all_masks: list[np.ndarray] = []
        last_event: dict | None = None

        iterator = self.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            return_masks=True,
            frame_step=frame_step,
        )
        for step, (frame_idx, obj_ids, masks) in enumerate(iterator):
            if return_full:
                frame_indices.append(int(frame_idx))
                all_masks.append(masks)
            event = {
                "type": "frame",
                "is_final": False,
                "frame_idx": int(frame_idx),
                "step": step,
                "obj_ids": list(obj_ids),
                "masks": masks,
            }
            last_event = event
            if throttle is None or step % throttle == 0:
                yield event

        if return_full:
            full_masks = np.stack(all_masks, axis=0) if all_masks else None
            yield {
                "type": "final",
                "is_final": True,
                "frame_idx": None,
                "frame_indices": np.asarray(frame_indices, dtype=np.int32),
                "obj_ids": [] if last_event is None else last_event["obj_ids"],
                "masks": full_masks,
            }

    def clear_all_prompts_in_frame(self, inference_state: dict, frame_idx: int, obj_id: int, need_output: bool = True):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        out_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = out_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            out_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)
        if not need_output:
            return None
        return frame_idx, inference_state["obj_ids"], self._output_masks(inference_state, [self._output_for_obj(inference_state, idx, frame_idx) for idx in range(len(inference_state["obj_ids"]))])

    def reset_state(self, inference_state: dict) -> None:
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

    def _reset_tracking_results(self, inference_state: dict) -> None:
        for value in inference_state["point_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["mask_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        for value in inference_state["temp_output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        for value in inference_state["frames_tracked_per_obj"].values():
            value.clear()
