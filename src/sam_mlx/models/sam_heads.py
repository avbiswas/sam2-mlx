import math

import mlx.core as mx
import mlx.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class SamMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False, activation: str = "relu"):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        self.sigmoid_output = sigmoid_output
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.gelu(x) if self.activation == "gelu" else nn.relu(x)
        return mx.sigmoid(x) if self.sigmoid_output else x


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.positional_encoding_gaussian_matrix = mx.zeros((2, num_pos_feats))

    def _pe_encoding(self, coords: mx.array) -> mx.array:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)

    def __call__(self, size: tuple[int, int]) -> mx.array:
        h, w = size
        y = (mx.arange(h, dtype=mx.float32) + 0.5) / h
        x = (mx.arange(w, dtype=mx.float32) + 0.5) / w
        yy = mx.broadcast_to(y[:, None], (h, w))
        xx = mx.broadcast_to(x[None, :], (h, w))
        pe = self._pe_encoding(mx.stack([xx, yy], axis=-1))
        return pe.transpose(2, 0, 1)

    def forward_with_coords(self, coords_input: mx.array, image_size: tuple[int, int]) -> mx.array:
        coords = coords_input.astype(mx.float32)
        scale = mx.array([image_size[1], image_size[0]], dtype=mx.float32)
        coords = coords / scale
        return self._pe_encoding(coords)


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, image_embedding_size: tuple[int, int] = (64, 64), input_image_size: tuple[int, int] = (1024, 1024), mask_in_chans: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(4)]
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling_0 = nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2)
        self.mask_downscaling_1 = LayerNorm2d(mask_in_chans // 4)
        self.mask_downscaling_3 = nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2)
        self.mask_downscaling_4 = LayerNorm2d(mask_in_chans)
        self.mask_downscaling_6 = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)

    def get_dense_pe(self) -> mx.array:
        return self.pe_layer(self.image_embedding_size)[None, ...]

    def _embed_points(self, points: mx.array, labels: mx.array, pad: bool) -> mx.array:
        points = points + 0.5
        if pad:
            padding_point = mx.zeros((points.shape[0], 1, 2), dtype=points.dtype)
            padding_label = -mx.ones((labels.shape[0], 1), dtype=labels.dtype)
            points = mx.concatenate([points, padding_point], axis=1)
            labels = mx.concatenate([labels, padding_label], axis=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding = mx.where(labels[..., None] == -1, mx.broadcast_to(self.not_a_point_embed.weight, point_embedding.shape), point_embedding)
        point_embedding = mx.where(labels[..., None] == 0, point_embedding + self.point_embeddings[0].weight, point_embedding)
        point_embedding = mx.where(labels[..., None] == 1, point_embedding + self.point_embeddings[1].weight, point_embedding)
        point_embedding = mx.where(labels[..., None] == 2, point_embedding + self.point_embeddings[2].weight, point_embedding)
        point_embedding = mx.where(labels[..., None] == 3, point_embedding + self.point_embeddings[3].weight, point_embedding)
        return point_embedding

    def _embed_masks(self, masks: mx.array) -> mx.array:
        x = masks.transpose(0, 2, 3, 1)
        x = self.mask_downscaling_0(x).transpose(0, 3, 1, 2)
        x = nn.gelu(self.mask_downscaling_1(x))
        x = self.mask_downscaling_3(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = nn.gelu(self.mask_downscaling_4(x))
        return self.mask_downscaling_6(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    def __call__(self, point_coords: mx.array | None, point_labels: mx.array | None, masks: mx.array | None = None) -> tuple[mx.array, mx.array]:
        bs = point_coords.shape[0] if point_coords is not None else (masks.shape[0] if masks is not None else 1)
        sparse = mx.zeros((bs, 0, self.embed_dim))
        if point_coords is not None and point_labels is not None:
            sparse = mx.concatenate([sparse, self._embed_points(point_coords, point_labels, pad=True)], axis=1)
        if masks is not None:
            dense = self._embed_masks(masks)
        else:
            dense = mx.broadcast_to(self.no_mask_embed.weight.reshape(1, -1, 1, 1), (bs, self.embed_dim, *self.image_embedding_size))
        return sparse, dense


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, kv_in_dim: int | None = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim or embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: mx.array) -> mx.array:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(0, 2, 1, 3)

    def _recombine_heads(self, x: mx.array) -> mx.array:
        b, h, n, c = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, n, h * c)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]))
        return self.out_proj(self._recombine_heads(out))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Attention(256, 8)
        self.norm1 = nn.LayerNorm(256)
        self.cross_attn_token_to_image = Attention(256, 8, downsample_rate=2)
        self.norm2 = nn.LayerNorm(256)
        self.mlp = SamMLP(256, 2048, 256, 2)
        self.norm3 = nn.LayerNorm(256)
        self.cross_attn_image_to_token = Attention(256, 8, downsample_rate=2)
        self.norm4 = nn.LayerNorm(256)
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(self, queries: mx.array, keys: mx.array, query_pe: mx.array, key_pe: mx.array) -> tuple[mx.array, mx.array]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.norm3(queries + self.mlp(queries))
        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [TwoWayAttentionBlock(skip_first_layer_pe=True), TwoWayAttentionBlock(skip_first_layer_pe=False)]
        self.final_attn_token_to_image = Attention(256, 8, downsample_rate=2)
        self.norm_final_attn = nn.LayerNorm(256)

    def __call__(self, image_embedding: mx.array, image_pe: mx.array, point_embedding: mx.array) -> tuple[mx.array, mx.array]:
        b, c, h, w = image_embedding.shape
        image_embedding = image_embedding.reshape(b, c, h * w).transpose(0, 2, 1)
        image_pe = image_pe.reshape(b, c, h * w).transpose(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(queries + point_embedding, keys + image_pe, keys))
        return queries, keys


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TwoWayTransformer()
        self.iou_token = nn.Embedding(1, 256)
        self.mask_tokens = nn.Embedding(4, 256)
        self.obj_score_token = nn.Embedding(1, 256)
        self.output_upscaling_0 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.output_upscaling_1 = LayerNorm2d(64)
        self.output_upscaling_3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_s0 = nn.Conv2d(256, 32, kernel_size=1)
        self.conv_s1 = nn.Conv2d(256, 64, kernel_size=1)
        self.output_hypernetworks_mlps = [SamMLP(256, 256, 32, 3) for _ in range(4)]
        self.iou_prediction_head = SamMLP(256, 256, 4, 3, sigmoid_output=True)
        self.pred_obj_score_head = SamMLP(256, 256, 1, 3)
        self.dynamic_multimask_via_stability = True
        self.dynamic_multimask_stability_delta = 0.05
        self.dynamic_multimask_stability_thresh = 0.98

    def project_high_res(self, fpn: list[mx.array]) -> list[mx.array]:
        feat_s0 = self.conv_s0(fpn[0].transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        feat_s1 = self.conv_s1(fpn[1].transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return [feat_s0, feat_s1]

    def predict_masks(self, image_embeddings: mx.array, image_pe: mx.array, sparse_prompt_embeddings: mx.array, dense_prompt_embeddings: mx.array, high_res_features: list[mx.array]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        output_tokens = mx.concatenate([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], axis=0)
        output_tokens = mx.broadcast_to(output_tokens[None, :, :], (sparse_prompt_embeddings.shape[0], output_tokens.shape[0], output_tokens.shape[1]))
        tokens = mx.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = mx.broadcast_to(image_pe, src.shape)
        b, c, h, w = src.shape
        hs, src_tokens = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2:6, :]
        src = src_tokens.transpose(0, 2, 1).reshape(b, c, h, w)

        feat_s0, feat_s1 = high_res_features
        x = self.output_upscaling_0(src.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        x = nn.gelu(self.output_upscaling_1(x + feat_s1))
        x = self.output_upscaling_3(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        upscaled_embedding = nn.gelu(x + feat_s0)

        hyper = mx.stack([mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)], axis=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper @ upscaled_embedding.reshape(b, c, h * w)).reshape(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _stability_scores(self, masks: mx.array) -> mx.array:
        flat = masks.reshape(*masks.shape[:2], -1)
        area_i = mx.sum(flat > self.dynamic_multimask_stability_delta, axis=-1).astype(mx.float32)
        area_u = mx.sum(flat > -self.dynamic_multimask_stability_delta, axis=-1).astype(mx.float32)
        return mx.where(area_u > 0, area_i / area_u, mx.ones_like(area_u))

    def _dynamic_multimask(self, masks: mx.array, iou_pred: mx.array) -> tuple[mx.array, mx.array]:
        multimask_logits = masks[:, 1:, :, :]
        multimask_iou = iou_pred[:, 1:]
        best = mx.argmax(multimask_iou, axis=-1)
        one_hot = mx.equal(mx.arange(multimask_logits.shape[1])[None, :], best[:, None]).astype(masks.dtype)
        best_logits = mx.sum(multimask_logits * one_hot[:, :, None, None], axis=1, keepdims=True)
        best_iou = mx.sum(multimask_iou * one_hot, axis=1, keepdims=True)

        single_logits = masks[:, 0:1, :, :]
        single_iou = iou_pred[:, 0:1]
        stable = self._stability_scores(single_logits) >= self.dynamic_multimask_stability_thresh
        return (
            mx.where(stable[:, :, None, None], single_logits, best_logits),
            mx.where(stable, single_iou, best_iou),
        )

    def __call__(self, image_embeddings: mx.array, image_pe: mx.array, sparse_prompt_embeddings: mx.array, dense_prompt_embeddings: mx.array, multimask_output: bool, high_res_features: list[mx.array]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, high_res_features)
        is_obj = object_score_logits > 0
        masks = mx.where(is_obj[:, :, None, None], masks, mx.full(masks.shape, -1024.0))
        if multimask_output:
            masks_out = masks[:, 1:, :, :]
            iou_out = iou_pred[:, 1:]
            sam_tokens = mask_tokens_out[:, 1:]
        elif self.dynamic_multimask_via_stability:
            masks_out, iou_out = self._dynamic_multimask(masks, iou_pred)
            sam_tokens = mask_tokens_out[:, 0:1]
        else:
            masks_out = masks[:, 0:1, :, :]
            iou_out = iou_pred[:, 0:1]
            sam_tokens = mask_tokens_out[:, 0:1]
        return masks_out, iou_out, sam_tokens, object_score_logits
