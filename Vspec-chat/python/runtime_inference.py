from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional dependency
    safe_open = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from vspec_cuda_bridge import (
        attention_single_f32,
        attention_fused_single_f32,
        attention_fused_single_f32_available,
        attention_single_f32_available,
        fused_linear_int3,
        fused_linear_int3_available,
        fused_linear_int4,
        fused_linear_int4_available,
        gemm_f32,
        gemm_f32_available,
        linear_f32,
        linear_f32_available,
        mul_f32,
        mul_f32_available,
        rmsnorm_f32,
        rmsnorm_f32_available,
        silu_f32,
        silu_f32_available,
    )
except Exception:  # pragma: no cover - optional dependency
    rmsnorm_f32 = None
    rmsnorm_f32_available = lambda: False
    linear_f32 = None
    linear_f32_available = lambda: False
    attention_single_f32 = None
    attention_fused_single_f32 = None
    attention_fused_single_f32_available = lambda: False
    attention_single_f32_available = lambda: False
    fused_linear_int3 = None
    fused_linear_int3_available = lambda: False
    fused_linear_int4 = None
    fused_linear_int4_available = lambda: False
    gemm_f32 = None
    gemm_f32_available = lambda: False
    silu_f32 = None
    silu_f32_available = lambda: False
    mul_f32 = None
    mul_f32_available = lambda: False

from model_adapters import ModelAdapter
from model_loader import WeightInfo


def _softmax(x: "np.ndarray") -> "np.ndarray":
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _rms_norm(x: "np.ndarray", weight: "np.ndarray", eps: float, bias: Optional["np.ndarray"]) -> "np.ndarray":
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    out = x / np.sqrt(mean_sq + eps)
    out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _silu(x: "np.ndarray") -> "np.ndarray":
    return x / (1.0 + np.exp(-x))


def _apply_rotary(q: "np.ndarray", k: "np.ndarray", position: int, rope_theta: float) -> tuple["np.ndarray", "np.ndarray"]:
    if q.shape[-1] % 2 != 0:
        return q, k
    dim = q.shape[-1]
    half = dim // 2
    idx = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (rope_theta ** (idx / half))
    angles = position * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = np.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = np.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot


def _softmax_torch(x: "torch.Tensor") -> "torch.Tensor":
    return torch.softmax(x, dim=-1)


def _rms_norm_torch(x: "torch.Tensor", weight: "torch.Tensor", eps: float, bias: Optional["torch.Tensor"]) -> "torch.Tensor":
    mean_sq = torch.mean(x * x, dim=-1, keepdim=True)
    out = x / torch.sqrt(mean_sq + eps)
    out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _silu_torch(x: "torch.Tensor") -> "torch.Tensor":
    return x / (1.0 + torch.exp(-x))


def _apply_rotary_torch(q: "torch.Tensor", k: "torch.Tensor", position: int, rope_theta: float) -> tuple["torch.Tensor", "torch.Tensor"]:
    if q.shape[-1] % 2 != 0:
        return q, k
    dim = q.shape[-1]
    half = dim // 2
    idx = torch.arange(half, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (idx / half))
    angles = position * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(dtype=q.dtype)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).to(dtype=k.dtype)
    return q_rot, k_rot


def _matmul_with_weight_dtype(x: "torch.Tensor", w_t: "torch.Tensor") -> "torch.Tensor":
    if x.dtype != w_t.dtype:
        x = x.to(dtype=w_t.dtype)
    return torch.matmul(x, w_t)


def _pack_signed_rowwise(q: "np.ndarray", bits: int) -> "np.ndarray":
    n, k = q.shape
    row_bytes = (k * bits + 7) // 8
    out = np.zeros((n, row_bytes), dtype=np.uint8)
    mask = (1 << bits) - 1

    for r in range(n):
        row = q[r]
        for i in range(k):
            code = int(row[i]) & mask
            bit_pos = i * bits
            byte_idx = bit_pos >> 3
            shift = bit_pos & 7
            out[r, byte_idx] |= np.uint8((code << shift) & 0xFF)
            if (8 - shift) < bits:
                out[r, byte_idx + 1] |= np.uint8((code >> (8 - shift)) & 0xFF)
    return out.reshape(-1)


def _quantize_weight_rowwise(weight: "np.ndarray", bits: int) -> tuple["np.ndarray", "np.ndarray"]:
    w = weight.astype(np.float32, copy=False)
    max_q = float((1 << (bits - 1)) - 1)
    min_q = float(-(1 << (bits - 1)))

    max_abs = np.max(np.abs(w), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / max_q, 1.0).astype(np.float32)
    q = np.round(w / scales[:, None])
    q = np.clip(q, min_q, max_q).astype(np.int8)

    packed = _pack_signed_rowwise(q, bits)
    return packed, scales


@dataclass
class SimpleRuntime:
    embed: "np.ndarray"
    lm_head: "np.ndarray"
    eos_token_id: Optional[int]

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if np is None:
            return []
        if not token_ids:
            token_ids = [0]
        embed_tokens = self.embed[token_ids]
        pooled = np.mean(embed_tokens, axis=0)
        logits = pooled @ self.lm_head
        return logits.astype(float).tolist()


@dataclass
class LayerWeights:
    wq: "np.ndarray"
    wk: "np.ndarray"
    wv: "np.ndarray"
    wo: "np.ndarray"
    w1: "np.ndarray"
    w2: "np.ndarray"
    w3: "np.ndarray"
    norm1: "np.ndarray"
    norm2: "np.ndarray"
    bq: Optional["np.ndarray"]
    bk: Optional["np.ndarray"]
    bv: Optional["np.ndarray"]
    bo: Optional["np.ndarray"]
    b1: Optional["np.ndarray"]
    b2: Optional["np.ndarray"]
    b3: Optional["np.ndarray"]
    norm1_bias: Optional["np.ndarray"]
    norm2_bias: Optional["np.ndarray"]
    packed: dict[str, tuple["np.ndarray", "np.ndarray", int, int]] = field(default_factory=dict)


@dataclass
class GenericTransformerRuntime:
    embed: "np.ndarray"
    lm_head: "np.ndarray"
    final_norm: "np.ndarray"
    layers: list[LayerWeights]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    eos_token_id: Optional[int]
    rope_theta: float
    position: int
    cache_k: list["np.ndarray"]
    cache_v: list["np.ndarray"]
    use_native_cuda_norm: bool
    fused_bits: int
    disable_fused_attention: bool

    def _forward_token(self, token_id: int, return_logits: bool) -> Optional[list[float]]:
        if np is None:
            return [] if return_logits else None

        x = self.embed[token_id].astype(np.float32)

        for idx, layer in enumerate(self.layers):
            if self.use_native_cuda_norm and rmsnorm_f32_available():
                x_norm = rmsnorm_f32(x[None, :], layer.norm1, self.rms_eps)[0]
            else:
                x_norm = _rms_norm(x, layer.norm1, self.rms_eps, layer.norm1_bias)

            def _linear_native(vec: "np.ndarray", w: "np.ndarray", key: str) -> "np.ndarray":
                if self.use_native_cuda_norm and self.fused_bits in {3, 4} and key in layer.packed:
                    packed_w, scales, bits, out_n = layer.packed[key]
                    if bits == 4 and fused_linear_int4_available():
                        return fused_linear_int4(vec, packed_w, scales, out_n)[0]
                    if bits == 3 and fused_linear_int3_available():
                        return fused_linear_int3(vec, packed_w, scales, out_n)[0]
                if self.use_native_cuda_norm and gemm_f32_available():
                    return gemm_f32(vec, w)[0]
                return vec @ w.T

            q = _linear_native(x_norm, layer.wq, "wq")
            k = _linear_native(x_norm, layer.wk, "wk")
            v = _linear_native(x_norm, layer.wv, "wv")

            if layer.bq is not None:
                q = q + layer.bq
            if layer.bk is not None:
                k = k + layer.bk
            if layer.bv is not None:
                v = v + layer.bv

            q = q.reshape(self.num_heads, self.head_dim)
            k = k.reshape(self.num_kv_heads, self.head_dim)
            v = v.reshape(self.num_kv_heads, self.head_dim)

            q, k = _apply_rotary(q, k, self.position, self.rope_theta)

            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = np.repeat(k, repeat, axis=0)
                v = np.repeat(v, repeat, axis=0)

            if len(self.cache_k) <= idx:
                self.cache_k.append(k[None, ...])
                self.cache_v.append(v[None, ...])
            else:
                self.cache_k[idx] = np.concatenate([self.cache_k[idx], k[None, ...]], axis=0)
                self.cache_v[idx] = np.concatenate([self.cache_v[idx], v[None, ...]], axis=0)

            keys = self.cache_k[idx]
            values = self.cache_v[idx]

            attn_out = []
            for h in range(self.num_heads):
                qh = q[h]
                kh = keys[:, h, :]
                vh = values[:, h, :]
                if self.use_native_cuda_norm and (not self.disable_fused_attention) and attention_fused_single_f32_available():
                    attn_out.append(attention_fused_single_f32(qh, kh, vh))
                elif self.use_native_cuda_norm and attention_single_f32_available():
                    attn_out.append(attention_single_f32(qh, kh, vh))
                else:
                    scores = (kh @ qh) / np.sqrt(self.head_dim)
                    scores = scores.reshape(1, -1)
                    probs = _softmax(scores)[0]
                    attn_out.append(probs @ vh)

            attn = np.concatenate(attn_out, axis=0)
            attn = _linear_native(attn, layer.wo, "wo")
            if layer.bo is not None:
                attn = attn + layer.bo
            x = x + attn

            if self.use_native_cuda_norm and rmsnorm_f32_available():
                x_norm = rmsnorm_f32(x[None, :], layer.norm2, self.rms_eps)[0]
            else:
                x_norm = _rms_norm(x, layer.norm2, self.rms_eps, layer.norm2_bias)
            gate = _linear_native(x_norm, layer.w1, "w1")
            if layer.b1 is not None:
                gate = gate + layer.b1
            if self.use_native_cuda_norm and silu_f32_available():
                gate = silu_f32(gate)
            else:
                gate = _silu(gate)

            up = _linear_native(x_norm, layer.w3, "w3")
            if layer.b3 is not None:
                up = up + layer.b3

            if self.use_native_cuda_norm and mul_f32_available():
                fused = mul_f32(gate, up)
            else:
                fused = gate * up

            ff = _linear_native(fused, layer.w2, "w2")
            if layer.b2 is not None:
                ff = ff + layer.b2
            x = x + ff

        self.position += 1

        if not return_logits:
            return None

        if self.use_native_cuda_norm and rmsnorm_f32_available():
            x_last = rmsnorm_f32(x[None, :], self.final_norm, self.rms_eps)[0]
        else:
            x_last = _rms_norm(x, self.final_norm, self.rms_eps, None)
        if self.use_native_cuda_norm and gemm_f32_available():
            logits = gemm_f32(x_last, self.lm_head)[0]
        else:
            logits = x_last @ self.lm_head
        return logits.astype(float).tolist()

    def prefill_tokens(self, token_ids: list[int]) -> None:
        if np is None or not token_ids:
            return
        for token_id in token_ids:
            self._forward_token(int(token_id), return_logits=False)

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if np is None or not token_ids:
            return []
        return self._forward_token(int(token_ids[-1]), return_logits=True) or []


@dataclass
class GenericTransformerRuntimeTorch:
    embed: "torch.Tensor"
    lm_head: "torch.Tensor"
    final_norm: "torch.Tensor"
    layers: list[LayerWeights]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    eos_token_id: Optional[int]
    rope_theta: float
    position: int
    cache_k: list["torch.Tensor"]
    cache_v: list["torch.Tensor"]

    def _forward_token(self, token_id: int, return_logits: bool) -> Optional[list[float]]:
        if torch is None:
            return [] if return_logits else None

        x = self.embed[token_id]

        for idx, layer in enumerate(self.layers):
            x_norm = _rms_norm_torch(x, layer.norm1, self.rms_eps, layer.norm1_bias)

            q = _matmul_with_weight_dtype(x_norm, layer.wq.t())
            k = _matmul_with_weight_dtype(x_norm, layer.wk.t())
            v = _matmul_with_weight_dtype(x_norm, layer.wv.t())

            if layer.bq is not None:
                q = q + layer.bq
            if layer.bk is not None:
                k = k + layer.bk
            if layer.bv is not None:
                v = v + layer.bv

            q = q.reshape(self.num_heads, self.head_dim)
            k = k.reshape(self.num_kv_heads, self.head_dim)
            v = v.reshape(self.num_kv_heads, self.head_dim)

            q, k = _apply_rotary_torch(q, k, self.position, self.rope_theta)

            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat(repeat, 1)
                v = v.repeat(repeat, 1)

            if len(self.cache_k) <= idx:
                self.cache_k.append(k.unsqueeze(0))
                self.cache_v.append(v.unsqueeze(0))
            else:
                self.cache_k[idx] = torch.cat([self.cache_k[idx], k.unsqueeze(0)], dim=0)
                self.cache_v[idx] = torch.cat([self.cache_v[idx], v.unsqueeze(0)], dim=0)

            keys = self.cache_k[idx]
            values = self.cache_v[idx]

            attn_out = []
            for h in range(self.num_heads):
                qh = q[h]
                kh = keys[:, h, :]
                vh = values[:, h, :]
                scores = torch.matmul(kh, qh) / torch.sqrt(torch.tensor(self.head_dim, device=kh.device, dtype=kh.dtype))
                probs = _softmax_torch(scores.view(1, -1))[0]
                attn_out.append(torch.matmul(probs, vh))

            attn = torch.cat(attn_out, dim=0)
            attn = _matmul_with_weight_dtype(attn, layer.wo.t())
            if layer.bo is not None:
                attn = attn + layer.bo
            x = x + attn

            x_norm = _rms_norm_torch(x, layer.norm2, self.rms_eps, layer.norm2_bias)
            gate = _matmul_with_weight_dtype(x_norm, layer.w1.t())
            if layer.b1 is not None:
                gate = gate + layer.b1
            gate = _silu_torch(gate)

            up = _matmul_with_weight_dtype(x_norm, layer.w3.t())
            if layer.b3 is not None:
                up = up + layer.b3

            ff = _matmul_with_weight_dtype(gate * up, layer.w2.t())
            if layer.b2 is not None:
                ff = ff + layer.b2
            x = x + ff

        self.position += 1

        if not return_logits:
            return None

        x_last = _rms_norm_torch(x, self.final_norm, self.rms_eps, None)
        logits = _matmul_with_weight_dtype(x_last, self.lm_head)
        return logits.float().cpu().tolist()

    def prefill_tokens(self, token_ids: list[int]) -> None:
        if torch is None or not token_ids:
            return
        with torch.no_grad():
            for token_id in token_ids:
                self._forward_token(int(token_id), return_logits=False)

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if torch is None or not token_ids:
            return []
        with torch.no_grad():
            return self._forward_token(int(token_ids[-1]), return_logits=True) or []


def _load_tensor(info: WeightInfo) -> Optional["np.ndarray"]:
    if np is None or safe_open is None:
        return None
    with safe_open(str(info.path), framework="np", device="cpu") as f:
        try:
            return f.get_tensor(info.name)
        except TypeError:
            if torch is None:
                return None
    if torch is None:
        return None
    with safe_open(str(info.path), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(info.name)
    return tensor.float().cpu().numpy()


def _load_tensor_torch(info: WeightInfo, device: str) -> Optional["torch.Tensor"]:
    if torch is None or safe_open is None:
        return None
    with safe_open(str(info.path), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(info.name)
    return tensor.to(device=device)


def _select_weight(weight_index: dict[str, WeightInfo], candidates: list[str]) -> Optional[WeightInfo]:
    for name in candidates:
        if name in weight_index:
            return weight_index[name]
    return None


def runtime_load_reason(weight_index: dict[str, WeightInfo], embed_names: list[str], lm_head_names: list[str]) -> str:
    if np is None or safe_open is None:
        return "missing numpy or safetensors"

    embed_info = _select_weight(weight_index, embed_names)
    lm_head_info = _select_weight(weight_index, lm_head_names)
    if not embed_info or not lm_head_info:
        return "missing embed or lm_head weights"

    needs_torch = False
    for info in (embed_info, lm_head_info):
        if str(info.dtype).lower() in {"bf16", "bfloat16"}:
            needs_torch = True

    if needs_torch and torch is None:
        return "missing torch for bfloat16 weights"

    return "unknown"


def _load_pair(weight_index: dict[str, WeightInfo], embed_names: list[str], lm_head_names: list[str]) -> Optional[SimpleRuntime]:
    embed_info = _select_weight(weight_index, embed_names)
    lm_head_info = _select_weight(weight_index, lm_head_names)
    if not embed_info or not lm_head_info:
        return None

    embed = _load_tensor(embed_info)
    lm_head = _load_tensor(lm_head_info)
    if embed is None or lm_head is None:
        return None

    if embed.ndim != 2 or lm_head.ndim != 2:
        return None

    embed_vocab, embed_dim = embed.shape
    if lm_head.shape[0] == embed_dim:
        pass
    elif lm_head.shape[1] == embed_dim:
        lm_head = lm_head.T
    else:
        return None

    return SimpleRuntime(embed=embed, lm_head=lm_head, eos_token_id=None)


def _load_first_available(weight_index: dict[str, WeightInfo], name: str) -> Optional["np.ndarray"]:
    info = weight_index.get(name)
    if not info:
        return None
    return _load_tensor(info)


def _load_first_available_torch(weight_index: dict[str, WeightInfo], name: str, device: str) -> Optional["torch.Tensor"]:
    info = weight_index.get(name)
    if not info:
        return None
    return _load_tensor_torch(info, device)


def _load_layer(weight_index: dict[str, WeightInfo], layer_idx: int) -> Optional[LayerWeights]:
    prefix = f"model.layers.{layer_idx}."

    q = _load_first_available(weight_index, prefix + "self_attn.q_proj.weight")
    k = _load_first_available(weight_index, prefix + "self_attn.k_proj.weight")
    v = _load_first_available(weight_index, prefix + "self_attn.v_proj.weight")

    qkv = _load_first_available(weight_index, prefix + "self_attn.qkv_proj.weight")
    if qkv is not None and q is None and k is None and v is None:
        q, k, v = np.split(qkv, 3, axis=0)

    o = _load_first_available(weight_index, prefix + "self_attn.o_proj.weight")
    if o is None:
        o = _load_first_available(weight_index, prefix + "self_attn.out_proj.weight")

    n1 = _load_first_available(weight_index, prefix + "input_layernorm.weight")
    if n1 is None:
        n1 = _load_first_available(weight_index, prefix + "attention_norm.weight")

    n2 = _load_first_available(weight_index, prefix + "post_attention_layernorm.weight")
    if n2 is None:
        n2 = _load_first_available(weight_index, prefix + "mlp_norm.weight")

    w1 = _load_first_available(weight_index, prefix + "mlp.gate_proj.weight")
    w2 = _load_first_available(weight_index, prefix + "mlp.down_proj.weight")
    w3 = _load_first_available(weight_index, prefix + "mlp.up_proj.weight")

    if w1 is None:
        w1 = _load_first_available(weight_index, prefix + "mlp.w1.weight")
    if w2 is None:
        w2 = _load_first_available(weight_index, prefix + "mlp.w2.weight")
    if w3 is None:
        w3 = _load_first_available(weight_index, prefix + "mlp.w3.weight")

    if any(x is None for x in (q, k, v, o, w1, w2, w3, n1, n2)):
        return None

    bq = _load_first_available(weight_index, prefix + "self_attn.q_proj.bias")
    bk = _load_first_available(weight_index, prefix + "self_attn.k_proj.bias")
    bv = _load_first_available(weight_index, prefix + "self_attn.v_proj.bias")
    bo = _load_first_available(weight_index, prefix + "self_attn.o_proj.bias")
    b1 = _load_first_available(weight_index, prefix + "mlp.gate_proj.bias")
    b2 = _load_first_available(weight_index, prefix + "mlp.down_proj.bias")
    b3 = _load_first_available(weight_index, prefix + "mlp.up_proj.bias")
    n1_bias = _load_first_available(weight_index, prefix + "input_layernorm.bias")
    n2_bias = _load_first_available(weight_index, prefix + "post_attention_layernorm.bias")

    layer = LayerWeights(
        wq=q,
        wk=k,
        wv=v,
        wo=o,
        w1=w1,
        w2=w2,
        w3=w3,
        norm1=n1,
        norm2=n2,
        bq=bq,
        bk=bk,
        bv=bv,
        bo=bo,
        b1=b1,
        b2=b2,
        b3=b3,
        norm1_bias=n1_bias,
        norm2_bias=n2_bias,
    )

    fused_bits = int(os.getenv("VSPEC_FUSED_BITS", "0") or "0")
    if fused_bits in {3, 4}:
        for key, w in {
            "wq": layer.wq,
            "wk": layer.wk,
            "wv": layer.wv,
            "wo": layer.wo,
            "w1": layer.w1,
            "w2": layer.w2,
            "w3": layer.w3,
        }.items():
            packed, scales = _quantize_weight_rowwise(w, fused_bits)
            layer.packed[key] = (packed, scales, fused_bits, int(w.shape[0]))

    return layer


def _load_layer_torch(weight_index: dict[str, WeightInfo], layer_idx: int, device: str) -> Optional[LayerWeights]:
    prefix = f"model.layers.{layer_idx}."

    q = _load_first_available_torch(weight_index, prefix + "self_attn.q_proj.weight", device)
    k = _load_first_available_torch(weight_index, prefix + "self_attn.k_proj.weight", device)
    v = _load_first_available_torch(weight_index, prefix + "self_attn.v_proj.weight", device)

    qkv = _load_first_available_torch(weight_index, prefix + "self_attn.qkv_proj.weight", device)
    if qkv is not None and q is None and k is None and v is None:
        q, k, v = torch.split(qkv, qkv.shape[0] // 3, dim=0)

    o = _load_first_available_torch(weight_index, prefix + "self_attn.o_proj.weight", device)
    if o is None:
        o = _load_first_available_torch(weight_index, prefix + "self_attn.out_proj.weight", device)

    n1 = _load_first_available_torch(weight_index, prefix + "input_layernorm.weight", device)
    if n1 is None:
        n1 = _load_first_available_torch(weight_index, prefix + "attention_norm.weight", device)

    n2 = _load_first_available_torch(weight_index, prefix + "post_attention_layernorm.weight", device)
    if n2 is None:
        n2 = _load_first_available_torch(weight_index, prefix + "mlp_norm.weight", device)

    w1 = _load_first_available_torch(weight_index, prefix + "mlp.gate_proj.weight", device)
    w2 = _load_first_available_torch(weight_index, prefix + "mlp.down_proj.weight", device)
    w3 = _load_first_available_torch(weight_index, prefix + "mlp.up_proj.weight", device)

    if w1 is None:
        w1 = _load_first_available_torch(weight_index, prefix + "mlp.w1.weight", device)
    if w2 is None:
        w2 = _load_first_available_torch(weight_index, prefix + "mlp.w2.weight", device)
    if w3 is None:
        w3 = _load_first_available_torch(weight_index, prefix + "mlp.w3.weight", device)

    if any(x is None for x in (q, k, v, o, w1, w2, w3, n1, n2)):
        return None

    bq = _load_first_available_torch(weight_index, prefix + "self_attn.q_proj.bias", device)
    bk = _load_first_available_torch(weight_index, prefix + "self_attn.k_proj.bias", device)
    bv = _load_first_available_torch(weight_index, prefix + "self_attn.v_proj.bias", device)
    bo = _load_first_available_torch(weight_index, prefix + "self_attn.o_proj.bias", device)
    b1 = _load_first_available_torch(weight_index, prefix + "mlp.gate_proj.bias", device)
    b2 = _load_first_available_torch(weight_index, prefix + "mlp.down_proj.bias", device)
    b3 = _load_first_available_torch(weight_index, prefix + "mlp.up_proj.bias", device)
    n1_bias = _load_first_available_torch(weight_index, prefix + "input_layernorm.bias", device)
    n2_bias = _load_first_available_torch(weight_index, prefix + "post_attention_layernorm.bias", device)

    return LayerWeights(
        wq=q,
        wk=k,
        wv=v,
        wo=o,
        w1=w1,
        w2=w2,
        w3=w3,
        norm1=n1,
        norm2=n2,
        bq=bq,
        bk=bk,
        bv=bv,
        bo=bo,
        b1=b1,
        b2=b2,
        b3=b3,
        norm1_bias=n1_bias,
        norm2_bias=n2_bias,
    )


def _load_generic_transformer(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    use_native_cuda_norm: bool,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[GenericTransformerRuntime]:
    embed = _load_pair(
        weight_index,
        [
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            "transformer.wte.weight",
        ],
        [
            "lm_head.weight",
            "output.weight",
            "transformer.lm_head.weight",
        ],
    )
    if embed is None:
        return None

    embed_weight = embed.embed
    lm_head = embed.lm_head

    final_norm = _load_first_available(weight_index, "model.norm.weight")
    if final_norm is None:
        final_norm = _load_first_available(weight_index, "norm.weight")
    if final_norm is None:
        return None

    num_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
    if num_layers == 0:
        num_layers = 1
    if max_layers:
        num_layers = min(num_layers, max_layers)

    layers: list[LayerWeights] = []
    if progress_cb is not None:
        progress_cb("layer_load", 0, num_layers)
    for idx in range(num_layers):
        layer = _load_layer(weight_index, idx)
        if layer is None:
            break
        layers.append(layer)
        if progress_cb is not None:
            progress_cb("layer_load", idx + 1, num_layers)

    if not layers:
        return None

    hidden = embed_weight.shape[1]
    num_heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
    num_kv_heads = int(config.get("num_key_value_heads", 0) or config.get("n_kv_head", 0) or num_heads)
    if num_heads <= 0:
        num_heads = 32
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    head_dim = hidden // num_heads

    rms_eps = float(config.get("rms_norm_eps", 1e-6))
    rope_theta = float(config.get("rope_theta", 10000.0))

    return GenericTransformerRuntime(
        embed=embed_weight,
        lm_head=lm_head,
        final_norm=final_norm,
        layers=layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_eps=rms_eps,
        eos_token_id=None,
        rope_theta=rope_theta,
        position=0,
        cache_k=[],
        cache_v=[],
        use_native_cuda_norm=use_native_cuda_norm,
        fused_bits=int(os.getenv("VSPEC_FUSED_BITS", "0") or "0"),
        disable_fused_attention=(os.getenv("VSPEC_DISABLE_FUSED_ATTN", "0").strip().lower() in {"1", "true", "yes", "on"}),
    )


def _load_generic_transformer_torch(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    device: str,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[GenericTransformerRuntimeTorch]:
    if torch is None:
        return None

    embed_info = _select_weight(weight_index, [
        "model.embed_tokens.weight",
        "tok_embeddings.weight",
        "transformer.wte.weight",
    ])
    lm_info = _select_weight(weight_index, [
        "lm_head.weight",
        "output.weight",
        "transformer.lm_head.weight",
    ])
    if embed_info is None or lm_info is None:
        return None

    embed_weight = _load_tensor_torch(embed_info, device)
    lm_head = _load_tensor_torch(lm_info, device)
    if embed_weight is None or lm_head is None:
        return None

    embed_vocab, embed_dim = embed_weight.shape
    if lm_head.shape[0] == embed_dim:
        pass
    elif lm_head.shape[1] == embed_dim:
        lm_head = lm_head.t()
    else:
        return None

    final_norm = _load_first_available_torch(weight_index, "model.norm.weight", device)
    if final_norm is None:
        final_norm = _load_first_available_torch(weight_index, "norm.weight", device)
    if final_norm is None:
        return None

    num_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
    if num_layers == 0:
        num_layers = 1
    if max_layers:
        num_layers = min(num_layers, max_layers)

    layers: list[LayerWeights] = []
    if progress_cb is not None:
        progress_cb("layer_load", 0, num_layers)
    for idx in range(num_layers):
        layer = _load_layer_torch(weight_index, idx, device)
        if layer is None:
            break
        layers.append(layer)
        if progress_cb is not None:
            progress_cb("layer_load", idx + 1, num_layers)

    if not layers:
        return None

    hidden = embed_weight.shape[1]
    num_heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
    num_kv_heads = int(config.get("num_key_value_heads", 0) or config.get("n_kv_head", 0) or num_heads)
    if num_heads <= 0:
        num_heads = 32
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    head_dim = hidden // num_heads

    rms_eps = float(config.get("rms_norm_eps", 1e-6))
    rope_theta = float(config.get("rope_theta", 10000.0))

    return GenericTransformerRuntimeTorch(
        embed=embed_weight,
        lm_head=lm_head,
        final_norm=final_norm,
        layers=layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_eps=rms_eps,
        eos_token_id=None,
        rope_theta=rope_theta,
        position=0,
        cache_k=[],
        cache_v=[],
    )


def build_runtime(adapter: ModelAdapter, weight_index: dict[str, WeightInfo]) -> Optional[SimpleRuntime]:
    return None


def build_generic_runtime(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    device: str,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[object]:
    if device == "torch-cuda" and torch is not None and torch.cuda.is_available():
        runtime = _load_generic_transformer_torch(config, weight_index, max_layers, device, progress_cb)
        if runtime is not None:
            return runtime
    use_native_cuda_norm = device in {"cuda", "cuda-native"}
    return _load_generic_transformer(config, weight_index, max_layers, use_native_cuda_norm, progress_cb)
