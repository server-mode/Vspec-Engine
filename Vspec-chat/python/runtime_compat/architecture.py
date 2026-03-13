from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RuntimeTarget:
    runtime_name: str
    normalized_model_type: str
    config: dict[str, Any]
    reason: str
    warnings: list[str] = field(default_factory=list)


def _tensor_name_set(tensor_names: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    out: set[str] = set()
    for name in list(tensor_names or []):
        try:
            out.add(str(name))
        except Exception:
            continue
    return out


def _weight_shape(weight_index: Optional[dict[str, Any]], candidates: list[str]) -> Optional[list[int]]:
    if not weight_index:
        return None
    for name in candidates:
        info = weight_index.get(name)
        if info is None:
            continue
        shape = getattr(info, "shape", None)
        if isinstance(shape, list) and shape:
            try:
                return [int(v) for v in shape]
            except Exception:
                continue
    return None


def _first_positive_int(*values: Any) -> int:
    for value in values:
        try:
            parsed = int(value)
        except Exception:
            parsed = 0
        if parsed > 0:
            return parsed
    return 0


def _has_any_prefix(names: set[str], prefixes: tuple[str, ...]) -> bool:
    for name in names:
        for prefix in prefixes:
            if name.startswith(prefix):
                return True
    return False


def _has_any_fragment(names: set[str], fragments: tuple[str, ...]) -> bool:
    for name in names:
        for fragment in fragments:
            if fragment in name:
                return True
    return False


def infer_qwen35_hybrid_config(config: dict[str, Any], weight_index: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    cfg = dict(config or {})
    issues: list[str] = []

    num_heads = _first_positive_int(cfg.get("num_attention_heads"), cfg.get("n_head"))
    num_kv_heads = _first_positive_int(cfg.get("num_key_value_heads"), cfg.get("n_kv_head"), num_heads)

    head_dim = _first_positive_int(cfg.get("head_dim"))
    if head_dim <= 0 and num_kv_heads > 0:
        k_shape = _weight_shape(weight_index, [
            "blk.0.attn_k.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ])
        if k_shape and len(k_shape) >= 1 and (k_shape[0] % num_kv_heads) == 0:
            head_dim = k_shape[0] // num_kv_heads
            cfg["head_dim"] = head_dim

    rope_dim = _first_positive_int(cfg.get("rope_dimension_count"), cfg.get("rope_dim"))
    if rope_dim <= 0 and head_dim > 0:
        rope_dim = head_dim
        cfg["rope_dimension_count"] = rope_dim

    linear_num_value_heads = _first_positive_int(cfg.get("linear_num_value_heads"))
    if linear_num_value_heads <= 0:
        beta_shape = _weight_shape(weight_index, ["blk.0.ssm_beta.weight"])
        a_shape = _weight_shape(weight_index, ["blk.0.ssm_a"])
        if beta_shape and len(beta_shape) >= 1:
            linear_num_value_heads = beta_shape[0]
        elif a_shape and len(a_shape) >= 1:
            linear_num_value_heads = a_shape[0]
        if linear_num_value_heads > 0:
            cfg["linear_num_value_heads"] = linear_num_value_heads

    linear_conv_kernel = _first_positive_int(cfg.get("linear_conv_kernel_dim"))
    if linear_conv_kernel <= 0:
        conv_shape = _weight_shape(weight_index, ["blk.0.ssm_conv1d.weight"])
        if conv_shape and len(conv_shape) >= 1:
            linear_conv_kernel = conv_shape[0]
            cfg["linear_conv_kernel_dim"] = linear_conv_kernel

    linear_value_head_dim = _first_positive_int(cfg.get("linear_value_head_dim"))
    if linear_value_head_dim <= 0 and linear_num_value_heads > 0:
        ssm_out_shape = _weight_shape(weight_index, ["blk.0.ssm_out.weight"])
        if ssm_out_shape and len(ssm_out_shape) >= 2:
            input_dim = ssm_out_shape[1]
            if input_dim > 0 and (input_dim % linear_num_value_heads) == 0:
                linear_value_head_dim = input_dim // linear_num_value_heads
                cfg["linear_value_head_dim"] = linear_value_head_dim

    linear_num_key_heads = _first_positive_int(cfg.get("linear_num_key_heads"), num_kv_heads)
    if linear_num_key_heads > 0:
        cfg["linear_num_key_heads"] = linear_num_key_heads

    linear_key_head_dim = _first_positive_int(cfg.get("linear_key_head_dim"))
    if linear_key_head_dim <= 0 and linear_num_key_heads > 0 and linear_value_head_dim > 0 and linear_num_value_heads > 0:
        qkv_shape = _weight_shape(weight_index, ["blk.0.attn_qkv.weight"])
        if qkv_shape and len(qkv_shape) >= 1:
            qkv_total = qkv_shape[0]
            remaining = qkv_total - (linear_num_value_heads * linear_value_head_dim)
            denom = 2 * linear_num_key_heads
            if remaining > 0 and denom > 0 and (remaining % denom) == 0:
                linear_key_head_dim = remaining // denom
                cfg["linear_key_head_dim"] = linear_key_head_dim

    required = {
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "rope_dimension_count": rope_dim,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_conv_kernel_dim": linear_conv_kernel,
    }
    for key, value in required.items():
        if int(value) <= 0:
            issues.append(f"missing_{key}")
        else:
            cfg[key] = int(value)

    return cfg, issues


def resolve_runtime_target(
    config: dict[str, Any],
    tensor_names: list[str] | tuple[str, ...] | set[str] | None,
    weight_index: Optional[dict[str, Any]] = None,
) -> RuntimeTarget:
    cfg = dict(config or {})
    names = _tensor_name_set(tensor_names)
    model_type = str(cfg.get("model_type", "") or "").strip().lower()

    if model_type == "gpt2":
        return RuntimeTarget(
            runtime_name="gpt2",
            normalized_model_type="gpt2",
            config=cfg,
            reason="explicit_gpt2_model_type",
        )

    has_qwen35_ssm = _has_any_fragment(names, (
        ".attn_qkv.weight",
        ".ssm_alpha.weight",
        ".ssm_beta.weight",
        ".ssm_conv1d.weight",
        ".ssm_out.weight",
    ))
    has_qwen35_attn = _has_any_fragment(names, (
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
    ))
    has_generic_transformer = _has_any_prefix(names, (
        "model.layers.",
        "transformer.h.",
    )) or _has_any_fragment(names, (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    ))

    if model_type == "qwen35" or has_qwen35_ssm:
        enriched_cfg, issues = infer_qwen35_hybrid_config(cfg, weight_index)
        if has_qwen35_ssm and not issues:
            enriched_cfg["model_type"] = "qwen35"
            return RuntimeTarget(
                runtime_name="qwen35",
                normalized_model_type="qwen35",
                config=enriched_cfg,
                reason="hybrid_qwen35_detected",
            )

        fallback_cfg = dict(enriched_cfg)
        fallback_cfg["model_type"] = "generic"
        warnings = ["qwen35_hybrid_fallback_to_generic"]
        warnings.extend(issues)
        return RuntimeTarget(
            runtime_name="generic",
            normalized_model_type="generic",
            config=fallback_cfg,
            reason=("qwen35_metadata_incomplete" if issues else "qwen35_without_ssm_tensors"),
            warnings=warnings,
        )

    if has_generic_transformer:
        generic_cfg = dict(cfg)
        generic_cfg["model_type"] = str(generic_cfg.get("model_type", "generic") or "generic")
        return RuntimeTarget(
            runtime_name="generic",
            normalized_model_type=str(generic_cfg.get("model_type", "generic") or "generic"),
            config=generic_cfg,
            reason="generic_transformer_detected",
        )

    return RuntimeTarget(
        runtime_name="generic",
        normalized_model_type=(model_type or "generic"),
        config=cfg,
        reason="default_generic_runtime",
    )
