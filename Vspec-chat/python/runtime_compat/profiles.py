from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RuntimeStabilityProfile:
    attention_logit_clip: float
    residual_feedback_gain: float
    residual_clamp_alpha: float
    logit_entropy_target: float
    logit_margin_floor: float
    logit_margin_gain: float
    ssm_warmup_tokens: int = 0


def _env_float(name: str, fallback: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return fallback
    try:
        parsed = float(value)
    except Exception:
        return fallback
    if parsed != parsed:
        return fallback
    return parsed


def _env_int(name: str, fallback: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return fallback
    try:
        return int(value)
    except Exception:
        return fallback


def _normalize_model_family(model_family: str | None) -> str:
    name = (model_family or "").strip().lower()
    if not name:
        return "generic"
    if "qwen3.5" in name or "qwen35" in name:
        return "qwen35"
    if "qwen" in name:
        return "qwen"
    if "llama" in name or "mistral" in name:
        return "llama"
    if "gpt2" in name:
        return "gpt2"
    return "generic"


def _resolve_family_defaults(model_family: str) -> RuntimeStabilityProfile:
    family = _normalize_model_family(model_family)
    if family == "gpt2":
        return RuntimeStabilityProfile(
            attention_logit_clip=20.0,
            residual_feedback_gain=0.06,
            residual_clamp_alpha=0.0,
            logit_entropy_target=0.0,
            logit_margin_floor=0.0,
            logit_margin_gain=0.0,
            ssm_warmup_tokens=0,
        )
    if family == "qwen":
        return RuntimeStabilityProfile(
            attention_logit_clip=24.0,
            residual_feedback_gain=0.10,
            residual_clamp_alpha=0.0,
            logit_entropy_target=0.0,
            logit_margin_floor=0.0,
            logit_margin_gain=0.0,
            ssm_warmup_tokens=0,
        )
    if family == "llama":
        return RuntimeStabilityProfile(
            attention_logit_clip=22.0,
            residual_feedback_gain=0.08,
            residual_clamp_alpha=0.0,
            logit_entropy_target=0.0,
            logit_margin_floor=0.0,
            logit_margin_gain=0.0,
            ssm_warmup_tokens=0,
        )
    if family == "qwen35":
        return RuntimeStabilityProfile(
            attention_logit_clip=24.0,
            residual_feedback_gain=0.12,
            residual_clamp_alpha=0.0,
            logit_entropy_target=0.0,
            logit_margin_floor=0.0,
            logit_margin_gain=0.0,
            ssm_warmup_tokens=8,
        )
    return RuntimeStabilityProfile(
        attention_logit_clip=22.0,
        residual_feedback_gain=0.08,
        residual_clamp_alpha=0.0,
        logit_entropy_target=0.0,
        logit_margin_floor=0.0,
        logit_margin_gain=0.0,
        ssm_warmup_tokens=0,
    )


def resolve_model_stability_profile(model_family: str | None = None) -> RuntimeStabilityProfile:
    base = _resolve_family_defaults(_normalize_model_family(model_family))
    env_prefix = _normalize_model_family(model_family).upper()
    if env_prefix == "GENERIC":
        env_prefix = "MODEL"

    return RuntimeStabilityProfile(
        attention_logit_clip=_env_float(f"VSPEC_{env_prefix}_ATTN_LOGIT_CLIP", _env_float("VSPEC_ATTN_LOGIT_CLIP", base.attention_logit_clip)),
        residual_feedback_gain=_env_float(f"VSPEC_{env_prefix}_RESIDUAL_FEEDBACK_GAIN", _env_float("VSPEC_RESIDUAL_FEEDBACK_GAIN", base.residual_feedback_gain)),
        residual_clamp_alpha=_env_float(f"VSPEC_{env_prefix}_RESIDUAL_CLAMP_ALPHA", _env_float("VSPEC_RESIDUAL_CLAMP_ALPHA", base.residual_clamp_alpha)),
        logit_entropy_target=_env_float(f"VSPEC_{env_prefix}_LOGIT_ENTROPY_TARGET", _env_float("VSPEC_LOGIT_ENTROPY_TARGET", base.logit_entropy_target)),
        logit_margin_floor=_env_float(f"VSPEC_{env_prefix}_LOGIT_MARGIN_FLOOR", _env_float("VSPEC_LOGIT_MARGIN_FLOOR", base.logit_margin_floor)),
        logit_margin_gain=_env_float(f"VSPEC_{env_prefix}_LOGIT_MARGIN_GAIN", _env_float("VSPEC_LOGIT_MARGIN_GAIN", base.logit_margin_gain)),
        ssm_warmup_tokens=max(0, _env_int(f"VSPEC_{env_prefix}_SSM_WARMUP_TOKENS", base.ssm_warmup_tokens)),
    )


def resolve_generic_stability_profile() -> RuntimeStabilityProfile:
    return resolve_model_stability_profile("generic")


def resolve_qwen35_stability_profile() -> RuntimeStabilityProfile:
    return resolve_model_stability_profile("qwen35")
