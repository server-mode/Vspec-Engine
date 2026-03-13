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


def resolve_generic_stability_profile() -> RuntimeStabilityProfile:
    return RuntimeStabilityProfile(
        attention_logit_clip=_env_float("VSPEC_ATTN_LOGIT_CLIP", 0.0),
        residual_feedback_gain=_env_float("VSPEC_RESIDUAL_FEEDBACK_GAIN", 0.0),
        residual_clamp_alpha=_env_float("VSPEC_RESIDUAL_CLAMP_ALPHA", 0.0),
        logit_entropy_target=_env_float("VSPEC_LOGIT_ENTROPY_TARGET", 0.0),
        logit_margin_floor=_env_float("VSPEC_LOGIT_MARGIN_FLOOR", 0.0),
        logit_margin_gain=_env_float("VSPEC_LOGIT_MARGIN_GAIN", 0.0),
        ssm_warmup_tokens=0,
    )


def resolve_qwen35_stability_profile() -> RuntimeStabilityProfile:
    generic = resolve_generic_stability_profile()
    return RuntimeStabilityProfile(
        attention_logit_clip=_env_float("VSPEC_QWEN35_ATTN_LOGIT_CLIP", generic.attention_logit_clip if generic.attention_logit_clip > 0.0 else 24.0),
        residual_feedback_gain=_env_float("VSPEC_QWEN35_RESIDUAL_FEEDBACK_GAIN", generic.residual_feedback_gain if generic.residual_feedback_gain > 0.0 else 0.12),
        residual_clamp_alpha=_env_float("VSPEC_QWEN35_RESIDUAL_CLAMP_ALPHA", generic.residual_clamp_alpha if generic.residual_clamp_alpha > 0.0 else 4.5),
        logit_entropy_target=_env_float("VSPEC_QWEN35_LOGIT_ENTROPY_TARGET", generic.logit_entropy_target if generic.logit_entropy_target > 0.0 else 7.5),
        logit_margin_floor=_env_float("VSPEC_QWEN35_LOGIT_MARGIN_FLOOR", generic.logit_margin_floor if generic.logit_margin_floor > 0.0 else 0.20),
        logit_margin_gain=_env_float("VSPEC_QWEN35_LOGIT_MARGIN_GAIN", generic.logit_margin_gain if generic.logit_margin_gain > 0.0 else 0.60),
        ssm_warmup_tokens=max(0, _env_int("VSPEC_QWEN35_SSM_WARMUP_TOKENS", 8)),
    )
