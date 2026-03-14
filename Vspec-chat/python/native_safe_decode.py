from __future__ import annotations

from runtime_inference import build_generic_runtime


def resolve_native_safe_max_layers(config: dict, requested_max_layers: int | None) -> int | None:
    try:
        total_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
    except Exception:
        total_layers = 0

    requested = int(requested_max_layers or 0)
    if total_layers <= 0:
        return requested_max_layers
    safe_floor = 20 if total_layers >= 32 else 16
    if requested <= 0:
        return min(total_layers, safe_floor)
    if requested < total_layers:
        return min(total_layers, max(requested, safe_floor))
    return requested


def build_native_safe_runtime(
    config: dict,
    weight_index: dict,
    requested_max_layers: int | None,
    device_preference: str = "cuda",
    progress_cb=None,
):
    safe_layers = resolve_native_safe_max_layers(config, requested_max_layers)
    device = "cuda" if str(device_preference) in {"cuda", "cuda-native"} else "cpu"
    use_native = device == "cuda"
    safe_config = dict(config)
    safe_config["vspec_force_runtime"] = "generic"
    runtime = build_generic_runtime(
        safe_config,
        weight_index,
        safe_layers,
        device,
        fused_bits_override=0,
        use_native_cuda_norm_override=use_native,
        progress_cb=progress_cb,
    )
    if runtime is None:
        runtime = build_generic_runtime(
            config,
            weight_index,
            safe_layers,
            device,
            fused_bits_override=0,
            use_native_cuda_norm_override=use_native,
            progress_cb=progress_cb,
        )
    if runtime is not None:
        try:
            setattr(runtime, "vspec_safe_lane", True)
        except Exception:
            pass
    return runtime