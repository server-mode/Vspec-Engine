from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
sys.path.insert(0, str(CHAT_PY))

from chat_prompt import build_prompt
from model_adapters import select_adapter
from model_loader import build_weight_index, collect_tensor_names, find_snapshot_dir, load_tokenizer, read_config, read_tokenizer_config
from runtime_inference import build_generic_runtime
from hardware_telemetry import build_hardware_report, capture_hardware_snapshot, summarize_hardware_usage


def _forward_logits_array(runtime, token_id: int) -> np.ndarray:
    if hasattr(runtime, "forward_logits_np"):
        logits = runtime.forward_logits_np([int(token_id)])
        if logits is None:
            return np.empty(0, dtype=np.float32)
        return np.asarray(logits, dtype=np.float32)
    logits = runtime.forward_logits([int(token_id)])
    if logits is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(logits, dtype=np.float32)


def _build_runtime(config: dict, weight_index: dict, max_layers: int, fused_bits: int, baseline_mode: str = "reference"):
    os.environ["VSPEC_FUSED_BITS"] = str(fused_bits)
    if fused_bits == 0 and baseline_mode == "performance":
        os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "1"
    else:
        os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "0"

    if fused_bits == 0:
        os.environ["VSPEC_LOWBIT_PROFILE"] = "baseline" if baseline_mode == "performance" else "aggressive"
    else:
        os.environ["VSPEC_LOWBIT_PROFILE"] = "aggressive"

    if fused_bits == 0:
        os.environ["VSPEC_PERFORMANCE_LOWBIT_DEFAULT"] = "1" if baseline_mode == "performance" else "0"
    else:
        os.environ["VSPEC_PERFORMANCE_LOWBIT_DEFAULT"] = "1"
    return build_generic_runtime(config, weight_index, max_layers=max_layers, device="cuda-native")


def _reset_runtime(runtime) -> None:
    if hasattr(runtime, "cache_k"):
        runtime.cache_k = []
    if hasattr(runtime, "cache_v"):
        runtime.cache_v = []
    if hasattr(runtime, "position"):
        runtime.position = 0


def _prepare_tokens(tokenizer, adapter, model_type: str, tok_cfg: dict, prompt: str, lang: str, chat_format: str) -> list[int]:
    prompt_for_model = build_prompt(prompt, model_type, tok_cfg, lang, chat_format)
    token_ids = list(tokenizer.encode(prompt_for_model).ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids
    return token_ids


def _run_decode_tps(
    runtime,
    token_ids: list[int],
    decode_steps: int,
    eos_token_id: int | None,
    warmup_decode_steps: int = 0,
) -> tuple[int, float, float]:
    def _run_decode_only(steps: int) -> int:
        ids_local = [int(t) for t in token_ids]
        generated_local = 0
        for _ in range(steps):
            logits_local = _forward_logits_array(runtime, ids_local[-1])
            if logits_local.size == 0:
                break
            nxt_local = int(np.argmax(logits_local))
            ids_local.append(nxt_local)
            generated_local += 1
            if eos_token_id is not None and nxt_local == eos_token_id:
                break
        return generated_local

    _reset_runtime(runtime)
    if len(token_ids) > 1:
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens([int(t) for t in token_ids[:-1]])
        else:
            for t in token_ids[:-1]:
                runtime.forward_logits([int(t)])

    if warmup_decode_steps > 0:
        _run_decode_only(warmup_decode_steps)
        _reset_runtime(runtime)
        if len(token_ids) > 1:
            if hasattr(runtime, "prefill_tokens"):
                runtime.prefill_tokens([int(t) for t in token_ids[:-1]])
            else:
                for t in token_ids[:-1]:
                    runtime.forward_logits([int(t)])

    ids = [int(t) for t in token_ids]
    generated = 0
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        logits = _forward_logits_array(runtime, ids[-1])
        if logits.size == 0:
            break
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        generated += 1
        if eos_token_id is not None and nxt == eos_token_id:
            break
    sec = time.perf_counter() - t0
    tps = (generated / sec) if sec > 0 else 0.0
    return generated, sec, tps


def _nll_from_logits(logits: list[float], target_id: int) -> float:
    if logits is None:
        return 0.0
    arr = np.asarray(logits, dtype=np.float32)
    if arr.size == 0 or target_id < 0 or target_id >= arr.shape[0]:
        return 0.0
    mx = float(np.max(arr))
    ex = np.exp(arr - mx)
    denom = float(np.sum(ex))
    if denom <= 0.0:
        return 0.0
    p = float(ex[target_id] / denom)
    if p <= 1e-12:
        p = 1e-12
    return -math.log(p)


def _perplexity_teacher_forcing(runtime, token_ids: list[int], max_eval_tokens: int) -> float:
    if len(token_ids) < 2:
        return 0.0

    _reset_runtime(runtime)
    steps = min(max_eval_tokens, len(token_ids) - 1)
    logits = _forward_logits_array(runtime, int(token_ids[0]))
    nll_total = 0.0

    for i in range(1, steps + 1):
        target = int(token_ids[i])
        nll_total += _nll_from_logits(logits, target)
        logits = _forward_logits_array(runtime, target)

    if steps == 0:
        return 0.0
    return float(math.exp(nll_total / float(steps)))


def _compute_projection_effective_bits(runtime) -> tuple[float, float, int, int]:
    proj_keys = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")
    packed_proj = 0
    total_proj = 0
    weighted_bits = 0.0
    weighted_count = 0

    for layer in runtime.layers:
        for key in proj_keys:
            w = getattr(layer, key)
            rows, cols = int(w.shape[0]), int(w.shape[1])
            count = rows * cols
            total_proj += 1
            if key in layer.packed:
                packed_proj += 1
                _, _, bits, _ = layer.packed[key]
                weighted_bits += float(bits) * count
            else:
                weighted_bits += 16.0 * count
            weighted_count += count

    effective_bits = (weighted_bits / weighted_count) if weighted_count > 0 else 0.0
    coverage = (packed_proj / total_proj) if total_proj > 0 else 0.0
    return effective_bits, coverage, packed_proj, total_proj


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 27 performance consolidation benchmark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-eval-tokens", type=int, default=24)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--prompt", default="Summarize in one line: Vspec runtime focuses on native low-bit inference.")
    parser.add_argument("--drift-threshold", type=float, default=4.0)
    parser.add_argument("--speed-ratio-target", type=float, default=0.95, help="lowbit_tps / baseline_tps minimum")
    parser.add_argument("--baseline-mode", choices=["reference", "performance"], default="reference", help="reference=full-precision baseline, performance=baseline uses adaptive lowbit manager")
    parser.add_argument("--baseline-tps-target", type=float, default=5.0, help="minimum baseline TPS when baseline-mode=performance")
    parser.add_argument("--output", default=str(ROOT / "logs" / "week27_performance_consolidation.json"))
    args = parser.parse_args()

    snapshot = find_snapshot_dir(Path(args.model_dir))
    config = read_config(snapshot)
    tok_cfg = read_tokenizer_config(snapshot)
    tokenizer = load_tokenizer(snapshot)
    if tokenizer is None:
        raise RuntimeError("tokenizer not found")

    tensor_names = collect_tensor_names(snapshot)
    weight_index = build_weight_index(snapshot)
    adapter = select_adapter(config, tensor_names)

    wall_t0 = time.perf_counter()
    hw_before = capture_hardware_snapshot(runtime=None, backend_hint="cuda-native")

    t_build0 = time.perf_counter()
    runtime_base = _build_runtime(
        config,
        weight_index,
        max_layers=args.max_layers,
        fused_bits=0,
        baseline_mode=args.baseline_mode,
    )
    t_build1 = time.perf_counter()
    runtime_low = _build_runtime(config, weight_index, max_layers=args.max_layers, fused_bits=3, baseline_mode="performance")
    t_build2 = time.perf_counter()

    if runtime_base is None or runtime_low is None:
        raise RuntimeError("runtime init failed")

    token_ids = _prepare_tokens(tokenizer, adapter, adapter.model_type, tok_cfg, args.prompt, args.lang, args.chat_format)

    base_gen, base_decode_sec, base_tps = _run_decode_tps(
        runtime_base,
        token_ids,
        args.decode_steps,
        adapter.eos_token_id,
        warmup_decode_steps=min(2, max(0, args.decode_steps // 4)),
    )
    low_gen, low_decode_sec, low_tps = _run_decode_tps(
        runtime_low,
        token_ids,
        args.decode_steps,
        adapter.eos_token_id,
        warmup_decode_steps=min(6, max(0, args.decode_steps // 2)),
    )

    ppl_base = _perplexity_teacher_forcing(runtime_base, token_ids, args.max_eval_tokens)
    ppl_low = _perplexity_teacher_forcing(runtime_low, token_ids, args.max_eval_tokens)
    drift_ratio = abs(ppl_low - ppl_base) / max(ppl_base, 1e-9)

    eff_bits, coverage, packed_proj, total_proj = _compute_projection_effective_bits(runtime_low)

    hw_after = capture_hardware_snapshot(runtime=runtime_low, backend_hint="cuda-native")
    total_sec = time.perf_counter() - wall_t0

    speed_ratio = (low_tps / base_tps) if base_tps > 0 else 0.0

    report = {
        "week": 27,
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "max_layers": int(args.max_layers),
        "baseline_mode": args.baseline_mode,
        "prompt_tokens": len(token_ids),
        "decode_steps": int(args.decode_steps),
        "timing": {
            "build_baseline_sec": float(t_build1 - t_build0),
            "build_lowbit_sec": float(t_build2 - t_build1),
            "decode_baseline_sec": float(base_decode_sec),
            "decode_lowbit_sec": float(low_decode_sec),
            "total_sec": float(total_sec),
        },
        "throughput": {
            "baseline_tps": float(base_tps),
            "lowbit_tps": float(low_tps),
            "speed_ratio_low_vs_base": float(speed_ratio),
            "baseline_generated": int(base_gen),
            "lowbit_generated": int(low_gen),
        },
        "vram": {
            "before_used_bytes": hw_before.get("vram_used_bytes"),
            "after_used_bytes": hw_after.get("vram_used_bytes"),
            "after_total_bytes": hw_after.get("vram_total_bytes"),
            "after_utilization_pct": hw_after.get("vram_utilization_pct"),
        },
        "effective_bits": {
            "estimate": float(eff_bits),
            "projection_coverage": float(coverage),
            "projection_packed": int(packed_proj),
            "projection_total": int(total_proj),
        },
        "perplexity": {
            "baseline": float(ppl_base),
            "lowbit": float(ppl_low),
            "drift_ratio": float(drift_ratio),
            "drift_threshold": float(args.drift_threshold),
        },
        "hardware": build_hardware_report(hw_before, hw_after),
    }

    if args.baseline_mode == "performance":
        report["kpi_speed_pass"] = bool(report["throughput"]["lowbit_tps"] >= float(args.baseline_tps_target))
    else:
        report["kpi_speed_pass"] = bool(report["throughput"]["speed_ratio_low_vs_base"] >= float(args.speed_ratio_target))
    report["kpi_baseline_tps_pass"] = bool(report["throughput"]["baseline_tps"] >= float(args.baseline_tps_target))
    report["kpi_effective_bits_lt4"] = bool(report["effective_bits"]["estimate"] < 4.0)
    report["kpi_perplexity_drift_pass"] = bool(report["perplexity"]["drift_ratio"] <= float(args.drift_threshold))
    if args.baseline_mode == "performance":
        report["kpi_week27_pass"] = bool(
            report["kpi_baseline_tps_pass"]
            and report["kpi_effective_bits_lt4"]
            and report["kpi_perplexity_drift_pass"]
        )
    else:
        report["kpi_week27_pass"] = bool(
            report["kpi_speed_pass"]
            and report["kpi_effective_bits_lt4"]
            and report["kpi_perplexity_drift_pass"]
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[hardware] {summarize_hardware_usage(hw_after)}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
