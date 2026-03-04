from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAT_PY))
sys.path.insert(0, str(THIS_DIR))

from chat_prompt import build_prompt
from model_adapters import select_adapter
from model_loader import build_weight_index, collect_tensor_names, find_snapshot_dir, load_tokenizer, read_config, read_tokenizer_config
from runtime_inference import build_generic_runtime
from vspec_chat import _apply_generation_controls, sample_from_logits
from hardware_telemetry import capture_hardware_snapshot


def _forward_logits_array(runtime, token_id: int) -> np.ndarray:
    if hasattr(runtime, "forward_logits_np"):
        logits = runtime.forward_logits_np([int(token_id)])
    else:
        logits = runtime.forward_logits([int(token_id)])
    if logits is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(logits, dtype=np.float32)


def _build_runtime(config: dict, weight_index: dict, max_layers: int, fused_bits: int):
    os.environ["VSPEC_FUSED_BITS"] = str(int(fused_bits))
    os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "0"
    os.environ["VSPEC_LOWBIT_PROFILE"] = "aggressive"
    return build_generic_runtime(config, weight_index, max_layers=max_layers, device="cuda-native")


def _reset_runtime(runtime) -> None:
    if hasattr(runtime, "cache_k"):
        runtime.cache_k = []
    if hasattr(runtime, "cache_v"):
        runtime.cache_v = []
    if hasattr(runtime, "cache_len"):
        runtime.cache_len = []
    if hasattr(runtime, "position"):
        runtime.position = 0


def _nll_from_logits(logits: np.ndarray, target_id: int) -> float:
    if logits.size == 0 or target_id < 0 or target_id >= logits.shape[0]:
        return 0.0
    mx = float(np.max(logits))
    ex = np.exp(logits - mx)
    denom = float(np.sum(ex))
    if denom <= 0.0:
        return 0.0
    p = float(ex[target_id] / denom)
    p = max(1e-12, p)
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


def _logit_drift(runtime_base, runtime_int4, token_ids: list[int], max_eval_tokens: int) -> dict:
    if len(token_ids) < 2:
        return {"steps": 0, "mean_l2": 0.0, "mean_l1": 0.0, "max_l2": 0.0}

    _reset_runtime(runtime_base)
    _reset_runtime(runtime_int4)

    steps = min(max_eval_tokens, len(token_ids) - 1)
    l2_vals: list[float] = []
    l1_vals: list[float] = []

    logits_b = _forward_logits_array(runtime_base, int(token_ids[0]))
    logits_q = _forward_logits_array(runtime_int4, int(token_ids[0]))

    for i in range(1, steps + 1):
        if logits_b.size == 0 or logits_q.size == 0:
            break
        n = min(logits_b.shape[0], logits_q.shape[0])
        if n == 0:
            break
        db = logits_b[:n]
        dq = logits_q[:n]
        diff = db - dq

        l2_vals.append(float(np.linalg.norm(diff) / max(np.linalg.norm(db), 1e-6)))
        l1_vals.append(float(np.mean(np.abs(diff))))

        nxt = int(token_ids[i])
        logits_b = _forward_logits_array(runtime_base, nxt)
        logits_q = _forward_logits_array(runtime_int4, nxt)

    if not l2_vals:
        return {"steps": 0, "mean_l2": 0.0, "mean_l1": 0.0, "max_l2": 0.0}

    return {
        "steps": len(l2_vals),
        "mean_l2": float(np.mean(l2_vals)),
        "mean_l1": float(np.mean(l1_vals)),
        "max_l2": float(np.max(l2_vals)),
    }


def _build_target_context_tokens(tokenizer, base_text: str, target_tokens: int) -> list[int]:
    chunk = list(tokenizer.encode(base_text).ids)
    if not chunk:
        return []

    out: list[int] = []
    while len(out) < target_tokens:
        out.extend(chunk)
    return out[:target_tokens]


def _decode_guarded(
    runtime,
    token_ids: list[int],
    decode_steps: int,
    eos_token_id: int | None,
    tokenizer,
    lang_mode: str,
    seed: int,
) -> tuple[list[int], float]:
    random.seed(seed)

    ids = [int(t) for t in token_ids]
    generated: list[int] = []
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        raw = runtime.forward_logits(ids)
        if not raw:
            break

        adjusted = _apply_generation_controls(
            raw,
            ids,
            repetition_penalty=1.16,
            repeat_window=64,
            no_repeat_ngram=3,
            entropy_floor=0.45,
            tail_flatten_std_floor=0.85,
            top12_margin_cap=7.5,
        )

        nxt = sample_from_logits(
            adjusted,
            temperature=0.85,
            top_k=40,
            greedy=False,
            tokenizer=tokenizer,
            lang_mode=lang_mode,
            lang_top_n=256,
        )
        nxt = int(nxt)
        generated.append(nxt)
        ids.append(nxt)
        if eos_token_id is not None and nxt == int(eos_token_id):
            break

    sec = time.perf_counter() - t0
    return generated, sec


def _run_context(runtime, context_tokens: list[int], decode_steps: int, eos_token_id: int | None, tokenizer, lang_mode: str, seed: int) -> dict:
    _reset_runtime(runtime)

    before = capture_hardware_snapshot(runtime=runtime, backend_hint="cuda-native")

    prefill_ids = context_tokens[:-1] if len(context_tokens) > 1 else context_tokens
    t0 = time.perf_counter()
    if prefill_ids:
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens([int(t) for t in prefill_ids])
        else:
            for t in prefill_ids:
                runtime.forward_logits([int(t)])
    prefill_sec = time.perf_counter() - t0

    generated, decode_sec = _decode_guarded(
        runtime,
        context_tokens,
        decode_steps,
        eos_token_id,
        tokenizer,
        lang_mode,
        seed,
    )

    after = capture_hardware_snapshot(runtime=runtime, backend_hint="cuda-native")

    text = tokenizer.decode(generated).strip()
    garbage = 0
    for ch in text:
        if ch == "�" or (ord(ch) < 32 and ch not in "\n\r\t"):
            garbage += 1

    unique_ratio = float(len(set(generated)) / max(1, len(generated))) if generated else 0.0
    bigrams = list(zip(generated[:-1], generated[1:])) if len(generated) >= 2 else []
    repeat_bigram_ratio = 0.0
    if bigrams:
        repeat_bigram_ratio = 1.0 - (len(set(bigrams)) / float(len(bigrams)))

    max_consecutive_repeat = 1 if generated else 0
    cur = 1
    for i in range(1, len(generated)):
        if generated[i] == generated[i - 1]:
            cur += 1
            max_consecutive_repeat = max(max_consecutive_repeat, cur)
        else:
            cur = 1

    return {
        "context_tokens": len(context_tokens),
        "prefill_seconds": prefill_sec,
        "prefill_tps": (len(prefill_ids) / prefill_sec) if prefill_sec > 0 else 0.0,
        "decode_tokens": len(generated),
        "decode_seconds": decode_sec,
        "decode_tps": (len(generated) / decode_sec) if decode_sec > 0 else 0.0,
        "hardware": {
            "before": before,
            "after": after,
            "vram_delta_bytes": (after.get("vram_used_bytes") - before.get("vram_used_bytes"))
            if isinstance(after.get("vram_used_bytes"), int) and isinstance(before.get("vram_used_bytes"), int)
            else None,
        },
        "output_sample": text,
        "output_garbage_ratio": float(garbage / max(1, len(text))),
        "output_meaningful": bool((len(text.strip()) > 16) and (garbage == 0)),
        "long_decode_stability": {
            "unique_ratio": unique_ratio,
            "repeat_bigram_ratio": float(repeat_bigram_ratio),
            "max_consecutive_repeat": int(max_consecutive_repeat),
            "stability_pass": bool(unique_ratio >= 0.12 and repeat_bigram_ratio <= 0.92 and max_consecutive_repeat <= 24),
        },
    }


def _kv_memory_estimate_bytes(context_tokens: int, layers: int, num_kv_heads: int, head_dim: int, block_size: int = 32) -> dict:
    kv_fp16 = context_tokens * layers * num_kv_heads * head_dim * 2 * 2

    packed_head_bytes = (head_dim * 3 + 7) // 8
    blocks_per_head = (head_dim + block_size - 1) // block_size
    scales_bytes_per_head = blocks_per_head * 4

    kv_int3_per_head = (packed_head_bytes + scales_bytes_per_head) * 2
    kv_int3 = context_tokens * layers * num_kv_heads * kv_int3_per_head

    return {
        "llamacpp_kv_fp16_bytes": int(kv_fp16),
        "vspec_kv_int3_bytes": int(kv_int3),
        "kv_savings_percent_vs_fp16": float((kv_fp16 - kv_int3) * 100.0 / max(1, kv_fp16)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Job3 long-context eval (4k/8k, VRAM/TPS, KV INT3 impact)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lang", default="vi")
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default=str(ROOT / "logs" / "job3_long_context_eval.json"))
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

    prompt_for_model = build_prompt(args.prompt, adapter.model_type, tok_cfg, args.lang, args.chat_format)
    base_seed_text = prompt_for_model + "\n" + "Hãy trả lời mạch lạc, không lặp, đúng ngữ nghĩa."

    t0 = time.perf_counter()
    runtime_base = _build_runtime(config, weight_index, args.max_layers, fused_bits=0)
    runtime_int4 = _build_runtime(config, weight_index, args.max_layers, fused_bits=4)
    if runtime_base is None or runtime_int4 is None:
        raise RuntimeError("runtime init failed")

    ctx_4k = _build_target_context_tokens(tokenizer, base_seed_text, 4096)
    ctx_8k = _build_target_context_tokens(tokenizer, base_seed_text, 8192)

    ppl_base = _perplexity_teacher_forcing(runtime_base, ctx_4k, args.max_eval_tokens)
    ppl_int4 = _perplexity_teacher_forcing(runtime_int4, ctx_4k, args.max_eval_tokens)
    drift = _logit_drift(runtime_base, runtime_int4, ctx_4k, args.max_eval_tokens)

    run_4k = _run_context(runtime_int4, ctx_4k, args.decode_steps, adapter.eos_token_id, tokenizer, args.lang, args.seed)
    run_8k = _run_context(runtime_int4, ctx_8k, args.decode_steps, adapter.eos_token_id, tokenizer, args.lang, args.seed)

    num_heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 32)
    num_kv_heads = int(config.get("num_key_value_heads", 0) or config.get("n_kv_head", 0) or num_heads)
    hidden = int(config.get("hidden_size", 0) or config.get("n_embd", 0) or 4096)
    head_dim = hidden // max(1, num_heads)

    kv_4k = _kv_memory_estimate_bytes(4096, args.max_layers, num_kv_heads, head_dim)
    kv_8k = _kv_memory_estimate_bytes(8192, args.max_layers, num_kv_heads, head_dim)

    def estimate_total_saving(run_ctx: dict, kv_est: dict) -> dict:
        measured_total = run_ctx["hardware"]["after"].get("vram_used_bytes")
        if not isinstance(measured_total, int):
            return {
                "estimated_total_llamacpp_bytes": None,
                "estimated_total_vspec_int3kv_bytes": None,
                "estimated_total_savings_percent": None,
            }

        kv_fp16 = kv_est["llamacpp_kv_fp16_bytes"]
        kv_int3 = kv_est["vspec_kv_int3_bytes"]
        non_kv = max(0, measured_total - kv_fp16)
        total_llama = non_kv + kv_fp16
        total_vspec = non_kv + kv_int3
        saving = (total_llama - total_vspec) * 100.0 / max(1, total_llama)
        return {
            "estimated_total_llamacpp_bytes": int(total_llama),
            "estimated_total_vspec_int3kv_bytes": int(total_vspec),
            "estimated_total_savings_percent": float(saving),
        }

    cmp_4k = estimate_total_saving(run_4k, kv_4k)
    cmp_8k = estimate_total_saving(run_8k, kv_8k)

    report = {
        "job": "job3",
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "runtime": {
            "baseline_fused_bits": int(getattr(runtime_base, "fused_bits", 0)),
            "int4_fused_bits": int(getattr(runtime_int4, "fused_bits", 0)),
            "forced_3bit": bool(int(getattr(runtime_int4, "fused_bits", 0)) == 3),
        },
        "pipeline_note": "INT3 weight storage -> on-device expand INT4 -> INT4 compute path is enabled in CUDA dispatch for INT3 weights.",
        "quality_metrics_4k": {
            "perplexity_base": float(ppl_base),
            "perplexity_int4": float(ppl_int4),
            "perplexity_delta": float(ppl_int4 - ppl_base),
            "perplexity_delta_ratio": float(abs(ppl_int4 - ppl_base) / max(ppl_base, 1e-9)),
            "logit_drift": drift,
        },
        "context_4k": run_4k,
        "context_8k": run_8k,
        "kv_int3_estimate": {
            "context_4k": kv_4k,
            "context_8k": kv_8k,
        },
        "llamacpp_comparison_estimate": {
            "context_4k": cmp_4k,
            "context_8k": cmp_8k,
        },
        "tps_impact": {
            "prefill_tps_ratio_8k_vs_4k": float(run_8k["prefill_tps"] / max(run_4k["prefill_tps"], 1e-9)),
            "decode_tps_ratio_8k_vs_4k": float(run_8k["decode_tps"] / max(run_4k["decode_tps"], 1e-9)),
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
