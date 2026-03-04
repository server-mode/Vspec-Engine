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
        return {
            "steps": 0,
            "mean_l2": 0.0,
            "mean_l1": 0.0,
            "max_l2": 0.0,
        }

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

        l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(db), 1e-6))
        l1 = float(np.mean(np.abs(diff)))
        l2_vals.append(l2)
        l1_vals.append(l1)

        nxt = int(token_ids[i])
        logits_b = _forward_logits_array(runtime_base, nxt)
        logits_q = _forward_logits_array(runtime_int4, nxt)

    if not l2_vals:
        return {
            "steps": 0,
            "mean_l2": 0.0,
            "mean_l1": 0.0,
            "max_l2": 0.0,
        }

    return {
        "steps": len(l2_vals),
        "mean_l2": float(np.mean(l2_vals)),
        "mean_l1": float(np.mean(l1_vals)),
        "max_l2": float(np.max(l2_vals)),
    }


def _decode(runtime, token_ids: list[int], decode_steps: int, eos_token_id: int | None) -> tuple[list[int], float]:
    _reset_runtime(runtime)
    if len(token_ids) > 1:
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens([int(t) for t in token_ids[:-1]])
        else:
            for t in token_ids[:-1]:
                runtime.forward_logits([int(t)])

    ids = [int(t) for t in token_ids]
    generated: list[int] = []
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        logits = _forward_logits_array(runtime, ids[-1])
        if logits.size == 0:
            break
        nxt = int(np.argmax(logits))
        generated.append(nxt)
        ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    sec = time.perf_counter() - t0
    return generated, sec


def _long_decode_stability(token_ids: list[int], generated: list[int], target_steps: int) -> dict:
    gen_n = len(generated)
    if gen_n == 0:
        return {
            "generated": 0,
            "target_steps": target_steps,
            "unique_ratio": 0.0,
            "repeat_bigram_ratio": 1.0,
            "max_consecutive_repeat": 0,
            "stability_pass": False,
        }

    unique_ratio = float(len(set(generated)) / max(gen_n, 1))

    bigrams = list(zip(generated[:-1], generated[1:])) if gen_n >= 2 else []
    repeat_bigram_ratio = 0.0
    if bigrams:
        repeat_bigram_ratio = 1.0 - (len(set(bigrams)) / float(len(bigrams)))

    max_consecutive_repeat = 1
    cur = 1
    for i in range(1, gen_n):
        if generated[i] == generated[i - 1]:
            cur += 1
            if cur > max_consecutive_repeat:
                max_consecutive_repeat = cur
        else:
            cur = 1

    enough_length = gen_n >= int(target_steps * 0.85)
    stability_pass = bool(
        enough_length
        and unique_ratio >= 0.12
        and repeat_bigram_ratio <= 0.92
        and max_consecutive_repeat <= 24
    )

    return {
        "generated": gen_n,
        "target_steps": target_steps,
        "unique_ratio": unique_ratio,
        "repeat_bigram_ratio": float(repeat_bigram_ratio),
        "max_consecutive_repeat": int(max_consecutive_repeat),
        "stability_pass": stability_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Job1 Phase3 eval: output + perplexity delta + logit drift + long decode stability")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lang", default="vi")
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--max-eval-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--long-decode-steps", type=int, default=256)
    parser.add_argument("--output", default=str(ROOT / "logs" / "job1_phase3_eval.json"))
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
    token_ids = list(tokenizer.encode(prompt_for_model).ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    t0 = time.perf_counter()
    runtime_base = _build_runtime(config, weight_index, args.max_layers, fused_bits=0)
    runtime_int4 = _build_runtime(config, weight_index, args.max_layers, fused_bits=4)
    if runtime_base is None or runtime_int4 is None:
        raise RuntimeError("runtime init failed")

    ppl_base = _perplexity_teacher_forcing(runtime_base, token_ids, args.max_eval_tokens)
    ppl_int4 = _perplexity_teacher_forcing(runtime_int4, token_ids, args.max_eval_tokens)
    ppl_delta = float(ppl_int4 - ppl_base)
    ppl_delta_ratio = float(abs(ppl_delta) / max(ppl_base, 1e-9))

    drift = _logit_drift(runtime_base, runtime_int4, token_ids, args.max_eval_tokens)

    gen_base, sec_base = _decode(runtime_base, token_ids, args.decode_steps, adapter.eos_token_id)
    gen_int4, sec_int4 = _decode(runtime_int4, token_ids, args.decode_steps, adapter.eos_token_id)

    long_gen_int4, long_sec_int4 = _decode(runtime_int4, token_ids, args.long_decode_steps, adapter.eos_token_id)
    long_stability = _long_decode_stability(token_ids, long_gen_int4, args.long_decode_steps)

    text_base = tokenizer.decode(gen_base).strip()
    text_int4 = tokenizer.decode(gen_int4).strip()
    text_int4_long = tokenizer.decode(long_gen_int4).strip()

    report = {
        "job": "job1",
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "prompt": args.prompt,
        "prompt_tokens": len(token_ids),
        "max_layers": int(args.max_layers),
        "runtime": {
            "baseline_fused_bits": int(getattr(runtime_base, "fused_bits", 0)),
            "int4_fused_bits": int(getattr(runtime_int4, "fused_bits", 0)),
        },
        "output": {
            "baseline_text": text_base,
            "int4_text": text_int4,
            "int4_long_text": text_int4_long,
        },
        "throughput": {
            "baseline_decode_tokens": len(gen_base),
            "baseline_decode_seconds": sec_base,
            "baseline_tps": (len(gen_base) / sec_base) if sec_base > 0 else 0.0,
            "int4_decode_tokens": len(gen_int4),
            "int4_decode_seconds": sec_int4,
            "int4_tps": (len(gen_int4) / sec_int4) if sec_int4 > 0 else 0.0,
            "int4_long_decode_tokens": len(long_gen_int4),
            "int4_long_decode_seconds": long_sec_int4,
            "int4_long_tps": (len(long_gen_int4) / long_sec_int4) if long_sec_int4 > 0 else 0.0,
        },
        "perplexity": {
            "baseline": ppl_base,
            "int4": ppl_int4,
            "delta": ppl_delta,
            "delta_ratio": ppl_delta_ratio,
        },
        "logit_drift": drift,
        "long_decode_stability": long_stability,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
