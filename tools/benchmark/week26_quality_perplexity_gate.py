from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
sys.path.insert(0, str(CHAT_PY))

from chat_prompt import build_prompt
from language_stability_guard import LanguageStabilityGuard
from model_adapters import select_adapter
from model_loader import build_weight_index, collect_tensor_names, find_snapshot_dir, load_tokenizer, read_config, read_tokenizer_config
from runtime_inference import build_generic_runtime
from hardware_telemetry import build_hardware_report, capture_hardware_snapshot, summarize_hardware_usage


@dataclass
class EvalCase:
    lang: str
    prompt: str


def _resolve_profile(profile: str) -> tuple[int, int, int, list[EvalCase]]:
    p = (profile or "quick").strip().lower()
    if p == "full":
        long_en = " ".join(["Vspec runtime focuses on efficient low-bit inference."] * 24)
        long_vi = " ".join(["Vspec runtime tập trung vào suy luận low-bit hiệu quả."] * 20)
        return (
            8,
            48,
            16,
            [
                EvalCase("en", f"Summarize briefly: {long_en}"),
                EvalCase("vi", f"Tóm tắt ngắn: {long_vi}"),
                EvalCase("zh", "请用一句话总结：Vspec运行时专注于低比特原生推理。"),
            ],
        )
    if p == "standard":
        return (
            6,
            24,
            10,
            [
                EvalCase("en", "One short sentence about Vspec native low-bit runtime."),
                EvalCase("vi", "Một câu ngắn về runtime native low-bit của Vspec."),
                EvalCase("zh", "用一句话说明Vspec低比特原生推理运行时。"),
            ],
        )
    return (
        4,
        20,
        8,
        [
            EvalCase("en", "One short sentence about Vspec low-bit runtime."),
            EvalCase("vi", "Một câu ngắn về runtime low-bit của Vspec."),
        ],
    )


def _reset_runtime(runtime) -> None:
    if hasattr(runtime, "cache_k"):
        runtime.cache_k = []
    if hasattr(runtime, "cache_v"):
        runtime.cache_v = []
    if hasattr(runtime, "position"):
        runtime.position = 0


def _build_runtime(config: dict, weight_index: dict, max_layers: int, fused_bits: int):
    os.environ["VSPEC_FUSED_BITS"] = str(fused_bits)
    os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "0"
    return build_generic_runtime(config, weight_index, max_layers=max_layers, device="cuda-native")


def _nll_from_logits(logits: list[float], target_id: int) -> float:
    if not logits or target_id < 0 or target_id >= len(logits):
        return 0.0
    arr = np.asarray(logits, dtype=np.float32)
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
    logits = runtime.forward_logits([int(token_ids[0])])
    nll_total = 0.0

    for i in range(1, steps + 1):
        target = int(token_ids[i])
        nll_total += _nll_from_logits(logits, target)
        logits = runtime.forward_logits([target])

    if steps == 0:
        return 0.0
    return float(math.exp(nll_total / float(steps)))


def _generate_greedy(runtime, token_ids: list[int], eos_token_id: int | None, max_steps: int) -> list[int]:
    _reset_runtime(runtime)
    if len(token_ids) > 1:
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens([int(t) for t in token_ids[:-1]])
        else:
            for t in token_ids[:-1]:
                runtime.forward_logits([int(t)])

    ids = [int(t) for t in token_ids]
    out = []
    for _ in range(max_steps):
        logits = runtime.forward_logits([ids[-1]])
        if not logits:
            break
        nxt = int(np.argmax(np.asarray(logits, dtype=np.float32)))
        out.append(nxt)
        ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return out


def _select_next_token(logits: list[float], tokenizer, guard: LanguageStabilityGuard, top_n: int = 128) -> int:
    if not logits:
        return 0
    arr = np.asarray(logits, dtype=np.float32)
    n = min(top_n, int(arr.shape[0]))
    if n <= 0:
        return int(np.argmax(arr))

    idx = np.argpartition(arr, -n)[-n:]
    idx = idx[np.argsort(arr[idx])[::-1]]

    best_tid = None
    best_score = None
    for tid in idx:
        text = tokenizer.decode([int(tid)]).strip()
        if not text or "�" in text:
            continue
        if any((ord(ch) < 32 and ch not in ("\t", "\n", "\r")) for ch in text):
            continue

        score = float(arr[int(tid)]) + float(guard.score_adjustment(text))
        if best_tid is None or score > best_score:
            best_tid = int(tid)
            best_score = score

        if guard.allow_text(text):
            return int(tid)

    if best_tid is not None:
        return best_tid
    return int(idx[0]) if len(idx) > 0 else int(np.argmax(arr))


def _generate_guarded(
    runtime,
    token_ids: list[int],
    eos_token_id: int | None,
    max_steps: int,
    tokenizer,
    prompt: str,
    lang: str,
) -> list[int]:
    _reset_runtime(runtime)
    if len(token_ids) > 1:
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens([int(t) for t in token_ids[:-1]])
        else:
            for t in token_ids[:-1]:
                runtime.forward_logits([int(t)])

    ids = [int(t) for t in token_ids]
    out = []
    prioritize_english = lang not in {"zh", "ja", "ko"}
    strictness = 0.72 if prioritize_english else 0.62
    guard = LanguageStabilityGuard(prompt=prompt, lang_mode=lang, strictness=strictness, prioritize_english=prioritize_english)
    for _ in range(max_steps):
        logits = runtime.forward_logits([ids[-1]])
        if not logits:
            break
        nxt = _select_next_token(logits, tokenizer, guard, top_n=128)
        out.append(nxt)
        ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return out


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _eval_case(
    case: EvalCase,
    tokenizer,
    adapter,
    tok_cfg: dict,
    baseline_rt,
    lowbit_rt,
    chat_format: str,
    max_eval_tokens: int,
    max_gen_tokens: int,
) -> dict:
    prompt_for_model = build_prompt(case.prompt, adapter.model_type, tok_cfg, case.lang, chat_format)
    token_ids = list(tokenizer.encode(prompt_for_model).ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    ppl_base = _perplexity_teacher_forcing(baseline_rt, token_ids, max_eval_tokens)
    ppl_low = _perplexity_teacher_forcing(lowbit_rt, token_ids, max_eval_tokens)
    drift = abs(ppl_low - ppl_base) / max(ppl_base, 1e-9)

    gen_base_ids = _generate_guarded(
        baseline_rt,
        token_ids,
        adapter.eos_token_id,
        max_gen_tokens,
        tokenizer,
        case.prompt,
        case.lang,
    )
    gen_low_ids = _generate_guarded(
        lowbit_rt,
        token_ids,
        adapter.eos_token_id,
        max_gen_tokens,
        tokenizer,
        case.prompt,
        case.lang,
    )
    txt_base = tokenizer.decode(gen_base_ids).strip()
    txt_low = tokenizer.decode(gen_low_ids).strip()

    sim = SequenceMatcher(None, _normalize_text(txt_base), _normalize_text(txt_low)).ratio()

    prioritize_english = case.lang not in {"zh", "ja", "ko"}
    strictness = 0.72 if prioritize_english else 0.62
    guard = LanguageStabilityGuard(prompt=case.prompt, lang_mode=case.lang, strictness=strictness, prioritize_english=prioritize_english)
    guard_base_ok = bool(guard.allow_text(txt_base))
    guard_low_ok = bool(guard.allow_text(txt_low))

    return {
        "lang": case.lang,
        "prompt_tokens": len(token_ids),
        "perplexity_baseline": ppl_base,
        "perplexity_lowbit": ppl_low,
        "perplexity_drift_ratio": drift,
        "text_similarity": sim,
        "guard_baseline_ok": guard_base_ok,
        "guard_lowbit_ok": guard_low_ok,
        "text_baseline": txt_base,
        "text_lowbit": txt_low,
    }


def _stability_gate(case_lang: str, similarity: float, threshold: float, guard_base_ok: bool, guard_low_ok: bool, drift_pass: bool) -> bool:
    if similarity >= threshold:
        return True
    if guard_base_ok and guard_low_ok:
        return True
    if case_lang in {"zh", "ja", "ko"} and guard_base_ok and drift_pass:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 26 quality guardrail + perplexity gate")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile", default="quick", choices=["quick", "standard", "full"], help="quick=fast iteration, standard=balanced, full=roadmap-like")
    parser.add_argument("--max-layers", type=int, default=0, help="0 = use profile default")
    parser.add_argument("--max-eval-tokens", type=int, default=0, help="0 = use profile default")
    parser.add_argument("--max-gen-tokens", type=int, default=0, help="0 = use profile default")
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--drift-threshold", type=float, default=4.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.10)
    parser.add_argument("--output", default=str(ROOT / "logs" / "week26_quality_perplexity_gate.json"))
    args = parser.parse_args()

    prof_layers, prof_eval, prof_gen, prof_cases = _resolve_profile(args.profile)
    max_layers = args.max_layers if args.max_layers > 0 else prof_layers
    max_eval_tokens = args.max_eval_tokens if args.max_eval_tokens > 0 else prof_eval
    max_gen_tokens = args.max_gen_tokens if args.max_gen_tokens > 0 else prof_gen

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
    baseline_rt = _build_runtime(config, weight_index, max_layers=max_layers, fused_bits=0)
    t_build1 = time.perf_counter()
    lowbit_rt = _build_runtime(config, weight_index, max_layers=max_layers, fused_bits=3)
    t_build2 = time.perf_counter()
    if baseline_rt is None or lowbit_rt is None:
        raise RuntimeError("runtime init failed")

    cases = prof_cases

    outputs = []
    drift_pass = 0
    stability_pass = 0
    t_eval0 = time.perf_counter()
    for c in cases:
        r = _eval_case(
            case=c,
            tokenizer=tokenizer,
            adapter=adapter,
            tok_cfg=tok_cfg,
            baseline_rt=baseline_rt,
            lowbit_rt=lowbit_rt,
            chat_format=args.chat_format,
            max_eval_tokens=max_eval_tokens,
            max_gen_tokens=max_gen_tokens,
        )
        r["drift_pass"] = bool(r["perplexity_drift_ratio"] <= args.drift_threshold)
        r["stability_pass"] = bool(
            _stability_gate(
                case_lang=r["lang"],
                similarity=float(r["text_similarity"]),
                threshold=float(args.similarity_threshold),
                guard_base_ok=bool(r["guard_baseline_ok"]),
                guard_low_ok=bool(r["guard_lowbit_ok"]),
                drift_pass=bool(r["drift_pass"]),
            )
        )
        if r["drift_pass"]:
            drift_pass += 1
        if r["stability_pass"]:
            stability_pass += 1
        outputs.append(r)
    t_eval1 = time.perf_counter()

    hw_after = capture_hardware_snapshot(runtime=lowbit_rt, backend_hint="cuda-native")
    wall_t1 = time.perf_counter()
    hardware = build_hardware_report(hw_before, hw_after)

    total = len(outputs)
    report = {
        "week": 26,
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "profile": args.profile,
        "max_layers": int(max_layers),
        "max_eval_tokens": int(max_eval_tokens),
        "max_gen_tokens": int(max_gen_tokens),
        "drift_threshold": float(args.drift_threshold),
        "similarity_threshold": float(args.similarity_threshold),
        "cases": outputs,
        "drift_pass_count": drift_pass,
        "stability_pass_count": stability_pass,
        "total_cases": total,
        "kpi_perplexity_gate_pass": bool(drift_pass == total),
        "kpi_quality_stable_pass": bool(stability_pass == total),
        "kpi_week26_pass": bool(drift_pass == total and stability_pass == total),
        "timing": {
            "build_baseline_sec": float(t_build1 - t_build0),
            "build_lowbit_sec": float(t_build2 - t_build1),
            "eval_sec": float(t_eval1 - t_eval0),
            "total_sec": float(wall_t1 - wall_t0),
        },
        "hardware": hardware,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[hardware] {summarize_hardware_usage(hw_after)}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
