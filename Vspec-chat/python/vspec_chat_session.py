import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional

from chat_prompt import build_prompt
from fast_output import FastOutputEngine, postprocess_output_text, resolve_speed_preset
from decode_optimization_module import DecodeOptimizationModule
from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from runtime_meaningful_response import RuntimeMeaningfulResponseAssurance
from runtime_lowbit_module import lowbit_projection_stats_reset, lowbit_projection_stats_snapshot
from runtime_threebit_module import ThreeBitRuntimeModule
from decode_contract import sanitize_and_validate_logits
from native_safe_decode import build_native_safe_runtime
from runtime_core_bridge import CoreDecodeSession, adaptive_step
from model_adapters import select_adapter
from model_loader import (
    build_weight_index,
    collect_tensor_names,
    find_snapshot_dir,
    load_tokenizer,
    read_config,
    read_tokenizer_config,
)
from runtime_inference import build_generic_runtime, runtime_matrix_bits_summary
from vspec_chat import _detect_lang, _is_runtime_fallback_text, _looks_gibberish_output
from lowbit_policy import build_layer_bits, effective_bits, summarize_layer_bits


def _resolve_quality_layer_floor(config: dict, requested_max_layers: int, device: str, unsafe_low_layers: bool) -> int:
    try:
        total_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
    except Exception:
        total_layers = 0
    if total_layers <= 0:
        return requested_max_layers
    if requested_max_layers <= 0:
        return requested_max_layers
    if unsafe_low_layers:
        return requested_max_layers
    if device not in {"cuda", "cuda-native", "torch-cuda"}:
        return requested_max_layers

    floor = 8
    if total_layers >= 40:
        floor = 12
    if total_layers >= 56:
        floor = 16

    if requested_max_layers < floor:
        return floor
    return requested_max_layers



def _progress(enabled: bool, pct: int, stage: str, detail: str = "") -> None:
    if not enabled:
        return
    msg = f"[session] {pct:>3}% | {stage}"
    if detail:
        msg += f" | {detail}"
    print(msg)



def _generate(
    prompt: str,
    tokenizer,
    adapter,
    runtime,
    tok_cfg,
    max_tokens: int,
    temperature: float,
    top_k: int,
    greedy: bool,
    lang_mode: str,
    lang_top_n: int,
    repetition_penalty: float,
    repeat_window: int,
    no_repeat_ngram: int,
    chat_format: str,
    stream: bool,
    show_progress: bool,
    disable_language_guard: bool,
    language_guard_strictness: float,
    prioritize_english: bool,
    structure_guard_strictness: float,
    disable_structure_guard: bool,
    decode_opt_mode: str,
    threebit_module: ThreeBitRuntimeModule,
    allow_semantic_rescue: bool,
    max_decode_seconds: float,
    max_retry_seconds: float,
    native_safe_fallback_builder: Optional[Callable[[], object]],
    torch_timeout_fallback_builder: Optional[Callable[[], object]],
) -> str:
    lowbit_projection_stats_reset()

    prompt_for_model = build_prompt(prompt, adapter.model_type, tok_cfg, lang_mode, chat_format)
    encoded = tokenizer.encode(prompt_for_model)
    token_ids = list(encoded.ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    assurance = RuntimeMeaningfulResponseAssurance(lang_mode, allow_semantic_rescue=allow_semantic_rescue)
    prompt_chars = len((prompt or "").strip())

    def _entropy_from_logits(logits_obj) -> float:
        try:
            vals = logits_obj.tolist() if hasattr(logits_obj, "tolist") else list(logits_obj)
            if not vals:
                return 0.0
            max_v = max(float(v) for v in vals)
            exps = [math.exp(max(-60.0, min(60.0, float(v) - max_v))) for v in vals]
            denom = max(1e-12, sum(exps))
            probs = [v / denom for v in exps]
            return float(-sum(p * math.log(max(1e-12, p)) for p in probs))
        except Exception:
            return 0.0

    def _is_meaningful_partial(candidate: str, generated_tokens: int) -> bool:
        out = (candidate or "").strip()
        if lang_mode == "en" and prioritize_english:
            letters = sum(1 for ch in out if ch.isalpha())
            words = [w for w in out.split() if w]
            if len(out) >= 12 and letters >= 6 and len(words) >= 2:
                return True
            if generated_tokens >= 8 and letters >= 6:
                return True
        if len(out) < 24:
            return False
        letters = sum(1 for ch in out if ch.isalpha())
        if (letters / max(1, len(out))) < 0.55:
            return False
        words = [w for w in out.split() if w]
        if len(words) < 4:
            return False
        return True

    def _resolve_decode_budget_seconds(prefill_tokens: int, layer_count: int) -> float:
        raw = float(max_decode_seconds)
        if raw > 0.0:
            return max(0.5, raw)
        if raw == 0.0:
            return 0.0
        work_units = max(1, (max(0, int(prefill_tokens)) + max(1, int(max_tokens))) * max(1, int(layer_count)))
        auto = 16.0 + (0.18 * float(max_tokens)) + (0.012 * float(prompt_chars)) + (0.008 * float(work_units))
        if max_tokens <= 64:
            auto = max(auto, 48.0)
        if lang_mode == "en" and prioritize_english:
            auto = max(auto, 60.0)
        return min(240.0, max(20.0, auto))

    def _resolve_retry_budget_seconds(decode_budget: float) -> float:
        raw = float(max_retry_seconds)
        if raw > 0.0:
            return max(0.5, raw)
        if raw == 0.0:
            return 0.0
        if decode_budget <= 0.0:
            return 10.0
        auto = decode_budget * 0.55
        return min(72.0, max(8.0, auto))

    def _resolve_budget_step_cap(requested_steps: int, decode_budget_seconds: float, prefill_tokens: int, layer_count: int) -> int:
        if requested_steps <= 0:
            return 1
        if decode_budget_seconds <= 0.0:
            return int(requested_steps)

        lowbit_plan = getattr(runtime, "lowbit_plan", None)
        lowbit_enabled = bool(getattr(lowbit_plan, "enabled", False))
        fused_bits = int(getattr(runtime, "fused_bits", 0) or 0)

        # Conservative latency estimate (seconds/token) to prevent hard timeout loops.
        token_latency = 0.012 + (0.0018 * float(max(1, int(layer_count))))
        if lowbit_enabled and fused_bits in {3, 4}:
            token_latency += 0.040
        elif lowbit_enabled:
            token_latency += 0.025
        if int(prefill_tokens) >= 256:
            token_latency *= 1.12
        token_latency = max(0.010, min(2.000, token_latency))

        reserve = max(3.0, min(20.0, decode_budget_seconds * 0.15))
        usable = max(0.5, decode_budget_seconds - reserve)
        cap = int(usable / token_latency)
        cap = max(12, cap)
        return max(1, min(int(requested_steps), cap))

    def _should_quality_retry(candidate: str) -> bool:
        repaired = assurance.repair(candidate, prompt)
        if _is_runtime_fallback_text(repaired, lang_mode):
            return True
        out = (repaired or "").strip()
        if not out or "�" in out:
            return True
        if lang_mode == "en" and prioritize_english:
            if len(out) < 12:
                return True
            return _looks_gibberish_output(repaired)
        return _looks_gibberish_output(repaired)

    def _prefill_runtime(runtime_obj, local_tokens: list[int]) -> None:
        if not hasattr(runtime_obj, "cache_k"):
            return

        def _cache_integrity_ok(expected_len: int) -> bool:
            if expected_len <= 0:
                return True
            if not hasattr(runtime_obj, "cache_len"):
                return True
            try:
                cache_len_list = list(getattr(runtime_obj, "cache_len") or [])
                layer_count = len(getattr(runtime_obj, "layers", []) or [])
                if layer_count <= 0:
                    return True
                if len(cache_len_list) < layer_count:
                    return False
                return all(int(cache_len_list[i]) == int(expected_len) for i in range(layer_count))
            except Exception:
                return False

        runtime_obj.cache_k = []
        runtime_obj.cache_v = []
        if hasattr(runtime_obj, "reset_core_kv_mirrors"):
            try:
                runtime_obj.reset_core_kv_mirrors()
            except Exception:
                pass
        if hasattr(runtime_obj, "cache_len"):
            runtime_obj.cache_len = []
        runtime_obj.position = 0

        prefill = local_tokens[:-1]
        total_prefill = len(prefill)
        if hasattr(runtime_obj, "prefill_tokens"):
            runtime_obj.prefill_tokens(prefill)
            if total_prefill > 0:
                _progress(show_progress, 45, "prefill", f"{total_prefill}/{total_prefill}")
            if _cache_integrity_ok(total_prefill):
                if hasattr(runtime_obj, "cache_len"):
                    cache_len_list = list(getattr(runtime_obj, "cache_len") or [])
                    if cache_len_list:
                        print(f"[session] prefill_cache_len_min= {int(min(cache_len_list))}")
                        print(f"[session] prefill_cache_len_max= {int(max(cache_len_list))}")
                return

            print("[session] prefill_cache_integrity= failed; replaying token-by-token")
            if hasattr(runtime_obj, "reset_core_kv_mirrors"):
                try:
                    runtime_obj.reset_core_kv_mirrors()
                except Exception:
                    pass
            runtime_obj.cache_k = []
            runtime_obj.cache_v = []
            if hasattr(runtime_obj, "cache_len"):
                runtime_obj.cache_len = []
            runtime_obj.position = 0
            for idx, tid in enumerate(prefill, start=1):
                runtime_obj.forward_logits([tid])
                if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                    pct = 10 + int((idx / total_prefill) * 35)
                    _progress(show_progress, min(45, pct), "prefill-replay", f"{idx}/{total_prefill}")
            if hasattr(runtime_obj, "cache_len"):
                cache_len_list = list(getattr(runtime_obj, "cache_len") or [])
                if cache_len_list:
                    print(f"[session] prefill_cache_len_min= {int(min(cache_len_list))}")
                    print(f"[session] prefill_cache_len_max= {int(max(cache_len_list))}")
            return

        for idx, tid in enumerate(prefill, start=1):
            runtime_obj.forward_logits([tid])
            if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                pct = 10 + int((idx / total_prefill) * 35)
                _progress(show_progress, min(45, pct), "prefill", f"{idx}/{total_prefill}")
        if hasattr(runtime_obj, "cache_len"):
            cache_len_list = list(getattr(runtime_obj, "cache_len") or [])
            if cache_len_list:
                print(f"[session] prefill_cache_len_min= {int(min(cache_len_list))}")
                print(f"[session] prefill_cache_len_max= {int(max(cache_len_list))}")

    def _decode_once(
        runtime_obj,
        local_tokens: list[int],
        local_temperature: float,
        local_top_k: int,
        local_lang_top_n: int,
        local_greedy: bool,
        local_rep_penalty: float,
        local_repeat_window: int,
        local_no_repeat_ngram: int,
        local_stream: bool,
        decode_budget_seconds: float,
        max_steps: int,
        disable_structure_guard_local: bool,
    ) -> tuple[str, int, float, dict | None, bool, bool]:
        debug_logits = os.getenv("VSPEC_DEBUG_LOGITS", "0").strip().lower() in {"1", "true", "yes", "on"}

        def _logits_health(logits_obj, step_idx: int) -> None:
            if not debug_logits:
                return
            try:
                vals = logits_obj.tolist() if hasattr(logits_obj, "tolist") else list(logits_obj)
                if not vals:
                    print(f"[session][logits] step={step_idx} empty=True")
                    return
                min_v = float(min(vals))
                max_v = float(max(vals))
                finite = True
                for v in vals:
                    fv = float(v)
                    if (fv != fv) or (fv == float("inf")) or (fv == float("-inf")):
                        finite = False
                        break
                print(f"[session][logits] step={step_idx} finite={finite} min={min_v:.6f} max={max_v:.6f}")
            except Exception as exc:
                print(f"[session][logits] step={step_idx} stats_failed={exc}")

        guard = None
        if not disable_language_guard:
            guard = LanguageStabilityGuard(
                prompt=prompt,
                lang_mode=lang_mode,
                strictness=language_guard_strictness,
                prioritize_english=prioritize_english,
            )

        structure_guard = None
        if not disable_structure_guard_local:
            structure_guard = LanguageStructureIntegrityManager(prompt=prompt, strictness=structure_guard_strictness)

        engine = FastOutputEngine(
            tokenizer=tokenizer,
            lang_mode=lang_mode,
            stream=local_stream,
            guard=guard,
            structure_guard=structure_guard,
        )
        decode_optimizer = DecodeOptimizationModule(
            repetition_penalty=local_rep_penalty,
            repeat_window=local_repeat_window,
            no_repeat_ngram=local_no_repeat_ngram,
            mode=decode_opt_mode,
        )
        decode_optimizer.seed_history(local_tokens)

        generated: list[int] = []
        start = time.perf_counter()
        engine.begin_stream()
        timed_out = False
        contract_failed = False

        total_steps = max(1, int(max_steps))
        core_decode = CoreDecodeSession.from_runtime(runtime_obj, total_steps)
        scheduler_enabled = core_decode.begin(prompt_tokens=max(0, len(local_tokens) - 1), max_new_tokens=total_steps)
        adaptive_entropy_prev = 0.0
        adaptive_latency_ms = 0.0
        adaptive_quality_drift = 0.0

        try:
            for step in range(total_steps):
                if scheduler_enabled:
                    quota = core_decode.next_quota()
                    if quota <= 0:
                        break
                else:
                    quota = 1

                elapsed = time.perf_counter() - start
                if decode_budget_seconds > 0.0 and elapsed >= decode_budget_seconds:
                    timed_out = True
                    break

                reached_eos = False
                for _ in range(quota):
                    token_t0 = time.perf_counter()
                    logits = decode_optimizer.fetch_logits(runtime_obj, local_tokens[-1], tokenizer.get_vocab_size())
                    logits, contract = sanitize_and_validate_logits(logits, tokenizer.get_vocab_size())
                    if not contract.ok:
                        print(
                            f"[session] decode_contract_ok= False reason={contract.reason} logits_len={contract.logits_len} expected_vocab={contract.expected_vocab_size}"
                        )
                        contract_failed = True
                        break
                    if step == 0 and contract.masked_tail > 0:
                        print(f"[session] decode_contract_masked_tail= {contract.masked_tail}")
                    if step == 0:
                        _logits_health(logits, step)
                    if decode_optimizer.logits_empty(logits):
                        if debug_logits:
                            print(f"[session][logits] step={step} empty_after_fetch=True")
                        break

                    logits = threebit_module.denoise_logits(logits, step)
                    logits = decode_optimizer.apply_generation_controls(logits, local_tokens)

                    entropy_now = _entropy_from_logits(logits)
                    entropy_collapse = max(0.0, adaptive_entropy_prev - entropy_now)
                    adaptive_entropy_prev = entropy_now
                    token_text = ""
                    try:
                        token_text = tokenizer.decode([int(local_tokens[-1])])
                    except Exception:
                        token_text = str(int(local_tokens[-1]))
                    decision = adaptive_step(
                        token_text=token_text,
                        token_entropy=entropy_now,
                        attention_entropy_collapse=entropy_collapse,
                        latency_ms=adaptive_latency_ms,
                        vram_pressure=0.5,
                        quality_drift=adaptive_quality_drift,
                        layer_type=0,
                    )

                    sample_temperature = threebit_module.auto_temperature(logits, local_temperature)
                    sample_top_k = int(local_top_k)
                    sample_greedy = bool(local_greedy)
                    if decision is not None:
                        setattr(runtime_obj, "adaptive_target_bits", int(decision.target_bits))
                        setattr(runtime_obj, "adaptive_routed_bits", int(decision.routed_bits))
                        setattr(runtime_obj, "adaptive_reduce_attention_depth", bool(decision.reduce_attention_depth))
                        setattr(runtime_obj, "adaptive_attention_depth_hint", int(decision.attention_depth_hint))
                        setattr(runtime_obj, "adaptive_enable_kv_compression", bool(decision.enable_kv_compression))
                        if decision.reduce_attention_depth and decision.attention_depth_hint > 0:
                            sample_top_k = min(sample_top_k, max(8, int(decision.attention_depth_hint) * 4))
                        if decision.skip_compute:
                            sample_greedy = True
                        bit_hint = int(decision.routed_bits if decision.routed_bits > 0 else decision.target_bits)
                        if bit_hint > 0:
                            temp_scale = max(0.65, min(1.0, bit_hint / 8.0))
                            sample_temperature = max(0.05, float(sample_temperature) * temp_scale)

                    next_id = engine.sample(logits, sample_temperature, sample_top_k, sample_greedy, local_lang_top_n)
                    generated.append(next_id)
                    local_tokens.append(next_id)
                    engine.stream_token(next_id)
                    decode_optimizer.observe_token(local_tokens)
                    adaptive_latency_ms = (time.perf_counter() - token_t0) * 1000.0
                    adaptive_quality_drift = min(1.0, 0.6 * adaptive_quality_drift + 0.4 * entropy_collapse)
                    reached_eos = adapter.eos_token_id is not None and next_id == adapter.eos_token_id
                    if scheduler_enabled:
                        core_decode.commit(1, reached_eos)
                    if reached_eos:
                        break

                if contract_failed or reached_eos:
                    break

                if (not local_stream) and (step == 0 or (step + 1) % max(1, total_steps // 4) == 0):
                    pct = 45 + int(((step + 1) / total_steps) * 50)
                    _progress(show_progress, min(95, pct), "decode", f"{step + 1}/{total_steps}")
        finally:
            engine.end_stream()
            if scheduler_enabled and core_decode.is_active():
                core_decode.cancel()
            core_decode.close()

        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated) if generated else ""
        text = postprocess_output_text(text, prompt, lang_mode)
        tps = (len(generated) / elapsed) if elapsed > 0 else 0.0
        return text, len(generated), tps, engine.structure_report(), timed_out, contract_failed

    base_tokens = list(token_ids)
    _prefill_runtime(runtime, base_tokens)
    layer_count = len(getattr(runtime, "layers", []) or [])
    prefill_tokens = max(0, len(base_tokens) - 1)
    decode_budget = _resolve_decode_budget_seconds(prefill_tokens, layer_count)
    retry_budget = _resolve_retry_budget_seconds(decode_budget)
    decode_step_cap = _resolve_budget_step_cap(max_tokens, decode_budget, prefill_tokens, layer_count)
    print(f"[session] decode_budget_seconds= {decode_budget:.1f}")
    print(f"[session] retry_budget_seconds= {retry_budget:.1f}")
    if decode_step_cap != int(max_tokens):
        print(f"[session] decode_step_cap= {decode_step_cap} (requested={int(max_tokens)})")

    effective_disable_structure_guard = bool(disable_structure_guard)
    if (not effective_disable_structure_guard) and prioritize_english and lang_mode == "en":
        effective_disable_structure_guard = True
        print("[session] structure_guard= off (english-priority)")
    else:
        print("[session] structure_guard=", "off" if effective_disable_structure_guard else "on")

    def _run_torch_rescue(
        reason: str,
        decode_steps: int,
        decode_temp: float,
        decode_top_k: int,
        decode_lang_top_n: int,
        decode_rep_penalty: float,
        decode_repeat_window: int,
        decode_no_repeat_ngram: int,
    ) -> tuple[str, int, float, dict | None, bool, bool] | None:
        if torch_timeout_fallback_builder is None:
            return None
        try:
            fallback_runtime = torch_timeout_fallback_builder()
        except Exception:
            fallback_runtime = None
        if fallback_runtime is None:
            return None
        print(f"[session] {reason}_fallback_runtime= torch-cuda")
        _prefill_runtime(fallback_runtime, base_tokens)
        return _decode_once(
            runtime_obj=fallback_runtime,
            local_tokens=list(base_tokens),
            local_temperature=decode_temp,
            local_top_k=decode_top_k,
            local_lang_top_n=decode_lang_top_n,
            local_greedy=True,
            local_rep_penalty=decode_rep_penalty,
            local_repeat_window=decode_repeat_window,
            local_no_repeat_ngram=decode_no_repeat_ngram,
            local_stream=False,
            decode_budget_seconds=retry_budget,
            max_steps=decode_steps,
            disable_structure_guard_local=effective_disable_structure_guard,
        )

    def _run_native_safe_rescue(
        reason: str,
        decode_steps: int,
        decode_temp: float,
        decode_top_k: int,
        decode_lang_top_n: int,
        decode_rep_penalty: float,
        decode_repeat_window: int,
        decode_no_repeat_ngram: int,
    ) -> tuple[str, int, float, dict | None, bool, bool] | None:
        if native_safe_fallback_builder is None:
            return None
        try:
            fallback_runtime = native_safe_fallback_builder()
        except Exception:
            fallback_runtime = None
        if fallback_runtime is None:
            return None
        print(f"[session] {reason}_fallback_runtime= native-safe")
        _prefill_runtime(fallback_runtime, base_tokens)
        return _decode_once(
            runtime_obj=fallback_runtime,
            local_tokens=list(base_tokens),
            local_temperature=decode_temp,
            local_top_k=decode_top_k,
            local_lang_top_n=decode_lang_top_n,
            local_greedy=True,
            local_rep_penalty=decode_rep_penalty,
            local_repeat_window=decode_repeat_window,
            local_no_repeat_ngram=decode_no_repeat_ngram,
            local_stream=False,
            decode_budget_seconds=retry_budget,
            max_steps=decode_steps,
            disable_structure_guard_local=True,
        )

    text, gen_count, tps, report, timed_out_first, contract_failed_first = _decode_once(
        runtime_obj=runtime,
        local_tokens=list(base_tokens),
        local_temperature=temperature,
        local_top_k=top_k,
        local_lang_top_n=lang_top_n,
        local_greedy=greedy,
        local_rep_penalty=repetition_penalty,
        local_repeat_window=repeat_window,
        local_no_repeat_ngram=no_repeat_ngram,
        local_stream=stream,
        decode_budget_seconds=decode_budget,
        max_steps=decode_step_cap,
        disable_structure_guard_local=effective_disable_structure_guard,
    )

    if (gen_count <= 0 or _is_runtime_fallback_text(text, lang_mode) or contract_failed_first) and retry_budget >= 0.5:
        rescue_top_k = max(1, min(8, top_k if top_k > 0 else 8))
        rescue_lang_top_n = max(32, min(lang_top_n, 96))
        rescue_temp = min(0.68, max(0.45, float(temperature) * 0.82))
        rescue_rep = max(1.18, repetition_penalty)
        rescue_window = max(48, repeat_window)
        rescue_ngram = max(3, no_repeat_ngram)
        rescue_steps = max(8, min(decode_step_cap, 64))

        rescue_result = _run_native_safe_rescue(
            reason="decode-failure",
            decode_steps=rescue_steps,
            decode_temp=rescue_temp,
            decode_top_k=rescue_top_k,
            decode_lang_top_n=rescue_lang_top_n,
            decode_rep_penalty=rescue_rep,
            decode_repeat_window=rescue_window,
            decode_no_repeat_ngram=rescue_ngram,
        )
        if rescue_result is None:
            rescue_result = _run_torch_rescue(
                reason="decode-failure",
                decode_steps=rescue_steps,
                decode_temp=rescue_temp,
                decode_top_k=rescue_top_k,
                decode_lang_top_n=rescue_lang_top_n,
                decode_rep_penalty=rescue_rep,
                decode_repeat_window=rescue_window,
                decode_no_repeat_ngram=rescue_ngram,
            )
        if rescue_result is not None:
            text2, gen_count2, tps2, report2, timed_out_retry, contract_failed_retry = rescue_result
            if (not timed_out_retry) and (not contract_failed_retry) and (not _should_quality_retry(text2)):
                text = text2
                gen_count = gen_count2
                tps = tps2
                report = report2
                timed_out_first = False
                contract_failed_first = False

    if timed_out_first or contract_failed_first:
        print(f"[session] decode_timeout= True (budget={decode_budget:.1f}s)")
        if not _is_meaningful_partial(text, gen_count):
            if retry_budget >= 0.5:
                _progress(show_progress, 96, "timeout-rescue", "auto retry with shorter decode window")
                rescue_top_k = max(1, min(6, top_k if top_k > 0 else 6))
                rescue_lang_top_n = max(24, min(lang_top_n, 80))
                rescue_temp = min(0.62, max(0.40, float(temperature) * 0.75))
                rescue_rep = max(1.20, repetition_penalty)
                rescue_window = max(40, repeat_window)
                rescue_ngram = max(3, no_repeat_ngram)
                rescue_steps = max(12, min(48, decode_step_cap // 2 if decode_step_cap > 1 else 12))

                rescue_result = _run_native_safe_rescue(
                    reason="timeout",
                    decode_steps=rescue_steps,
                    decode_temp=rescue_temp,
                    decode_top_k=rescue_top_k,
                    decode_lang_top_n=rescue_lang_top_n,
                    decode_rep_penalty=rescue_rep,
                    decode_repeat_window=rescue_window,
                    decode_no_repeat_ngram=rescue_ngram,
                )
                if rescue_result is None:
                    rescue_result = _run_torch_rescue(
                        reason="timeout",
                        decode_steps=rescue_steps,
                        decode_temp=rescue_temp,
                        decode_top_k=rescue_top_k,
                        decode_lang_top_n=rescue_lang_top_n,
                        decode_rep_penalty=rescue_rep,
                        decode_repeat_window=rescue_window,
                        decode_no_repeat_ngram=rescue_ngram,
                    )
                if rescue_result is None:
                    _prefill_runtime(runtime, base_tokens)
                    rescue_result = _decode_once(
                        runtime_obj=runtime,
                        local_tokens=list(base_tokens),
                        local_temperature=rescue_temp,
                        local_top_k=rescue_top_k,
                        local_lang_top_n=rescue_lang_top_n,
                        local_greedy=True,
                        local_rep_penalty=rescue_rep,
                        local_repeat_window=rescue_window,
                        local_no_repeat_ngram=rescue_ngram,
                        local_stream=False,
                        decode_budget_seconds=retry_budget,
                        max_steps=rescue_steps,
                        disable_structure_guard_local=effective_disable_structure_guard,
                    )

                text2, gen_count2, tps2, report2, timed_out_retry, contract_failed_retry = rescue_result
                if timed_out_retry:
                    print(f"[session] retry_timeout= True (budget={retry_budget:.1f}s)")
                if (not timed_out_retry) and (not contract_failed_retry) and (not _should_quality_retry(text2)):
                    text = text2
                    gen_count = gen_count2
                    tps = tps2
                    report = report2
                else:
                    text = assurance.repair("", prompt)
                    gen_count = 0
                    tps = 0.0
            else:
                text = assurance.repair("", prompt)
                gen_count = 0
                tps = 0.0

    first_bad = _should_quality_retry(text)
    if first_bad and retry_budget >= 0.5 and (not timed_out_first):
        _progress(show_progress, 96, "quality-retry", "detected noisy output, retrying with safe decode")
        safe_top_k = max(1, min(8, top_k if top_k > 0 else 8))
        safe_lang_top_n = max(32, min(lang_top_n, 96))
        safe_temp = min(0.68, max(0.45, float(temperature) * 0.82))
        safe_rep = max(1.18, repetition_penalty)
        safe_window = max(48, repeat_window)
        safe_ngram = max(3, no_repeat_ngram)

        rescue_result = _run_native_safe_rescue(
            reason="quality",
            decode_steps=max(8, min(decode_step_cap, 64)),
            decode_temp=safe_temp,
            decode_top_k=safe_top_k,
            decode_lang_top_n=safe_lang_top_n,
            decode_rep_penalty=safe_rep,
            decode_repeat_window=safe_window,
            decode_no_repeat_ngram=safe_ngram,
        )
        if rescue_result is None:
            rescue_result = _run_torch_rescue(
                reason="quality",
                decode_steps=max(8, min(decode_step_cap, 64)),
                decode_temp=safe_temp,
                decode_top_k=safe_top_k,
                decode_lang_top_n=safe_lang_top_n,
                decode_rep_penalty=safe_rep,
                decode_repeat_window=safe_window,
                decode_no_repeat_ngram=safe_ngram,
            )
        if rescue_result is None:
            _prefill_runtime(runtime, base_tokens)
            rescue_result = _decode_once(
                runtime_obj=runtime,
                local_tokens=list(base_tokens),
                local_temperature=safe_temp,
                local_top_k=safe_top_k,
                local_lang_top_n=safe_lang_top_n,
                local_greedy=True,
                local_rep_penalty=safe_rep,
                local_repeat_window=safe_window,
                local_no_repeat_ngram=safe_ngram,
                local_stream=False,
                decode_budget_seconds=retry_budget,
                max_steps=max(8, min(decode_step_cap, 64)),
                disable_structure_guard_local=effective_disable_structure_guard,
            )

        text2, gen_count2, tps2, report2, timed_out_retry, contract_failed_retry = rescue_result
        if timed_out_retry:
            print(f"[session] retry_timeout= True (budget={retry_budget:.1f}s)")
            if not _is_meaningful_partial(text2, gen_count2):
                text2 = assurance.repair("", prompt)
                gen_count2 = 0
                tps2 = 0.0
        if (not timed_out_retry) and (not contract_failed_retry) and (not _should_quality_retry(text2)):
            text = text2
            gen_count = gen_count2
            tps = tps2
            report = report2
        else:
            text = assurance.repair(text2, prompt)
            gen_count = gen_count2
            tps = tps2
            report = report2
    else:
        text = assurance.repair(text, prompt)

    _progress(show_progress, 100, "done", f"{gen_count} tok | {tps:.2f} tok/s")
    if report is not None:
        print("[session] structure_integrity_pass=", report.get("integrity_pass"))
        print("[session] structure_section_coverage=", round(float(report.get("section_coverage", 0.0)), 4))
    lowbit_stats = lowbit_projection_stats_snapshot()
    print("[session] lowbit_exec_calls=", int(lowbit_stats.get("calls", 0)))
    print("[session] lowbit_exec_lowbit_calls=", int(lowbit_stats.get("lowbit_calls", 0)))
    print("[session] lowbit_exec_fallback_gemm_calls=", int(lowbit_stats.get("fallback_gemm_calls", 0)))
    print("[session] lowbit_exec_fallback_matmul_calls=", int(lowbit_stats.get("fallback_matmul_calls", 0)))
    return text



def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Vspec chat session (loads model once)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cuda-native", "torch-cuda"])
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--lang", default="auto", choices=["auto", "vi", "en"])
    parser.add_argument("--lang-top-n", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repeat-window", type=int, default=32)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--chat-format", default="auto", choices=["auto", "plain", "chatml", "llama3", "alpaca"])
    parser.add_argument("--speed-preset", default="fast", choices=["normal", "fast", "ultra"])
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-bits", type=int, default=3, choices=[0, 2, 3, 4, 8, 16])
    parser.add_argument("--fused-bits", type=int, default=3, choices=[0, 3, 4, 16])
    parser.add_argument("--disable-language-guard", action="store_true")
    parser.add_argument("--language-guard-strictness", type=float, default=0.72)
    parser.add_argument("--no-prioritize-english", action="store_true")
    parser.add_argument("--disable-structure-guard", action="store_true")
    parser.add_argument("--structure-guard-strictness", type=float, default=0.72)
    parser.add_argument("--decode-opt-mode", default="optimized", choices=["stable", "optimized"])
    parser.add_argument("--enable-3bit-runtime-module", action="store_true")
    parser.add_argument(
        "--allow-semantic-rescue",
        action="store_true",
        help="Deprecated no-op: synthetic rescue responses are disabled by integrity policy",
    )
    parser.add_argument("--unsafe-low-layers", action="store_true")
    parser.add_argument(
        "--max-decode-seconds",
        type=float,
        default=-1.0,
        help="<0 auto, =0 disable timeout, >0 fixed budget per decode pass",
    )
    parser.add_argument(
        "--max-retry-seconds",
        type=float,
        default=-1.0,
        help="<0 auto, =0 disable retry, >0 fixed retry budget",
    )
    parser.add_argument(
        "--disable-timeout-torch-fallback",
        action="store_true",
        help="Disable automatic torch-cuda retry when native decode times out",
    )
    args = parser.parse_args()

    fused_bits_env = int(args.fused_bits)
    if fused_bits_env not in {0, 3, 4}:
        fused_bits_env = 0
    os.environ["VSPEC_FUSED_BITS"] = str(fused_bits_env)
    if (
        fused_bits_env == 4
        and str(args.device) in {"cuda", "cuda-native"}
        and not os.getenv("VSPEC_INT4_COMPUTE_MODE", "").strip()
    ):
        os.environ["VSPEC_INT4_COMPUTE_MODE"] = "native"

    random.seed(args.seed)
    show_progress = not args.no_progress

    _progress(show_progress, 5, "snapshot", "finding snapshot")
    snapshot = find_snapshot_dir(Path(args.model_dir))
    _progress(show_progress, 15, "config", "loading config/tokenizer")
    config = read_config(snapshot)
    requested_layers = int(args.max_layers)
    args.max_layers = _resolve_quality_layer_floor(config, requested_layers, str(args.device), bool(args.unsafe_low_layers))
    if requested_layers > 0 and args.max_layers != requested_layers:
        print(f"[session] quality_guard_max_layers_adjusted= {requested_layers} -> {args.max_layers}")
    model_type = str(config.get("model_type", "") or "").lower()
    if model_type == "gpt2":
        if int(args.fused_bits) == 3:
            args.fused_bits = 4
            os.environ["VSPEC_FUSED_BITS"] = "4"
            print("[session] auto_adjust_fused_bits= 3 -> 4 for gpt2 stability")
        if int(args.target_bits) == 3:
            args.target_bits = 4
            print("[session] auto_adjust_target_bits= 3 -> 4 for gpt2 stability")

    tok_cfg = read_tokenizer_config(snapshot)
    tokenizer = load_tokenizer(snapshot)
    tensor_names = collect_tensor_names(snapshot)
    _progress(show_progress, 35, "weights", "index model weights")
    weight_index = build_weight_index(snapshot)
    adapter = select_adapter(config, tensor_names)

    speed = resolve_speed_preset(args.speed_preset)
    top_k = max(0, min(args.top_k, speed.top_k)) if args.top_k > 0 else speed.top_k
    lang_top_n = max(16, min(args.lang_top_n, speed.lang_top_n))
    repetition_penalty = min(args.repetition_penalty, speed.repetition_penalty)
    no_repeat_ngram = min(args.no_repeat_ngram, speed.no_repeat_ngram) if speed.no_repeat_ngram > 0 else 0
    repeat_window = min(args.repeat_window, speed.repeat_window)

    threebit_module = ThreeBitRuntimeModule(
        enabled=(args.enable_3bit_runtime_module or int(args.fused_bits) == 3 or int(args.target_bits) == 3),
        fused_bits=args.fused_bits,
        target_bits=args.target_bits,
    )
    tuned = threebit_module.tune_sampling(
        top_k=top_k,
        lang_top_n=lang_top_n,
        repetition_penalty=repetition_penalty,
        repeat_window=repeat_window,
    )
    top_k = tuned.top_k
    lang_top_n = tuned.lang_top_n
    repetition_penalty = tuned.repetition_penalty
    repeat_window = tuned.repeat_window

    _progress(show_progress, 60, "runtime", f"build {args.device}")

    def _runtime_progress(stage: str, current: int, total: int) -> None:
        if not show_progress:
            return
        if stage != "layer_load" or total <= 0:
            return
        pct = 61 + int((current / total) * 35)
        _progress(show_progress, min(96, pct), "runtime-load", f"layer {current}/{total}")

    runtime = build_generic_runtime(
        config,
        weight_index,
        args.max_layers,
        args.device,
        progress_cb=_runtime_progress,
    )
    if runtime is None:
        print("[session] runtime init failed")
        return

    runtime.eos_token_id = adapter.eos_token_id
    print("[session] decode_opt_mode=", args.decode_opt_mode)
    print("[session] runtime_3bit_module=", tuned.active)
    if tuned.active:
        print("[session] runtime_3bit_tuned_top_k=", tuned.top_k)
        print("[session] runtime_3bit_tuned_lang_top_n=", tuned.lang_top_n)
        print("[session] runtime_3bit_tuned_repetition_penalty=", round(tuned.repetition_penalty, 4))
        print("[session] runtime_3bit_tuned_repeat_window=", tuned.repeat_window)
    if hasattr(runtime, "fused_bits"):
        print("[session] runtime_effective_fused_bits=", int(getattr(runtime, "fused_bits", 0)))
        print("[session] storage_int3_active=", int(getattr(runtime, "fused_bits", 0)) == 3)
    if hasattr(runtime, "lowbit_plan") and getattr(runtime, "lowbit_plan") is not None:
        lowbit_plan = getattr(runtime, "lowbit_plan")
        print("[session] runtime_lowbit_enabled=", bool(getattr(lowbit_plan, "enabled", False)))
        print("[session] runtime_lowbit_reason=", getattr(lowbit_plan, "reason", "unknown"))
    print("[session] quant_source_format=", getattr(runtime, "quant_source_format", "unknown"))
    print("[session] quant_source_quantized=", bool(getattr(runtime, "quant_source_quantized", False)))
    print("[session] quant_runtime_disabled=", bool(getattr(runtime, "quant_runtime_disabled", False)))
    print("[session] quant_policy_reason=", getattr(runtime, "quant_policy_reason", "unknown"))
    if hasattr(runtime, "layers"):
        try:
            matrix_bits, exec_eff_bits, _, lowbit_coverage = runtime_matrix_bits_summary(runtime.layers)
            print("[session] lowbit_mode=runtime")
            print("[session] target_bits=", args.target_bits)
            print("[session] matrix_bits=", matrix_bits)
            print("[session] effective_bits_estimate=", round(exec_eff_bits, 4))
            print("[session] lowbit_coverage=", round(lowbit_coverage, 4))
        except Exception:
            layer_bits = build_layer_bits(len(runtime.layers), args.target_bits)
            print("[session] lowbit_mode=policy-fallback")
            print("[session] target_bits=", args.target_bits)
            print("[session] layer_bits=", summarize_layer_bits(layer_bits))
            print("[session] effective_bits_estimate=", round(effective_bits(layer_bits), 4))
            print("[session] note=policy telemetry fallback")

    _progress(show_progress, 100, "ready", "model loaded; enter prompt")
    print("[session] type /exit to quit")

    torch_runtime_cache = {"value": None, "attempted": False}
    native_safe_runtime_cache = {"value": None, "attempted": False}

    def _native_safe_fallback_builder():
        if native_safe_runtime_cache["attempted"]:
            return native_safe_runtime_cache["value"]
        native_safe_runtime_cache["attempted"] = True
        try:
            fallback_runtime = build_native_safe_runtime(config, weight_index, args.max_layers, str(args.device), None)
        except Exception:
            fallback_runtime = None
        if fallback_runtime is not None and hasattr(fallback_runtime, "eos_token_id"):
            fallback_runtime.eos_token_id = adapter.eos_token_id
        native_safe_runtime_cache["value"] = fallback_runtime
        return fallback_runtime

    def _torch_timeout_fallback_builder():
        if args.disable_timeout_torch_fallback:
            return None
        if str(args.device) not in {"cuda", "cuda-native"}:
            return None
        if torch_runtime_cache["attempted"]:
            return torch_runtime_cache["value"]
        torch_runtime_cache["attempted"] = True
        try:
            fallback_runtime = build_generic_runtime(
                config,
                weight_index,
                args.max_layers,
                "torch-cuda",
                progress_cb=None,
            )
        except Exception:
            fallback_runtime = None
        if fallback_runtime is not None and hasattr(fallback_runtime, "eos_token_id"):
            fallback_runtime.eos_token_id = adapter.eos_token_id
        torch_runtime_cache["value"] = fallback_runtime
        return fallback_runtime

    while True:
        try:
            prompt = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[session] bye")
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit"}:
            print("[session] bye")
            break

        lang_mode = args.lang if args.lang != "auto" else _detect_lang(prompt)
        _progress(show_progress, 1, "request", f"lang={lang_mode}")
        out = _generate(
            prompt=prompt,
            tokenizer=tokenizer,
            adapter=adapter,
            runtime=runtime,
            tok_cfg=tok_cfg,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
            greedy=args.greedy,
            lang_mode=lang_mode,
            lang_top_n=lang_top_n,
            repetition_penalty=repetition_penalty,
            repeat_window=repeat_window,
            no_repeat_ngram=no_repeat_ngram,
            chat_format=args.chat_format,
            stream=(not args.no_stream),
            show_progress=show_progress,
            disable_language_guard=args.disable_language_guard,
            language_guard_strictness=args.language_guard_strictness,
            prioritize_english=(not args.no_prioritize_english),
            structure_guard_strictness=args.structure_guard_strictness,
            disable_structure_guard=args.disable_structure_guard,
            decode_opt_mode=args.decode_opt_mode,
            threebit_module=threebit_module,
            allow_semantic_rescue=args.allow_semantic_rescue,
            max_decode_seconds=args.max_decode_seconds,
            max_retry_seconds=args.max_retry_seconds,
            native_safe_fallback_builder=_native_safe_fallback_builder,
            torch_timeout_fallback_builder=_torch_timeout_fallback_builder,
        )
        if args.no_stream:
            print("assistant>", out)


if __name__ == "__main__":
    main()
