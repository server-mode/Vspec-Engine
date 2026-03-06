import argparse
import os
import random
import time
from pathlib import Path

from chat_prompt import build_prompt
from fast_output import FastOutputEngine, postprocess_output_text, resolve_speed_preset
from decode_optimization_module import DecodeOptimizationModule
from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from runtime_meaningful_response import RuntimeMeaningfulResponseAssurance
from runtime_threebit_module import ThreeBitRuntimeModule
from model_adapters import select_adapter
from model_loader import (
    build_weight_index,
    collect_tensor_names,
    find_snapshot_dir,
    load_tokenizer,
    read_config,
    read_tokenizer_config,
)
from runtime_inference import build_generic_runtime
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
) -> str:
    prompt_for_model = build_prompt(prompt, adapter.model_type, tok_cfg, lang_mode, chat_format)
    encoded = tokenizer.encode(prompt_for_model)
    token_ids = list(encoded.ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    assurance = RuntimeMeaningfulResponseAssurance(lang_mode, allow_semantic_rescue=allow_semantic_rescue)

    def _is_meaningful_partial(candidate: str) -> bool:
        out = (candidate or "").strip()
        if len(out) < 24:
            return False
        letters = sum(1 for ch in out if ch.isalpha())
        if (letters / max(1, len(out))) < 0.55:
            return False
        words = [w for w in out.split() if w]
        if len(words) < 4:
            return False
        return True

    prompt_chars = len((prompt or "").strip())

    def _resolve_decode_budget_seconds() -> float:
        raw = float(max_decode_seconds)
        if raw > 0.0:
            return max(0.5, raw)
        if raw == 0.0:
            return 0.0
        auto = 12.0 + (0.16 * float(max_tokens)) + (0.025 * float(prompt_chars))
        if max_tokens <= 64:
            auto = max(auto, 24.0)
        return min(90.0, max(16.0, auto))

    def _resolve_retry_budget_seconds(decode_budget: float) -> float:
        raw = float(max_retry_seconds)
        if raw > 0.0:
            return max(0.5, raw)
        if raw == 0.0:
            return 0.0
        if decode_budget <= 0.0:
            return 8.0
        auto = decode_budget * 0.35
        return min(24.0, max(6.0, auto))

    def _prefill_runtime(local_tokens: list[int]) -> None:
        if not hasattr(runtime, "cache_k"):
            return
        runtime.cache_k = []
        runtime.cache_v = []
        runtime.position = 0
        prefill = local_tokens[:-1]
        total_prefill = len(prefill)
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens(prefill)
            if total_prefill > 0:
                _progress(show_progress, 45, "prefill", f"{total_prefill}/{total_prefill}")
            return
        for idx, tid in enumerate(prefill, start=1):
            runtime.forward_logits([tid])
            if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                pct = 10 + int((idx / total_prefill) * 35)
                _progress(show_progress, min(45, pct), "prefill", f"{idx}/{total_prefill}")

    def _decode_once(
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
    ) -> tuple[str, int, float, dict | None, bool]:
        guard = None
        if not disable_language_guard:
            guard = LanguageStabilityGuard(
                prompt=prompt,
                lang_mode=lang_mode,
                strictness=language_guard_strictness,
                prioritize_english=prioritize_english,
            )

        structure_guard = None
        if not disable_structure_guard:
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

        generated = []
        start = time.perf_counter()
        engine.begin_stream()
        timed_out = False

        total_steps = max(1, int(max_steps))
        for step in range(total_steps):
            elapsed = time.perf_counter() - start
            if decode_budget_seconds > 0.0 and elapsed >= decode_budget_seconds:
                timed_out = True
                break
            logits = decode_optimizer.fetch_logits(runtime, local_tokens[-1], tokenizer.get_vocab_size())
            if decode_optimizer.logits_empty(logits):
                break

            logits = threebit_module.denoise_logits(logits, step)
            logits = decode_optimizer.apply_generation_controls(logits, local_tokens)
            sample_temperature = threebit_module.auto_temperature(logits, local_temperature)
            next_id = engine.sample(logits, sample_temperature, local_top_k, local_greedy, local_lang_top_n)
            generated.append(next_id)
            local_tokens.append(next_id)
            engine.stream_token(next_id)
            decode_optimizer.observe_token(local_tokens)

            if adapter.eos_token_id is not None and next_id == adapter.eos_token_id:
                break
            if (not local_stream) and (step == 0 or (step + 1) % max(1, total_steps // 4) == 0):
                pct = 45 + int(((step + 1) / total_steps) * 50)
                _progress(show_progress, min(95, pct), "decode", f"{step + 1}/{total_steps}")

        engine.end_stream()
        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated) if generated else ""
        text = postprocess_output_text(text, prompt, lang_mode)
        tps = (len(generated) / elapsed) if elapsed > 0 else 0.0
        return text, len(generated), tps, engine.structure_report(), timed_out

    base_tokens = list(token_ids)
    _prefill_runtime(base_tokens)
    decode_budget = _resolve_decode_budget_seconds()
    retry_budget = _resolve_retry_budget_seconds(decode_budget)
    print(f"[session] decode_budget_seconds= {decode_budget:.1f}")
    print(f"[session] retry_budget_seconds= {retry_budget:.1f}")

    text, gen_count, tps, report, timed_out_first = _decode_once(
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
        max_steps=max_tokens,
    )

    first_bad = _is_runtime_fallback_text(text, lang_mode) or _looks_gibberish_output(text)
    if timed_out_first:
        print(f"[session] decode_timeout= True (budget={decode_budget:.1f}s)")
        if not _is_meaningful_partial(text):
            if retry_budget >= 0.5:
                _progress(show_progress, 96, "timeout-rescue", "auto retry with shorter decode window")
                rescue_top_k = max(1, min(6, top_k if top_k > 0 else 6))
                rescue_lang_top_n = max(24, min(lang_top_n, 80))
                rescue_temp = min(0.62, max(0.40, float(temperature) * 0.75))
                rescue_rep = max(1.20, repetition_penalty)
                rescue_window = max(40, repeat_window)
                rescue_ngram = max(3, no_repeat_ngram)
                rescue_steps = max(12, min(48, max_tokens // 2 if max_tokens > 1 else 12))

                _prefill_runtime(base_tokens)
                text2, gen_count2, tps2, report2, timed_out_retry = _decode_once(
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
                )
                if timed_out_retry:
                    print(f"[session] retry_timeout= True (budget={retry_budget:.1f}s)")
                if (not timed_out_retry) and (not (_is_runtime_fallback_text(text2, lang_mode) or _looks_gibberish_output(text2))):
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

    first_bad = _is_runtime_fallback_text(text, lang_mode) or _looks_gibberish_output(text)
    if first_bad and retry_budget >= 0.5 and (not timed_out_first):
        _progress(show_progress, 96, "quality-retry", "detected noisy output, retrying with safe decode")
        safe_top_k = max(1, min(8, top_k if top_k > 0 else 8))
        safe_lang_top_n = max(32, min(lang_top_n, 96))
        safe_temp = min(0.68, max(0.45, float(temperature) * 0.82))
        safe_rep = max(1.18, repetition_penalty)
        safe_window = max(48, repeat_window)
        safe_ngram = max(3, no_repeat_ngram)

        _prefill_runtime(base_tokens)
        text2, gen_count2, tps2, report2, timed_out_retry = _decode_once(
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
            max_steps=max(8, min(max_tokens, 64)),
        )
        if timed_out_retry:
            print(f"[session] retry_timeout= True (budget={retry_budget:.1f}s)")
            if not _is_meaningful_partial(text2):
                text2 = assurance.repair("", prompt)
                gen_count2 = 0
                tps2 = 0.0
        if (not timed_out_retry) and (not (_is_runtime_fallback_text(text2, lang_mode) or _looks_gibberish_output(text2))):
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
    parser.add_argument("--target-bits", type=int, default=3, choices=[0, 2, 3, 4])
    parser.add_argument("--fused-bits", type=int, default=3, choices=[0, 3, 4])
    parser.add_argument("--disable-language-guard", action="store_true")
    parser.add_argument("--language-guard-strictness", type=float, default=0.72)
    parser.add_argument("--no-prioritize-english", action="store_true")
    parser.add_argument("--disable-structure-guard", action="store_true")
    parser.add_argument("--structure-guard-strictness", type=float, default=0.72)
    parser.add_argument("--decode-opt-mode", default="optimized", choices=["stable", "optimized"])
    parser.add_argument("--enable-3bit-runtime-module", action="store_true")
    parser.add_argument("--allow-semantic-rescue", action="store_true", help="Deprecated no-op: synthetic rescue responses are disabled by integrity policy")
    parser.add_argument("--unsafe-low-layers", action="store_true")
    parser.add_argument("--max-decode-seconds", type=float, default=-1.0, help="<0 auto, =0 disable timeout, >0 fixed budget per decode pass")
    parser.add_argument("--max-retry-seconds", type=float, default=-1.0, help="<0 auto, =0 disable retry, >0 fixed retry budget")
    args = parser.parse_args()

    os.environ["VSPEC_FUSED_BITS"] = str(args.fused_bits)

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
    tok_cfg = read_tokenizer_config(snapshot)
    tokenizer = load_tokenizer(snapshot)
    tensor_names = collect_tensor_names(snapshot)
    _progress(show_progress, 35, "weights", "index headers")
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

    runtime = build_generic_runtime(config, weight_index, args.max_layers, args.device, _runtime_progress)
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
    if args.target_bits > 0 and hasattr(runtime, "layers"):
        layer_bits = build_layer_bits(len(runtime.layers), args.target_bits)
        print("[session] lowbit_mode=policy")
        print("[session] target_bits=", args.target_bits)
        print("[session] layer_bits=", summarize_layer_bits(layer_bits))
        print("[session] effective_bits_estimate=", round(effective_bits(layer_bits), 4))
        print("[session] note=policy telemetry only; current kernels stay quality-first")
    _progress(show_progress, 100, "ready", "model loaded; enter prompt")

    print("[session] type /exit to quit")
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
        )
        if args.no_stream:
            print("assistant>", out)


if __name__ == "__main__":
    main()
