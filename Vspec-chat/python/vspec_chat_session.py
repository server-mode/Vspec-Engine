import argparse
import random
import time
from pathlib import Path

from chat_prompt import build_prompt
from fast_output import FastOutputEngine, postprocess_output_text, resolve_speed_preset
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
from vspec_chat import _apply_generation_controls, _detect_lang
from lowbit_policy import build_layer_bits, effective_bits, summarize_layer_bits


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
) -> str:
    prompt_for_model = build_prompt(prompt, adapter.model_type, tok_cfg, lang_mode, chat_format)
    encoded = tokenizer.encode(prompt_for_model)
    token_ids = list(encoded.ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    if hasattr(runtime, "cache_k"):
        runtime.cache_k = []
        runtime.cache_v = []
        runtime.position = 0
        prefill = token_ids[:-1]
        total_prefill = len(prefill)
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens(prefill)
            if total_prefill > 0:
                _progress(show_progress, 45, "prefill", f"{total_prefill}/{total_prefill}")
        else:
            for idx, tid in enumerate(prefill, start=1):
                runtime.forward_logits([tid])
                if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                    pct = 10 + int((idx / total_prefill) * 35)
                    _progress(show_progress, min(45, pct), "prefill", f"{idx}/{total_prefill}")

    engine = FastOutputEngine(tokenizer=tokenizer, lang_mode=lang_mode, stream=stream)
    engine.begin_stream()
    generated = []
    start = time.perf_counter()

    for step in range(max_tokens):
        logits = runtime.forward_logits(token_ids)
        logits = _apply_generation_controls(logits, token_ids, repetition_penalty, repeat_window, no_repeat_ngram)
        next_id = engine.sample(logits, temperature, top_k, greedy, lang_top_n)
        generated.append(next_id)
        token_ids.append(next_id)
        engine.stream_token(next_id)

        if adapter.eos_token_id is not None and next_id == adapter.eos_token_id:
            break
        if (not stream) and (step == 0 or (step + 1) % max(1, max_tokens // 4) == 0):
            pct = 45 + int(((step + 1) / max_tokens) * 50)
            _progress(show_progress, min(95, pct), "decode", f"{step + 1}/{max_tokens}")

    engine.end_stream()
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated)
    text = postprocess_output_text(text, prompt, lang_mode)
    tps = (len(generated) / elapsed) if elapsed > 0 else 0.0
    _progress(show_progress, 100, "done", f"{len(generated)} tok | {tps:.2f} tok/s")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Vspec chat session (loads model once)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--lang", default="auto", choices=["auto", "vi", "en"])
    parser.add_argument("--lang-top-n", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repeat-window", type=int, default=32)
    parser.add_argument("--no-repeat-ngram", type=int, default=1)
    parser.add_argument("--chat-format", default="auto", choices=["auto", "plain", "chatml", "llama3", "alpaca"])
    parser.add_argument("--speed-preset", default="fast", choices=["normal", "fast", "ultra"])
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-bits", type=int, default=0, choices=[0, 2, 3, 4])
    args = parser.parse_args()

    random.seed(args.seed)
    show_progress = not args.no_progress

    _progress(show_progress, 5, "snapshot", "finding snapshot")
    snapshot = find_snapshot_dir(Path(args.model_dir))
    _progress(show_progress, 15, "config", "loading config/tokenizer")
    config = read_config(snapshot)
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
        )
        if args.no_stream:
            print("assistant>", out)


if __name__ == "__main__":
    main()
