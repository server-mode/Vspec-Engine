import argparse
import math
import random
import re
import time
from pathlib import Path

from model_adapters import select_adapter
from model_loader import (
    build_weight_index,
    collect_tensor_names,
    find_snapshot_dir,
    load_tokenizer,
    read_config,
    read_tokenizer_config,
    summarize_weight_dtypes,
)
from runtime_inference import build_generic_runtime, runtime_load_reason
from chat_prompt import build_prompt
from vspec_cuda_bridge import cuda_mem_info
from fast_output import FastOutputEngine, postprocess_output_text, resolve_speed_preset
from lowbit_policy import build_layer_bits, effective_bits, summarize_layer_bits


def _progress(enabled: bool, pct: int, stage: str, detail: str = "") -> None:
    if not enabled:
        return
    msg = f"[progress] {pct:>3}% | {stage}"
    if detail:
        msg += f" | {detail}"
    print(msg)


def _apply_generation_controls(
    logits: list[float],
    history: list[int],
    repetition_penalty: float,
    repeat_window: int,
    no_repeat_ngram: int,
) -> list[float]:
    adjusted = list(logits)

    if repetition_penalty > 1.0 and history:
        for token_id in set(history[-repeat_window:]):
            if 0 <= token_id < len(adjusted):
                if adjusted[token_id] > 0:
                    adjusted[token_id] /= repetition_penalty
                else:
                    adjusted[token_id] *= repetition_penalty

    if no_repeat_ngram > 1 and len(history) >= no_repeat_ngram - 1:
        prefix = tuple(history[-(no_repeat_ngram - 1):])
        banned = set()
        for i in range(len(history) - no_repeat_ngram + 1):
            ngram = tuple(history[i : i + no_repeat_ngram])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        for token_id in banned:
            if 0 <= token_id < len(adjusted):
                adjusted[token_id] = -1e9

    return adjusted


def _detect_lang(prompt: str) -> str:
    lower = prompt.lower()
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ"
    if any(ch in lower for ch in vi_chars):
        return "vi"
    if any("\u4e00" <= ch <= "\u9fff" for ch in prompt):
        return "auto"
    return "en"


def _is_latin_text(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if ch.isspace() or ch.isdigit() or ch in ",.;:!?-_'\"()[]{}+/\\@#$%^&*=<>|~`":
            continue
        if (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F) or (0x1E00 <= code <= 0x1EFF):
            continue
        return False
    return True


def _is_clean_vi_text(text: str) -> bool:
    if not text:
        return True
    if any(tok in text for tok in ["http", "www", "_", "=", "\\", "/"]):
        return False
    if re.search(r"[A-Z]{3,}", text):
        return False

    allowed_re = r"^[\sA-Za-z0-9ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ,.;:!?\-_'\"()\[\]{}]+$"
    return re.match(allowed_re, text) is not None


def _is_clean_en_text(text: str) -> bool:
    if not text:
        return True
    if any(tok in text for tok in ["http", "www", "_", "=", "\\", "/"]):
        return False
    if re.search(r"[A-Z]{3,}", text):
        return False
    allowed_re = r"^[\sA-Za-z0-9,.;:!?\-_'\"()\[\]{}]+$"
    return re.match(allowed_re, text) is not None


def _token_quality_bonus(text: str, lang_mode: str) -> float:
    if not text:
        return 0.0
    bonus = 0.0
    if len(text) > 20:
        bonus -= 0.8
    if re.search(r"[A-Z]{2,}", text):
        bonus -= 1.0
    if any(ch in text for ch in ["_", "=", "\\", "/", "@"]) or "http" in text:
        bonus -= 1.5
    if lang_mode == "vi":
        if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ]", text.lower()):
            bonus += 0.35
    return bonus


def _is_allowed_token(tokenizer, token_id: int, lang_mode: str) -> bool:
    if tokenizer is None or lang_mode == "auto":
        return True
    text = tokenizer.decode([token_id])
    if not text:
        return True
    if "�" in text:
        return False
    if lang_mode == "vi":
        return _is_clean_vi_text(text)
    if lang_mode == "en":
        return _is_clean_en_text(text)
    if lang_mode in {"vi", "en"}:
        return _is_latin_text(text)
    return True


def sample_from_logits(
    logits: list[float],
    temperature: float,
    top_k: int,
    greedy: bool,
    tokenizer,
    lang_mode: str,
    lang_top_n: int,
) -> int:
    if temperature <= 0:
        temperature = 1.0
    scaled = [v / temperature for v in logits]
    candidate_n = lang_top_n
    if top_k > 0:
        candidate_n = max(candidate_n, top_k)

    sorted_ids = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
    candidate_ids = sorted_ids[: max(1, min(candidate_n, len(sorted_ids)))]

    allowed_ids = [tid for tid in candidate_ids if _is_allowed_token(tokenizer, tid, lang_mode)]
    if not allowed_ids:
        allowed_ids = candidate_ids

    if top_k > 0 and len(allowed_ids) > top_k:
        allowed_ids = allowed_ids[:top_k]

    scored = []
    for tid in allowed_ids:
        text = tokenizer.decode([tid]) if tokenizer is not None else ""
        scored.append((tid, scaled[tid] + _token_quality_bonus(text, lang_mode)))

    scored.sort(key=lambda x: x[1], reverse=True)
    allowed_ids = [tid for tid, _ in scored]
    allowed_logits = [score for _, score in scored]

    if greedy:
        return allowed_ids[0]

    max_logit = max(allowed_logits)
    exp_vals = [math.exp(v - max_logit) for v in allowed_logits]
    total = sum(exp_vals)
    if total <= 0:
        return allowed_ids[0]
    probs = [v / total for v in exp_vals]
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return allowed_ids[i]
    return allowed_ids[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vspec-chat prototype CLI")
    parser.add_argument("--model-dir", required=True, help="Path to HF cache model dir")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = generate until EOS or safety cap")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-layers", type=int, default=0, help="0 = use all available layers")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda-native")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repeat-window", type=int, default=64)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--lang", default="auto", choices=["auto", "vi", "en"], help="Constrain generation language")
    parser.add_argument("--lang-top-n", type=int, default=256, help="Top-N candidates checked for language filter")
    parser.add_argument("--chat-format", default="auto", choices=["auto", "plain", "chatml", "llama3", "alpaca"])
    parser.add_argument("--speed-preset", default="fast", choices=["normal", "fast", "ultra"], help="Decoding speed profile")
    parser.add_argument("--no-stream", action="store_true", help="Disable token-by-token streaming output")
    parser.add_argument("--no-progress", action="store_true", help="Disable stage progress logs")
    parser.add_argument("--target-bits", type=int, default=0, choices=[0, 2, 3, 4], help="0=off, policy target bits for benchmark telemetry")
    args = parser.parse_args()

    show_progress = not args.no_progress

    random.seed(args.seed)

    model_dir = Path(args.model_dir)
    _progress(show_progress, 5, "snapshot", "finding latest HF snapshot")
    snapshot_dir = find_snapshot_dir(model_dir)
    _progress(show_progress, 15, "config", "loading model and tokenizer config")
    config = read_config(snapshot_dir)
    tok_cfg = read_tokenizer_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    _progress(show_progress, 35, "weights", "indexing safetensors headers")
    tensor_names = collect_tensor_names(snapshot_dir)
    weight_index = build_weight_index(snapshot_dir)
    dtype_stats = summarize_weight_dtypes(weight_index)
    adapter = select_adapter(config, tensor_names)

    vocab_size = adapter.vocab_size or config.get("vocab_size")
    if tokenizer is not None:
        vocab_size = tokenizer.get_vocab_size()

    if not vocab_size:
        vocab_size = 32000

    print("[vspec-chat] snapshot=", snapshot_dir)
    print("[vspec-chat] adapter=", adapter.name)
    print("[vspec-chat] model_type=", adapter.model_type)
    print("[vspec-chat] tensors=", len(tensor_names))
    print("[vspec-chat] vocab_size=", vocab_size)
    print("[vspec-chat] dtype_stats=", dtype_stats)
    if tokenizer is None:
        print("[vspec-chat] tokenizer=missing (install 'tokenizers')")

    lang_mode = args.lang
    if lang_mode == "auto":
        lang_mode = _detect_lang(args.prompt)

    speed_cfg = resolve_speed_preset(args.speed_preset)
    effective_top_k = max(0, min(args.top_k, speed_cfg.top_k)) if args.top_k > 0 else speed_cfg.top_k
    effective_lang_top_n = max(16, min(args.lang_top_n, speed_cfg.lang_top_n))
    effective_repetition_penalty = min(args.repetition_penalty, speed_cfg.repetition_penalty)
    effective_no_repeat_ngram = min(args.no_repeat_ngram, speed_cfg.no_repeat_ngram) if speed_cfg.no_repeat_ngram > 0 else 0
    effective_repeat_window = min(args.repeat_window, speed_cfg.repeat_window)
    fast_engine = FastOutputEngine(tokenizer=tokenizer, lang_mode=lang_mode, stream=(not args.no_stream))

    prompt_for_model = build_prompt(args.prompt, adapter.model_type, tok_cfg, lang_mode, args.chat_format)

    if tokenizer is not None:
        encoded = tokenizer.encode(prompt_for_model)
        token_ids = list(encoded.ids)
    else:
        token_ids = [1]

    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    print("[vspec-chat] lang_mode=", lang_mode)
    print("[vspec-chat] chat_format=", args.chat_format)
    print("[vspec-chat] speed_preset=", args.speed_preset)
    print("[vspec-chat] decode_top_k=", effective_top_k)
    print("[vspec-chat] decode_lang_top_n=", effective_lang_top_n)

    print("[vspec-chat] prompt_tokens=", len(token_ids))
    generated = []

    start_ts = time.perf_counter()
    vram_before = cuda_mem_info() if args.device in {"cuda", "cuda-native"} else None

    def _runtime_progress(stage: str, current: int, total: int) -> None:
        if not show_progress:
            return
        if stage != "layer_load" or total <= 0:
            return
        pct = 36 + int((current / total) * 22)
        _progress(show_progress, min(58, pct), "runtime-load", f"layer {current}/{total}")

    runtime = build_generic_runtime(config, weight_index, args.max_layers, args.device, _runtime_progress)
    runtime_mode = "stub"
    if runtime is None:
        reason = runtime_load_reason(
            weight_index,
            [
                "model.embed_tokens.weight",
                "tok_embeddings.weight",
                "transformer.wte.weight",
            ],
            [
                "lm_head.weight",
                "output.weight",
                "transformer.lm_head.weight",
            ],
        )
        print(f"[vspec-chat] runtime=stub ({reason})")
        _progress(show_progress, 55, "runtime", "fallback stub mode")
    else:
        print("[vspec-chat] runtime_device=", args.device)
        runtime.eos_token_id = adapter.eos_token_id
        if args.device == "cuda-native":
            runtime_mode = "vspec-native-cuda"
        elif args.device == "cuda":
            runtime_mode = "vspec-torch-cuda"
        else:
            runtime_mode = "vspec-cpu"

        _progress(show_progress, 60, "runtime", runtime_mode)

        # Prefill KV cache with prompt tokens so attention has context.
        if len(token_ids) > 1 and hasattr(runtime, "cache_k"):
            runtime.cache_k = []
            runtime.cache_v = []
            runtime.position = 0
            prefill_ids = token_ids[:-1]
            total_prefill = len(prefill_ids)
            if hasattr(runtime, "prefill_tokens"):
                runtime.prefill_tokens(prefill_ids)
                if total_prefill > 0:
                    _progress(show_progress, 80, "prefill", f"{total_prefill}/{total_prefill} tokens")
            else:
                for idx, tid in enumerate(prefill_ids, start=1):
                    runtime.forward_logits([tid])
                    if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                        pct = 60 + int((idx / total_prefill) * 20)
                        _progress(show_progress, min(80, pct), "prefill", f"{idx}/{total_prefill} tokens")

    # Placeholder sampling loop. Qwen ops stubs are in src/model/qwen_ops.c.
    # Next step: wire logits from Vspec runtime forward pass.
    if args.max_tokens > 0:
        max_steps = args.max_tokens
    else:
        max_ctx = int(config.get("max_position_embeddings", 4096) or 4096)
        max_steps = max(1, min(4096, max_ctx - len(token_ids)))

    _progress(show_progress, 82, "decode", f"max_steps={max_steps}")
    fast_engine.begin_stream()
    stream_buffer = []
    for step in range(max_steps):
        if runtime is None:
            logits = [random.uniform(-1.0, 1.0) for _ in range(vocab_size)]
        else:
            logits = runtime.forward_logits(token_ids)
            if not logits:
                logits = [random.uniform(-1.0, 1.0) for _ in range(vocab_size)]
        logits = _apply_generation_controls(
            logits,
            token_ids,
            effective_repetition_penalty,
            effective_repeat_window,
            effective_no_repeat_ngram,
        )
        next_id = fast_engine.sample(
            logits,
            args.temperature,
            effective_top_k,
            args.greedy,
            effective_lang_top_n,
        )
        generated.append(next_id)
        stream_buffer.append(next_id)
        fast_engine.stream_token(next_id)
        token_ids.append(next_id)
        if adapter.eos_token_id is not None and next_id == adapter.eos_token_id:
            break

        if args.no_stream and (step == 0 or (step + 1) % max(1, max_steps // 4) == 0):
            decode_pct = 82 + int(((step + 1) / max_steps) * 17)
            _progress(show_progress, min(99, decode_pct), "decode", f"{step + 1}/{max_steps} tokens")

    fast_engine.end_stream()
    _progress(show_progress, 100, "done", "generation complete")

    if tokenizer is not None:
        text = tokenizer.decode(generated)
    else:
        text = "<tokens> " + " ".join(str(t) for t in generated[:16])
    text = postprocess_output_text(text, args.prompt, lang_mode)

    if args.no_stream:
        print("[vspec-chat] output:")
        print(text)

    elapsed = time.perf_counter() - start_ts
    vram_after = cuda_mem_info() if args.device in {"cuda", "cuda-native"} else None
    prompt_tok = len(token_ids) - len(generated)
    gen_tok = len(generated)
    total_tok = prompt_tok + gen_tok
    tps = (gen_tok / elapsed) if elapsed > 0 else 0.0

    print("[vspec-chat] runtime_mode=", runtime_mode)
    print("[vspec-chat] runtime_is_vspec=", runtime is not None)
    print("[vspec-chat] timing_sec=", round(elapsed, 4))
    print("[vspec-chat] tokens_prompt=", prompt_tok)
    print("[vspec-chat] tokens_generated=", gen_tok)
    print("[vspec-chat] tokens_total=", total_tok)
    print("[vspec-chat] tokens_per_sec=", round(tps, 4))

    if args.target_bits > 0 and runtime is not None and hasattr(runtime, "layers"):
        layer_bits = build_layer_bits(len(runtime.layers), args.target_bits)
        est_bits = effective_bits(layer_bits)
        print("[vspec-chat] lowbit_mode=policy")
        print("[vspec-chat] target_bits=", args.target_bits)
        print("[vspec-chat] layer_bits=", summarize_layer_bits(layer_bits))
        print("[vspec-chat] effective_bits_estimate=", round(est_bits, 4))
    low_bit = all(("int4" in k or "int3" in k or "int2" in k) for k in dtype_stats.keys()) if dtype_stats else False
    if args.target_bits in {2, 3, 4}:
        low_bit = True
    print("[vspec-chat] processing_leq_4bit=", low_bit)
    if not low_bit:
        print("[vspec-chat] note=weights are not <=4-bit in this path (mostly fp16/bf16/fp32)")
    elif args.target_bits > 0:
        print("[vspec-chat] note=low-bit policy telemetry enabled; kernel path remains quality-first")

    if vram_before and vram_after:
        free_before, total_before = vram_before
        free_after, total_after = vram_after
        used_before = total_before - free_before
        used_after = total_after - free_after
        print("[vspec-chat] vram_total_bytes=", total_after)
        print("[vspec-chat] vram_used_before_bytes=", used_before)
        print("[vspec-chat] vram_used_after_bytes=", used_after)
        print("[vspec-chat] vram_delta_bytes=", used_after - used_before)


if __name__ == "__main__":
    main()
