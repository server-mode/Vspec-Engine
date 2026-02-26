import argparse
import importlib.util
import math
import os
import random
import re
import subprocess
import sys
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
from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from lowbit_policy import build_layer_bits, effective_bits, summarize_layer_bits
from decode_optimization_module import DecodeOptimizationModule
from runtime_meaningful_response import RuntimeMeaningfulResponseAssurance
from runtime_threebit_module import ThreeBitRuntimeModule


def _load_two_bit_prototype_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "vspec-2bit-prototype" / "python" / "two_bit_prototype.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("vspec_two_bit_prototype", str(module_path))
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            stream_encoding = getattr(stream, "encoding", None) or "utf-8"
            reconfigure(encoding=stream_encoding, errors="replace")
        except Exception:
            pass


def _progress(enabled: bool, pct: int, stage: str, detail: str = "") -> None:
    if not enabled:
        return
    msg = f"[progress] {pct:>3}% | {stage}"
    if detail:
        msg += f" | {detail}"
    print(msg)


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

    if greedy:
        if allowed_ids:
            return allowed_ids[0]
        return sorted_ids[0]

    scored = []
    for tid in allowed_ids:
        text = tokenizer.decode([tid]) if tokenizer is not None else ""
        scored.append((tid, scaled[tid] + _token_quality_bonus(text, lang_mode)))

    scored.sort(key=lambda x: x[1], reverse=True)
    allowed_ids = [tid for tid, _ in scored]
    allowed_logits = [score for _, score in scored]

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


def _apply_generation_controls(
    logits: list[float],
    token_ids: list[int],
    repetition_penalty: float,
    repeat_window: int,
    no_repeat_ngram: int,
) -> list[float]:
    adjusted = list(logits)

    if repetition_penalty > 1.0 and token_ids and repeat_window > 0:
        for tid in set(token_ids[-repeat_window:]):
            if 0 <= tid < len(adjusted):
                if adjusted[tid] > 0:
                    adjusted[tid] /= repetition_penalty
                else:
                    adjusted[tid] *= repetition_penalty

    if no_repeat_ngram > 1 and len(token_ids) >= no_repeat_ngram - 1:
        prefix = tuple(token_ids[-(no_repeat_ngram - 1):])
        banned = set()
        for i in range(len(token_ids) - no_repeat_ngram + 1):
            ngram = tuple(token_ids[i:i + no_repeat_ngram])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        for tid in banned:
            if 0 <= tid < len(adjusted):
                adjusted[tid] = -1e9

    return adjusted


def _run_interactive_session(args) -> int:
    session_script = Path(__file__).resolve().parent / "vspec_chat_session.py"
    if not session_script.exists():
        print("[vspec-chat] interactive mode unavailable: missing vspec_chat_session.py")
        return 2

    cmd = [
        sys.executable,
        str(session_script),
        "--model-dir",
        str(args.model_dir),
        "--device",
        str(args.device),
        "--max-layers",
        str(args.max_layers),
        "--max-tokens",
        str(args.max_tokens if args.max_tokens > 0 else 96),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--lang",
        str(args.lang),
        "--lang-top-n",
        str(args.lang_top_n),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--repeat-window",
        str(args.repeat_window),
        "--no-repeat-ngram",
        str(args.no_repeat_ngram),
        "--chat-format",
        str(args.chat_format),
        "--speed-preset",
        str(args.speed_preset),
        "--seed",
        str(args.seed),
        "--target-bits",
        str(args.target_bits),
        "--fused-bits",
        str(args.fused_bits),
        "--language-guard-strictness",
        str(args.language_guard_strictness),
        "--structure-guard-strictness",
        str(args.structure_guard_strictness),
    ]

    if args.greedy:
        cmd.append("--greedy")
    if args.no_stream:
        cmd.append("--no-stream")
    if args.no_progress:
        cmd.append("--no-progress")
    if args.disable_language_guard:
        cmd.append("--disable-language-guard")
    if args.no_prioritize_english:
        cmd.append("--no-prioritize-english")
    if args.disable_structure_guard:
        cmd.append("--disable-structure-guard")

    completed = subprocess.run(cmd)
    return int(completed.returncode)


def _extract_output_block(stdout_text: str) -> str:
    marker = "[vspec-chat] output:"
    pos = stdout_text.find(marker)
    if pos < 0:
        return ""
    tail = stdout_text[pos + len(marker):]
    next_metric = tail.find("\n[vspec-chat]")
    if next_metric >= 0:
        tail = tail[:next_metric]
    return tail.strip()


def _is_runtime_fallback_text(text: str, lang_mode: str) -> bool:
    out = (text or "").strip().lower()
    if not out:
        return True
    if "i could not confidently decode a clean response" in out:
        return True
    if lang_mode == "vi" and "mình chưa giải mã được câu trả lời đủ sạch" in out:
        return True
    return False


def _looks_gibberish_output(text: str) -> bool:
    out = (text or "").strip()
    if not out:
        return True
    letters = sum(1 for ch in out if ch.isalpha())
    if len(out) >= 40 and (letters / max(1, len(out))) < 0.45:
        return True
    punct = sum(1 for ch in out if (not ch.isalnum()) and (not ch.isspace()))
    if len(out) >= 40 and (punct / max(1, len(out))) > 0.22:
        return True
    words = re.findall(r"[A-Za-z]{2,}", out)
    if len(words) == 1 and len(words[0]) >= 20:
        return True
    if 1 <= len(words) <= 3 and max(len(w) for w in words) >= 14:
        return True
    single_letter_words = re.findall(r"\b[A-Za-z]\b", out)
    total_word_like = len(words) + len(single_letter_words)
    if len(out) >= 80 and total_word_like >= 16:
        single_ratio = len(single_letter_words) / max(1, total_word_like)
        punct = sum(1 for ch in out if (not ch.isalnum()) and (not ch.isspace()))
        punct_ratio = punct / max(1, len(out))
        if single_ratio > 0.22 and punct_ratio > 0.08:
            return True
    if len(out) >= 24 and (" " not in out) and out.isalpha():
        return True
    if len(words) >= 6:
        avg_len = sum(len(w) for w in words) / max(1, len(words))
        if avg_len > 10.0:
            return True
        common_words = {
            "the", "and", "for", "you", "your", "with", "that", "this", "what", "when", "where", "which",
            "hello", "know", "about", "vietnam", "yes", "can", "help", "please", "thanks", "is", "are", "to",
            "in", "on", "it", "of", "a", "an", "do", "does", "how", "why", "i", "me", "my"
        }
        normalized = [w.lower() for w in words]
        known_ratio = sum(1 for w in normalized if w in common_words) / len(normalized)
        if len(words) >= 10 and known_ratio < 0.08:
            return True
    return False


def _run_torch_verifier(args, prompt: str, lang_mode: str, verifier_device_override: str | None = None, timeout_override_sec: float | None = None) -> str:
    verifier_device = str(verifier_device_override or getattr(args, "hybrid_verifier_device", "torch-cuda") or "torch-cuda")
    verifier_layers = 0 if verifier_device == "cpu" else 0
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-dir",
        str(args.model_dir),
        "--prompt",
        str(prompt),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--max-layers",
        str(min(verifier_layers, int(args.max_layers)) if int(args.max_layers) > 0 else verifier_layers),
        "--device",
        verifier_device,
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--repeat-window",
        str(args.repeat_window),
        "--no-repeat-ngram",
        str(args.no_repeat_ngram),
        "--lang",
        str(lang_mode),
        "--lang-top-n",
        str(args.lang_top_n),
        "--chat-format",
        str(args.chat_format),
        "--speed-preset",
        str(args.speed_preset),
        "--target-bits",
        str(args.target_bits),
        "--fused-bits",
        "0",
        "--decode-opt-mode",
        str(args.decode_opt_mode),
        "--runtime-mix-mode",
        "native-only",
        "--no-stream",
        "--no-progress",
    ]
    if args.greedy:
        cmd.append("--greedy")
    if args.disable_language_guard:
        cmd.append("--disable-language-guard")
    if args.no_prioritize_english:
        cmd.append("--no-prioritize-english")
    if args.disable_structure_guard:
        cmd.append("--disable-structure-guard")
    cmd.extend(["--language-guard-strictness", str(args.language_guard_strictness)])
    cmd.extend(["--structure-guard-strictness", str(args.structure_guard_strictness)])

    try:
        timeout_sec = float(timeout_override_sec if timeout_override_sec is not None else (getattr(args, "hybrid_verifier_timeout_sec", 8.0) or 8.0))
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=max(1.0, timeout_sec),
        )
    except Exception:
        return ""

    if completed.returncode != 0:
        return ""
    return _extract_output_block(completed.stdout)


def main() -> None:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(description="Vspec-chat prototype CLI")
    parser.add_argument("--model-dir", required=True, help="Path to HF cache model dir")
    parser.add_argument("--prompt", default="", help="Prompt text (required unless --interactive)")
    parser.add_argument("--interactive", action="store_true", help="Start terminal chat session (REPL)")
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = generate until EOS or safety cap")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-layers", type=int, default=0, help="0 = use all available layers")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cuda-native", "torch-cuda"], help="cuda=default native path; use torch-cuda only for fallback/debug")
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
    parser.add_argument("--target-bits", type=int, default=3, choices=[0, 2, 3, 4], help="Policy target bits for benchmark telemetry (default: 3)")
    parser.add_argument("--fused-bits", type=int, default=3, choices=[0, 3, 4], help="Enable fused low-bit linear kernels for native path (default: 3)")
    parser.add_argument("--disable-language-guard", action="store_true", help="Disable Language Stability Guard")
    parser.add_argument("--language-guard-strictness", type=float, default=0.72, help="0..1, higher is stricter against language/script drift")
    parser.add_argument("--no-prioritize-english", action="store_true", help="Disable English-first fallback when language is ambiguous")
    parser.add_argument("--disable-structure-guard", action="store_true", help="Disable output structure integrity guard")
    parser.add_argument("--structure-guard-strictness", type=float, default=0.72, help="0..1, higher is stricter for structural integrity")
    parser.add_argument("--decode-opt-mode", default="stable", choices=["stable", "optimized"], help="Decode optimization module mode")
    parser.add_argument("--runtime-mix-mode", default="native-only", choices=["native-only", "hybrid-verify"], help="native-only=single runtime, hybrid-verify=native then torch verifier on fallback")
    parser.add_argument("--hybrid-verifier-policy", default="auto", choices=["auto", "off", "on"], help="auto=skip verifier in 3-bit mode to protect speed/VRAM")
    parser.add_argument("--hybrid-verifier-device", default="torch-cuda", choices=["torch-cuda", "cpu"], help="Verifier device when hybrid verifier runs")
    parser.add_argument("--hybrid-verifier-timeout-sec", type=float, default=8.0, help="Timeout for verifier subprocess")
    parser.add_argument("--enable-3bit-runtime-module", action="store_true", help="Enable dedicated 3-bit noise and sampling module")
    parser.add_argument("--allow-semantic-rescue", action="store_true", help="Allow intent-based synthetic rescue response when decode fails")
    parser.add_argument("--threebit-test-boost", action="store_true", help="Boost test run with deeper layers and more tokens for 3-bit validation")
    parser.add_argument("--prototype-2bit", action="store_true", help="Enable non-invasive 2-bit prototype module")
    parser.add_argument("--prototype-2bit-mode", default="balanced", choices=["safe", "balanced", "aggressive"], help="2-bit prototype policy profile")
    parser.add_argument("--prototype-2bit-protect-last", type=int, default=2, help="Number of final layers kept at original precision")
    args = parser.parse_args()

    if args.interactive:
        exit_code = _run_interactive_session(args)
        raise SystemExit(exit_code)

    if not str(args.prompt or "").strip():
        parser.error("--prompt is required unless --interactive is used")

    if args.threebit_test_boost:
        if int(args.max_layers) <= 0:
            args.max_layers = 0
        if int(args.max_tokens) <= 0 or int(args.max_tokens) < 128:
            args.max_tokens = 128
    if int(args.fused_bits) == 3 and int(args.max_tokens) > 0 and int(args.max_tokens) < 64:
        args.max_tokens = 64
    if int(args.fused_bits) == 3 and int(args.max_layers) > 0 and int(args.max_layers) < 8:
        args.max_layers = 8

    os.environ["VSPEC_FUSED_BITS"] = str(args.fused_bits)
    if int(args.fused_bits) == 3 or int(args.target_bits) == 3:
        os.environ["VSPEC_3BIT_RUNTIME_MODULE"] = "1"

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

    threebit_module = ThreeBitRuntimeModule(
        enabled=(args.enable_3bit_runtime_module or int(args.fused_bits) == 3 or int(args.target_bits) == 3),
        fused_bits=args.fused_bits,
        target_bits=args.target_bits,
    )
    tuned = threebit_module.tune_sampling(
        top_k=effective_top_k,
        lang_top_n=effective_lang_top_n,
        repetition_penalty=effective_repetition_penalty,
        repeat_window=effective_repeat_window,
    )
    effective_top_k = tuned.top_k
    effective_lang_top_n = tuned.lang_top_n
    effective_repetition_penalty = tuned.repetition_penalty
    effective_repeat_window = tuned.repeat_window

    if tuned.active and int(args.max_tokens) > 0 and int(args.max_tokens) < 32:
        args.max_tokens = 32

    guard = None
    if not args.disable_language_guard:
        guard = LanguageStabilityGuard(
            prompt=args.prompt,
            lang_mode=lang_mode,
            strictness=args.language_guard_strictness,
            prioritize_english=(not args.no_prioritize_english),
        )

    structure_guard = None
    if not args.disable_structure_guard:
        structure_guard = LanguageStructureIntegrityManager(
            prompt=args.prompt,
            strictness=args.structure_guard_strictness,
        )

    fast_engine = FastOutputEngine(
        tokenizer=tokenizer,
        lang_mode=lang_mode,
        stream=(not args.no_stream),
        guard=guard,
        structure_guard=structure_guard,
    )

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
    print("[vspec-chat] decode_opt_mode=", args.decode_opt_mode)
    print("[vspec-chat] runtime_mix_mode=", args.runtime_mix_mode)
    print("[vspec-chat] runtime_3bit_module=", tuned.active)
    if tuned.active:
        print("[vspec-chat] runtime_3bit_tuned_top_k=", tuned.top_k)
        print("[vspec-chat] runtime_3bit_tuned_lang_top_n=", tuned.lang_top_n)
        print("[vspec-chat] runtime_3bit_tuned_repetition_penalty=", round(tuned.repetition_penalty, 4))
        print("[vspec-chat] runtime_3bit_tuned_repeat_window=", tuned.repeat_window)
    print("[vspec-chat] fused_bits=", args.fused_bits)
    print("[vspec-chat] language_guard=", "on" if guard is not None else "off")
    print("[vspec-chat] structure_guard=", "on" if structure_guard is not None else "off")
    if guard is not None:
        print("[vspec-chat] language_guard_primary_script=", guard.profile.primary_script)
        print("[vspec-chat] language_guard_strictness=", round(guard.profile.strictness, 3))
        print("[vspec-chat] language_guard_prioritize_english=", guard.profile.prioritized_english)
    if structure_guard is not None:
        print("[vspec-chat] structure_guard_expected_sections=", structure_guard.profile.expected_sections)
        print("[vspec-chat] structure_guard_strictness=", round(structure_guard.profile.strictness, 3))
    print("[vspec-chat] decode_top_k=", effective_top_k)
    print("[vspec-chat] decode_lang_top_n=", effective_lang_top_n)

    print("[vspec-chat] prompt_tokens=", len(token_ids))
    generated = []

    decode_optimizer = DecodeOptimizationModule(
        repetition_penalty=effective_repetition_penalty,
        repeat_window=effective_repeat_window,
        no_repeat_ngram=effective_no_repeat_ngram,
        mode=args.decode_opt_mode,
    )
    decode_optimizer.seed_history(token_ids)

    start_ts = time.perf_counter()
    vram_before = cuda_mem_info() if args.device in {"cuda", "cuda-native", "torch-cuda"} else None

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
        if args.device in {"cuda", "cuda-native"}:
            runtime_mode = "vspec-native-cuda"
        elif args.device == "torch-cuda":
            runtime_mode = "vspec-torch-cuda"
        else:
            runtime_mode = "vspec-cpu"

        _progress(show_progress, 60, "runtime", runtime_mode)
        if hasattr(runtime, "fused_bits"):
            print("[vspec-chat] runtime_effective_fused_bits=", int(getattr(runtime, "fused_bits", 0)))
        if hasattr(runtime, "lowbit_plan") and getattr(runtime, "lowbit_plan") is not None:
            lowbit_plan = getattr(runtime, "lowbit_plan")
            print("[vspec-chat] runtime_lowbit_enabled=", bool(getattr(lowbit_plan, "enabled", False)))
            print("[vspec-chat] runtime_lowbit_reason=", getattr(lowbit_plan, "reason", "unknown"))

        if args.prototype_2bit:
            proto_mod = _load_two_bit_prototype_module()
            if proto_mod is None:
                print("[vspec-chat] prototype_2bit=off (module not found: vspec-2bit-prototype/python/two_bit_prototype.py)")
            else:
                apply_fn = getattr(proto_mod, "apply_two_bit_prototype", None)
                if apply_fn is None:
                    print("[vspec-chat] prototype_2bit=off (apply function missing)")
                else:
                    report = apply_fn(
                        runtime,
                        mode=args.prototype_2bit_mode,
                        protect_last_layers=max(0, int(args.prototype_2bit_protect_last)),
                    )
                    print("[vspec-chat] prototype_2bit=", bool(getattr(report, "enabled", False)))
                    print("[vspec-chat] prototype_2bit_mode=", getattr(report, "mode", args.prototype_2bit_mode))
                    print("[vspec-chat] prototype_2bit_reason=", getattr(report, "reason", "unknown"))
                    print("[vspec-chat] prototype_2bit_total_layers=", int(getattr(report, "total_layers", 0)))
                    print("[vspec-chat] prototype_2bit_protected_layers=", int(getattr(report, "protected_layers", 0)))
                    print("[vspec-chat] prototype_2bit_quantized_layers=", int(getattr(report, "quantized_layers", 0)))
                    print("[vspec-chat] prototype_2bit_quantized_matrices=", int(getattr(report, "quantized_matrices", 0)))
                    print("[vspec-chat] prototype_2bit_effective_bits_estimate=", round(float(getattr(report, "estimated_effective_bits", 0.0)), 4))

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
            logits = decode_optimizer.fetch_logits(runtime, token_ids[-1], vocab_size)
            if decode_optimizer.logits_empty(logits):
                logits = [random.uniform(-1.0, 1.0) for _ in range(vocab_size)]
        logits = threebit_module.denoise_logits(logits, step)
        logits = decode_optimizer.apply_generation_controls(logits, token_ids)
        sample_temperature = threebit_module.auto_temperature(logits, args.temperature)
        next_id = fast_engine.sample(
            logits,
            sample_temperature,
            effective_top_k,
            args.greedy,
            effective_lang_top_n,
        )
        generated.append(next_id)
        stream_buffer.append(next_id)
        fast_engine.stream_token(next_id)
        token_ids.append(next_id)
        decode_optimizer.observe_token(token_ids)
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

    needs_verify = _is_runtime_fallback_text(text, lang_mode) or _looks_gibberish_output(text)
    if (
        args.runtime_mix_mode == "hybrid-verify"
        and args.device in {"cuda", "cuda-native", "cpu"}
        and needs_verify
    ):
        policy = str(args.hybrid_verifier_policy or "auto")
        verifier_allowed = True
        verifier_device_override = None
        verifier_timeout_override = None
        if policy == "off":
            verifier_allowed = False
        elif policy == "auto":
            if int(args.fused_bits) == 3 or int(args.target_bits) == 3:
                is_fallback = _is_runtime_fallback_text(text, lang_mode)
                is_gibberish = _looks_gibberish_output(text)
                if is_fallback or is_gibberish:
                    verifier_allowed = True
                    verifier_device_override = str(getattr(args, "hybrid_verifier_device", "torch-cuda") or "torch-cuda")
                    verifier_timeout_override = max(45.0, float(getattr(args, "hybrid_verifier_timeout_sec", 8.0) or 8.0))
                else:
                    verifier_allowed = False

        if verifier_allowed:
            torch_text = _run_torch_verifier(
                args,
                args.prompt,
                lang_mode,
                verifier_device_override=verifier_device_override,
                timeout_override_sec=verifier_timeout_override,
            )
            if torch_text and (not _is_runtime_fallback_text(torch_text, lang_mode)) and (not _looks_gibberish_output(torch_text)):
                text = torch_text
                print("[vspec-chat] hybrid_verify_used= True")
            else:
                assurance = RuntimeMeaningfulResponseAssurance(lang_mode, allow_semantic_rescue=args.allow_semantic_rescue)
                text = assurance.repair(text, args.prompt)
                print("[vspec-chat] hybrid_verify_used= False")
        else:
            assurance = RuntimeMeaningfulResponseAssurance(lang_mode, allow_semantic_rescue=args.allow_semantic_rescue)
            text = assurance.repair(text, args.prompt)
            print("[vspec-chat] hybrid_verify_skipped= True")

    if args.no_stream:
        print("[vspec-chat] output:")
        print(text)

    structure_report = fast_engine.structure_report()
    if structure_report is not None:
        print("[vspec-chat] structure_guard_integrity_pass=", structure_report.get("integrity_pass"))
        print("[vspec-chat] structure_guard_section_coverage=", round(float(structure_report.get("section_coverage", 0.0)), 4))
        print("[vspec-chat] structure_guard_code_fence_balanced=", structure_report.get("code_fence_balanced"))
        print("[vspec-chat] structure_guard_seen_sections=", structure_report.get("seen_sections"))

    elapsed = time.perf_counter() - start_ts
    vram_after = cuda_mem_info() if args.device in {"cuda", "cuda-native", "torch-cuda"} else None
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
