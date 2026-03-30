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
from runtime_inference import build_generic_runtime, runtime_load_reason, runtime_matrix_bits_summary
from chat_prompt import build_prompt
from vspec_cuda_bridge import cuda_mem_info
from fast_output import FastOutputEngine, postprocess_output_text, resolve_speed_preset
from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from lowbit_policy import build_layer_bits, effective_bits, summarize_layer_bits
from decode_optimization_module import DecodeOptimizationModule
from runtime_meaningful_response import RuntimeMeaningfulResponseAssurance
from runtime_lowbit_module import lowbit_projection_stats_reset, lowbit_projection_stats_snapshot
from runtime_threebit_module import ThreeBitRuntimeModule
from decode_phase1_contract import DecodeState, PythonDecodeOrchestrator
from decode_phase2_prefill import run_prefill_with_core_scheduler
from decode_phase3_step_dispatch import Phase3StepDispatcher
from native_safe_decode import resolve_native_safe_max_layers
from runtime_core_bridge import (
    CoreDecodeSession,
    CoreNativeForwardContext,
    CoreNativeDecodeLoop,
    adaptive_step,
    native_anf_observe_activations,
    native_anf_observe_quality,
    native_anf_prototype_enabled,
    native_anf_report,
)
from generation_batch_driver import CoreBatchGenerationDriver, ManagedGenerationRequest


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
    print(msg, flush=True)


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


def _anf_quality_proxies(logits_obj, entropy_now: float) -> tuple[float, float, float]:
    try:
        vals = logits_obj.tolist() if hasattr(logits_obj, "tolist") else list(logits_obj)
        if not vals:
            return 0.0, 0.0, 0.0
        n = float(len(vals))
        mean = sum(float(v) for v in vals) / n
        var = sum((float(v) - mean) * (float(v) - mean) for v in vals) / n
        std = math.sqrt(max(0.0, var))
        residual_rms = min(2.0, max(0.0, std / 4.0))
        entropy_collapse = min(1.0, max(0.0, entropy_now / max(1.0, math.log(max(2.0, n)))))
        activation_norm_drift = min(1.0, max(0.0, abs(mean) / 8.0 + std / 16.0))
        return residual_rms, entropy_collapse, activation_norm_drift
    except Exception:
        return 0.0, 0.0, 0.0


def _resolve_budget_step_cap(
    requested_steps: int,
    decode_budget_seconds: float,
    prefill_tokens: int,
    layer_count: int,
    lowbit_enabled: bool,
    fused_bits: int,
) -> int:
    if requested_steps <= 0:
        return 1
    if decode_budget_seconds <= 0.0:
        return int(requested_steps)

    # Conservative latency estimate (seconds/token) to keep decode within budget.
    token_latency = 0.012 + (0.0018 * float(max(1, int(layer_count))))
    if lowbit_enabled and int(fused_bits) in {3, 4}:
        token_latency += 0.040
    elif lowbit_enabled:
        token_latency += 0.025
    if int(prefill_tokens) >= 256:
        token_latency *= 1.12
    token_latency = max(0.010, min(2.000, token_latency))

    reserve = max(3.0, min(20.0, float(decode_budget_seconds) * 0.15))
    usable = max(0.5, float(decode_budget_seconds) - reserve)
    cap = int(usable / token_latency)
    cap = max(12, cap)
    return max(1, min(int(requested_steps), cap))


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
    if re.search(r"\b(path|size|state|windows|println|return|case|break|if|else|switch|class|public|private|void)\b", text.lower()):
        bonus -= 1.6
    if "\t" in text or "\r" in text or "\n" in text:
        bonus -= 1.2
    if re.search(r"[{}();]", text):
        bonus -= 0.7
    if lang_mode == "vi":
        if re.search(r"\b(path|size|state|windows|println|return|case|break|if|else)\b", text.lower()):
            bonus -= 0.8
        if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ]", text.lower()):
            bonus += 0.35
        if re.search(r"\b(và|là|của|bạn|tôi|được|không|xin|chào)\b", text.lower()):
            bonus += 0.25
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
    entropy_floor: float = 0.45,
    tail_flatten_std_floor: float = 0.85,
    top12_margin_cap: float = 7.5,
) -> list[float]:
    adjusted = list(logits)

    if not adjusted:
        return adjusted

    mean_v = sum(adjusted) / float(len(adjusted))
    var_v = 0.0
    for value in adjusted:
        d = value - mean_v
        var_v += d * d
    var_v /= float(max(1, len(adjusted)))
    std_v = math.sqrt(max(var_v, 1e-12))

    if std_v < tail_flatten_std_floor:
        sharpen = min(2.5, tail_flatten_std_floor / max(std_v, 1e-6))
        adjusted = [mean_v + (value - mean_v) * sharpen for value in adjusted]

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

    candidate_n = min(64, len(adjusted))
    if candidate_n >= 2:
        sorted_ids = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
        top_ids = sorted_ids[:candidate_n]
        top_logits = [adjusted[i] for i in top_ids]
        max_logit = max(top_logits)

        exp_vals = [math.exp(v - max_logit) for v in top_logits]
        total = sum(exp_vals)
        if total > 0.0:
            probs = [v / total for v in exp_vals]
            entropy = 0.0
            for p in probs:
                if p > 1e-12:
                    entropy += -p * math.log(p)
            max_entropy = math.log(float(candidate_n)) if candidate_n > 1 else 1.0
            norm_entropy = entropy / max(max_entropy, 1e-6)

            if norm_entropy < entropy_floor:
                relax = min(0.35, entropy_floor - norm_entropy)
                for idx in top_ids:
                    adjusted[idx] *= (1.0 - relax)

            first = top_ids[0]
            second = top_ids[1]
            margin = adjusted[first] - adjusted[second]
            if margin > top12_margin_cap:
                adjusted[first] -= (margin - top12_margin_cap)

    if len(token_ids) >= 3:
        recent = token_ids[-3:]
        if recent[0] == recent[1] == recent[2]:
            tid = recent[-1]
            if 0 <= tid < len(adjusted):
                adjusted[tid] -= 2.5

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
        "--decode-opt-mode",
        str(args.decode_opt_mode),
        "--max-decode-seconds",
        str(args.max_decode_seconds),
        "--max-retry-seconds",
        str(args.max_retry_seconds),
        "--allow-semantic-rescue",
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
    if args.enable_3bit_runtime_module:
        cmd.append("--enable-3bit-runtime-module")
    if args.unsafe_low_layers:
        cmd.append("--unsafe-low-layers")
    completed = subprocess.run(cmd)
    return int(completed.returncode)


def _run_prompt_file_batch(
    args,
    runtime,
    tokenizer,
    adapter,
    tok_cfg,
    threebit_module: ThreeBitRuntimeModule,
    effective_top_k: int,
    effective_lang_top_n: int,
    effective_repetition_penalty: float,
    effective_repeat_window: int,
    effective_no_repeat_ngram: int,
    show_progress: bool,
) -> int:
    prompts_path = Path(args.prompts_file)
    raw_lines = prompts_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    prompts = [line.strip() for line in raw_lines if line.strip()]
    if not prompts:
        print(f"[vspec-chat] batch_prompts=empty file={prompts_path}")
        return 2

    max_prompt_chars = max(len(prompt) for prompt in prompts)
    if float(args.max_decode_seconds) > 0.0:
        decode_budget_seconds = max(0.5, float(args.max_decode_seconds))
    elif float(args.max_decode_seconds) == 0.0:
        decode_budget_seconds = 0.0
    else:
        layer_count = len(getattr(runtime, "layers", []) or []) if runtime is not None else 0
        work_units = max(1, (max_prompt_chars + max(1, int(args.max_tokens))) * max(1, layer_count))
        auto_budget = 18.0 + (0.16 * float(max(1, int(args.max_tokens)))) + (0.006 * float(max_prompt_chars)) + (0.003 * float(work_units))
        decode_budget_seconds = min(240.0, max(24.0, auto_budget))

    requests: list[ManagedGenerationRequest] = []
    for prompt in prompts:
        req_lang = str(args.lang)
        if req_lang == "auto":
            req_lang = _detect_lang(prompt)
        prompt_for_model = build_prompt(prompt, adapter.model_type, tok_cfg, req_lang, args.chat_format)
        if tokenizer is not None:
            encoded = tokenizer.encode(prompt_for_model)
            token_ids = list(encoded.ids)
        else:
            token_ids = [1]
        if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
            token_ids = [adapter.bos_token_id] + token_ids
        requests.append(
            ManagedGenerationRequest(
                prompt=prompt,
                lang_mode=req_lang,
                token_ids=token_ids,
                max_new_tokens=max(1, int(args.max_tokens) if int(args.max_tokens) > 0 else 96),
                temperature=float(args.temperature),
                top_k=effective_top_k,
                greedy=bool(args.greedy),
                lang_top_n=effective_lang_top_n,
                repetition_penalty=effective_repetition_penalty,
                repeat_window=effective_repeat_window,
                no_repeat_ngram=effective_no_repeat_ngram,
                stream=False,
                disable_language_guard=bool(args.disable_language_guard),
                language_guard_strictness=float(args.language_guard_strictness),
                prioritize_english=bool(not args.no_prioritize_english),
                structure_guard_strictness=float(args.structure_guard_strictness),
                disable_structure_guard=bool(args.disable_structure_guard or (req_lang == "en" and not args.no_prioritize_english)),
                decode_opt_mode=str(args.decode_opt_mode),
            )
        )

    def _batch_progress(stage: str, current: int, total: int) -> None:
        if not show_progress or total <= 0:
            return
        pct = int((float(current) / float(max(1, total))) * 100.0)
        print(f"[vspec-chat][batch] {stage}= {current}/{total} ({pct}%)")

    driver = CoreBatchGenerationDriver(
        runtime=runtime,
        tokenizer=tokenizer,
        adapter=adapter,
        threebit_module=threebit_module,
        decode_budget_seconds=decode_budget_seconds,
        progress_cb=_batch_progress,
    )
    result = driver.run(requests)
    print(f"[vspec-chat] batch_requests= {len(result.requests)}")
    print(f"[vspec-chat] batch_core_scheduler= {'on' if result.used_core_batcher else 'off'}")
    if result.stats:
        print(f"[vspec-chat] batch_stats= {result.stats}")

    output_chunks: list[str] = []
    for idx, req in enumerate(result.requests, start=1):
        status = "ok"
        if req.contract_failed:
            status = "contract-failed"
        elif req.timed_out:
            status = "timed-out"
        print(f"[vspec-chat][batch][{idx}] status= {status} lang= {req.lang_mode} generated= {len(req.generated)} tps= {req.tokens_per_second:.2f}")
        print(f"[vspec-chat][batch][{idx}] prompt= {req.prompt}")
        print(f"[vspec-chat][batch][{idx}] output:")
        print(req.output_text)
        output_chunks.append(f"### Request {idx}\nPrompt: {req.prompt}\n\n{req.output_text}\n")

    if str(getattr(args, "batch_output_file", "") or "").strip():
        Path(args.batch_output_file).write_text("\n".join(output_chunks), encoding="utf-8")
        print(f"[vspec-chat] batch_output_file= {args.batch_output_file}")
    return 0


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
    if out.startswith("[vspec-decode-error]"):
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
    if len(out) >= 32 and out.count("\n") <= 1:
        quote_ratio = (out.count("\"") + out.count("'")) / max(1, len(out))
        comma_ratio = out.count(",") / max(1, len(out))
        if quote_ratio > 0.08 or comma_ratio > 0.10:
            return True
    letters = sum(1 for ch in out if ch.isalpha())
    if len(out) >= 40 and (letters / max(1, len(out))) < 0.45:
        return True
    punct = sum(1 for ch in out if (not ch.isalnum()) and (not ch.isspace()))
    if len(out) >= 40 and (punct / max(1, len(out))) > 0.22:
        return True
    words = re.findall(r"[A-Za-z]{2,}", out)
    if len(words) >= 16:
        normalized = [w.lower() for w in words]
        unique_ratio = len(set(normalized)) / max(1, len(normalized))
        if unique_ratio < 0.45:
            return True
        short_words = [w for w in normalized if len(w) <= 5]
        if short_words:
            counts = {}
            for w in short_words:
                counts[w] = counts.get(w, 0) + 1
            top_freq = max(counts.values()) if counts else 0
            if top_freq / max(1, len(short_words)) > 0.22:
                return True
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
        "--greedy",
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


def _run_native_safe_verifier(args, prompt: str, lang_mode: str, timeout_override_sec: float | None = None) -> str:
    env = os.environ.copy()
    env["VSPEC_DISABLE_NATIVE_SAFE_VERIFY"] = "1"
    safe_layers = resolve_native_safe_max_layers(read_config(find_snapshot_dir(Path(args.model_dir))), args.max_layers)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-dir",
        str(args.model_dir),
        "--prompt",
        str(prompt),
        "--max-tokens",
        str(min(max(12, int(args.max_tokens or 12)), 32)),
        "--temperature",
        str(min(0.55, max(0.20, float(args.temperature) * 0.70))),
        "--top-k",
        str(max(1, min(8, int(args.top_k or 8)))),
        "--seed",
        str(args.seed),
        "--max-layers",
        str(int(safe_layers or 0)),
        "--device",
        "cuda" if str(args.device) in {"cuda", "cuda-native"} else "cpu",
        "--repetition-penalty",
        str(max(1.18, float(args.repetition_penalty))),
        "--repeat-window",
        str(max(48, int(args.repeat_window))),
        "--no-repeat-ngram",
        str(max(3, int(args.no_repeat_ngram))),
        "--lang",
        str(lang_mode),
        "--lang-top-n",
        str(max(32, min(int(args.lang_top_n), 96))),
        "--chat-format",
        str(args.chat_format),
        "--speed-preset",
        "normal",
        "--target-bits",
        "16",
        "--fused-bits",
        "0",
        "--decode-opt-mode",
        "stable",
        "--runtime-mix-mode",
        "native-only",
        "--no-stream",
        "--no-progress",
        "--greedy",
    ]
    if args.disable_language_guard:
        cmd.append("--disable-language-guard")
    if args.no_prioritize_english:
        cmd.append("--no-prioritize-english")
    cmd.append("--disable-structure-guard")
    cmd.extend(["--language-guard-strictness", str(args.language_guard_strictness)])
    cmd.extend(["--structure-guard-strictness", str(args.structure_guard_strictness)])

    try:
        timeout_sec = float(timeout_override_sec if timeout_override_sec is not None else 30.0)
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
            timeout=max(5.0, timeout_sec),
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
    parser.add_argument("--prompts-file", default="", help="Text file with one prompt per line for continuous-batch generation")
    parser.add_argument("--batch-output-file", default="", help="Optional file to save batched outputs")
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
    parser.add_argument("--target-bits", type=int, default=3, choices=[0, 2, 3, 4, 8, 16], help="Policy target bits for telemetry (3/4=low-bit, 8/16=high-precision)")
    parser.add_argument("--fused-bits", type=int, default=3, choices=[0, 3, 4, 16], help="Enable fused low-bit linear kernels (16 = high-precision passthrough diagnostic)")
    parser.add_argument("--disable-language-guard", action="store_true", help="Disable Language Stability Guard")
    parser.add_argument("--language-guard-strictness", type=float, default=0.72, help="0..1, higher is stricter against language/script drift")
    parser.add_argument("--no-prioritize-english", action="store_true", help="Disable English-first fallback when language is ambiguous")
    parser.add_argument("--disable-structure-guard", action="store_true", help="Disable output structure integrity guard")
    parser.add_argument("--structure-guard-strictness", type=float, default=0.72, help="0..1, higher is stricter for structural integrity")
    parser.add_argument("--decode-opt-mode", default="optimized", choices=["stable", "optimized"], help="Decode optimization module mode")
    parser.add_argument("--max-decode-seconds", type=float, default=-1.0, help="<0 auto, =0 disable timeout, >0 fixed decode budget")
    parser.add_argument("--max-retry-seconds", type=float, default=-1.0, help="Interactive retry budget: <0 auto, =0 disable, >0 fixed")
    parser.add_argument("--runtime-mix-mode", default="native-only", choices=["native-only", "hybrid-verify"], help="native-only=single runtime, hybrid-verify=native then torch verifier on fallback")
    parser.add_argument("--hybrid-verifier-policy", default="auto", choices=["auto", "off", "on"], help="auto=skip verifier in 3-bit mode to protect speed/VRAM")
    parser.add_argument("--hybrid-verifier-device", default="torch-cuda", choices=["torch-cuda", "cpu"], help="Verifier device when hybrid verifier runs")
    parser.add_argument("--hybrid-verifier-timeout-sec", type=float, default=8.0, help="Timeout for verifier subprocess")
    parser.add_argument("--enable-3bit-runtime-module", action="store_true", help="Enable dedicated 3-bit noise and sampling module")
    parser.add_argument("--allow-semantic-rescue", action="store_true", help="Deprecated no-op: synthetic rescue responses are disabled by integrity policy")
    parser.add_argument("--threebit-test-boost", action="store_true", help="Boost test run with deeper layers and more tokens for 3-bit validation")
    parser.add_argument("--prototype-2bit", action="store_true", help="Enable non-invasive 2-bit prototype module")
    parser.add_argument("--prototype-2bit-mode", default="balanced", choices=["safe", "balanced", "aggressive"], help="2-bit prototype policy profile")
    parser.add_argument("--prototype-2bit-protect-last", type=int, default=2, help="Number of final layers kept at original precision")
    parser.add_argument("--unsafe-low-layers", action="store_true", help="Allow very low max-layers even if quality may collapse")
    args = parser.parse_args()

    batch_mode = bool(str(args.prompts_file or "").strip())

    if args.interactive:
        if batch_mode:
            parser.error("--interactive cannot be combined with --prompts-file")
        exit_code = _run_interactive_session(args)
        raise SystemExit(exit_code)

    if (not batch_mode) and (not str(args.prompt or "").strip()):
        parser.error("--prompt is required unless --interactive or --prompts-file is used")

    requested_max_tokens = int(args.max_tokens)
    requested_max_layers_post_parse = int(args.max_layers)

    if args.threebit_test_boost:
        if int(args.max_layers) <= 0:
            args.max_layers = 0
        if int(args.max_tokens) <= 0 or int(args.max_tokens) < 128:
            args.max_tokens = 128
    if int(args.fused_bits) == 3 and int(args.max_tokens) > 0 and int(args.max_tokens) < 64:
        args.max_tokens = 64
    if int(args.fused_bits) == 3 and int(args.max_layers) > 0 and int(args.max_layers) < 8:
        args.max_layers = 8

    fused_bits_env = int(args.fused_bits)
    if fused_bits_env not in {0, 3, 4}:
        fused_bits_env = 0
    os.environ["VSPEC_FUSED_BITS"] = str(fused_bits_env)
    os.environ.setdefault("VSPEC_INT4_PRE_REGISTER", "1")
    os.environ.setdefault("VSPEC_CUDA_GRAPH_CAPTURE", "1")
    os.environ.setdefault("VSPEC_NATIVE_GRAPH_SIG_MODE", "shape-only")
    os.environ.setdefault("VSPEC_INT4_BLOCKWISE_ENABLE", "1")
    os.environ.setdefault("VSPEC_INT4_BLOCK_SIZE", "32")
    prototype_anf_mode = (
        os.getenv("VSPEC_CHAT_PROTOTYPE", "0").strip().lower() in {"1", "true", "yes", "on"}
        and os.getenv("VSPEC_ENABLE_ANF", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    os.environ.setdefault("VSPEC_CUBLAS_CACHE_SIZE", "0" if prototype_anf_mode else "16")
    os.environ.setdefault("VSPEC_INT4_BRIDGE_CACHE_CAP", "256")
    os.environ.setdefault("VSPEC_DISABLE_PY_KV_SHADOW", "1" if prototype_anf_mode else "0")
    if prototype_anf_mode:
        os.environ.setdefault("VSPEC_ANF_TCC_ENABLE", "1")
        os.environ.setdefault("VSPEC_ANF_MAX_HOT_RATIO", "0.10")
        os.environ.setdefault("VSPEC_ANF_ACTIVATION_THRESHOLD", "1.10")
    os.environ.setdefault("VSPEC_C_SAMPLER_REQUIRED", "1")
    os.environ.setdefault("VSPEC_USE_C_SAMPLER", "1")
    os.environ.setdefault("VSPEC_PREFILL_CORE_SCHED", "1")
    if os.getenv("VSPEC_CUBLAS_CACHE_SIZE", "16").strip() in {"0", "off", "OFF"}:
        os.environ.setdefault("VSPEC_INT4_COMPUTE_MODE", "native")
    if (
        fused_bits_env == 4
        and str(args.device) in {"cuda", "cuda-native"}
        and not os.getenv("VSPEC_INT4_COMPUTE_MODE", "").strip()
    ):
        os.environ["VSPEC_INT4_COMPUTE_MODE"] = "native"
    if int(args.fused_bits) == 3 or int(args.target_bits) == 3:
        os.environ["VSPEC_3BIT_RUNTIME_MODULE"] = "1"

    show_progress = not args.no_progress

    random.seed(args.seed)

    model_dir = Path(args.model_dir)
    _progress(show_progress, 5, "snapshot", "finding latest HF snapshot")
    snapshot_dir = find_snapshot_dir(model_dir)

    def _resolve_native_model_file(snapshot_dir: Path) -> str | None:
        preferred = sorted(snapshot_dir.glob("model-*.safetensors"))
        if preferred:
            return str(preferred[0])
        any_safe = sorted(snapshot_dir.glob("*.safetensors"))
        if any_safe:
            return str(any_safe[0])
        return None

    native_model_file = _resolve_native_model_file(snapshot_dir)
    _progress(show_progress, 15, "config", "loading model and tokenizer config")
    config = read_config(snapshot_dir)
    requested_layers = int(args.max_layers)
    args.max_layers = _resolve_quality_layer_floor(config, requested_layers, str(args.device), bool(args.unsafe_low_layers))
    if requested_layers > 0 and args.max_layers != requested_layers:
        print(f"[vspec-chat] quality_guard_max_layers_adjusted= {requested_layers} -> {args.max_layers}")
    tok_cfg = read_tokenizer_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    model_type = str(config.get("model_type", "") or "").lower()
    if model_type == "gpt2":
        if int(args.fused_bits) == 3:
            args.fused_bits = 4
            os.environ["VSPEC_FUSED_BITS"] = "4"
            print("[vspec-chat] auto_adjust_fused_bits= 3 -> 4 for gpt2 stability")
        if int(args.target_bits) == 3:
            args.target_bits = 4
            print("[vspec-chat] auto_adjust_target_bits= 3 -> 4 for gpt2 stability")
        if requested_max_tokens > 0:
            args.max_tokens = requested_max_tokens
        if requested_max_layers_post_parse > 0:
            args.max_layers = requested_max_layers_post_parse
    _progress(show_progress, 35, "weights", "indexing model weights")
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
    if (not batch_mode) and lang_mode == "auto":
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
    structure_guard = None
    effective_disable_structure_guard = bool(args.disable_structure_guard)
    fast_engine = None
    token_ids: list[int] = []
    generated: list[int] = []
    decode_optimizer = None
    if not batch_mode:
        if not args.disable_language_guard:
            guard = LanguageStabilityGuard(
                prompt=args.prompt,
                lang_mode=lang_mode,
                strictness=args.language_guard_strictness,
                prioritize_english=(not args.no_prioritize_english),
            )

        if (not effective_disable_structure_guard) and (lang_mode == "en") and (not args.no_prioritize_english):
            effective_disable_structure_guard = True
        if not effective_disable_structure_guard:
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

        decode_optimizer = DecodeOptimizationModule(
            repetition_penalty=effective_repetition_penalty,
            repeat_window=effective_repeat_window,
            no_repeat_ngram=effective_no_repeat_ngram,
            mode=args.decode_opt_mode,
        )
        decode_optimizer.seed_history(token_ids)

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
    if not batch_mode:
        if effective_disable_structure_guard and (not args.disable_structure_guard) and (lang_mode == "en") and (not args.no_prioritize_english):
            print("[vspec-chat] structure_guard= off (english-priority)")
        else:
            print("[vspec-chat] structure_guard=", "on" if structure_guard is not None else "off")
        if guard is not None:
            print("[vspec-chat] language_guard_primary_script=", guard.profile.primary_script)
            print("[vspec-chat] language_guard_strictness=", round(guard.profile.strictness, 3))
            print("[vspec-chat] language_guard_prioritize_english=", guard.profile.prioritized_english)
        if structure_guard is not None:
            print("[vspec-chat] structure_guard_expected_sections=", structure_guard.profile.expected_sections)
            print("[vspec-chat] structure_guard_strictness=", round(structure_guard.profile.strictness, 3))
    else:
        print("[vspec-chat] structure_guard= per-request")
    print("[vspec-chat] decode_top_k=", effective_top_k)
    print("[vspec-chat] decode_lang_top_n=", effective_lang_top_n)

    if not batch_mode:
        print("[vspec-chat] prompt_tokens=", len(token_ids))
    else:
        print("[vspec-chat] batch_mode= prompts-file")

    start_ts = time.perf_counter()
    runtime_load_elapsed = 0.0
    prefill_elapsed = 0.0
    prefill_tokens_total = 0
    prefill_core_scheduler_used = False
    prefill_core_steps = 0
    vram_before = cuda_mem_info() if args.device in {"cuda", "cuda-native", "torch-cuda"} else None

    def _runtime_progress(stage: str, current: int, total: int) -> None:
        if not show_progress:
            return
        if total <= 0:
            return
        if stage == "layer_load":
            pct = 36 + int((current / total) * 22)
            _progress(show_progress, min(58, pct), "runtime-load", f"layer {current}/{total}")
            return
        if stage == "int4_pre_register":
            pct = 58 + int((current / total) * 6)
            _progress(show_progress, min(64, pct), "runtime-int4", f"pre-register {current}/{total}")

    runtime_load_t0 = time.perf_counter()
    runtime = build_generic_runtime(
        config,
        weight_index,
        args.max_layers,
        args.device,
        progress_cb=_runtime_progress,
    )
    runtime_load_elapsed = max(0.0, time.perf_counter() - runtime_load_t0)
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

        if batch_mode:
            exit_code = _run_prompt_file_batch(
                args=args,
                runtime=runtime,
                tokenizer=tokenizer,
                adapter=adapter,
                tok_cfg=tok_cfg,
                threebit_module=threebit_module,
                effective_top_k=effective_top_k,
                effective_lang_top_n=effective_lang_top_n,
                effective_repetition_penalty=effective_repetition_penalty,
                effective_repeat_window=effective_repeat_window,
                effective_no_repeat_ngram=effective_no_repeat_ngram,
                show_progress=show_progress,
            )
            raise SystemExit(exit_code)

        def _cache_integrity_ok(runtime_obj, expected_len: int) -> bool:
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

        # Prefill KV cache with prompt tokens so attention has context.
        if len(token_ids) > 1 and hasattr(runtime, "cache_k"):
            prefill_ts = time.perf_counter()
            if hasattr(runtime, "reset_core_kv_mirrors"):
                try:
                    runtime.reset_core_kv_mirrors()
                except Exception:
                    pass
            runtime.cache_k = []
            runtime.cache_v = []
            if hasattr(runtime, "cache_len"):
                runtime.cache_len = []
            runtime.position = 0
            prefill_ids = token_ids[:-1]
            total_prefill = len(prefill_ids)
            prefill_tokens_total = int(total_prefill)
            if hasattr(runtime, "prefill_tokens"):
                run_prefill_direct = True
                prefill_sched_requested = os.getenv("VSPEC_PREFILL_CORE_SCHED", "0").strip().lower() not in {"0", "false", "no", "off"}
                use_prefill_core_sched = bool(prefill_sched_requested)
                if use_prefill_core_sched and total_prefill > 0:
                    print("[vspec-chat] prefill_core_scheduler= on")
                    result = run_prefill_with_core_scheduler(
                        runtime,
                        prefill_ids,
                        progress_cb=lambda cur, tot: _progress(
                            show_progress,
                            min(80, 60 + int((cur / max(1, tot)) * 20)),
                            "prefill-core",
                            f"{cur}/{tot} tokens",
                        ),
                    )
                    if result.used_core_scheduler:
                        run_prefill_direct = False
                        prefill_core_scheduler_used = True
                        prefill_core_steps = int(result.core_steps)
                        print("[vspec-chat] prefill_core_reserved_vram=", int(result.reserved_vram))
                    else:
                        print("[vspec-chat] prefill_core_fallback=", str(result.reason or "unknown"))

                if run_prefill_direct:
                    runtime.prefill_tokens(prefill_ids)
                if total_prefill > 0:
                    _progress(show_progress, 80, "prefill", f"{total_prefill}/{total_prefill} tokens")
                if not _cache_integrity_ok(runtime, total_prefill):
                    print("[vspec-chat] prefill_cache_integrity= failed; replaying in chunks")
                    if hasattr(runtime, "reset_core_kv_mirrors"):
                        try:
                            runtime.reset_core_kv_mirrors()
                        except Exception:
                            pass
                    runtime.cache_k = []
                    runtime.cache_v = []
                    if hasattr(runtime, "cache_len"):
                        runtime.cache_len = []
                    runtime.position = 0
                    try:
                        replay_chunk = max(1, int(os.getenv("VSPEC_PREFILL_REPLAY_CHUNK", "64") or "64"))
                    except Exception:
                        replay_chunk = 64
                    if hasattr(runtime, "prefill_tokens"):
                        for start in range(0, total_prefill, replay_chunk):
                            end = min(total_prefill, start + replay_chunk)
                            runtime.prefill_tokens(prefill_ids[start:end])
                            idx = end
                            if total_prefill > 0 and (idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                                pct = 60 + int((idx / total_prefill) * 20)
                                _progress(show_progress, min(80, pct), "prefill-replay", f"{idx}/{total_prefill} tokens")
                    else:
                        for idx, tid in enumerate(prefill_ids, start=1):
                            runtime.forward_logits([tid])
                            if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                                pct = 60 + int((idx / total_prefill) * 20)
                                _progress(show_progress, min(80, pct), "prefill-replay", f"{idx}/{total_prefill} tokens")
            else:
                for idx, tid in enumerate(prefill_ids, start=1):
                    runtime.forward_logits([tid])
                    if total_prefill > 0 and (idx == 1 or idx == total_prefill or idx % max(1, total_prefill // 4) == 0):
                        pct = 60 + int((idx / total_prefill) * 20)
                        _progress(show_progress, min(80, pct), "prefill", f"{idx}/{total_prefill} tokens")
            prefill_elapsed = max(0.0, time.perf_counter() - prefill_ts)
            if hasattr(runtime, "cache_len"):
                try:
                    cache_len_list = list(getattr(runtime, "cache_len") or [])
                    if cache_len_list:
                        print("[vspec-chat] prefill_cache_len_min=", int(min(cache_len_list)))
                        print("[vspec-chat] prefill_cache_len_max=", int(max(cache_len_list)))
                except Exception:
                    pass

    # Placeholder sampling loop. Qwen ops stubs are in src/model/qwen_ops.c.
    # Next step: wire logits from Vspec runtime forward pass.
    if args.max_tokens > 0:
        max_steps = args.max_tokens
    else:
        max_ctx = int(config.get("max_position_embeddings", 4096) or 4096)
        max_steps = max(1, min(4096, max_ctx - len(token_ids)))

    _progress(show_progress, 82, "decode", f"max_steps={max_steps}")
    lowbit_projection_stats_reset()
    fast_engine.begin_stream()
    stream_buffer = []
    decode_contract_failed = False
    decode_start_ts = time.perf_counter()
    if float(args.max_decode_seconds) > 0.0:
        decode_budget_seconds = max(0.5, float(args.max_decode_seconds))
    elif float(args.max_decode_seconds) == 0.0:
        decode_budget_seconds = 0.0
    else:
        prompt_chars = len(str(args.prompt or "").strip())
        prompt_tokens = max(0, len(token_ids) - 1)
        layer_count = len(getattr(runtime, "layers", []) or []) if runtime is not None else 0
        work_units = max(1, (prompt_tokens + max_steps) * max(1, layer_count))
        auto_budget = 16.0 + (0.18 * float(max_steps)) + (0.012 * float(prompt_chars)) + (0.008 * float(work_units))
        if max_steps <= 64:
            auto_budget = max(auto_budget, 48.0)
        if (lang_mode == "en") and (not args.no_prioritize_english):
            auto_budget = max(auto_budget, 60.0)
        decode_budget_seconds = min(240.0, max(20.0, auto_budget))
    print(f"[vspec-chat] decode_budget_seconds= {decode_budget_seconds:.1f}")
    lowbit_plan = getattr(runtime, "lowbit_plan", None) if runtime is not None else None
    lowbit_enabled = bool(getattr(lowbit_plan, "enabled", False))
    fused_bits_runtime = int(getattr(runtime, "fused_bits", args.fused_bits) or args.fused_bits) if runtime is not None else int(args.fused_bits)
    decode_step_cap = _resolve_budget_step_cap(
        requested_steps=max_steps,
        decode_budget_seconds=decode_budget_seconds,
        prefill_tokens=max(0, len(token_ids) - 1),
        layer_count=len(getattr(runtime, "layers", []) or []) if runtime is not None else 0,
        lowbit_enabled=lowbit_enabled,
        fused_bits=fused_bits_runtime,
    )
    if decode_step_cap != int(max_steps):
        print(f"[vspec-chat] decode_step_cap= {decode_step_cap} (requested={int(max_steps)})")
    max_steps = decode_step_cap
    decode_state = DecodeState(prompt_tokens=max(0, len(token_ids) - 1), max_new_tokens=int(max_steps))
    phase3_dispatcher = Phase3StepDispatcher(
        runtime=runtime,
        decode_optimizer=decode_optimizer,
        expected_vocab_size=vocab_size,
    )
    phase1_orchestrator = PythonDecodeOrchestrator(
        state=decode_state,
        runtime=runtime,
        decode_optimizer=decode_optimizer,
        expected_vocab_size=vocab_size,
        scheduler_enabled=scheduler_enabled,
        core_decode=core_decode,
        step_dispatcher=phase3_dispatcher,
    )
    phase1_orchestrator.prefill(token_ids)
    native_cpp_loop = os.getenv("VSPEC_NATIVE_CPP_LOOP", "1").strip().lower() in {"1", "true", "yes", "on"}

    def _runtime_graph_signature(runtime_obj, prompt_tokens: int, decode_steps: int) -> int:
        graph_mode = os.getenv("VSPEC_NATIVE_GRAPH_SIG_MODE", "shape-only").strip().lower()
        try:
            prompt_bucket = max(1, int(os.getenv("VSPEC_GRAPH_SIG_PROMPT_BUCKET", "64") or "64"))
        except Exception:
            prompt_bucket = 64
        try:
            decode_bucket = max(1, int(os.getenv("VSPEC_GRAPH_SIG_DECODE_BUCKET", "16") or "16"))
        except Exception:
            decode_bucket = 16

        if graph_mode == "strict":
            p_sig = int(prompt_tokens)
            d_sig = int(decode_steps)
        elif graph_mode == "shape-only":
            p_sig = 0
            d_sig = 0
        else:
            p_sig = int(max(0, prompt_tokens) // prompt_bucket)
            d_sig = int((max(1, decode_steps) + decode_bucket - 1) // decode_bucket)

        if runtime_obj is None:
            return int((p_sig * 1315423911 + d_sig * 2654435761) & 0xFFFFFFFFFFFFFFFF)
        layer_count = len(getattr(runtime_obj, "layers", []) or [])
        hidden = 0
        try:
            embed = getattr(runtime_obj, "embed", None)
            if embed is not None:
                hidden = int(embed.shape[1])
        except Exception:
            hidden = 0
        num_heads = int(getattr(runtime_obj, "num_heads", 0) or 0)
        num_kv_heads = int(getattr(runtime_obj, "num_kv_heads", 0) or 0)
        fused_bits_sig = int(getattr(runtime_obj, "fused_bits", 0) or 0)
        sig = 1469598103934665603
        for part in (layer_count, hidden, num_heads, num_kv_heads, fused_bits_sig, p_sig, d_sig):
            sig ^= int(part) & 0xFFFFFFFFFFFFFFFF
            sig = (sig * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return int(sig)

    if native_cpp_loop:
        core_decode = CoreNativeDecodeLoop.from_runtime(runtime, max_steps)
        graph_sig = _runtime_graph_signature(runtime, max(0, len(token_ids) - 1), max_steps)
        scheduler_enabled = core_decode.begin(
            prompt_tokens=max(0, len(token_ids) - 1),
            max_new_tokens=max_steps,
            graph_signature=graph_sig,
        )
    else:
        core_decode = CoreDecodeSession.from_runtime(runtime, max_steps)
        scheduler_enabled = core_decode.begin(prompt_tokens=max(0, len(token_ids) - 1), max_new_tokens=max_steps)
    if scheduler_enabled:
        scheduler_mode = "native-cpp-loop" if native_cpp_loop else "decode-session"
        print(f"[vspec-chat] core_scheduler= on reserve_bytes={core_decode.reserve_bytes} mode={scheduler_mode}")
    adaptive_entropy_prev = 0.0
    adaptive_latency_ms = 0.0
    adaptive_quality_drift = 0.0
    native_blend_calls = 0
    native_blend_failures = 0
    native_forward_enabled = os.getenv("VSPEC_NATIVE_FORWARD_BLEND", "1").strip().lower() in {"1", "true", "yes", "on"}
    native_forward_ctx = None
    if native_forward_enabled and native_model_file:
        try:
            seed_raw = os.getenv("VSPEC_NATIVE_FORWARD_SEED", str(int(args.seed)))
            seed_val = int(seed_raw)
        except Exception:
            seed_val = int(args.seed)
        try:
            native_forward_ctx = CoreNativeForwardContext(native_model_file, seed=seed_val)
            if not native_forward_ctx.available:
                native_forward_ctx = None
        except Exception:
            native_forward_ctx = None

    anf_enabled = native_anf_prototype_enabled()
    for step in range(max_steps):
        if scheduler_enabled:
            quota = core_decode.next_quota()
            if quota <= 0:
                break
        else:
            quota = 1
        decode_elapsed = time.perf_counter() - decode_start_ts
        if decode_budget_seconds > 0.0 and decode_elapsed >= decode_budget_seconds:
            print(f"[vspec-chat] decode_timeout= True (budget={decode_budget_seconds:.1f}s)")
            decode_state.mark_timeout()
            break
        reached_eos = False
        for _ in range(quota):
            token_t0 = time.perf_counter()
            step_result = phase1_orchestrator.step(token_ids[-1])
            if not step_result.ok:
                print(
                    f"[vspec-chat] decode_contract_ok= False reason={step_result.reason} logits_len=0 expected_vocab={vocab_size}"
                )
                decode_contract_failed = True
                break
            logits = step_result.logits
            if step == 0 and int(step_result.masked_tail or 0) > 0:
                print(f"[vspec-chat] decode_contract_masked_tail= {int(step_result.masked_tail)}")
            if decode_contract_failed:
                break
            logits = threebit_module.denoise_logits(logits, step)
            logits = decode_optimizer.apply_generation_controls(logits, token_ids)
            sample_temperature = threebit_module.auto_temperature(logits, args.temperature)

            entropy_now = _entropy_from_logits(logits)
            entropy_collapse = max(0.0, adaptive_entropy_prev - entropy_now)
            adaptive_entropy_prev = entropy_now
            if anf_enabled:
                try:
                    native_anf_observe_activations(logits)
                    rrms, ecollapse, norm_drift = _anf_quality_proxies(logits, entropy_now)
                    native_anf_observe_quality(rrms, ecollapse, norm_drift)
                except Exception:
                    pass
            token_text = ""
            try:
                token_text = tokenizer.decode([int(token_ids[-1])])
            except Exception:
                token_text = str(int(token_ids[-1]))
            vram_pressure = 0.5
            try:
                mem_stats = cuda_mem_info()
                if isinstance(mem_stats, dict):
                    used = float(mem_stats.get("used", 0.0) or 0.0)
                    total = float(mem_stats.get("total", 0.0) or 0.0)
                    if total > 0.0:
                        vram_pressure = max(0.0, min(1.0, used / total))
            except Exception:
                vram_pressure = 0.5
            decision = adaptive_step(
                token_text=token_text,
                token_entropy=entropy_now,
                attention_entropy_collapse=entropy_collapse,
                latency_ms=adaptive_latency_ms,
                vram_pressure=vram_pressure,
                quality_drift=adaptive_quality_drift,
                layer_type=0,
            )
            sample_top_k = int(effective_top_k)
            sample_greedy = bool(args.greedy)
            if decision is not None and runtime is not None:
                setattr(runtime, "adaptive_target_bits", int(decision.target_bits))
                setattr(runtime, "adaptive_routed_bits", int(decision.routed_bits))
                setattr(runtime, "adaptive_reduce_attention_depth", bool(decision.reduce_attention_depth))
                setattr(runtime, "adaptive_attention_depth_hint", int(decision.attention_depth_hint))
                setattr(runtime, "adaptive_enable_kv_compression", bool(decision.enable_kv_compression))
                if decision.reduce_attention_depth and decision.attention_depth_hint > 0:
                    sample_top_k = min(sample_top_k, max(8, int(decision.attention_depth_hint) * 4))
                if decision.skip_compute:
                    sample_greedy = True
                bit_hint = int(decision.routed_bits if decision.routed_bits > 0 else decision.target_bits)
                if bit_hint > 0:
                    temp_scale = max(0.65, min(1.0, bit_hint / 8.0))
                    sample_temperature = max(0.05, float(sample_temperature) * temp_scale)

            if native_forward_ctx is not None and native_forward_ctx.available:
                try:
                    blend_alpha = float(os.getenv("VSPEC_NATIVE_FORWARD_BLEND_ALPHA", "0.18") or "0.18")
                except Exception:
                    blend_alpha = 0.18
                blend_alpha = max(0.0, min(1.0, blend_alpha))
                if blend_alpha > 0.0:
                    try:
                        blend_topk = int(os.getenv("VSPEC_NATIVE_FORWARD_BLEND_TOPK", "24") or "24")
                    except Exception:
                        blend_topk = 24
                    blend_topk = max(4, min(128, blend_topk))

                    try:
                        vocab_vals = logits.tolist() if hasattr(logits, "tolist") else list(logits)
                    except Exception:
                        vocab_vals = []

                    if vocab_vals:
                        import heapq

                        cand_k = min(len(vocab_vals), max(blend_topk, sample_top_k))
                        cand_ids = heapq.nlargest(cand_k, range(len(vocab_vals)), key=vocab_vals.__getitem__)
                        base_scores = [float(vocab_vals[i]) for i in cand_ids]
                        blended = native_forward_ctx.blend_candidates(
                            prompt=str(args.prompt or ""),
                            produced_tokens=len(generated),
                            candidate_ids=[int(i) for i in cand_ids],
                            base_scores=base_scores,
                            blend=blend_alpha,
                        )
                        if blended is not None and len(blended) == len(cand_ids):
                            for idx, score in zip(cand_ids, blended):
                                logits[idx] = float(score)
                            native_blend_calls += 1
                        else:
                            native_blend_failures += 1
                    else:
                        native_blend_failures += 1
                else:
                    native_blend_failures += 1

            next_id = phase1_orchestrator.sample(
                logits,
                fast_engine.sample,
                sample_temperature,
                sample_top_k,
                sample_greedy,
                effective_lang_top_n,
            )
            generated.append(next_id)
            stream_buffer.append(next_id)
            fast_engine.stream_token(next_id)
            token_ids.append(next_id)
            decode_optimizer.observe_token(token_ids)
            adaptive_latency_ms = (time.perf_counter() - token_t0) * 1000.0
            adaptive_quality_drift = min(1.0, 0.6 * adaptive_quality_drift + 0.4 * entropy_collapse)
            reached_eos = adapter.eos_token_id is not None and next_id == adapter.eos_token_id
            phase1_orchestrator.commit(next_id, reached_eos)
            if reached_eos:
                break

        if decode_contract_failed or reached_eos:
            break

        if args.no_stream and (step == 0 or (step + 1) % max(1, max_steps // 4) == 0):
            decode_pct = 82 + int(((step + 1) / max_steps) * 17)
            _progress(show_progress, min(99, decode_pct), "decode", f"{step + 1}/{max_steps} tokens")

    if scheduler_enabled:
        try:
            core_decode.cancel()
        except Exception:
            pass

    if native_cpp_loop:
        try:
            loop_stats = core_decode.stats()
            if loop_stats:
                print("[vspec-chat] graph_signature=", int(loop_stats.get("graph_signature", 0)))
                print("[vspec-chat] graph_reuse_hits=", int(loop_stats.get("graph_reuse_hits", 0)))
                print("[vspec-chat] graph_reuse_misses=", int(loop_stats.get("graph_reuse_misses", 0)))
                print("[vspec-chat] graph_capture_enabled=", int(loop_stats.get("graph_capture_enabled", 0)))
                print("[vspec-chat] graph_captures=", int(loop_stats.get("graph_captures", 0)))
                print("[vspec-chat] graph_replays=", int(loop_stats.get("graph_replays", 0)))
                print("[vspec-chat] graph_cached_signatures=", int(loop_stats.get("graph_cached_signatures", 0)))
                print("[vspec-chat] native_loop_steps=", int(loop_stats.get("steps", 0)))
        except Exception:
            pass

    if native_forward_ctx is not None:
        try:
            native_forward_ctx.close()
        except Exception:
            pass
    print("[vspec-chat] native_forward_blend_calls=", int(native_blend_calls))
    print("[vspec-chat] native_forward_blend_failures=", int(native_blend_failures))
    phase4_sampler_stats = fast_engine.phase4_sampler_report() if fast_engine is not None else {}
    print("[vspec-chat] phase4_sampler_calls=", int(phase4_sampler_stats.get("calls", 0)))
    print("[vspec-chat] phase4_c_sampler_calls=", int(phase4_sampler_stats.get("c_sampler_calls", 0)))
    print("[vspec-chat] phase4_python_sampler_calls=", int(phase4_sampler_stats.get("python_sampler_calls", 0)))
    print("[vspec-chat] phase4_sampler_parity_checks=", int(phase4_sampler_stats.get("parity_checks", 0)))
    print("[vspec-chat] phase4_sampler_parity_mismatch=", int(phase4_sampler_stats.get("parity_mismatch", 0)))
    print("[vspec-chat] phase4_sampler_parity_fallbacks=", int(phase4_sampler_stats.get("parity_fallbacks", 0)))
    print("[vspec-chat] phase3_step_calls=", int(phase3_dispatcher.stats.get("calls", 0)))
    print("[vspec-chat] phase3_c_step_calls=", int(phase3_dispatcher.stats.get("c_step_calls", 0)))
    print("[vspec-chat] phase3_python_step_calls=", int(phase3_dispatcher.stats.get("python_step_calls", 0)))
    print("[vspec-chat] phase3_parity_checks=", int(phase3_dispatcher.stats.get("parity_checks", 0)))
    print("[vspec-chat] phase3_parity_failures=", int(phase3_dispatcher.stats.get("parity_failures", 0)))
    print("[vspec-chat] phase3_parity_fallbacks=", int(phase3_dispatcher.stats.get("parity_fallbacks", 0)))
    print(
        "[vspec-chat] phase1_decode_state=",
        f"prefill_done:{int(decode_state.prefill_done)} generated:{int(decode_state.generated_tokens)} finished:{int(decode_state.finished)} reason:{decode_state.finish_reason or 'none'}",
    )

    if anf_enabled:
        try:
            anf_report = native_anf_report() or {}
            if anf_report:
                print("[vspec-chat] anf_available=", int(anf_report.get("anf_available", 0)))
                print("[vspec-chat] anf_mode=", int(anf_report.get("anf_mode", 0)))
                print("[vspec-chat] anf_hot_ratio_avg=", float(anf_report.get("hot_ratio_avg", 0.0)))
                print("[vspec-chat] anf_tokens_observed=", int(anf_report.get("tokens_observed", 0)))
                print("[vspec-chat] anf_skip_ratio_avg=", float(anf_report.get("skip_ratio_avg", 0.0)))
                print("[vspec-chat] anf_cascade_depth_max=", int(anf_report.get("cascade_depth_max", 0)))
                print("[vspec-chat] anf_forced_fallback_count=", int(anf_report.get("forced_fallback_count", 0)))
                print("[vspec-chat] anf_silent_stop_count=", int(anf_report.get("silent_stop_count", 0)))
        except Exception:
            pass
    core_decode.close()

    fast_engine.end_stream()
    _progress(show_progress, 100, "done", "generation complete")

    if tokenizer is not None:
        text = tokenizer.decode(generated)
    else:
        text = "<tokens> " + " ".join(str(t) for t in generated[:16])
    text = postprocess_output_text(text, args.prompt, lang_mode)

    needs_verify = (
        decode_contract_failed
        or _is_runtime_fallback_text(text, lang_mode)
        or _looks_gibberish_output(text)
    )

    def _verifier_headroom_ok() -> bool:
        if args.device not in {"cuda", "cuda-native", "torch-cuda"}:
            return True
        try:
            min_free_gb = float(os.getenv("VSPEC_VERIFIER_MIN_FREE_GB", "2.0") or "2.0")
        except Exception:
            min_free_gb = 2.0
        try:
            mem = cuda_mem_info()
            if not mem:
                return True
            free_bytes = int(mem[0])
            return free_bytes >= int(max(0.5, min_free_gb) * (1024**3))
        except Exception:
            return True

    disable_native_safe_verify = os.getenv("VSPEC_DISABLE_NATIVE_SAFE_VERIFY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if (not disable_native_safe_verify) and needs_verify and args.device in {"cuda", "cuda-native", "cpu"}:
        if not _verifier_headroom_ok():
            print("[vspec-chat] native_safe_verify_skipped= low_vram_headroom")
        else:
            native_safe_text = _run_native_safe_verifier(args, args.prompt, lang_mode)
            assurance = RuntimeMeaningfulResponseAssurance(lang_mode, allow_semantic_rescue=args.allow_semantic_rescue)
            native_safe_text = assurance.repair(native_safe_text, args.prompt) if native_safe_text else ""
            if native_safe_text and (not _is_runtime_fallback_text(native_safe_text, lang_mode)):
                text = native_safe_text
                needs_verify = False
                print("[vspec-chat] native_safe_verify_used= True")
            else:
                print("[vspec-chat] native_safe_verify_used= False")

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
            if not _verifier_headroom_ok():
                print("[vspec-chat] torch_verifier_skipped= low_vram_headroom")
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

    end_ts = time.perf_counter()
    elapsed = end_ts - start_ts
    decode_elapsed = max(0.0, end_ts - decode_start_ts)
    vram_after = cuda_mem_info() if args.device in {"cuda", "cuda-native", "torch-cuda"} else None
    prompt_tok = len(token_ids) - len(generated)
    gen_tok = len(generated)
    total_tok = prompt_tok + gen_tok
    tps_total = (gen_tok / elapsed) if elapsed > 0 else 0.0
    tps_decode = (gen_tok / decode_elapsed) if decode_elapsed > 0 else 0.0

    print("[vspec-chat] runtime_mode=", runtime_mode)
    print("[vspec-chat] runtime_is_vspec=", runtime is not None)
    if runtime is not None:
        print("[vspec-chat] quant_source_format=", getattr(runtime, "quant_source_format", "unknown"))
        print("[vspec-chat] quant_source_quantized=", bool(getattr(runtime, "quant_source_quantized", False)))
        print("[vspec-chat] quant_runtime_disabled=", bool(getattr(runtime, "quant_runtime_disabled", False)))
        print("[vspec-chat] quant_policy_reason=", getattr(runtime, "quant_policy_reason", "unknown"))
    print("[vspec-chat] timing_sec=", round(elapsed, 4))
    print("[vspec-chat] runtime_load_timing_sec=", round(runtime_load_elapsed, 4))
    print("[vspec-chat] prefill_timing_sec=", round(prefill_elapsed, 4))
    print("[vspec-chat] prefill_core_scheduler=", prefill_core_scheduler_used)
    print("[vspec-chat] prefill_core_steps=", int(prefill_core_steps))
    print("[vspec-chat] decode_timing_sec=", round(decode_elapsed, 4))
    print("[vspec-chat] tokens_prompt=", prompt_tok)
    print("[vspec-chat] tokens_generated=", gen_tok)
    print("[vspec-chat] tokens_total=", total_tok)
    print("[vspec-chat] tokens_per_sec=", round(tps_total, 4))
    print("[vspec-chat] decode_tokens_per_sec=", round(tps_decode, 4))
    prefill_tps = (prefill_tokens_total / prefill_elapsed) if prefill_elapsed > 0.0 else 0.0
    print("[vspec-chat] prefill_tokens=", int(prefill_tokens_total))
    print("[vspec-chat] prefill_tokens_per_sec=", round(prefill_tps, 4))

    low_bit = False
    if runtime is not None and hasattr(runtime, "layers"):
        try:
            matrix_bits, exec_eff_bits, has_lowbit, lowbit_coverage = runtime_matrix_bits_summary(runtime.layers)
            print("[vspec-chat] lowbit_mode=runtime")
            print("[vspec-chat] target_bits=", args.target_bits)
            print("[vspec-chat] matrix_bits=", matrix_bits)
            print("[vspec-chat] effective_bits_estimate=", round(exec_eff_bits, 4))
            print("[vspec-chat] lowbit_coverage=", round(lowbit_coverage, 4))
            low_bit = bool(has_lowbit)
        except Exception:
            layer_bits = build_layer_bits(len(runtime.layers), args.target_bits)
            est_bits = effective_bits(layer_bits)
            print("[vspec-chat] lowbit_mode=policy-fallback")
            print("[vspec-chat] target_bits=", args.target_bits)
            print("[vspec-chat] layer_bits=", summarize_layer_bits(layer_bits))
            print("[vspec-chat] effective_bits_estimate=", round(est_bits, 4))
            low_bit = any(v <= 4 for v in layer_bits)
    print("[vspec-chat] processing_leq_4bit=", low_bit)
    if not low_bit:
        print("[vspec-chat] note=weights are not <=4-bit in this path (mostly fp16/bf16/fp32)")
    elif args.target_bits > 0:
        int4_mode = os.getenv("VSPEC_INT4_PRECISION_MODE", "balanced").strip().lower() or "balanced"
        print(f"[vspec-chat] note=low-bit policy telemetry enabled; int4_precision_mode={int4_mode}")

    if vram_before and vram_after:
        free_before, total_before = vram_before
        free_after, total_after = vram_after
        used_before = total_before - free_before
        used_after = total_after - free_after
        print("[vspec-chat] vram_total_bytes=", total_after)
        print("[vspec-chat] vram_used_before_bytes=", used_before)
        print("[vspec-chat] vram_used_after_bytes=", used_after)
        print("[vspec-chat] vram_delta_bytes=", used_after - used_before)

    lowbit_stats = lowbit_projection_stats_snapshot()
    print("[vspec-chat] int4_pre_registered=", int(getattr(runtime, "int4_pre_registered", 0) or 0) if runtime is not None else 0)
    print("[vspec-chat] int4_pre_register_failures=", int(getattr(runtime, "int4_pre_register_failures", 0) or 0) if runtime is not None else 0)
    print("[vspec-chat] lowbit_exec_calls=", int(lowbit_stats.get("calls", 0)))
    print("[vspec-chat] lowbit_exec_lowbit_calls=", int(lowbit_stats.get("lowbit_calls", 0)))
    print("[vspec-chat] lowbit_exec_int4_registered_calls=", int(lowbit_stats.get("int4_registered_calls", 0)))
    print("[vspec-chat] lowbit_exec_int4_registered_many_calls=", int(lowbit_stats.get("int4_registered_many_calls", 0)))
    print("[vspec-chat] lowbit_exec_int4_registered_registers=", int(lowbit_stats.get("int4_registered_registers", 0)))
    print("[vspec-chat] lowbit_exec_int4_registered_failures=", int(lowbit_stats.get("int4_registered_failures", 0)))
    print("[vspec-chat] lowbit_exec_fallback_gemm_calls=", int(lowbit_stats.get("fallback_gemm_calls", 0)))
    print("[vspec-chat] lowbit_exec_fallback_matmul_calls=", int(lowbit_stats.get("fallback_matmul_calls", 0)))
    print("[vspec-chat] kv_core_mirror_enabled=", bool(getattr(runtime, "kv_core_mirror_enabled", False)) if runtime is not None else False)
    print("[vspec-chat] kv_python_shadow_disabled=", bool(getattr(runtime, "kv_python_shadow_disabled", False)) if runtime is not None else False)
    print("[vspec-chat] cublas_cache_size_hint=", os.getenv("VSPEC_CUBLAS_CACHE_SIZE", "16"))
    print("[vspec-chat] c_sampler_required=", os.getenv("VSPEC_C_SAMPLER_REQUIRED", "1"))


if __name__ == "__main__":
    main()
