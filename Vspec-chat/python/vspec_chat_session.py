import argparse
import math
import os
import random
import re
import subprocess
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
from decode_phase1_contract import DecodeState, PythonDecodeOrchestrator
from decode_phase2_prefill import run_prefill_with_core_scheduler
from decode_phase3_step_dispatch import Phase3StepDispatcher
from decode_phase5_daemon import Phase5TurnReport, SessionCoreDaemonSupervisor
from native_safe_decode import build_native_safe_runtime
from runtime_core_bridge import (
    CoreDecodeSession,
    CoreNativeDecodeLoop,
    CoreNativeForwardContext,
    adaptive_step,
    native_anf_observe_activations,
    native_anf_observe_quality,
    native_anf_prototype_enabled,
    native_anf_report,
)
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
from vspec_cuda_bridge import cuda_device_capability, cuda_mem_info, int4_compute_mode, int4_tensorcore_available
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
    print(msg, flush=True)


def _detect_lang(prompt: str) -> str:
    lower = prompt.lower()
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ"
    if any(ch in lower for ch in vi_chars):
        return "vi"
    if any("\u4e00" <= ch <= "\u9fff" for ch in prompt):
        return "auto"
    return "en"


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


def _fallback_cuda_capability() -> Optional[tuple[int, int, int]]:
    # Fallback path when cuda bridge is unavailable: query nvidia-smi / torch for CC.
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            rows = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if rows:
                first = rows[0]
                if "." in first:
                    maj_s, min_s = first.split(".", 1)
                    return max(0, int(maj_s)), max(0, int(min_s)), 0
    except Exception:
        pass

    try:
        import torch

        if torch is not None and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            sms = int(torch.cuda.get_device_properties(0).multi_processor_count)
            return int(major), int(minor), sms
    except Exception:
        pass

    return None



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
    native_loop_handle: Optional[CoreNativeDecodeLoop],
    native_forward_ctx: Optional[CoreNativeForwardContext],
    native_model_file: Optional[str],
    phase5_observer: Optional[Callable[[Phase5TurnReport], None]],
) -> str:
    lowbit_projection_stats_reset()

    assurance = RuntimeMeaningfulResponseAssurance(lang_mode, allow_semantic_rescue=allow_semantic_rescue)
    prompt_chars = len((prompt or "").strip())

    def _run_full_native_turn() -> Optional[str]:
        enabled = os.getenv("VSPEC_FULL_NATIVE_C", "1").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return None
        if not native_model_file:
            print("[session] full_native_c= unavailable (missing native model file)")
            return None

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "build" / "Release" / "vspec_native_internal_loop_chat.exe",
            repo_root / "build" / "Debug" / "vspec_native_internal_loop_chat.exe",
            repo_root / "build" / "vspec_native_internal_loop_chat",
        ]
        exe_path = next((p for p in candidates if p.exists()), None)
        if exe_path is None:
            print("[session] full_native_c= unavailable (native executable not found)")
            return None

        timeout_s = 0.0
        try:
            timeout_s = float(max_decode_seconds)
        except Exception:
            timeout_s = 0.0
        if timeout_s <= 0.0:
            timeout_s = 90.0

        cmd = [str(exe_path), str(native_model_file), str(prompt), str(max(1, int(max_tokens)))]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            print(f"[session] full_native_c= failed ({exc})")
            return None

        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        output_line = None
        for ln in lines:
            if ln.startswith("[native-chat] output:"):
                output_line = ln
                break

        if proc.returncode != 0 or output_line is None:
            stderr_tail = (proc.stderr or "").strip()
            if stderr_tail:
                print(f"[session] full_native_c= failed returncode={proc.returncode} stderr={stderr_tail[:200]}")
            else:
                print(f"[session] full_native_c= failed returncode={proc.returncode}")
            return None

        text_out = output_line.split(":", 1)[1].strip()
        print(f"[session] full_native_c= on backend={exe_path.name}")
        for ln in lines:
            if ln.startswith("[native-chat] model_family=") or ln.startswith("[native-chat] produced_tokens="):
                print(f"[session] {ln.replace('[native-chat] ', '')}")
        return text_out

    def _native_safe_headroom_ok() -> bool:
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

    def _is_numeric_like_text(text: str) -> bool:
        out = (text or "").strip()
        if not out:
            return False
        compact = out.replace(" ", "")
        return re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", compact) is not None

    def _resolve_min_decode_tokens(prompt_text: str, language: str) -> int:
        text = (prompt_text or "").strip()
        words = len([w for w in text.split() if w])
        chars = len(text)
        if chars <= 36 or words <= 8:
            return 2
        if chars <= 90 or words <= 18:
            return 6 if language == "en" else 5
        return 10 if language == "en" else 8

    brief_numeric_mode = False

    def _looks_short_numeric_prompt(prompt_text: str) -> bool:
        s = (prompt_text or "").strip().lower()
        if not s:
            return False
        numeric_markers = (
            "how much",
            "bao nhieu",
            "result",
            "ket qua",
            "answer only",
            "chi can dap an",
            "just number",
            "chi so",
        )
        has_expr = any(ch.isdigit() for ch in s) and any(op in s for op in ("+", "-", "*", "/", "=", "%"))
        return has_expr or any(marker in s for marker in numeric_markers)

    def _requires_number_only(prompt_text: str) -> bool:
        s = (prompt_text or "").strip().lower()
        if not s:
            return False
        strict_markers = (
            "answer only with the number",
            "answer with just number",
            "just number",
            "only the number",
            "chi can so",
            "chi tra loi bang so",
            "chi so",
        )
        return any(marker in s for marker in strict_markers)

    def _looks_incomplete_stub(text: str) -> bool:
        out = (text or "").strip()
        if not out:
            return True
        if out.endswith((":", ";", ",", "-")) and len(out) <= 96:
            return True
        if out.count("(") != out.count(")") or out.count("[") != out.count("]") or out.count("{") != out.count("}"):
            return True
        if out.count('"') % 2 == 1:
            return True
        return False

    decode_min_tokens = _resolve_min_decode_tokens(prompt, lang_mode)
    brief_answer_mode = decode_min_tokens <= 2
    brief_numeric_mode = _looks_short_numeric_prompt(prompt)

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
        if brief_numeric_mode and _is_numeric_like_text(out):
            return True
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

        # Calibrated latency estimate (seconds/token) to prevent optimistic caps.
        base_latency = float(os.getenv("VSPEC_TOKEN_LATENCY_BASE_SEC", "0.42") or "0.42")
        per_layer_latency = float(os.getenv("VSPEC_TOKEN_LATENCY_PER_LAYER_SEC", "0.0055") or "0.0055")
        token_latency = base_latency + (per_layer_latency * float(max(1, int(layer_count))))
        if lowbit_enabled and fused_bits in {3, 4}:
            token_latency += 0.06
        elif lowbit_enabled:
            token_latency += 0.04
        if int(prefill_tokens) >= 256:
            token_latency *= 1.12
        token_latency = max(0.05, min(3.0, token_latency))

        reserve = max(3.0, min(20.0, decode_budget_seconds * 0.15))
        usable = max(0.5, decode_budget_seconds - reserve)
        cap = int(usable / token_latency)
        cap = max(12, cap)
        if brief_numeric_mode:
            cap = min(cap, 12)
        return max(1, min(int(requested_steps), cap))

    def _should_quality_retry(candidate: str) -> bool:
        repaired = assurance.repair(candidate, prompt)
        if _is_runtime_fallback_text(repaired, lang_mode):
            return True
        out = (repaired or "").strip()
        if not out or "�" in out:
            return True
        if _looks_incomplete_stub(out):
            return True
        if brief_answer_mode and len(out) <= 24 and (not _looks_gibberish_output(out)):
            return False
        if brief_numeric_mode:
            if _is_numeric_like_text(out):
                return False
            if len(out) <= 24 and (not _looks_gibberish_output(out)):
                return False
        if lang_mode == "en" and prioritize_english:
            if len(out) < 12:
                return True
            return _looks_gibberish_output(repaired)
        return _looks_gibberish_output(repaired)

    def _finalize_clean_output(candidate: str) -> str:
        repaired = assurance.repair(candidate, prompt)
        out = (repaired or "").strip()
        if _is_runtime_fallback_text(out, lang_mode):
            return out
        if _requires_number_only(prompt):
            if not _is_numeric_like_text(out):
                print("[session] output_filter= number-only-enforced")
                return assurance.repair("", prompt)
            return out
        if _looks_gibberish_output(out) or _looks_incomplete_stub(out):
            print("[session] output_filter= gibberish-blocked")
            return assurance.repair("", prompt)
        return out

    native_text = _run_full_native_turn()
    if native_text is not None:
        text = _finalize_clean_output(native_text)
        _progress(show_progress, 100, "done", "native-full-c")
        return text

    prompt_for_model = build_prompt(prompt, adapter.model_type, tok_cfg, lang_mode, chat_format)
    encoded = tokenizer.encode(prompt_for_model)
    token_ids = list(encoded.ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

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
            run_prefill_direct = True
            prefill_sched_requested = os.getenv("VSPEC_PREFILL_CORE_SCHED", "0").strip().lower() not in {"0", "false", "no", "off"}
            use_prefill_core_sched = bool(prefill_sched_requested)
            if use_prefill_core_sched and total_prefill > 0:
                print("[session] prefill_core_scheduler= on")
                result = run_prefill_with_core_scheduler(
                    runtime_obj,
                    prefill,
                    progress_cb=lambda cur, tot: _progress(
                        show_progress,
                        min(45, 10 + int((cur / max(1, tot)) * 35)),
                        "prefill-core",
                        f"{cur}/{tot}",
                    ),
                )
                if result.used_core_scheduler:
                    run_prefill_direct = False
                    print(f"[session] prefill_core_steps= {int(result.core_steps)}")
                    print(f"[session] prefill_core_reserved_vram= {int(result.reserved_vram)}")
                else:
                    print(f"[session] prefill_core_fallback= {str(result.reason or 'unknown')}")

            if run_prefill_direct:
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
        min_decode_tokens_local: int,
        disable_structure_guard_local: bool,
        native_loop_override: Optional[CoreNativeDecodeLoop],
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
        native_blend_calls = 0
        native_blend_failures = 0
        flash_calls_before = int(getattr(runtime_obj, "phase3_flash_attn_calls", 0) or 0)
        fused_calls_before = int(getattr(runtime_obj, "phase3_fused_attn_calls", 0) or 0)
        scalar_calls_before = int(getattr(runtime_obj, "phase3_scalar_attn_calls", 0) or 0)
        cpu_calls_before = int(getattr(runtime_obj, "phase3_cpu_attn_calls", 0) or 0)

        total_steps = max(1, int(max_steps))
        decode_state = DecodeState(
            prompt_tokens=max(0, len(local_tokens) - 1),
            max_new_tokens=int(total_steps),
        )
        native_cpp_loop = os.getenv("VSPEC_NATIVE_CPP_LOOP", "1").strip().lower() in {"1", "true", "yes", "on"}

        def _runtime_graph_signature(runtime_ref, prompt_tokens: int, decode_steps: int) -> int:
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

            if runtime_ref is None:
                return int((p_sig * 1315423911 + d_sig * 2654435761) & 0xFFFFFFFFFFFFFFFF)
            layer_count = len(getattr(runtime_ref, "layers", []) or [])
            hidden = 0
            try:
                embed = getattr(runtime_ref, "embed", None)
                if embed is not None:
                    hidden = int(embed.shape[1])
            except Exception:
                hidden = 0
            num_heads = int(getattr(runtime_ref, "num_heads", 0) or 0)
            num_kv_heads = int(getattr(runtime_ref, "num_kv_heads", 0) or 0)
            fused_bits_sig = int(getattr(runtime_ref, "fused_bits", 0) or 0)
            sig = 1469598103934665603
            for part in (layer_count, hidden, num_heads, num_kv_heads, fused_bits_sig, p_sig, d_sig):
                sig ^= int(part) & 0xFFFFFFFFFFFFFFFF
                sig = (sig * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            return int(sig)

        owns_native_loop = False
        if native_cpp_loop:
            if native_loop_override is not None:
                core_decode = native_loop_override
            else:
                core_decode = CoreNativeDecodeLoop.from_runtime(runtime_obj, total_steps)
                owns_native_loop = True
            graph_sig = _runtime_graph_signature(runtime_obj, max(0, len(local_tokens) - 1), total_steps)
            scheduler_enabled = core_decode.begin(
                prompt_tokens=max(0, len(local_tokens) - 1),
                max_new_tokens=total_steps,
                graph_signature=graph_sig,
            )
        else:
            core_decode = CoreDecodeSession.from_runtime(runtime_obj, total_steps)
            scheduler_enabled = core_decode.begin(prompt_tokens=max(0, len(local_tokens) - 1), max_new_tokens=total_steps)
        phase1_orchestrator = PythonDecodeOrchestrator(
            state=decode_state,
            runtime=runtime_obj,
            decode_optimizer=decode_optimizer,
            expected_vocab_size=tokenizer.get_vocab_size(),
            scheduler_enabled=scheduler_enabled,
            core_decode=core_decode,
            step_dispatcher=Phase3StepDispatcher(
                runtime=runtime_obj,
                decode_optimizer=decode_optimizer,
                expected_vocab_size=tokenizer.get_vocab_size(),
            ),
        )
        phase1_orchestrator.prefill(local_tokens)
        adaptive_entropy_prev = 0.0
        adaptive_latency_ms = 0.0
        adaptive_quality_drift = 0.0
        anf_enabled = native_anf_prototype_enabled()

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
                    decode_state.mark_timeout()
                    break

                reached_eos = False
                for _ in range(quota):
                    token_t0 = time.perf_counter()
                    step_result = phase1_orchestrator.step(local_tokens[-1])
                    if not step_result.ok:
                        print(
                            f"[session] decode_contract_ok= False reason={step_result.reason} logits_len=0 expected_vocab={tokenizer.get_vocab_size()}"
                        )
                        contract_failed = True
                        break
                    logits = step_result.logits
                    if step == 0 and int(step_result.masked_tail or 0) > 0:
                        print(f"[session] decode_contract_masked_tail= {int(step_result.masked_tail)}")
                    if step == 0:
                        _logits_health(logits, step)

                    logits = threebit_module.denoise_logits(logits, step)
                    logits = decode_optimizer.apply_generation_controls(logits, local_tokens)

                    if adapter.eos_token_id is not None and len(generated) < int(max(0, min_decode_tokens_local)):
                        eos_id = int(adapter.eos_token_id)
                        try:
                            if 0 <= eos_id < len(logits):
                                logits[eos_id] = -1e9
                        except Exception:
                            pass

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

                    native_blend_enabled = os.getenv("VSPEC_NATIVE_FORWARD_BLEND", "1").strip().lower() in {"1", "true", "yes", "on"}
                    if native_blend_enabled and native_forward_ctx is not None and native_forward_ctx.available:
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
                                    prompt=prompt,
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
                        engine.sample,
                        sample_temperature,
                        sample_top_k,
                        sample_greedy,
                        local_lang_top_n,
                    )
                    generated.append(next_id)
                    local_tokens.append(next_id)
                    engine.stream_token(next_id)
                    decode_optimizer.observe_token(local_tokens)
                    adaptive_latency_ms = (time.perf_counter() - token_t0) * 1000.0
                    adaptive_quality_drift = min(1.0, 0.6 * adaptive_quality_drift + 0.4 * entropy_collapse)
                    reached_eos = adapter.eos_token_id is not None and next_id == adapter.eos_token_id
                    phase1_orchestrator.commit(next_id, reached_eos)
                    if reached_eos:
                        break

                if contract_failed or reached_eos:
                    break

                if (not local_stream) and (step == 0 or (step + 1) % max(1, total_steps // 4) == 0):
                    pct = 45 + int(((step + 1) / total_steps) * 50)
                    _progress(show_progress, min(95, pct), "decode", f"{step + 1}/{total_steps}")
        finally:
            engine.end_stream()
            if scheduler_enabled:
                try:
                    core_decode.cancel()
                except Exception:
                    pass
            if native_cpp_loop:
                try:
                    loop_stats = core_decode.stats()
                    if loop_stats:
                        print(f"[session] graph_reuse_hits= {int(loop_stats.get('graph_reuse_hits', 0))}")
                        print(f"[session] graph_reuse_misses= {int(loop_stats.get('graph_reuse_misses', 0))}")
                        print(f"[session] graph_capture_enabled= {int(loop_stats.get('graph_capture_enabled', 0))}")
                        print(f"[session] graph_captures= {int(loop_stats.get('graph_captures', 0))}")
                        print(f"[session] graph_replays= {int(loop_stats.get('graph_replays', 0))}")
                        print(f"[session] graph_cached_signatures= {int(loop_stats.get('graph_cached_signatures', 0))}")
                except Exception:
                    pass
            if anf_enabled:
                try:
                    anf_report = native_anf_report() or {}
                    if anf_report:
                        print(f"[session] anf_available= {int(anf_report.get('anf_available', 0))}")
                        print(f"[session] anf_mode= {int(anf_report.get('anf_mode', 0))}")
                        print(f"[session] anf_hot_ratio_avg= {float(anf_report.get('hot_ratio_avg', 0.0))}")
                        print(f"[session] anf_tokens_observed= {int(anf_report.get('tokens_observed', 0))}")
                        print(f"[session] anf_skip_ratio_avg= {float(anf_report.get('skip_ratio_avg', 0.0))}")
                        print(f"[session] anf_cascade_depth_max= {int(anf_report.get('cascade_depth_max', 0))}")
                        print(f"[session] anf_forced_fallback_count= {int(anf_report.get('forced_fallback_count', 0))}")
                        print(f"[session] anf_silent_stop_count= {int(anf_report.get('silent_stop_count', 0))}")
                except Exception:
                    pass
            if native_cpp_loop:
                if owns_native_loop:
                    core_decode.close()
            else:
                core_decode.close()

        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated) if generated else ""
        text = postprocess_output_text(text, prompt, lang_mode)
        tps = (len(generated) / elapsed) if elapsed > 0 else 0.0
        decode_report = engine.structure_report() or {}
        decode_report["phase1_generated_tokens"] = int(decode_state.generated_tokens)
        decode_report["phase1_finish_reason"] = str(decode_state.finish_reason or "")
        decode_report["phase1_contract_failed"] = bool(decode_state.contract_failed)
        phase4_stats = engine.phase4_sampler_report() if engine is not None else {}
        decode_report["phase4_sampler_calls"] = int(phase4_stats.get("calls", 0) or 0)
        decode_report["phase4_c_sampler_calls"] = int(phase4_stats.get("c_sampler_calls", 0) or 0)
        decode_report["phase4_python_sampler_calls"] = int(phase4_stats.get("python_sampler_calls", 0) or 0)
        decode_report["phase4_sampler_parity_checks"] = int(phase4_stats.get("parity_checks", 0) or 0)
        decode_report["phase4_sampler_parity_mismatch"] = int(phase4_stats.get("parity_mismatch", 0) or 0)
        decode_report["phase4_sampler_parity_fallbacks"] = int(phase4_stats.get("parity_fallbacks", 0) or 0)
        phase3_stats = getattr(phase1_orchestrator, "step_dispatcher", None)
        phase3_stats = getattr(phase3_stats, "stats", {}) if phase3_stats is not None else {}
        decode_report["phase3_step_calls"] = int(phase3_stats.get("calls", 0) or 0)
        decode_report["phase3_c_step_calls"] = int(phase3_stats.get("c_step_calls", 0) or 0)
        decode_report["phase3_python_step_calls"] = int(phase3_stats.get("python_step_calls", 0) or 0)
        decode_report["phase3_parity_checks"] = int(phase3_stats.get("parity_checks", 0) or 0)
        decode_report["phase3_parity_failures"] = int(phase3_stats.get("parity_failures", 0) or 0)
        decode_report["phase3_parity_fallbacks"] = int(phase3_stats.get("parity_fallbacks", 0) or 0)
        decode_report["native_forward_blend_calls"] = int(native_blend_calls)
        decode_report["native_forward_blend_failures"] = int(native_blend_failures)
        decode_report["flash_attn_calls"] = int((int(getattr(runtime_obj, "phase3_flash_attn_calls", 0) or 0) - flash_calls_before))
        decode_report["fused_attn_calls"] = int((int(getattr(runtime_obj, "phase3_fused_attn_calls", 0) or 0) - fused_calls_before))
        decode_report["scalar_attn_calls"] = int((int(getattr(runtime_obj, "phase3_scalar_attn_calls", 0) or 0) - scalar_calls_before))
        decode_report["cpu_attn_calls"] = int((int(getattr(runtime_obj, "phase3_cpu_attn_calls", 0) or 0) - cpu_calls_before))
        return text, len(generated), tps, decode_report, timed_out, contract_failed

    base_tokens = list(token_ids)
    _prefill_runtime(runtime, base_tokens)
    layer_count = len(getattr(runtime, "layers", []) or [])
    prefill_tokens = max(0, len(base_tokens) - 1)
    decode_budget = _resolve_decode_budget_seconds(prefill_tokens, layer_count)
    retry_budget = _resolve_retry_budget_seconds(decode_budget)
    decode_step_cap = _resolve_budget_step_cap(max_tokens, decode_budget, prefill_tokens, layer_count)
    decode_temperature = temperature
    decode_top_k = top_k
    decode_lang_top_n = lang_top_n
    decode_greedy = greedy
    if brief_numeric_mode:
        decode_temperature = min(0.35, max(0.05, float(temperature) * 0.55))
        decode_top_k = max(1, min(8, int(top_k) if int(top_k) > 0 else 8))
        decode_lang_top_n = max(16, min(int(lang_top_n), 48))
        decode_greedy = True
        decode_step_cap = min(decode_step_cap, 12)
    print(f"[session] decode_budget_seconds= {decode_budget:.1f}")
    print(f"[session] retry_budget_seconds= {retry_budget:.1f}")
    if decode_step_cap != int(max_tokens):
        print(f"[session] decode_step_cap= {decode_step_cap} (requested={int(max_tokens)})")
    if brief_numeric_mode:
        print("[session] concise_numeric_mode= on")

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
            min_decode_tokens_local=decode_min_tokens,
            disable_structure_guard_local=effective_disable_structure_guard,
            native_loop_override=None,
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
        if not _native_safe_headroom_ok():
            print(f"[session] {reason}_fallback_runtime= skipped-low-vram")
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
            min_decode_tokens_local=decode_min_tokens,
            disable_structure_guard_local=True,
            native_loop_override=None,
        )

    text, gen_count, tps, report, timed_out_first, contract_failed_first = _decode_once(
        runtime_obj=runtime,
        local_tokens=list(base_tokens),
        local_temperature=decode_temperature,
        local_top_k=decode_top_k,
        local_lang_top_n=decode_lang_top_n,
        local_greedy=decode_greedy,
        local_rep_penalty=repetition_penalty,
        local_repeat_window=repeat_window,
        local_no_repeat_ngram=no_repeat_ngram,
        local_stream=stream,
        decode_budget_seconds=decode_budget,
        max_steps=decode_step_cap,
        min_decode_tokens_local=decode_min_tokens,
        disable_structure_guard_local=effective_disable_structure_guard,
        native_loop_override=native_loop_handle,
    )

    if (gen_count <= 0 or _is_runtime_fallback_text(text, lang_mode) or contract_failed_first) and retry_budget >= 0.5:
        rescue_top_k = max(1, min(8, top_k if top_k > 0 else 8))
        rescue_lang_top_n = max(32, min(lang_top_n, 96))
        rescue_temp = min(0.68, max(0.45, float(temperature) * 0.82))
        rescue_rep = max(1.18, repetition_penalty)
        rescue_window = max(48, repeat_window)
        rescue_ngram = max(3, no_repeat_ngram)
        rescue_steps = max(8, min(decode_step_cap, 64))

        if brief_numeric_mode:
            rescue_top_k = max(1, min(4, rescue_top_k))
            rescue_lang_top_n = max(16, min(rescue_lang_top_n, 32))
            rescue_temp = min(0.28, rescue_temp)
            rescue_steps = max(2, min(rescue_steps, 8))

        rescue_result = None
        if brief_numeric_mode:
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
                min_decode_tokens_local=decode_min_tokens,
                disable_structure_guard_local=effective_disable_structure_guard,
                native_loop_override=native_loop_handle,
            )
        if rescue_result is None:
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
                        min_decode_tokens_local=decode_min_tokens,
                        disable_structure_guard_local=effective_disable_structure_guard,
                        native_loop_override=native_loop_handle,
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
                min_decode_tokens_local=decode_min_tokens,
                disable_structure_guard_local=effective_disable_structure_guard,
                native_loop_override=native_loop_handle,
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

    text = _finalize_clean_output(text)

    _progress(show_progress, 100, "done", f"{gen_count} tok | {tps:.2f} tok/s")
    native_blend_calls_report = 0
    native_blend_failures_report = 0
    if report is not None:
        print("[session] structure_integrity_pass=", report.get("integrity_pass"))
        print("[session] structure_section_coverage=", round(float(report.get("section_coverage", 0.0)), 4))
        native_blend_calls_report = int(report.get("native_forward_blend_calls", 0) or 0)
        native_blend_failures_report = int(report.get("native_forward_blend_failures", 0) or 0)
        print("[session] flash_attn_calls=", int(report.get("flash_attn_calls", 0) or 0))
        print("[session] fused_attn_calls=", int(report.get("fused_attn_calls", 0) or 0))
        print("[session] scalar_attn_calls=", int(report.get("scalar_attn_calls", 0) or 0))
        print("[session] cpu_attn_calls=", int(report.get("cpu_attn_calls", 0) or 0))
    print("[session] native_forward_blend_calls=", native_blend_calls_report)
    print("[session] native_forward_blend_failures=", native_blend_failures_report)
    lowbit_stats = lowbit_projection_stats_snapshot()
    print("[session] lowbit_exec_calls=", int(lowbit_stats.get("calls", 0)))
    print("[session] lowbit_exec_lowbit_calls=", int(lowbit_stats.get("lowbit_calls", 0)))
    print("[session] lowbit_exec_int4_registered_calls=", int(lowbit_stats.get("int4_registered_calls", 0)))
    print("[session] lowbit_exec_int4_registered_many_calls=", int(lowbit_stats.get("int4_registered_many_calls", 0)))
    print("[session] lowbit_exec_int4_registered_registers=", int(lowbit_stats.get("int4_registered_registers", 0)))
    print("[session] lowbit_exec_int4_registered_failures=", int(lowbit_stats.get("int4_registered_failures", 0)))
    print("[session] lowbit_exec_fallback_gemm_calls=", int(lowbit_stats.get("fallback_gemm_calls", 0)))
    print("[session] lowbit_exec_fallback_matmul_calls=", int(lowbit_stats.get("fallback_matmul_calls", 0)))
    int4_cached_stats = lowbit_stats.get("int4_cached_bridge", {}) if isinstance(lowbit_stats, dict) else {}
    if isinstance(int4_cached_stats, dict) and int4_cached_stats:
        print("[session] int4_cached_dispatch_calls=", int(int4_cached_stats.get("dispatch_calls", 0)))
        print("[session] int4_cached_dispatch_hits=", int(int4_cached_stats.get("dispatch_hits", 0)))
        print("[session] int4_cached_dispatch_misses=", int(int4_cached_stats.get("dispatch_misses", 0)))
        print("[session] int4_cached_register_calls=", int(int4_cached_stats.get("register_calls", 0)))
        print("[session] int4_cached_register_reuse=", int(int4_cached_stats.get("register_reuse", 0)))
        print("[session] int4_cached_register_evictions=", int(int4_cached_stats.get("register_evictions", 0)))

    if phase5_observer is not None:
        try:
            phase5_observer(
                Phase5TurnReport(
                    timed_out=bool(timed_out_first),
                    contract_failed=bool(contract_failed_first),
                    generated_tokens=int(gen_count),
                )
            )
        except Exception:
            pass
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
    os.environ.setdefault("VSPEC_DISABLE_PY_KV_SHADOW", "1")
    if prototype_anf_mode:
        os.environ.setdefault("VSPEC_ANF_TCC_ENABLE", "1")
        os.environ.setdefault("VSPEC_ANF_MAX_HOT_RATIO", "0.10")
        os.environ.setdefault("VSPEC_ANF_ACTIVATION_THRESHOLD", "1.10")
    os.environ.setdefault("VSPEC_C_SAMPLER_REQUIRED", "1")
    os.environ.setdefault("VSPEC_USE_C_SAMPLER", "1")
    os.environ.setdefault("VSPEC_PREFILL_CORE_SCHED", "1")
    os.environ.setdefault("VSPEC_ENABLE_PHASE5_DAEMON", "1")
    if os.getenv("VSPEC_CUBLAS_CACHE_SIZE", "16").strip() in {"0", "off", "OFF"}:
        os.environ.setdefault("VSPEC_INT4_COMPUTE_MODE", "native")
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

    def _resolve_native_model_file(snapshot_dir: Path) -> Optional[str]:
        preferred = sorted(snapshot_dir.glob("model-*.safetensors"))
        if preferred:
            return str(preferred[0])
        any_safe = sorted(snapshot_dir.glob("*.safetensors"))
        if any_safe:
            return str(any_safe[0])
        return None

    def _resolve_native_chat_executable() -> Optional[Path]:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "build" / "Release" / "vspec_native_internal_loop_chat.exe",
            repo_root / "build" / "Debug" / "vspec_native_internal_loop_chat.exe",
            repo_root / "build" / "vspec_native_internal_loop_chat",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _run_native_turn(native_exe: Path, model_file: str, prompt_text: str, max_steps: int, timeout_s: float) -> Optional[str]:
        cmd = [str(native_exe), str(model_file), str(prompt_text), str(max(1, int(max_steps)))]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1.0, float(timeout_s)),
                check=False,
            )
        except Exception as exc:
            print(f"[session] full_native_c= failed ({exc})")
            return None

        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        output_line = None
        for ln in lines:
            if ln.startswith("[native-chat] output:"):
                output_line = ln
                break

        if proc.returncode != 0 or output_line is None:
            stderr_tail = (proc.stderr or "").strip()
            if stderr_tail:
                print(f"[session] full_native_c= failed returncode={proc.returncode} stderr={stderr_tail[:200]}")
            else:
                print(f"[session] full_native_c= failed returncode={proc.returncode}")
            return None

        print(f"[session] full_native_c= on backend={native_exe.name}")
        for ln in lines:
            if ln.startswith("[native-chat] model_family=") or ln.startswith("[native-chat] produced_tokens="):
                print(f"[session] {ln.replace('[native-chat] ', '')}")
        return output_line.split(":", 1)[1].strip()

    native_model_file = _resolve_native_model_file(snapshot)
    native_full_enabled = os.getenv("VSPEC_FULL_NATIVE_C", "1").strip().lower() in {"1", "true", "yes", "on"}
    native_bypass_runtime = os.getenv("VSPEC_FULL_NATIVE_BYPASS_RUNTIME", "1").strip().lower() in {"1", "true", "yes", "on"}
    native_exe = _resolve_native_chat_executable() if native_full_enabled else None

    if native_full_enabled and native_bypass_runtime and native_model_file and native_exe is not None:
        _progress(show_progress, 100, "ready", "full-native-c session")
        print("[session] full_native_c_session= on (runtime bootstrap skipped)")
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
            timeout_s = float(args.max_decode_seconds) if float(args.max_decode_seconds) > 0.0 else 90.0
            out = _run_native_turn(native_exe, native_model_file, prompt, int(args.max_tokens), timeout_s)
            if not out:
                out = "[vspec-decode-error] Generation failed on this turn; native full-C path returned no output."
            print("assistant>", out)
        return
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
        if total <= 0:
            return
        if stage == "layer_load":
            pct = 61 + int((current / total) * 31)
            _progress(show_progress, min(92, pct), "runtime-load", f"layer {current}/{total}")
            return
        if stage == "int4_pre_register":
            pct = 92 + int((current / total) * 4)
            _progress(show_progress, min(96, pct), "runtime-int4", f"pre-register {current}/{total}")

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
    print("[session] int4_pre_register_enabled=", os.getenv("VSPEC_INT4_PRE_REGISTER", "1").strip().lower() in {"1", "true", "yes", "on"})
    print("[session] int4_pre_registered=", int(getattr(runtime, "int4_pre_registered", 0) or 0))
    print("[session] int4_pre_register_failures=", int(getattr(runtime, "int4_pre_register_failures", 0) or 0))
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
    print("[session] kv_core_mirror_enabled=", bool(getattr(runtime, "kv_core_mirror_enabled", False)))
    print("[session] kv_core_mirror_count=", int(getattr(runtime, "kv_core_mirror_count", 0) or 0))
    print("[session] kv_python_shadow_disabled=", bool(getattr(runtime, "kv_python_shadow_disabled", False)))
    print("[session] cublas_cache_size_hint=", os.getenv("VSPEC_CUBLAS_CACHE_SIZE", "16"))
    print("[session] c_sampler_required=", os.getenv("VSPEC_C_SAMPLER_REQUIRED", "1"))
    print("[session] flash_attention_min_tokens=", int(getattr(runtime, "flash_attention_min_tokens", 0) or 0))
    print("[session] flash_attention_block_tokens=", int(getattr(runtime, "flash_attention_block_tokens", 0) or 0))
    print("[session] flash_attention_available=", bool(args.device in {"cuda", "cuda-native"}))
    print("[session] cuda_graph_capture_enabled=", os.getenv("VSPEC_CUDA_GRAPH_CAPTURE", "1").strip().lower() in {"1", "true", "yes", "on"})
    print("[session] native_graph_sig_mode=", os.getenv("VSPEC_NATIVE_GRAPH_SIG_MODE", "shape-only"))
    cap = cuda_device_capability()
    if cap is None:
        cap = _fallback_cuda_capability()
    if cap is not None:
        print(f"[session] cuda_cc= {cap[0]}.{cap[1]} sm_count={cap[2]}")
    else:
        print("[session] cuda_cc= unknown")
    print("[session] int4_tensorcore_available=", bool(int4_tensorcore_available()))
    print("[session] int4_compute_mode=", int4_compute_mode())
    int4_blockwise_enabled = os.getenv("VSPEC_INT4_BLOCKWISE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
    int4_block_size_raw = os.getenv("VSPEC_INT4_BLOCK_SIZE", "32").strip()
    try:
        int4_block_size = max(32, ((int(int4_block_size_raw) + 31) // 32) * 32)
    except Exception:
        int4_block_size = 32
    print("[session] int4_blockwise_enabled=", bool(int4_blockwise_enabled))
    print("[session] int4_block_size=", int(int4_block_size))
    prequant_gguf_fastpath = bool(
        str(getattr(runtime, "quant_source_format", "")).lower() == "gguf"
        and bool(getattr(runtime, "quant_runtime_disabled", False))
    )
    print("[session] prequant_gguf_fastpath=", prequant_gguf_fastpath)
    try:
        from vspec_cuda_bridge import rmsnorm_f32_available, silu_f32_available, mul_f32_available  # local import to avoid startup hard-fail

        print("[session] fused_ops_rmsnorm=", bool(rmsnorm_f32_available()))
        print("[session] fused_ops_silu=", bool(silu_f32_available()))
        print("[session] fused_ops_mul=", bool(mul_f32_available()))
    except Exception:
        print("[session] fused_ops_rmsnorm=", False)
        print("[session] fused_ops_silu=", False)
        print("[session] fused_ops_mul=", False)

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

    phase5_daemon = SessionCoreDaemonSupervisor(
        runtime=runtime,
        max_tokens=int(args.max_tokens),
        native_model_file=native_model_file,
        seed=int(args.seed),
    )
    phase5_daemon.start()
    phase5_status = phase5_daemon.status()
    print("[session] phase5_daemon_enabled=", int(bool(phase5_status.get("enabled", False))))
    print("[session] phase5_daemon_loop_available=", int(bool(phase5_status.get("loop_available", False))))
    print("[session] phase5_daemon_forward_ctx_available=", int(bool(phase5_status.get("forward_ctx_available", False))))

    def _observe_phase5_turn(turn_report: Phase5TurnReport) -> None:
        phase5_daemon.observe_turn(turn_report)
        st = phase5_daemon.status()
        print("[session] phase5_turns_total=", int(st.get("turns_total", 0) or 0))
        print("[session] phase5_turns_timeout=", int(st.get("turns_timeout", 0) or 0))
        print("[session] phase5_turns_contract_failed=", int(st.get("turns_contract_failed", 0) or 0))
        print("[session] phase5_daemon_restarts=", int(st.get("restarts", 0) or 0))

    try:
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
                native_loop_handle=phase5_daemon.loop_handle,
                native_forward_ctx=phase5_daemon.forward_ctx,
                native_model_file=native_model_file,
                phase5_observer=_observe_phase5_turn,
            )
            if args.no_stream:
                print("assistant>", out)
    finally:
        phase5_daemon.close()


if __name__ == "__main__":
    main()
