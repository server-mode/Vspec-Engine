from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
class CaseResult:
    lang: str
    prompt: str
    text_baseline: str
    text_3bit: str
    similarity: float
    guard_baseline_ok: bool
    guard_3bit_ok: bool


def _build_runtime(config: dict, weight_index: dict, max_layers: int, fused_bits: int):
    os.environ["VSPEC_FUSED_BITS"] = str(fused_bits)
    os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "0"
    return build_generic_runtime(config, weight_index, max_layers=max_layers, device="cuda-native")


def _generate_text(runtime, tokenizer, adapter, tok_cfg: dict, prompt: str, lang: str, chat_format: str, max_steps: int) -> str:
    prompt_for_model = build_prompt(prompt, adapter.model_type, tok_cfg, lang, chat_format)
    ids = list(tokenizer.encode(prompt_for_model).ids)
    if adapter.bos_token_id is not None and (not ids or ids[0] != adapter.bos_token_id):
        ids = [adapter.bos_token_id] + ids

    if len(ids) > 1 and hasattr(runtime, "cache_k"):
        runtime.cache_k = []
        runtime.cache_v = []
        runtime.position = 0
        prefill = ids[:-1]
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens(prefill)
        else:
            for tid in prefill:
                runtime.forward_logits([tid])

    generated = []
    for _ in range(max_steps):
        logits = runtime.forward_logits([ids[-1]])
        if not logits:
            break
        next_id = int(np.argmax(np.asarray(logits, dtype=np.float32)))
        generated.append(next_id)
        ids.append(next_id)
        if adapter.eos_token_id is not None and next_id == adapter.eos_token_id:
            break

    text = tokenizer.decode(generated).strip()
    return text


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _run_case(runtime_base, runtime_3bit, tokenizer, adapter, tok_cfg: dict, prompt: str, lang: str, chat_format: str, max_steps: int) -> CaseResult:
    text_base = _generate_text(runtime_base, tokenizer, adapter, tok_cfg, prompt, lang, chat_format, max_steps)
    text_3bit = _generate_text(runtime_3bit, tokenizer, adapter, tok_cfg, prompt, lang, chat_format, max_steps)

    guard = LanguageStabilityGuard(prompt=prompt, lang_mode=lang, strictness=0.72, prioritize_english=True)
    b_ok = guard.allow_text(text_base)
    q_ok = guard.allow_text(text_3bit)
    sim = SequenceMatcher(None, _normalize_text(text_base), _normalize_text(text_3bit)).ratio()

    return CaseResult(
        lang=lang,
        prompt=prompt,
        text_baseline=text_base,
        text_3bit=text_3bit,
        similarity=float(sim),
        guard_baseline_ok=bool(b_ok),
        guard_3bit_ok=bool(q_ok),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 24 quality gate: language/script stability for 3-bit path")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--output", default=str(ROOT / "logs" / "week24_quality_gate.json"))
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

    runtime_base = _build_runtime(config, weight_index, max_layers=args.max_layers, fused_bits=0)
    runtime_3bit = _build_runtime(config, weight_index, max_layers=args.max_layers, fused_bits=3)
    if runtime_base is None or runtime_3bit is None:
        raise RuntimeError("runtime init failed")

    hw_before = capture_hardware_snapshot(runtime=runtime_base, backend_hint="cuda-native")

    cases = [
        ("en", "Summarize this in one short sentence: Vspec runtime focuses on low-bit native inference."),
        ("vi", "Tóm tắt ngắn: Vspec runtime tập trung vào suy luận native low-bit."),
    ]

    outputs: list[dict] = []
    pass_count = 0
    for lang, prompt in cases:
        result = _run_case(runtime_base, runtime_3bit, tokenizer, adapter, tok_cfg, prompt, lang, args.chat_format, args.decode_steps)
        case_pass = (result.guard_3bit_ok and result.similarity >= 0.25)
        if case_pass:
            pass_count += 1
        outputs.append({
            "lang": result.lang,
            "prompt": result.prompt,
            "text_baseline": result.text_baseline,
            "text_3bit": result.text_3bit,
            "similarity": result.similarity,
            "guard_baseline_ok": result.guard_baseline_ok,
            "guard_3bit_ok": result.guard_3bit_ok,
            "case_pass": case_pass,
        })

    report = {
        "week": 24,
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "max_layers": int(args.max_layers),
        "decode_steps": int(args.decode_steps),
        "cases": outputs,
        "pass_count": pass_count,
        "total_cases": len(outputs),
        "kpi_quality_gate_pass": bool(pass_count == len(outputs)),
        "hardware": build_hardware_report(hw_before, capture_hardware_snapshot(runtime=runtime_3bit, backend_hint="cuda-native")),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[hardware] {summarize_hardware_usage(report['hardware']['after'])}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
