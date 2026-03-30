from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str((ROOT / "tools" / "cli").resolve()))

from vspec_runner import VspecRunArgs, run_once  # type: ignore[import-not-found]


def main() -> int:
    model = os.getenv(
        "VSPEC_PHASE4_MODEL",
        "C:/Users/Long/.cache/huggingface/hub/models--sshleifer--tiny-gpt2/snapshots/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
    )

    os.environ["VSPEC_USE_C_SAMPLER"] = os.getenv("VSPEC_USE_C_SAMPLER", "1")
    os.environ["VSPEC_SAMPLER_PARITY_SHADOW"] = os.getenv("VSPEC_SAMPLER_PARITY_SHADOW", "1")
    os.environ["VSPEC_SAMPLER_PARITY_FALLBACK"] = os.getenv("VSPEC_SAMPLER_PARITY_FALLBACK", "0")

    args = VspecRunArgs(
        model=model,
        prompt="Xin chao, tom tat 2 y ve phase 4.",
        prompts_file="",
        batch_output_file="",
        device="cpu",
        fused_bits=None,
        target_bits=None,
        max_layers=0,
        max_tokens=12,
        max_decode_seconds=25.0,
        max_retry_seconds=-1.0,
        temperature=0.8,
        top_k=20,
        repetition_penalty=1.1,
        repeat_window=32,
        no_repeat_ngram=1,
        speed_preset="fast",
        lang="vi",
        stream=False,
        unsafe_low_layers=False,
    )

    result = run_once(args)
    metrics = result.get("metrics", {}) or {}
    report = {
        "ok": bool(result.get("ok", False)),
        "returncode": int(result.get("returncode", 1)),
        "model": model,
        "text_preview": str(result.get("text", ""))[:180],
        "phase4_sampler_calls": metrics.get("phase4_sampler_calls", ""),
        "phase4_c_sampler_calls": metrics.get("phase4_c_sampler_calls", ""),
        "phase4_python_sampler_calls": metrics.get("phase4_python_sampler_calls", ""),
        "phase4_sampler_parity_checks": metrics.get("phase4_sampler_parity_checks", ""),
        "phase4_sampler_parity_mismatch": metrics.get("phase4_sampler_parity_mismatch", ""),
        "phase4_sampler_parity_fallbacks": metrics.get("phase4_sampler_parity_fallbacks", ""),
        "decode_contract_ok": metrics.get("decode_contract_ok", ""),
        "phase1_contract_failed": metrics.get("phase1_contract_failed", ""),
    }

    out_path = ROOT / "logs" / "job62_phase4_runtime_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

    phase4_calls = str(report.get("phase4_sampler_calls", "")).strip()
    phase4_c_calls = str(report.get("phase4_c_sampler_calls", "")).strip()
    pass_gate = bool(report["ok"]) and phase4_calls not in {"", "0"} and phase4_c_calls not in {"", "0"}
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
