from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
import sys
sys.path.insert(0, str(CHAT_PY))

from model_loader import find_snapshot_dir, read_config, read_tokenizer_config, load_tokenizer, build_weight_index
from model_adapters import select_adapter
from runtime_inference import build_generic_runtime
from chat_prompt import build_prompt
from hardware_telemetry import build_hardware_report, capture_hardware_snapshot, summarize_hardware_usage


def _build_runtime(config: dict, weight_index: dict, disable_fused: bool, max_layers: int):
    os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "1" if disable_fused else "0"
    os.environ.setdefault("VSPEC_FUSED_BITS", "4")
    return build_generic_runtime(config, weight_index, max_layers=max_layers, device="cuda-native")


def _run_decode(runtime, token_ids: list[int], decode_steps: int) -> dict:
    if len(token_ids) > 1 and hasattr(runtime, "cache_k"):
        runtime.cache_k = []
        runtime.cache_v = []
        runtime.position = 0
        prefill_ids = token_ids[:-1]
        t0 = time.perf_counter()
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens(prefill_ids)
        else:
            for tid in prefill_ids:
                runtime.forward_logits([tid])
        prefill_sec = time.perf_counter() - t0
    else:
        prefill_sec = 0.0

    ids = list(token_ids)
    t1 = time.perf_counter()
    for _ in range(decode_steps):
        logits = runtime.forward_logits([ids[-1]])
        if not logits:
            break
        next_id = int(np.argmax(np.asarray(logits, dtype=np.float32)))
        ids.append(next_id)
    decode_sec = time.perf_counter() - t1

    return {
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "decode_tokens": decode_steps,
        "decode_tps": (decode_steps / decode_sec) if decode_sec > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 23 decode-only A/B benchmark (fused attention ON/OFF)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Hello benchmark")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--output", default=str(ROOT / "logs" / "week23_decode_ab.json"))
    args = parser.parse_args()

    snapshot = find_snapshot_dir(Path(args.model_dir))
    config = read_config(snapshot)
    tok_cfg = read_tokenizer_config(snapshot)
    tokenizer = load_tokenizer(snapshot)
    if tokenizer is None:
        raise RuntimeError("tokenizer not found; install tokenizers or provide valid snapshot")

    weight_index = build_weight_index(snapshot)
    adapter = select_adapter(config, list(weight_index.keys()))
    prompt_for_model = build_prompt(args.prompt, adapter.model_type, tok_cfg, args.lang, args.chat_format)
    encoded = tokenizer.encode(prompt_for_model)
    token_ids = list(encoded.ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    rt_on = _build_runtime(config, weight_index, disable_fused=False, max_layers=args.max_layers)
    rt_off = _build_runtime(config, weight_index, disable_fused=True, max_layers=args.max_layers)
    if rt_on is None or rt_off is None:
        raise RuntimeError("runtime init failed for one of the A/B cases")

    hw_before = capture_hardware_snapshot(runtime=rt_on, backend_hint="cuda-native")

    on_metrics = _run_decode(rt_on, token_ids, args.decode_steps)
    off_metrics = _run_decode(rt_off, token_ids, args.decode_steps)
    hw_after = capture_hardware_snapshot(runtime=rt_on, backend_hint="cuda-native")

    speedup = (off_metrics["decode_sec"] / on_metrics["decode_sec"]) if on_metrics["decode_sec"] > 0 else 0.0
    delta_pct = ((off_metrics["decode_sec"] - on_metrics["decode_sec"]) / off_metrics["decode_sec"] * 100.0) if off_metrics["decode_sec"] > 0 else 0.0

    result = {
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "prompt_tokens": len(token_ids),
        "decode_steps": args.decode_steps,
        "max_layers": args.max_layers,
        "fused_on": on_metrics,
        "fused_off": off_metrics,
        "hardware": build_hardware_report(hw_before, hw_after),
        "speedup_vs_off": speedup,
        "decode_latency_reduction_pct": delta_pct,
        "kpi_week23_pass": bool(on_metrics["decode_sec"] < off_metrics["decode_sec"]),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[hardware] {summarize_hardware_usage(hw_after)}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
