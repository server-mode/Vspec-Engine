from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
sys.path.insert(0, str(CHAT_PY))

from chat_prompt import build_prompt
from model_adapters import select_adapter
from model_loader import build_weight_index, collect_tensor_names, find_snapshot_dir, load_tokenizer, read_config, read_tokenizer_config
from runtime_inference import build_generic_runtime

PROJ_KEYS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


def _compute_projection_effective_bits(runtime) -> tuple[float, float, int, int]:
    packed_proj = 0
    total_proj = 0
    weighted_bits = 0.0
    weighted_count = 0

    for layer in runtime.layers:
        for key in PROJ_KEYS:
            w = getattr(layer, key)
            rows, cols = int(w.shape[0]), int(w.shape[1])
            count = rows * cols
            total_proj += 1
            if key in layer.packed:
                packed_proj += 1
                _, _, bits, _ = layer.packed[key]
                weighted_bits += float(bits) * count
                weighted_count += count
            else:
                weighted_bits += 16.0 * count
                weighted_count += count

    effective_bits = (weighted_bits / weighted_count) if weighted_count > 0 else 0.0
    coverage = (packed_proj / total_proj) if total_proj > 0 else 0.0
    return effective_bits, coverage, packed_proj, total_proj


def _run_short_decode(runtime, token_ids: list[int], steps: int) -> tuple[int, float]:
    if len(token_ids) > 1 and hasattr(runtime, "cache_k"):
        runtime.cache_k = []
        runtime.cache_v = []
        runtime.position = 0
        prefill = token_ids[:-1]
        if hasattr(runtime, "prefill_tokens"):
            runtime.prefill_tokens(prefill)
        else:
            for tid in prefill:
                runtime.forward_logits([tid])

    ids = list(token_ids)
    t0 = time.perf_counter()
    generated = 0
    for _ in range(steps):
        logits = runtime.forward_logits(ids)
        if not logits:
            break
        next_id = int(np.argmax(np.asarray(logits, dtype=np.float32)))
        ids.append(next_id)
        generated += 1
    decode_sec = time.perf_counter() - t0
    return generated, decode_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 24 KPI: end-to-end 3-bit execution check")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Hello from Week 24 benchmark")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--chat-format", default="plain")
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--output", default=str(ROOT / "logs" / "week24_3bit_kpi.json"))
    args = parser.parse_args()

    os.environ["VSPEC_FUSED_BITS"] = "3"
    os.environ["VSPEC_DISABLE_FUSED_ATTN"] = "0"

    snapshot = find_snapshot_dir(Path(args.model_dir))
    config = read_config(snapshot)
    tok_cfg = read_tokenizer_config(snapshot)
    tokenizer = load_tokenizer(snapshot)
    if tokenizer is None:
        raise RuntimeError("tokenizer not found")

    tensor_names = collect_tensor_names(snapshot)
    weight_index = build_weight_index(snapshot)
    adapter = select_adapter(config, tensor_names)

    runtime = build_generic_runtime(config, weight_index, max_layers=args.max_layers, device="cuda-native")
    if runtime is None:
        raise RuntimeError("failed to init runtime")

    prompt_for_model = build_prompt(args.prompt, adapter.model_type, tok_cfg, args.lang, args.chat_format)
    token_ids = list(tokenizer.encode(prompt_for_model).ids)
    if adapter.bos_token_id is not None and (not token_ids or token_ids[0] != adapter.bos_token_id):
        token_ids = [adapter.bos_token_id] + token_ids

    eff_bits, coverage, packed_proj, total_proj = _compute_projection_effective_bits(runtime)
    generated, decode_sec = _run_short_decode(runtime, token_ids, args.decode_steps)

    result = {
        "week": 24,
        "model_dir": str(args.model_dir),
        "snapshot": str(snapshot),
        "runtime_mode": "vspec-native-cuda",
        "fused_bits": int(getattr(runtime, "fused_bits", 0)),
        "max_layers": int(args.max_layers),
        "prompt_tokens": len(token_ids),
        "decode_steps_requested": int(args.decode_steps),
        "decode_steps_generated": int(generated),
        "decode_sec": float(decode_sec),
        "projection_3bit_coverage": float(coverage),
        "projection_3bit_packed": int(packed_proj),
        "projection_total": int(total_proj),
        "effective_bits_estimate": float(eff_bits),
        "kpi_run_large_model_3bit": bool(generated > 0 and args.max_layers >= 4),
        "kpi_effective_bits_lt4": bool(eff_bits < 4.0),
        "kpi_week24_pass": bool((generated > 0 and args.max_layers >= 4) and (eff_bits < 4.0)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
