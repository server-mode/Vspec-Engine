from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = ROOT / "research" / "llama.cpp-b8234" / "llama.cpp-b8234" / "gguf-py"
CHAT_PY = ROOT / "Vspec-chat" / "python"

if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))
if str(CHAT_PY) not in sys.path:
    sys.path.insert(0, str(CHAT_PY))

import gguf  # noqa: E402
from model_loader import build_weight_index  # noqa: E402
from runtime_inference import _load_tensor  # noqa: E402


def _write_test_gguf(path: Path, qtype, weight: np.ndarray) -> None:
    writer = gguf.GGUFWriter(str(path), arch="qwen3")
    writer.add_name("gguf-quant-test")
    writer.add_block_count(1)
    writer.add_context_length(32)
    writer.add_embedding_length(int(weight.shape[1]))
    writer.add_feed_forward_length(max(16, int(weight.shape[1] * 2)))
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["<pad>", "hello", " world"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    try:
        qdata = gguf.quants.quantize(weight.astype(np.float32), qtype)
    except Exception:
        byte_shape = gguf.quants.quant_shape_to_byte_shape(weight.shape, qtype)
        rng = np.random.default_rng(20260308 + int(qtype.value))
        qdata = rng.integers(0, 256, size=byte_shape, dtype=np.uint8)
    writer.add_tensor("blk.0.attn_q.weight", qdata, raw_shape=qdata.shape, raw_dtype=qtype)

    writer.write_header_to_file(path=path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _run_case(qtype, seed: int) -> tuple[float, tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((32, 256), dtype=np.float32) * 0.25).astype(np.float32)

    fd, name = tempfile.mkstemp(suffix=f".{qtype.name.lower()}.gguf")
    os.close(fd)
    path = Path(name)
    path.unlink(missing_ok=True)

    try:
        _write_test_gguf(path, qtype, weight)
        index = build_weight_index(path)
        loaded = _load_tensor(index["model.layers.0.self_attn.q_proj.weight"])
        if loaded is None:
            raise RuntimeError(f"loader returned None for {qtype.name}")
        try:
            qdata = gguf.quants.quantize(weight, qtype)
        except Exception:
            byte_shape = gguf.quants.quant_shape_to_byte_shape(weight.shape, qtype)
            rng_ref = np.random.default_rng(20260308 + int(qtype.value))
            qdata = rng_ref.integers(0, 256, size=byte_shape, dtype=np.uint8)
        ref = gguf.quants.dequantize(qdata, qtype).astype(np.float32, copy=False)
        ref_safe = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        loaded_safe = np.nan_to_num(loaded, nan=0.0, posinf=0.0, neginf=0.0)
        err = float(np.max(np.abs(ref_safe - loaded_safe)))
        return err, tuple(int(v) for v in loaded.shape)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    cases = [
        (gguf.GGMLQuantizationType.Q4_0, 2026030801),
        (gguf.GGMLQuantizationType.Q4_K, 2026030802),
        (gguf.GGMLQuantizationType.Q5_K, 2026030803),
        (gguf.GGMLQuantizationType.Q6_K, 2026030804),
        (gguf.GGMLQuantizationType.Q8_0, 2026030805),
    ]

    worst = 0.0
    for qtype, seed in cases:
        err, shape = _run_case(qtype, seed)
        worst = max(worst, err)
        print(f"{qtype.name}: max_abs_err={err:.6f} shape={shape}")

    threshold = float(os.getenv("VSPEC_GGUF_VALIDATE_ABS_THRESHOLD", "0.0005"))
    print(f"summary: worst_max_abs_err={worst:.6f}")
    if worst > threshold:
        print(f"FAIL: worst abs error {worst:.6f} > threshold {threshold:.6f}")
        return 1
    print("PASS: GGUF quant loader validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())