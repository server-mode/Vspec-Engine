import argparse
import json
import re
from pathlib import Path


def _dtype_bits(dtype: str) -> int:
    key = dtype.upper()
    if key in {"F32", "FLOAT32"}:
        return 32
    if key in {"F16", "FLOAT16", "BF16", "BFLOAT16"}:
        return 16
    if key in {"F64", "FLOAT64"}:
        return 64
    if key in {"I8", "INT8", "U8", "UINT8"}:
        return 8
    if key in {"I16", "INT16", "U16", "UINT16"}:
        return 16
    if key in {"I32", "INT32", "U32", "UINT32"}:
        return 32
    if key in {"I64", "INT64", "U64", "UINT64"}:
        return 64
    if key in {"I4", "INT4"}:
        return 4
    if key in {"I3", "INT3"}:
        return 3
    if key in {"I2", "INT2"}:
        return 2
    return 0


def _precision_bytes(name: str) -> int:
    key = name.lower()
    if key in {"fp16", "f16", "bf16"}:
        return 2
    if key in {"fp32", "f32"}:
        return 4
    if key in {"fp64", "f64"}:
        return 8
    raise ValueError(f"unsupported precision: {name}")


def _bytes_from_bits(count: int, bits: int) -> int:
    if bits <= 0:
        return 0
    total_bits = count * bits
    return (total_bits + 7) // 8


def _count_elements(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _parse_perf_from_text(text: str) -> tuple[int, float]:
    tokens = 0
    seconds = 0.0

    token_match = re.search(r"tokens\s*[=:]\s*(\d+)", text, flags=re.IGNORECASE)
    if token_match:
        tokens = int(token_match.group(1))

    seconds_match = re.search(r"seconds\s*[=:]\s*([0-9.]+)", text, flags=re.IGNORECASE)
    if seconds_match:
        seconds = float(seconds_match.group(1))

    tps_match = re.search(r"tokens_per_sec\s*[=:]\s*([0-9.]+)", text, flags=re.IGNORECASE)
    if tps_match and tokens == 0 and seconds == 0.0:
        tps = float(tps_match.group(1))
        return int(tps * 1000), 1000.0

    return tokens, seconds


def _parse_extra_metrics(text: str) -> dict:
    metrics: dict = {}

    drift_match = re.search(r"perplexity_drift\s*[=:]\s*([0-9.]+)", text, flags=re.IGNORECASE)
    if drift_match:
        metrics["perplexity_drift"] = float(drift_match.group(1))

    occ_match = re.search(r"sm_occupancy\s*[=:]\s*([0-9.]+)\s*%?", text, flags=re.IGNORECASE)
    if occ_match:
        metrics["sm_occupancy_percent"] = float(occ_match.group(1))

    bw_match = re.search(r"memory_bandwidth\s*[=:]\s*([0-9.]+)\s*%?", text, flags=re.IGNORECASE)
    if bw_match:
        metrics["memory_bandwidth_percent"] = float(bw_match.group(1))

    stall_match = re.search(r"warp_stall_reason\s*[=:]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if stall_match:
        metrics["warp_stall_reason"] = stall_match.group(1).strip()

    seq_match = re.search(r"sequence_scaling\s*[=:]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if seq_match:
        metrics["sequence_scaling"] = seq_match.group(1).strip()

    return metrics


def _parse_perf_from_file(path: Path) -> tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _parse_perf_from_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom LLM benchmark report for Vspec runtime")
    parser.add_argument("--model-id", required=True, help="Model identifier (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--ir", required=True, help="Path to IR json")
    parser.add_argument("--baseline-precision", default="fp16", help="Baseline weight precision (fp16/fp32/bf16)")
    parser.add_argument("--vspec-bits", type=int, default=4, help="Default bits for Vspec weights")
    parser.add_argument("--force-vspec-bits", action="store_true", help="Ignore tensor dtypes and apply vspec-bits to all weights")
    parser.add_argument("--kv-tokens", type=int, default=0, help="KV cache tokens")
    parser.add_argument("--kv-heads", type=int, default=0, help="KV heads")
    parser.add_argument("--kv-head-dim", type=int, default=0, help="KV head dimension")
    parser.add_argument("--kv-precision", default="fp16", help="KV precision (fp16/fp32/bf16)")
    parser.add_argument("--vspec-tokens", type=int, default=0, help="Generated tokens by Vspec run")
    parser.add_argument("--vspec-seconds", type=float, default=0.0, help="Elapsed seconds for Vspec run")
    parser.add_argument("--baseline-tokens", type=int, default=0, help="Generated tokens by baseline run")
    parser.add_argument("--baseline-seconds", type=float, default=0.0, help="Elapsed seconds for baseline run")
    parser.add_argument("--vspec-log", help="Path to Vspec run log; supports tokens/seconds parsing")
    parser.add_argument("--baseline-log", help="Path to baseline run log; supports tokens/seconds parsing")
    parser.add_argument("--perplexity-drift", type=float, help="Override perplexity drift metric")
    parser.add_argument("--sm-occupancy", type=float, help="Override SM occupancy percent")
    parser.add_argument("--memory-bandwidth", type=float, help="Override memory bandwidth percent")
    parser.add_argument("--warp-stall-reason", help="Override warp stall reason")
    parser.add_argument("--sequence-scaling", help="Override sequence scaling result")
    parser.add_argument("--output", help="Write report to json file")
    args = parser.parse_args()

    ir = json.loads(Path(args.ir).read_text(encoding="utf-8"))
    tensors = ir.get("tensors", [])

    baseline_bytes = _precision_bytes(args.baseline_precision)
    kv_bytes = _precision_bytes(args.kv_precision)

    vspec_log_metrics = {}
    baseline_log_metrics = {}

    if args.vspec_log:
        try:
            log_text = Path(args.vspec_log).read_text(encoding="utf-8", errors="ignore")
            tokens, seconds = _parse_perf_from_text(log_text)
            vspec_log_metrics = _parse_extra_metrics(log_text)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))
        if tokens > 0 and seconds > 0.0:
            args.vspec_tokens = tokens
            args.vspec_seconds = seconds

    if args.baseline_log:
        try:
            log_text = Path(args.baseline_log).read_text(encoding="utf-8", errors="ignore")
            tokens, seconds = _parse_perf_from_text(log_text)
            baseline_log_metrics = _parse_extra_metrics(log_text)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))
        if tokens > 0 and seconds > 0.0:
            args.baseline_tokens = tokens
            args.baseline_seconds = seconds

    total_elements = 0
    vspec_weight_bytes = 0
    unknown_dtypes = {}

    for tensor in tensors:
        dtype = tensor.get("dtype", "")
        shape = tensor.get("shape", [])
        count = _count_elements(shape)
        total_elements += count

        if args.force_vspec_bits:
            bits = args.vspec_bits
        else:
            bits = _dtype_bits(dtype)
            if bits == 0:
                unknown_dtypes[dtype] = unknown_dtypes.get(dtype, 0) + 1
                bits = args.vspec_bits
        vspec_weight_bytes += _bytes_from_bits(count, bits)

    baseline_weight_bytes = total_elements * baseline_bytes

    kv_cache_bytes = 0
    if args.kv_tokens > 0 and args.kv_heads > 0 and args.kv_head_dim > 0:
        kv_cache_bytes = args.kv_tokens * args.kv_heads * args.kv_head_dim * 2 * kv_bytes

    baseline_total = baseline_weight_bytes + kv_cache_bytes
    vspec_total = vspec_weight_bytes + kv_cache_bytes

    report = {
        "model_id": args.model_id,
        "tensor_count": len(tensors),
        "total_elements": total_elements,
        "baseline_precision": args.baseline_precision,
        "vspec_bits": args.vspec_bits,
        "unknown_dtypes": unknown_dtypes,
        "memory_bytes": {
            "baseline_weights": baseline_weight_bytes,
            "vspec_weights": vspec_weight_bytes,
            "kv_cache": kv_cache_bytes,
            "baseline_total": baseline_total,
            "vspec_total": vspec_total,
        },
        "memory_savings": {},
        "throughput": {},
        "metrics": {},
    }

    if baseline_total > 0:
        savings = baseline_total - vspec_total
        report["memory_savings"] = {
            "bytes": savings,
            "percent": (savings / baseline_total) * 100.0,
        }

    if args.vspec_tokens > 0 and args.vspec_seconds > 0.0:
        report["throughput"]["vspec_tokens_per_sec"] = args.vspec_tokens / args.vspec_seconds
    if args.baseline_tokens > 0 and args.baseline_seconds > 0.0:
        report["throughput"]["baseline_tokens_per_sec"] = args.baseline_tokens / args.baseline_seconds

    vspec_tps = report["throughput"].get("vspec_tokens_per_sec")
    baseline_tps = report["throughput"].get("baseline_tokens_per_sec")
    if vspec_tps and baseline_tps:
        report["throughput"]["speedup"] = vspec_tps / baseline_tps

    if vspec_log_metrics:
        report["metrics"].update(vspec_log_metrics)
    if baseline_log_metrics:
        for key, value in baseline_log_metrics.items():
            if key not in report["metrics"]:
                report["metrics"][key] = value

    if args.perplexity_drift is not None:
        report["metrics"]["perplexity_drift"] = args.perplexity_drift
    if args.sm_occupancy is not None:
        report["metrics"]["sm_occupancy_percent"] = args.sm_occupancy
    if args.memory_bandwidth is not None:
        report["metrics"]["memory_bandwidth_percent"] = args.memory_bandwidth
    if args.warp_stall_reason:
        report["metrics"]["warp_stall_reason"] = args.warp_stall_reason
    if args.sequence_scaling:
        report["metrics"]["sequence_scaling"] = args.sequence_scaling

    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
