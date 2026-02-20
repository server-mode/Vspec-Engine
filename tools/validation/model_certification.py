import argparse
import csv
import json
from pathlib import Path


def _dtype_size(dtype: str) -> int:
    key = dtype.upper()
    if key in {"F32", "FLOAT32"}:
        return 4
    if key in {"F16", "BF16", "FLOAT16", "BFLOAT16"}:
        return 2
    if key in {"F64", "FLOAT64"}:
        return 8
    if key in {"I8", "INT8", "U8", "UINT8"}:
        return 1
    if key in {"I16", "INT16", "U16", "UINT16"}:
        return 2
    if key in {"I32", "INT32", "U32", "UINT32"}:
        return 4
    if key in {"I64", "INT64", "U64", "UINT64"}:
        return 8
    if key in {"I4", "INT4"}:
        return 1
    if key in {"I3", "INT3"}:
        return 1
    return 0


def _load_csv(path: Path) -> list[list[float]]:
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    return rows


def _flatten(rows: list[list[float]]) -> tuple[list[float], int, int]:
    if not rows:
        return [], 0, 0
    vocab = len(rows[0])
    for row in rows:
        if len(row) != vocab:
            raise ValueError("CSV rows must have consistent column counts")
    flat = [value for row in rows for value in row]
    return flat, vocab, len(rows)


def _perplexity_from_logits(logits: list[float], vocab: int, steps: int) -> float:
    if not logits or vocab == 0 or steps == 0:
        return 0.0
    total_nll = 0.0
    for t in range(steps):
        offset = t * vocab
        row = logits[offset : offset + vocab]
        max_logit = max(row)
        denom = sum(pow(2.718281828, v - max_logit) for v in row)
        log_sum_exp = max_logit + (0.0 if denom == 0.0 else __import__("math").log(denom))
        total_nll += log_sum_exp - row[0]
    mean_nll = total_nll / float(steps)
    return float(__import__("math").exp(mean_nll))


def _drift(baseline: list[float], test: list[float]) -> dict:
    if not baseline or not test or len(baseline) != len(test):
        return {"mean_abs": 0.0, "max_abs": 0.0, "mean_rel": 0.0}
    sum_abs = 0.0
    sum_rel = 0.0
    max_abs = 0.0
    for b, t in zip(baseline, test):
        diff = abs(t - b)
        sum_abs += diff
        if diff > max_abs:
            max_abs = diff
        denom = abs(b) if abs(b) > 1e-6 else 1.0
        sum_rel += diff / denom
    count = float(len(baseline))
    return {
        "mean_abs": sum_abs / count,
        "max_abs": max_abs,
        "mean_rel": sum_rel / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model certification harness (Phase 3)")
    parser.add_argument("--model-id", required=True, help="Model identifier (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--ir", required=True, help="Path to IR json")
    parser.add_argument("--baseline-logits", help="CSV logits from baseline run")
    parser.add_argument("--test-logits", help="CSV logits from test run")
    parser.add_argument("--tokens", type=int, default=0, help="Generated tokens for throughput calc")
    parser.add_argument("--seconds", type=float, default=0.0, help="Elapsed seconds for throughput calc")
    parser.add_argument("--output", help="Write report to json file")
    args = parser.parse_args()

    ir = json.loads(Path(args.ir).read_text(encoding="utf-8"))
    tensors = ir.get("tensors", [])

    weight_bytes = 0
    unknown_dtypes = {}
    for tensor in tensors:
        dtype = tensor.get("dtype", "")
        shape = tensor.get("shape", [])
        count = 1
        for dim in shape:
            count *= int(dim)
        size = _dtype_size(dtype)
        if size == 0:
            unknown_dtypes[dtype] = unknown_dtypes.get(dtype, 0) + 1
            continue
        weight_bytes += count * size

    report = {
        "model_id": args.model_id,
        "tensor_count": len(tensors),
        "weight_bytes": weight_bytes,
        "unknown_dtypes": unknown_dtypes,
        "metrics": {},
    }

    if args.tokens > 0 and args.seconds > 0.0:
        report["metrics"]["tokens_per_sec"] = args.tokens / args.seconds

    if args.baseline_logits and args.test_logits:
        baseline_rows = _load_csv(Path(args.baseline_logits))
        test_rows = _load_csv(Path(args.test_logits))
        baseline, vocab, steps = _flatten(baseline_rows)
        test, _, _ = _flatten(test_rows)
        report["metrics"]["perplexity_baseline"] = _perplexity_from_logits(baseline, vocab, steps)
        report["metrics"]["perplexity_test"] = _perplexity_from_logits(test, vocab, steps)
        report["metrics"]["drift"] = _drift(baseline, test)

    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
