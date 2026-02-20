from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_manifest(path: Path) -> list[dict]:
    tensors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, dtype, shape = line.split("|", 2)
        dims = [int(x.strip()) for x in shape.split(",") if x.strip()]
        tensors.append({"name": name, "dtype": dtype, "shape": dims})
    return tensors


def _parse_safetensors(path: Path) -> list[dict]:
    raw = path.read_bytes()
    hlen = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))

    tensors = []
    for name, obj in header.items():
        if name == "__metadata__":
            continue
        tensors.append(
            {
                "name": name,
                "dtype": obj.get("dtype", ""),
                "shape": obj.get("shape", []),
                "data_offsets": obj.get("data_offsets", [0, 0]),
            }
        )
    return tensors


def _to_ir(tensors: list[dict]) -> dict:
    return {
        "ir_version": "0.1",
        "tensors": tensors,
        "graph": {
            "nodes": [
                {"id": 0, "op": "linear", "inputs": ["x", "w0"], "output": "h0"},
                {"id": 1, "op": "attention", "inputs": ["h0", "kv"], "output": "y"},
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert manifest/safetensors header to Vspec IR JSON")
    parser.add_argument("--input", required=True, help="Input .vpt or .safetensors")
    parser.add_argument("--output", required=True, help="Output IR json path")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() in {".vpt", ".manifest"}:
        tensors = _parse_manifest(in_path)
    elif in_path.suffix.lower() == ".safetensors":
        tensors = _parse_safetensors(in_path)
    else:
        raise SystemExit(f"Unsupported input format: {in_path.suffix}")

    ir = _to_ir(tensors)
    Path(args.output).write_text(json.dumps(ir, indent=2), encoding="utf-8")
    print(f"converted {len(tensors)} tensors -> {args.output}")


if __name__ == "__main__":
    main()
