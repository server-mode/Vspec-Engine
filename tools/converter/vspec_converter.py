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


def _select_snapshot_dir(root: Path) -> Path | None:
    snapshot_root = root / "snapshots"
    if not snapshot_root.exists():
        return None
    candidates = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _collect_safetensors_from_dir(root: Path) -> list[Path]:
    snapshot_dir = _select_snapshot_dir(root)
    search_root = snapshot_dir if snapshot_dir else root
    paths = [p for p in search_root.rglob("*.safetensors") if ".no_exist" not in str(p)]
    return sorted(paths)


def _merge_tensor_headers(paths: list[Path]) -> list[dict]:
    by_name: dict[str, dict] = {}
    for path in paths:
        for tensor in _parse_safetensors(path):
            name = tensor.get("name", "")
            if name and name not in by_name:
                by_name[name] = tensor
    return list(by_name.values())


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
    if in_path.is_dir():
        shards = _collect_safetensors_from_dir(in_path)
        if not shards:
            raise SystemExit(f"No .safetensors files found under: {in_path}")
        tensors = _merge_tensor_headers(shards)
    elif in_path.suffix.lower() in {".vpt", ".manifest"}:
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
