from __future__ import annotations

import argparse
import json
from pathlib import Path

from vspec_cli_common import collect_model_files, find_model_root, read_vspec_manifest


def _print_manifest(path: Path) -> None:
    m = read_vspec_manifest(path)
    print(f"[vspec-info] package={path}")
    print(f"[vspec-info] format={m.get('format')} version={m.get('version')}")
    print(f"[vspec-info] model_name={m.get('model_name')}")
    print(f"[vspec-info] source={m.get('source')}")
    quant = m.get("quant", {})
    print(f"[vspec-info] quant_label={quant.get('label')} target_bits={quant.get('target_bits')}")
    print(f"[vspec-info] files={len(m.get('files', []))}")


def _print_model_dir(path: Path) -> None:
    root = find_model_root(path)
    files = collect_model_files(root)
    print(f"[vspec-info] model_root={root}")
    print(f"[vspec-info] files={len(files)}")
    config = root / "config.json"
    if config.exists():
        try:
            cfg = json.loads(config.read_text(encoding="utf-8"))
            hidden = cfg.get("hidden_size", cfg.get("n_embd"))
            layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
            heads = cfg.get("num_attention_heads", cfg.get("n_head"))
            model_type = cfg.get("model_type", "")
            print(f"[vspec-info] model_type={model_type} hidden_size={hidden} layers={layers} heads={heads}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect .vspec package or raw model directory")
    parser.add_argument("model", help="Path to .vspec or model directory")
    args = parser.parse_args()

    p = Path(args.model)
    if not p.exists():
        raise SystemExit(f"Path not found: {p}")

    if p.is_file() and p.suffix.lower() == ".vspec":
        _print_manifest(p)
    else:
        _print_model_dir(p)


if __name__ == "__main__":
    main()
