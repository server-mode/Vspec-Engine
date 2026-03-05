from __future__ import annotations

import argparse
from pathlib import Path

from vspec_cli_common import create_vspec_package, maybe_download_hf_repo


def _resolve_input(args) -> tuple[Path, str, str]:
    if args.hf:
        source = maybe_download_hf_repo(args.hf)
        model_name = args.name or args.hf.split("/")[-1]
        source_hint = f"hf:{args.hf}"
        return source, model_name, source_hint

    if not args.input:
        raise SystemExit("Either --hf or --input is required")

    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"Input model path not found: {p}")

    model_name = args.name or p.name
    source_hint = str(p)
    return p, model_name, source_hint


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HuggingFace/local model into .vspec package")
    parser.add_argument("--hf", default="", help="HF repo id, e.g. meta-llama/Llama-3-8B")
    parser.add_argument("--input", default="", help="Local model directory or snapshot path")
    parser.add_argument("--output", required=True, help="Output .vspec file")
    parser.add_argument("--quant", default="q3", choices=["q2", "q3", "q4", "fp16"], help="Target quant profile tag")
    parser.add_argument("--name", default="", help="Override model name metadata")
    args = parser.parse_args()

    model_path, model_name, source_hint = _resolve_input(args)
    out_path = Path(args.output)

    manifest = create_vspec_package(
        input_model_path=model_path,
        output_path=out_path,
        model_name=model_name,
        quant_label=args.quant,
        source_hint=source_hint,
    )

    print(f"[vspec-convert] output={out_path}")
    print(f"[vspec-convert] model_name={manifest.get('model_name')}")
    print(f"[vspec-convert] files={len(manifest.get('files', []))}")
    print(f"[vspec-convert] target_bits={manifest.get('quant', {}).get('target_bits')}")


if __name__ == "__main__":
    main()
