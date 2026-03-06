from __future__ import annotations

import argparse
import json

from vspec_runner import VspecRunArgs, run_interactive, run_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vspec model quickly from .vspec or model directory")
    parser.add_argument("model", nargs="?", default="", help="Path to .vspec package or model directory")
    parser.add_argument("-m", "--model", dest="model_opt", default="", help="Path to .vspec package or model directory")
    parser.add_argument("-p", "--prompt", default="", help="Prompt text")
    parser.add_argument("--chat", action="store_true", help="Interactive terminal chat mode")

    parser.add_argument("--device", default="", choices=["", "cpu", "cuda", "cuda-native", "torch-cuda"])
    parser.add_argument("--fused-bits", type=int, choices=[0, 3, 4], default=None)
    parser.add_argument("--target-bits", type=int, choices=[0, 2, 3, 4], default=None)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-decode-seconds", type=float, default=-1.0, help="<0 auto, =0 disable timeout, >0 fixed")
    parser.add_argument("--max-retry-seconds", type=float, default=-1.0, help="<0 auto, =0 disable retry, >0 fixed")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repeat-window", type=int, default=64)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--speed-preset", default="fast", choices=["normal", "fast", "ultra"])
    parser.add_argument("--lang", default="auto", choices=["auto", "vi", "en"])
    parser.add_argument("--unsafe-low-layers", action="store_true", help="Allow very low max-layers even if response quality may collapse")
    parser.add_argument("--stream", action="store_true", help="Stream tokens for single prompt mode")
    parser.add_argument("--json", action="store_true", help="Emit JSON output for automation")

    args = parser.parse_args()

    model = (args.model_opt or args.model or "").strip()
    if not model:
        parser.error("model path is required. Use positional model or --model")

    run_args = VspecRunArgs(
        model=model,
        prompt=args.prompt,
        device=(args.device or None),
        fused_bits=args.fused_bits,
        target_bits=args.target_bits,
        max_layers=args.max_layers,
        max_tokens=args.max_tokens,
        max_decode_seconds=args.max_decode_seconds,
        max_retry_seconds=args.max_retry_seconds,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repeat_window=args.repeat_window,
        no_repeat_ngram=args.no_repeat_ngram,
        speed_preset=args.speed_preset,
        lang=args.lang,
        stream=args.stream,
        unsafe_low_layers=bool(args.unsafe_low_layers),
    )

    if args.chat:
        code = run_interactive(run_args)
        raise SystemExit(code)

    if not args.prompt.strip():
        parser.error("--prompt is required unless --chat is used")

    result = run_once(run_args)

    if args.json:
        payload = {
            "ok": result["ok"],
            "returncode": result["returncode"],
            "text": result["text"],
            "metrics": result["metrics"],
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        if result["text"]:
            print(result["text"])
        else:
            print(result["stdout"])

    raise SystemExit(0 if result["ok"] else int(result["returncode"]))


if __name__ == "__main__":
    main()
