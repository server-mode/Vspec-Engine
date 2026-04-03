from __future__ import annotations

import argparse
import json
import os

from vspec_runner import VspecRunArgs, run_interactive, run_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vspec model quickly from .vspec or model directory")
    parser.add_argument("model", nargs="?", default="", help="Path to .vspec package or model directory")
    parser.add_argument("-m", "--model", dest="model_opt", default="", help="Path to .vspec package or model directory")
    parser.add_argument("-p", "--prompt", default="", help="Prompt text")
    parser.add_argument("--prompts-file", default="", help="Text file with one prompt per line for continuous-batch generation")
    parser.add_argument("--batch-output-file", default="", help="Optional file to save batched outputs")
    parser.add_argument("--chat", action="store_true", help="Interactive terminal chat mode")

    parser.add_argument("--device", default="", choices=["", "cpu", "cuda", "cuda-native", "torch-cuda"])
    parser.add_argument("--fused-bits", type=int, choices=[0, 3, 4, 16], default=None)
    parser.add_argument("--target-bits", type=int, choices=[0, 2, 3, 4, 8, 16], default=None)
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
    parser.add_argument("--strict-native", action="store_true", help="Require native-real backend only (no Python compute fallback)")
    parser.add_argument("--enable-anf", action="store_true", help="Enable ANF telemetry and routing")
    parser.add_argument("--anf-mode", default="shadow", choices=["off", "shadow", "active"], help="ANF routing mode when enabled")
    parser.add_argument("--native-full-transformer", action="store_true", help="Enable full transformer native-forward scoring path in C")
    parser.add_argument("--native-full-layer-limit", type=int, default=0, help="0=all detected native layers, >0 limits native full-forward layers")
    parser.add_argument("--native-full-context-limit", type=int, default=0, help="0=full context, >0 limits context tokens used by native full-forward")
    parser.add_argument("--native-c-logits-provider", action="store_true", help="Use native C top-k logits provider during decode")
    parser.add_argument("--native-c-logits-topk", type=int, default=64, help="Top-k candidate count from native C logits provider")
    parser.add_argument("--native-c-strict", action="store_true", help="Require native C logits provider during decode and fail on fallback")

    args = parser.parse_args()

    if args.strict_native:
        os.environ["VSPEC_FORCE_NATIVE_REAL"] = "1"
        os.environ["VSPEC_REQUIRE_NATIVE_REAL"] = "1"
        os.environ.setdefault("VSPEC_NATIVE_BACKEND", "native-real")

    model = (args.model_opt or args.model or "").strip()
    if not model:
        parser.error("model path is required. Use positional model or --model")

    run_args = VspecRunArgs(
        model=model,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        batch_output_file=args.batch_output_file,
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
        enable_anf=bool(args.enable_anf),
        anf_mode=str(args.anf_mode),
        native_full_transformer=bool(args.native_full_transformer),
        native_full_layer_limit=int(args.native_full_layer_limit),
        native_full_context_limit=int(args.native_full_context_limit),
        native_c_logits_provider=bool(args.native_c_logits_provider),
        native_c_logits_topk=int(args.native_c_logits_topk),
        native_c_strict=bool(args.native_c_strict),
    )

    if args.chat:
        if str(args.prompts_file or "").strip():
            parser.error("--prompts-file cannot be combined with --chat")
        code = run_interactive(run_args)
        raise SystemExit(code)

    if (not str(args.prompts_file or "").strip()) and (not args.prompt.strip()):
        parser.error("--prompt is required unless --chat or --prompts-file is used")

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
