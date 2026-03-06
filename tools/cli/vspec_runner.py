from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from vspec_cli_common import auto_device, auto_fused_bits, auto_target_bits, chat_python_dir, resolve_model_for_runtime


@dataclass
class VspecRunArgs:
    model: str
    prompt: str
    device: str | None = None
    fused_bits: int | None = None
    target_bits: int | None = None
    max_layers: int = 0
    max_tokens: int = 128
    max_decode_seconds: float = -1.0
    max_retry_seconds: float = -1.0
    temperature: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.15
    repeat_window: int = 64
    no_repeat_ngram: int = 3
    speed_preset: str = "fast"
    lang: str = "auto"
    stream: bool = False
    unsafe_low_layers: bool = False


def _build_chat_cmd(args: VspecRunArgs, interactive: bool) -> list[str]:
    model_dir = resolve_model_for_runtime(args.model)
    device = args.device or auto_device()
    fused_bits = auto_fused_bits() if args.fused_bits is None else int(args.fused_bits)
    target_bits = auto_target_bits() if args.target_bits is None else int(args.target_bits)

    script = chat_python_dir() / "vspec_chat.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--device",
        str(device),
        "--fused-bits",
        str(fused_bits),
        "--target-bits",
        str(target_bits),
        "--max-layers",
        str(int(args.max_layers)),
        "--max-tokens",
        str(int(args.max_tokens)),
        "--max-decode-seconds",
        str(float(args.max_decode_seconds)),
        "--max-retry-seconds",
        str(float(args.max_retry_seconds)),
        "--temperature",
        str(float(args.temperature)),
        "--top-k",
        str(int(args.top_k)),
        "--repetition-penalty",
        str(float(args.repetition_penalty)),
        "--repeat-window",
        str(int(args.repeat_window)),
        "--no-repeat-ngram",
        str(int(args.no_repeat_ngram)),
        "--speed-preset",
        str(args.speed_preset),
        "--lang",
        str(args.lang),
        "--no-progress",
    ]

    if args.unsafe_low_layers:
        cmd.append("--unsafe-low-layers")

    if interactive:
        cmd.append("--interactive")
        cmd.append("--no-stream")
    else:
        cmd.extend(["--prompt", args.prompt])
        if not args.stream:
            cmd.append("--no-stream")

    return cmd


def run_interactive(args: VspecRunArgs) -> int:
    cmd = _build_chat_cmd(args, interactive=True)
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def run_once(args: VspecRunArgs) -> dict:
    cmd = _build_chat_cmd(args, interactive=False)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    text = ""
    marker = "[vspec-chat] output:"
    idx = stdout.find(marker)
    if idx >= 0:
        tail = stdout[idx + len(marker) :]
        lines = []
        for line in tail.splitlines():
            if line.startswith("[vspec-chat]"):
                break
            lines.append(line)
        text = "\n".join(lines).strip()

    metrics: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("[vspec-chat]") and "=" in line:
            raw = line[len("[vspec-chat]") :].strip()
            key, value = raw.split("=", 1)
            metrics[key.strip()] = value.strip()

    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "text": text,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": metrics,
        "cmd": cmd,
    }


def run_once_json(args: VspecRunArgs) -> str:
    result = run_once(args)
    payload = {
        "ok": result["ok"],
        "returncode": result["returncode"],
        "text": result["text"],
        "metrics": result["metrics"],
    }
    return json.dumps(payload, ensure_ascii=False)
