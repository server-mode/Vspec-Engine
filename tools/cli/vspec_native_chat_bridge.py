#!/usr/bin/env python3
"""Thin Python bridge: menu/tokenizer only, inference in native C executable."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for tokenizer loading") from exc

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=True)
    if tok is None:
        raise RuntimeError("failed to load tokenizer from model directory")
    return tok


def _find_native_decoder(root: Path) -> Path:
    candidates = [
        root / "build" / "Release" / "vspec_native_real_decode.exe",
        root / "build" / "Debug" / "vspec_native_real_decode.exe",
        root / "build" / "vspec_native_real_decode.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("native decoder binary not found. Build target: vspec_native_real_decode")


def _parse_generated_ids(stdout: str) -> List[int]:
    marker = "[native-real-decode] generated_ids="
    ids_line = ""
    for line in stdout.splitlines():
        if line.startswith(marker):
            ids_line = line[len(marker) :].strip()
    if not ids_line:
        raise RuntimeError("native decoder did not return generated_ids")
    out: List[int] = []
    for part in ids_line.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _run_decode(
    decoder_exe: Path,
    model_dir: Path,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_id: int,
) -> List[int]:
    csv_ids = ",".join(str(x) for x in prompt_ids)
    cmd = [
        str(decoder_exe),
        str(model_dir),
        csv_ids,
        str(max_new_tokens),
        str(eos_token_id),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            "native decode failed\n"
            f"returncode={cp.returncode}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )
    return _parse_generated_ids(cp.stdout)


def _build_prompt(tokenizer, messages: List[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback text format for tokenizers without chat template.
    parts = []
    for m in messages:
        parts.append(f"{m['role']}: {m['content']}")
    parts.append("assistant:")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Native C chat bridge")
    ap.add_argument("--model-dir", required=True, help="Hugging Face model directory")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    model_dir = Path(args.model_dir).resolve()

    tokenizer = _load_tokenizer(model_dir)
    decoder_exe = _find_native_decoder(repo_root)

    eos = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else -1

    print("[native-bridge] inference_backend=vspec_native_real_decode (C)")
    print(f"[native-bridge] model_dir={model_dir}")
    print("[native-bridge] type /exit to quit")

    messages: List[dict] = []

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break

        messages.append({"role": "user", "content": user_text})
        prompt_text = _build_prompt(tokenizer, messages)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if not prompt_ids:
            print("assistant> [error] prompt encoding produced empty token ids")
            continue

        try:
            gen_ids = _run_decode(
                decoder_exe=decoder_exe,
                model_dir=model_dir,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos,
            )
        except Exception as exc:
            print(f"assistant> [error] {exc}")
            continue

        if not gen_ids:
            print("assistant> [error] native decoder returned no tokens")
            continue

        assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not assistant_text:
            assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
        if not assistant_text:
            assistant_text = "[token_ids] " + ",".join(str(x) for x in gen_ids)

        print(f"assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
