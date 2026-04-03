from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from vspec_cli_common import auto_device, auto_fused_bits, auto_target_bits, chat_python_dir, resolve_model_for_runtime


_TOKENIZER_CACHE: dict[str, object] = {}


def _is_true_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _apply_python_quality_guards(
    decoded_pieces: list[str],
    prompt: str,
    lang_mode: str,
) -> tuple[str, dict[str, str]]:
    metrics: dict[str, str] = {
        "py_quality_guards_applied": "0",
        "py_quality_guard_blocked_pieces": "0",
    }

    # User requested quality guards remain in Python. Keep them on by default.
    require_py_guards = _is_true_env("VSPEC_REQUIRE_PY_QUALITY_GUARDS", default=True)
    if not decoded_pieces:
        metrics["py_quality_guards_applied"] = "1"
        return "", metrics

    try:
        guard_root = chat_python_dir()

        def _load_symbol(module_name: str, file_name: str, symbol_name: str):
            module_path = guard_root / file_name
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load module spec: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            sym = getattr(module, symbol_name, None)
            if sym is None:
                raise RuntimeError(f"missing symbol {symbol_name} in {module_path}")
            return sym

        LanguageStabilityGuard = _load_symbol(
            module_name="vspec_lang_stability_guard",
            file_name="language_stability_guard.py",
            symbol_name="LanguageStabilityGuard",
        )
        LanguageStructureIntegrityManager = _load_symbol(
            module_name="vspec_lang_structure_guard",
            file_name="language_structure_guard.py",
            symbol_name="LanguageStructureIntegrityManager",
        )
        MeaningfulOutputGuard = _load_symbol(
            module_name="vspec_meaningful_output_guard",
            file_name="meaningful_output_guard.py",
            symbol_name="MeaningfulOutputGuard",
        )

        disable_lang_guard = _is_true_env("VSPEC_DISABLE_LANGUAGE_GUARD", default=False)
        disable_structure_guard = _is_true_env("VSPEC_DISABLE_STRUCTURE_GUARD", default=False)
        lang_strictness = float(os.getenv("VSPEC_LANGUAGE_GUARD_STRICTNESS", "0.72") or "0.72")
        structure_strictness = float(os.getenv("VSPEC_STRUCTURE_GUARD_STRICTNESS", "0.72") or "0.72")

        lang_guard = None if disable_lang_guard else LanguageStabilityGuard(
            prompt=prompt or "",
            lang_mode=(lang_mode or "auto"),
            strictness=lang_strictness,
            prioritize_english=True,
        )
        structure_guard = None if disable_structure_guard else LanguageStructureIntegrityManager(
            prompt=prompt or "",
            strictness=structure_strictness,
        )
        meaningful_guard = MeaningfulOutputGuard(lang_mode=(lang_mode or "auto"))

        accepted: list[str] = []
        blocked = 0
        for piece in decoded_pieces:
            part = str(piece or "")
            allow = True
            if lang_guard is not None and not lang_guard.allow_text(part):
                allow = False
            if allow and structure_guard is not None and not structure_guard.allow_text(part):
                allow = False
            if allow and not meaningful_guard.allow_text(part):
                allow = False

            if allow:
                accepted.append(part)
                if structure_guard is not None:
                    structure_guard.observe_text(part)
                meaningful_guard.observe_text(part)
            else:
                blocked += 1

        metrics["py_quality_guards_applied"] = "1"
        metrics["py_quality_guard_blocked_pieces"] = str(int(blocked))
        if blocked > 0:
            metrics["py_quality_guard_status"] = "filtered"
        else:
            metrics["py_quality_guard_status"] = "pass"
        return "".join(accepted).strip(), metrics
    except Exception as exc:
        if require_py_guards:
            raise RuntimeError(f"python quality guards unavailable: {exc}") from exc
        return "".join(decoded_pieces).strip(), metrics


@dataclass
class VspecRunArgs:
    model: str
    prompt: str
    prompts_file: str = ""
    batch_output_file: str = ""
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
    enable_anf: bool = False
    anf_mode: str = "shadow"
    native_full_transformer: bool = False
    native_full_layer_limit: int = 0
    native_full_context_limit: int = 0
    native_c_logits_provider: bool = False
    native_c_logits_topk: int = 64
    native_c_strict: bool = False


def _build_chat_cmd(args: VspecRunArgs, interactive: bool) -> list[str]:
    model_dir = resolve_model_for_runtime(args.model)
    device = args.device or auto_device()
    fused_bits = auto_fused_bits() if args.fused_bits is None else int(args.fused_bits)
    target_bits = auto_target_bits() if args.target_bits is None else int(args.target_bits)

    script_name = "vspec_chat_session.py" if interactive else "vspec_chat.py"
    script = chat_python_dir() / script_name
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
    ]

    # Keep progress visible for interactive session bootstrap; forcing --no-progress
    # makes model load look like a hang at "launching interactive chat...".
    if not interactive:
        cmd.append("--no-progress")

    if args.unsafe_low_layers:
        cmd.append("--unsafe-low-layers")

    if args.enable_anf:
        cmd.append("--enable-anf")
        cmd.extend(["--anf-mode", str(args.anf_mode)])

    if args.native_full_transformer:
        cmd.append("--native-full-transformer")
    if int(args.native_full_layer_limit) > 0:
        cmd.extend(["--native-full-layer-limit", str(int(args.native_full_layer_limit))])
    if int(args.native_full_context_limit) > 0:
        cmd.extend(["--native-full-context-limit", str(int(args.native_full_context_limit))])

    if args.native_c_logits_provider:
        cmd.append("--native-c-logits-provider")
        if int(args.native_c_logits_topk) > 0:
            cmd.extend(["--native-c-logits-topk", str(int(args.native_c_logits_topk))])
    if args.native_c_strict:
        cmd.append("--native-c-strict")

    if interactive:
        cmd.append("--no-stream")
        if _is_true_env("VSPEC_CHAT_CLAUDE_STYLE", default=False) and (not _is_true_env("VSPEC_CHAT_SHOW_PROGRESS", default=False)):
            cmd.append("--no-progress")
    else:
        if str(args.prompts_file or "").strip():
            cmd.extend(["--prompts-file", str(args.prompts_file)])
            if str(args.batch_output_file or "").strip():
                cmd.extend(["--batch-output-file", str(args.batch_output_file)])
        else:
            cmd.extend(["--prompt", args.prompt])
        if not args.stream:
            cmd.append("--no-stream")

    return cmd


def _native_real_enabled() -> bool:
    mode = str(os.getenv("VSPEC_CHAT_MODE", "")).strip().lower()
    backend = str(os.getenv("VSPEC_NATIVE_BACKEND", "")).strip().lower()
    force = str(os.getenv("VSPEC_FORCE_NATIVE_REAL", "")).strip().lower()
    if force in {"1", "true", "yes", "on"}:
        return True
    if backend in {"native-real", "real", "c"}:
        return True
    return mode == "native"


def _native_real_required() -> bool:
    strict = str(os.getenv("VSPEC_REQUIRE_NATIVE_REAL", "")).strip().lower()
    return strict in {"1", "true", "yes", "on"}


def _resolve_native_real_decode_exe() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "Release" / "vspec_native_real_decode.exe",
        root / "build" / "Debug" / "vspec_native_real_decode.exe",
        root / "build" / "vspec_native_real_decode",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _get_tokenizer(model_dir: Path):
    key = str(model_dir.resolve())
    cached = _TOKENIZER_CACHE.get(key)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=True)
    _TOKENIZER_CACHE[key] = tok
    return tok


def _run_native_real_once(args: VspecRunArgs, model_dir: Path | None = None, tokenizer=None) -> dict:
    model_dir = model_dir or resolve_model_for_runtime(args.model)
    exe = _resolve_native_real_decode_exe()
    if exe is None:
        return {
            "ok": False,
            "returncode": 2,
            "text": "",
            "stdout": "",
            "stderr": "native backend unavailable: build/Release/vspec_native_real_decode.exe not found",
            "metrics": {},
            "cmd": [],
        }

    tokenizer = tokenizer or _get_tokenizer(model_dir)
    prompt_ids = tokenizer.encode(args.prompt or "", add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [int(getattr(tokenizer, "bos_token_id", 1) or 1)]
    token_csv = ",".join(str(int(t)) for t in prompt_ids)
    eos_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)

    cmd = [
        str(exe),
        str(model_dir),
        token_csv,
        str(max(1, int(args.max_tokens))),
        str(max(0, eos_token_id)),
    ]
    timeout_sec_raw = str(os.getenv("VSPEC_NATIVE_REAL_TIMEOUT_SEC", "90")).strip()
    try:
        timeout_sec = max(5.0, float(timeout_sec_raw))
    except Exception:
        timeout_sec = 90.0
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "returncode": 124,
            "text": "",
            "stdout": (exc.stdout or ""),
            "stderr": f"native-real decode timeout after {timeout_sec:.1f}s",
            "metrics": {
                "backend": "native-real-c-decode",
                "engine_cmd": " ".join(cmd),
                "elapsed_ms": f"{elapsed_ms:.1f}",
                "timeout_sec": f"{timeout_sec:.1f}",
            },
            "cmd": cmd,
        }

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    metrics: dict[str, str] = {}
    generated_ids: list[int] = []

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("[native-real-decode] generated_count="):
            metrics["generated_count"] = line.split("=", 1)[1].strip()
        elif line.startswith("[native-real-decode] mode="):
            metrics["native_mode"] = line.split("=", 1)[1].strip()
        elif line.startswith("[native-real-decode] generated_ids="):
            raw = line.split("=", 1)[1].strip()
            if raw:
                try:
                    generated_ids = [int(x) for x in raw.split(",") if x.strip()]
                except Exception:
                    generated_ids = []

    text = ""
    decoded_pieces: list[str] = []
    if generated_ids:
        decoded_pieces = [tokenizer.decode([int(tid)]) for tid in generated_ids]
        try:
            text, qg_metrics = _apply_python_quality_guards(
                decoded_pieces=decoded_pieces,
                prompt=str(args.prompt or ""),
                lang_mode=str(args.lang or "auto"),
            )
            metrics.update(qg_metrics)
        except Exception as exc:
            if _is_true_env("VSPEC_REQUIRE_PY_QUALITY_GUARDS", default=True):
                return {
                    "ok": False,
                    "returncode": 3,
                    "text": "",
                    "stdout": stdout,
                    "stderr": f"python quality guards failed: {exc}",
                    "metrics": {
                        "backend": "native-real-c-decode",
                        "py_quality_guards_applied": "0",
                        "py_quality_guard_status": "error",
                        "py_quality_guard_error": str(exc),
                        "engine_cmd": " ".join(cmd),
                    },
                    "cmd": cmd,
                }
            text = "".join(decoded_pieces).strip()
            metrics["py_quality_guards_applied"] = "0"
            metrics["py_quality_guard_status"] = "error"
            metrics["py_quality_guard_error"] = str(exc)

    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "text": text,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": {
            **metrics,
            "backend": "native-real-c-decode",
            "engine_cmd": " ".join(cmd),
            "elapsed_ms": f"{(time.perf_counter() - t0) * 1000.0:.1f}",
        },
        "cmd": cmd,
    }


def run_interactive(args: VspecRunArgs) -> int:
    if _native_real_required() and not _native_real_enabled():
        print("[vspec-run] strict-native: VSPEC_REQUIRE_NATIVE_REAL=1 but native-real backend is not enabled")
        return 2

    if _native_real_enabled():
        exe = _resolve_native_real_decode_exe()
        if exe is None:
            print("[vspec-run] strict/native-real requested but vspec_native_real_decode executable was not found")
            return 2
        model_dir = resolve_model_for_runtime(args.model)
        tokenizer = _get_tokenizer(model_dir)
        print("[vspec-run] native_real=on (C decode loop)")
        print(f"[vspec-run] engine={exe}")
        cap_raw = str(os.getenv("VSPEC_NATIVE_INTERACTIVE_MAX_TOKENS", "0")).strip()
        try:
            interactive_cap = int(cap_raw)
        except Exception:
            interactive_cap = 0
        if interactive_cap > 0:
            print(f"[vspec-run] interactive_max_tokens_cap={interactive_cap}")
        else:
            print("[vspec-run] interactive_max_tokens_cap=off")
        print("[vspec-run] type /exit to quit")
        while True:
            try:
                prompt = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[vspec-run] bye")
                return 0
            if not prompt:
                continue
            if prompt.lower() in {"/exit", "exit", "quit"}:
                print("[vspec-run] bye")
                return 0
            turn_args = VspecRunArgs(**{**args.__dict__, "prompt": prompt})
            if interactive_cap > 0 and int(turn_args.max_tokens) > interactive_cap:
                turn_args.max_tokens = int(interactive_cap)
            print(f"[vspec-run] generating... max_tokens={int(turn_args.max_tokens)}")
            result = _run_native_real_once(turn_args, model_dir=model_dir, tokenizer=tokenizer)
            if result["ok"] and str(result.get("text", "")).strip():
                print("assistant>", result["text"])
            elif result["ok"]:
                print("assistant> [native-real] generated empty output")
            else:
                err = str(result.get("stderr", "") or result.get("stdout", "") or "native-real decode failed")
                print("assistant> [native-real-error]", err[:240])
            ems = str((result.get("metrics", {}) or {}).get("elapsed_ms", "")).strip()
            if ems:
                print(f"[vspec-run] elapsed_ms={ems}")
    cmd = _build_chat_cmd(args, interactive=True)
    timeout_raw = str(os.getenv("VSPEC_INTERACTIVE_START_TIMEOUT_SEC", "300")).strip()
    try:
        timeout_sec = max(30.0, float(timeout_raw))
    except Exception:
        timeout_sec = 300.0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.run(cmd, timeout=timeout_sec, env=env)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        print(f"[vspec-run] interactive startup timed out after {timeout_sec:.1f}s")
        print("[vspec-run] hint: verify model snapshot, CUDA availability, and VSPEC runtime bridge health")
        return 124


def run_once(args: VspecRunArgs) -> dict:
    if _native_real_required() and not _native_real_enabled():
        return {
            "ok": False,
            "returncode": 2,
            "text": "",
            "stdout": "",
            "stderr": "strict-native enabled but native-real backend is not enabled",
            "metrics": {"backend": "strict-native-error"},
            "cmd": [],
        }

    if _native_real_enabled():
        return _run_native_real_once(args)

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
