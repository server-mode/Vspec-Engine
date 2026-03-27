from __future__ import annotations

import json
import os
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from vspec_runner import VspecRunArgs, run_interactive
from vspec_cli_common import detect_cuda_available, repo_root, resolve_model_for_runtime


@dataclass
class ModelCandidate:
    path: Path
    kind: str


SCAN_CACHE_SCHEMA = 1


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _is_windows() -> bool:
    return os.name == "nt"


def _fixed_drive_roots() -> list[Path]:
    if not _is_windows():
        return [Path("/")]

    roots: list[Path] = []
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:/")
        if root.exists():
            roots.append(root)
    return roots


def _skip_dir(name: str) -> bool:
    n = name.lower()
    return n in {
        "$recycle.bin",
        "$windows.~bt",
        "$windows.~ws",
        "appdata",
        "bin",
        "boot",
        "dev",
        "intel",
        "lib",
        "lib64",
        "msys64",
        "obj",
        "opt",
        "proc",
        "program files",
        "program files (x86)",
        "programdata",
        "recovery",
        "run",
        "sbin",
        "site-packages",
        "swapfile.sys",
        "system volume information",
        "tmp",
        "usr",
        "var",
        "windows",
        "winsxs",
        "__pycache__",
        ".cache",
        ".git",
        ".venv",
        "node_modules",
    }


def _looks_like_model_dir(files: list[str]) -> bool:
    names = set(files)
    if "config.json" not in names:
        return False
    for name in names:
        lower = name.lower()
        if lower.endswith(".safetensors"):
            return True
        if lower.endswith(".gguf"):
            return True
        if lower == "pytorch_model.bin":
            return True
        if lower.endswith(".safetensors.index.json"):
            return True
    return False


def _scan_mode() -> str:
    mode = os.getenv("VSPEC_MODEL_SCAN_MODE", "auto").strip().lower()
    if mode in {"auto", "everything", "turbo", "full"}:
        return mode
    return "auto"


def _model_roots_turbo() -> list[Path]:
    roots: list[Path] = []

    def _push(p: Path) -> None:
        if p.exists() and p not in roots:
            roots.append(p)

    user_profile = Path(os.getenv("USERPROFILE", "")).resolve() if os.getenv("USERPROFILE") else None
    local_app_data = Path(os.getenv("LOCALAPPDATA", "")).resolve() if os.getenv("LOCALAPPDATA") else None

    if user_profile is not None:
        _push(user_profile / ".cache" / "huggingface" / "hub")
        _push(user_profile / ".cache" / "huggingface")
        _push(user_profile / "models")
        _push(user_profile / "Downloads")
        _push(user_profile / "Desktop")

    if local_app_data is not None:
        _push(local_app_data / "huggingface" / "hub")

    _push(repo_root())
    _push(repo_root() / "logs")

    custom = os.getenv("VSPEC_MODEL_SCAN_ROOTS", "").strip()
    if custom:
        for raw in custom.split(";"):
            raw = raw.strip()
            if raw:
                _push(Path(raw))

    return roots


def _cap_candidates(candidates: list[ModelCandidate], limit: int) -> list[ModelCandidate]:
    if limit <= 0:
        return candidates
    return candidates[:limit]


def _everything_cli_path() -> Path | None:
    exe = which("es.exe")
    if exe:
        return Path(exe)

    candidates = [
        Path(os.getenv("ProgramFiles", "")) / "Everything" / "es.exe",
        Path(os.getenv("ProgramFiles(x86)", "")) / "Everything" / "es.exe",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _everything_query(es_path: Path, query: str, timeout_seconds: float = 12.0) -> list[Path]:
    cmd = [str(es_path), "-n", "20000", "-sort", "path", query]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, timeout_seconds),
            check=False,
        )
    except Exception:
        return []

    if proc.returncode not in {0, 1}:
        return []

    out: list[Path] = []
    for line in (proc.stdout or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        out.append(Path(raw))
    return out


def _scan_with_everything() -> list[ModelCandidate]:
    es_path = _everything_cli_path()
    if es_path is None:
        return []

    print(f"[startup] using Everything index: {es_path}")

    found: list[ModelCandidate] = []
    seen: set[str] = set()

    for p in _everything_query(es_path, "*.vspec"):
        try:
            if p.exists() and p.is_file():
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(ModelCandidate(path=p, kind="vspec"))
        except Exception:
            continue

    weight_query = "*.safetensors|*.gguf|pytorch_model.bin|*.safetensors.index.json"
    parent_dirs: set[str] = set()
    for p in _everything_query(es_path, weight_query):
        try:
            if p.exists() and p.is_file():
                parent_dirs.add(str(p.parent.resolve()))
        except Exception:
            continue

    for raw_dir in sorted(parent_dirs):
        d = Path(raw_dir)
        try:
            if _candidate_has_runtime_artifacts(d):
                key = str(d.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(ModelCandidate(path=d, kind="model_dir"))
        except Exception:
            continue

    found.sort(key=lambda x: str(x.path).lower())
    limit = int(os.getenv("VSPEC_MODEL_SCAN_MAX_RESULTS", "600") or "600")
    return _cap_candidates(found, max(50, limit))


def _scan_models_turbo() -> list[ModelCandidate]:
    roots = _model_roots_turbo()
    print("[startup] turbo scan roots:")
    for r in roots:
        print(f"  - {r}")

    found: list[ModelCandidate] = []
    seen: set[str] = set()
    limit = int(os.getenv("VSPEC_MODEL_SCAN_MAX_RESULTS", "600") or "600")

    for root in roots:
        for p in root.rglob("*.vspec"):
            try:
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(ModelCandidate(path=p, kind="vspec"))
                    if len(found) >= limit:
                        found.sort(key=lambda x: str(x.path).lower())
                        return found
            except Exception:
                continue

    for root in roots:
        for cfg in root.rglob("config.json"):
            try:
                d = cfg.parent.resolve()
                key = str(d)
                if key in seen:
                    continue
                if _candidate_has_runtime_artifacts(d):
                    seen.add(key)
                    found.append(ModelCandidate(path=d, kind="model_dir"))
                    if len(found) >= limit:
                        found.sort(key=lambda x: str(x.path).lower())
                        return found
            except Exception:
                continue

    found.sort(key=lambda x: str(x.path).lower())
    return found


def _cache_file_path() -> Path:
    p = repo_root() / "logs" / "model_scan_cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _cache_ttl_seconds() -> int:
    raw = os.getenv("VSPEC_MODEL_SCAN_CACHE_TTL_SECONDS", "21600").strip()
    try:
        return max(60, int(raw))
    except Exception:
        return 21600


def _allow_stale_cache() -> bool:
    raw = os.getenv("VSPEC_MODEL_SCAN_ALLOW_STALE", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_scan_cache() -> tuple[list[ModelCandidate] | None, bool]:
    cache_file = _cache_file_path()
    if not cache_file.exists():
        return None, False

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None, False

    if int(payload.get("schema", 0) or 0) != SCAN_CACHE_SCHEMA:
        return None, False

    ts = float(payload.get("timestamp", 0.0) or 0.0)
    if ts <= 0.0:
        return None, False

    age = time.time() - ts
    stale = age > float(_cache_ttl_seconds())

    out: list[ModelCandidate] = []
    for item in payload.get("candidates", []) or []:
        try:
            p = Path(str(item.get("path", "")))
            kind = str(item.get("kind", "model_dir"))
            if p.exists() and kind in {"model_dir", "vspec"}:
                out.append(ModelCandidate(path=p, kind=kind))
        except Exception:
            continue
    if not out:
        return None, False

    out.sort(key=lambda x: str(x.path).lower())
    return out, stale


def _save_scan_cache(candidates: list[ModelCandidate]) -> None:
    cache_file = _cache_file_path()
    payload = {
        "schema": SCAN_CACHE_SCHEMA,
        "timestamp": time.time(),
        "candidates": [
            {"path": str(c.path), "kind": c.kind}
            for c in candidates
        ],
    }
    try:
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _scan_models_with_cache() -> list[ModelCandidate]:
    cached, stale = _load_scan_cache()
    if cached is not None and (not stale):
        print(f"[startup] model cache hit: {len(cached)} candidate(s)")
        return cached

    if cached is not None and stale and _allow_stale_cache():
        print(f"[startup] model cache stale but reused for fast startup: {len(cached)} candidate(s)")
        return cached

    mode = _scan_mode()
    if mode in {"auto", "everything"}:
        scanned = _scan_with_everything()
        if scanned:
            _save_scan_cache(scanned)
            print("[startup] model cache updated")
            return scanned
        if mode == "everything":
            print("[startup] Everything scan returned no result; fallback to turbo scan")

    if mode in {"auto", "turbo", "everything"}:
        scanned = _scan_models_turbo()
    else:
        print("[startup] full scan mode enabled")
        scanned = _scan_models_all_drives()

    _save_scan_cache(scanned)
    print("[startup] model cache updated")
    return scanned


def _candidate_has_runtime_artifacts(model_path: Path) -> bool:
    if not model_path.exists() or not model_path.is_dir():
        return False
    if not (model_path / "config.json").exists():
        return False

    for p in model_path.iterdir():
        n = p.name.lower()
        if n.endswith(".safetensors") or n.endswith(".gguf") or n == "pytorch_model.bin" or n.endswith(".safetensors.index.json"):
            return True
    return False


def _is_candidate_valid_for_backend(candidate: ModelCandidate, backend: str, cuda_available: bool) -> bool:
    if not candidate.path.exists():
        return False
    if backend == "cuda" and (not cuda_available):
        return False

    resolved = candidate.path
    if candidate.kind == "vspec":
        try:
            resolved = resolve_model_for_runtime(str(candidate.path))
        except Exception:
            return False

    return _candidate_has_runtime_artifacts(resolved)


def _choose_filter_options() -> tuple[bool, str]:
    print("\nBackend filter (show only models valid for selected backend) is OFF by default.")
    raw = input("Enable backend filter? [y/N]: ").strip().lower()
    if raw not in {"y", "yes"}:
        return False, "cpu"

    print("Choose backend for filter:")
    print("  1. CPU")
    print("  2. CUDA")
    while True:
        backend_raw = input("Backend [1/2, default 1]: ").strip()
        if backend_raw in {"", "1"}:
            return True, "cpu"
        if backend_raw == "2":
            return True, "cuda"
        print("Please enter 1 or 2.")


def _scan_models_all_drives() -> list[ModelCandidate]:
    found: list[ModelCandidate] = []
    seen: set[str] = set()

    roots = _fixed_drive_roots()
    print("[startup] scanning local drives for models...")

    for root in roots:
        print(f"[startup] scanning {root}")
        for current_root, dirs, files in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = [d for d in dirs if not _skip_dir(d)]

            current = Path(current_root)
            lower_files = [f.lower() for f in files]

            for idx, f in enumerate(files):
                if lower_files[idx].endswith(".vspec"):
                    p = current / f
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append(ModelCandidate(path=p, kind="vspec"))

            if _looks_like_model_dir(files):
                key = str(current.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(ModelCandidate(path=current, kind="model_dir"))

    found.sort(key=lambda x: str(x.path).lower())
    return found


def _choose_model(candidates: list[ModelCandidate]) -> ModelCandidate | None:
    if not candidates:
        print("[startup] no model found on this machine.")
        return None

    print("\nAvailable models:")
    for i, candidate in enumerate(candidates, start=1):
        print(f"  {i:>3}. [{candidate.kind}] {candidate.path}")

    while True:
        raw = input("\nSelect model index (0 to exit): ").strip()
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(raw)
        if idx == 0:
            return None
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print("Index out of range.")


def _apply_backend_filter(candidates: list[ModelCandidate], enabled: bool, backend: str) -> list[ModelCandidate]:
    if not enabled:
        print("[startup] backend filter: OFF")
        return candidates

    cuda_available = bool(detect_cuda_available())
    print(f"[startup] backend filter: ON ({backend.upper()})")
    if backend == "cuda" and (not cuda_available):
        print("[startup] CUDA backend not available on this machine.")
        return []

    filtered: list[ModelCandidate] = []
    for candidate in candidates:
        if _is_candidate_valid_for_backend(candidate, backend, cuda_available):
            filtered.append(candidate)

    print(f"[startup] backend filter result: {len(filtered)}/{len(candidates)} candidate(s)")
    return filtered


def _choose_function() -> str:
    print("Select function:")
    print("  1. Chat")
    print("  2. Benchmark")
    print("  0. Exit")

    while True:
        raw = input("Your choice: ").strip()
        if raw == "1":
            return "chat"
        if raw == "2":
            return "benchmark"
        if raw == "0":
            return "exit"
        print("Please select 1, 2 or 0.")


def _run_benchmark() -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "tools" / "cli" / "vspec_benchmark.py"
    cmd = [sys.executable, str(script)]
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _run_chat(selected_model: ModelCandidate) -> int:
    os.environ["VSPEC_TARGET_BITS"] = "3"
    os.environ["VSPEC_FUSED_BITS"] = "4"
    os.environ["VSPEC_3BIT_RUNTIME_MODULE"] = "1"
    os.environ["VSPEC_3BIT_ATTN_QK_BITS"] = "4"
    os.environ["VSPEC_3BIT_ATTN_PROJ_BITS"] = "3"
    os.environ["VSPEC_3BIT_MLP_BITS"] = "3"
    os.environ["VSPEC_3BIT_LM_HEAD_BITS"] = "4"
    os.environ["VSPEC_PRECISION_DOWNGRADE_TRIGGER"] = "0.70"
    os.environ["VSPEC_CACHE_COMPRESSION_TRIGGER"] = "0.78"
    os.environ["VSPEC_PER_MODEL_ADAPTIVE_BIT_CAP"] = "3"

    print("\n[startup] Chat runtime defaults:")
    print("  - Runtime compute: INT4 (fused-bits=4)")
    print("  - Storage policy: 3-bit low VRAM mode")
    print("  - target-bits=3")
    print("  - VSPEC_3BIT_RUNTIME_MODULE=1")
    print("  - VSPEC_3BIT_ATTN_QK_BITS=4")
    print("  - VSPEC_3BIT_ATTN_PROJ_BITS=3")
    print("  - VSPEC_3BIT_MLP_BITS=3")
    print("  - VSPEC_3BIT_LM_HEAD_BITS=4")
    print("  - VSPEC_PRECISION_DOWNGRADE_TRIGGER=0.70")
    print("  - VSPEC_CACHE_COMPRESSION_TRIGGER=0.78")
    print("  - VSPEC_PER_MODEL_ADAPTIVE_BIT_CAP=3")
    print("\n[startup] launching interactive chat...\n")

    native_repl_enabled = os.getenv("VSPEC_NATIVE_CHAT_REPL", "1").strip().lower() in {"1", "true", "yes", "on"}
    if native_repl_enabled:
        root = Path(__file__).resolve().parents[2]
        native_exe_candidates = [
            root / "build" / "Release" / "vspec_native_session_chat.exe",
            root / "build" / "Debug" / "vspec_native_session_chat.exe",
            root / "build" / "vspec_native_session_chat",
        ]
        native_exe = next((p for p in native_exe_candidates if p.exists()), None)

        resolved_model = selected_model.path
        if selected_model.kind == "vspec":
            try:
                resolved_model = Path(resolve_model_for_runtime(str(selected_model.path)))
            except Exception:
                resolved_model = selected_model.path

        native_model = None
        if resolved_model.exists() and resolved_model.is_dir():
            shards = sorted(resolved_model.glob("model-*.safetensors"))
            if shards:
                native_model = shards[0]
            else:
                safes = sorted(resolved_model.glob("*.safetensors"))
                if safes:
                    native_model = safes[0]

        if native_exe is not None and native_model is not None:
            print(f"[startup] native chat repl enabled: {native_exe.name}")
            proc = subprocess.run([str(native_exe), str(native_model), "256"])
            return int(proc.returncode)

        print("[startup] native chat repl requested but unavailable, fallback to python session.")

    run_args = VspecRunArgs(
        model=str(selected_model.path),
        prompt="",
        device=None,
        fused_bits=4,
        target_bits=3,
        max_layers=0,
        max_tokens=256,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.15,
        repeat_window=64,
        no_repeat_ngram=3,
        speed_preset="fast",
        lang="auto",
        stream=False,
        unsafe_low_layers=False,
    )
    return run_interactive(run_args)


def main() -> None:
    _clear_screen()
    print("Welcome to Vspec Engine")
    print("[startup] engine bootstrap completed successfully.")
    print("[startup] Option 1 scans all local drives for available models.")

    candidates = _scan_models_with_cache()
    print(f"[startup] model scan complete: {len(candidates)} candidate(s) found.")

    filter_enabled, filter_backend = _choose_filter_options()
    candidates = _apply_backend_filter(candidates, filter_enabled, filter_backend)

    selected = _choose_model(candidates)
    if selected is None:
        raise SystemExit(0)

    _clear_screen()
    print("Welcome to Vspec Engine")
    print(f"Selected model: {selected.path}\n")

    action = _choose_function()
    if action == "exit":
        raise SystemExit(0)
    if action == "benchmark":
        raise SystemExit(_run_benchmark())
    raise SystemExit(_run_chat(selected))


if __name__ == "__main__":
    main()
