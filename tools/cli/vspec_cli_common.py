from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

VPEC_MANIFEST_NAME = "vspec_manifest.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def chat_python_dir() -> Path:
    return repo_root() / "Vspec-chat" / "python"


def ensure_chat_import_path() -> None:
    p = str(chat_python_dir())
    if p not in sys.path:
        sys.path.insert(0, p)


def detect_cuda_available() -> bool:
    try:
        ensure_chat_import_path()
        from vspec_cuda_bridge import rmsnorm_f32_available

        return bool(rmsnorm_f32_available())
    except Exception:
        return False


def auto_device() -> str:
    env = os.getenv("VSPEC_DEVICE", "").strip().lower()
    if env in {"cpu", "cuda", "cuda-native", "torch-cuda"}:
        return env
    return "cuda" if detect_cuda_available() else "cpu"


def auto_fused_bits() -> int:
    env = os.getenv("VSPEC_FUSED_BITS", "").strip()
    if env in {"0", "3", "4"}:
        return int(env)
    return 4 if detect_cuda_available() else 0


def auto_target_bits() -> int:
    env = os.getenv("VSPEC_TARGET_BITS", "").strip()
    if env in {"0", "2", "3", "4"}:
        return int(env)
    return 3


def quant_label_to_bits(label: str) -> int:
    normalized = (label or "q3").strip().lower()
    table = {
        "q2": 2,
        "q3": 3,
        "q4": 4,
        "int2": 2,
        "int3": 3,
        "int4": 4,
        "fp16": 0,
        "bf16": 0,
        "none": 0,
    }
    return int(table.get(normalized, 3))


def _select_snapshot_dir(root: Path) -> Path:
    snapshot_root = root / "snapshots"
    if not snapshot_root.exists():
        return root
    candidates = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not candidates:
        return root
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_model_root(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Model path not found: {input_path}")
    if input_path.is_file() and input_path.suffix.lower() == ".safetensors":
        return input_path.parent
    if input_path.is_file():
        return input_path.parent
    return _select_snapshot_dir(input_path)


def collect_model_files(model_root: Path) -> list[Path]:
    include_names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
    }

    files: list[Path] = []
    for name in include_names:
        p = model_root / name
        if p.exists() and p.is_file():
            files.append(p)

    for p in sorted(model_root.glob("*.safetensors")):
        files.append(p)
    for p in sorted(model_root.glob("*.safetensors.index.json")):
        files.append(p)

    if not files:
        raise RuntimeError(f"No model artifacts found under {model_root}")
    return files


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def create_vspec_package(
    input_model_path: Path,
    output_path: Path,
    model_name: str,
    quant_label: str,
    source_hint: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_root = find_model_root(input_model_path)
    model_files = collect_model_files(model_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quant_bits = quant_label_to_bits(quant_label)
    file_entries: list[dict[str, Any]] = []

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in model_files:
            arc_name = f"model/{f.name}"
            zf.write(f, arc_name)
            file_entries.append(
                {
                    "name": f.name,
                    "arcname": arc_name,
                    "size": int(f.stat().st_size),
                    "sha1": _sha1_file(f),
                }
            )

        manifest = {
            "format": "vspec",
            "version": "1.0",
            "model_name": model_name,
            "source": source_hint,
            "quant": {"label": quant_label, "target_bits": quant_bits},
            "model_dir_in_package": "model",
            "files": file_entries,
            "runtime_defaults": {
                "device": "auto",
                "fused_bits": auto_fused_bits(),
                "target_bits": max(0, min(4, quant_bits if quant_bits in {2, 3, 4} else auto_target_bits())),
            },
            "manual_override": {
                "supported_cli": [
                    "--device",
                    "--fused-bits",
                    "--target-bits",
                    "--max-layers",
                    "--max-tokens",
                    "--temperature",
                    "--top-k",
                    "--repetition-penalty",
                    "--repeat-window",
                    "--no-repeat-ngram",
                    "--speed-preset",
                    "--lang",
                ],
                "supported_env": [
                    "VSPEC_DEVICE",
                    "VSPEC_FUSED_BITS",
                    "VSPEC_TARGET_BITS",
                    "VSPEC_FORCE_TENSORCORE_4BIT",
                    "VSPEC_PRECISION_DOWNGRADE_TRIGGER",
                    "VSPEC_CACHE_COMPRESSION_TRIGGER",
                    "VSPEC_PER_MODEL_ADAPTIVE_BIT_CAP",
                ],
            },
        }
        if extra_metadata:
            manifest.update(extra_metadata)
        zf.writestr(VPEC_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))

    return manifest


def read_vspec_manifest(vspec_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(vspec_path, "r") as zf:
        try:
            raw = zf.read(VPEC_MANIFEST_NAME)
        except KeyError as exc:
            raise RuntimeError(f"Missing {VPEC_MANIFEST_NAME} in {vspec_path}") from exc
    return json.loads(raw.decode("utf-8"))


def extraction_cache_root() -> Path:
    root = os.getenv("VSPEC_PACKAGE_CACHE_DIR", "").strip()
    if root:
        p = Path(root)
    else:
        p = repo_root() / "logs" / "vspec_package_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _artifact_key(vspec_path: Path) -> str:
    st = vspec_path.stat()
    key = f"{vspec_path.resolve()}::{st.st_size}::{int(st.st_mtime)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def extract_vspec_to_cache(vspec_path: Path) -> Path:
    cache_root = extraction_cache_root()
    key = _artifact_key(vspec_path)
    dst = cache_root / key
    done = dst / ".ready"
    model_dir = dst / "model"

    if done.exists() and model_dir.exists():
        return model_dir

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(vspec_path, "r") as zf:
        zf.extractall(dst)

    done.write_text("ok", encoding="utf-8")
    return model_dir


def resolve_model_for_runtime(model_arg: str) -> Path:
    p = Path(model_arg)
    if p.exists() and p.is_file() and p.suffix.lower() == ".vspec":
        return extract_vspec_to_cache(p)
    if p.exists():
        return p
    raise FileNotFoundError(f"Model path not found: {model_arg}")


def maybe_download_hf_repo(repo_id: str, local_dir: Path | None = None) -> Path:
    local = local_dir or (repo_root() / "logs" / "hf_models")
    local.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to convert from --hf repo id. Install with: pip install huggingface_hub"
        ) from exc

    snapshot_path = snapshot_download(repo_id=repo_id, local_dir=str(local / repo_id.replace("/", "__")), local_dir_use_symlinks=False)
    return Path(snapshot_path)


def temp_file_path(prefix: str, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(name)
