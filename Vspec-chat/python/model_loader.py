import json
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from gguf_support import get_gguf_archive, is_gguf_path
from runtime_core_bridge import canonical_weight_name

try:
    from tokenizers import Tokenizer
except Exception:  # pragma: no cover - optional dependency
    Tokenizer = None

try:
    _transformers_mod = importlib.import_module("transformers")
    AutoTokenizer = getattr(_transformers_mod, "AutoTokenizer", None)
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None


class _EncodeResult:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _TransformersTokenizerAdapter:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str):
        ids = self._tokenizer.encode(str(text), add_special_tokens=False)
        return _EncodeResult([int(v) for v in ids])

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(
            [int(v) for v in ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def decode_batch(self, ids_batch: list[list[int]]) -> list[str]:
        out: list[str] = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def get_vocab_size(self) -> int:
        return int(len(self._tokenizer))


def find_snapshot_dir(model_dir: Path) -> Path:
    if is_gguf_path(model_dir):
        return model_dir
    snapshot_root = model_dir / "snapshots"
    if not snapshot_root.exists():
        return model_dir
    candidates = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not candidates:
        return model_dir
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_config(snapshot_dir: Path) -> dict:
    if is_gguf_path(snapshot_dir):
        return get_gguf_archive(snapshot_dir).config()
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_tokenizer_config(snapshot_dir: Path) -> dict:
    if is_gguf_path(snapshot_dir):
        return get_gguf_archive(snapshot_dir).tokenizer_config()
    tok_cfg_path = snapshot_dir / "tokenizer_config.json"
    if not tok_cfg_path.exists():
        return {}
    try:
        return json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_tokenizer(snapshot_dir: Path) -> Optional[object]:
    if is_gguf_path(snapshot_dir):
        return get_gguf_archive(snapshot_dir).load_tokenizer()
    if Tokenizer is not None:
        tok_path = snapshot_dir / "tokenizer.json"
        if tok_path.exists():
            return Tokenizer.from_file(str(tok_path))

    if AutoTokenizer is None:
        return None

    try:
        hf_tok = AutoTokenizer.from_pretrained(str(snapshot_dir), use_fast=True, trust_remote_code=True)
        return _TransformersTokenizerAdapter(hf_tok)
    except Exception:
        return None


def _parse_safetensors_header(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 8:
        return {}
    hlen = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 : 8 + hlen].decode("utf-8"))


def parse_safetensors_header_names(path: Path) -> list[str]:
    header = _parse_safetensors_header(path)
    return [k for k in header.keys() if k != "__metadata__"]


def pick_safetensors(snapshot_dir: Path) -> list[Path]:
    if is_gguf_path(snapshot_dir):
        return []
    return sorted([p for p in snapshot_dir.rglob("*.safetensors") if ".no_exist" not in str(p)])


def collect_tensor_names(snapshot_dir: Path) -> list[str]:
    if is_gguf_path(snapshot_dir):
        return get_gguf_archive(snapshot_dir).list_tensor_names()
    names: list[str] = []
    seen = set()
    for path in pick_safetensors(snapshot_dir):
        for name in parse_safetensors_header_names(path):
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


@dataclass
class WeightInfo:
    name: str
    dtype: str
    shape: list[int]
    path: Path
    source_format: str = "safetensors"


def build_weight_index(snapshot_dir: Path) -> dict[str, WeightInfo]:
    if is_gguf_path(snapshot_dir):
        archive = get_gguf_archive(snapshot_dir)
        index: dict[str, WeightInfo] = {}
        for tensor in archive.reader.tensors:
            shape = [int(x) for x in reversed(tensor.shape.tolist())]
            weight = WeightInfo(
                name=tensor.name,
                dtype=str(tensor.tensor_type.name).lower(),
                shape=shape,
                path=snapshot_dir,
                source_format="gguf",
            )
            index[tensor.name] = weight
            mapped = archive.map_tensor_name(tensor.name)
            if mapped and mapped not in index:
                index[mapped] = WeightInfo(
                    name=tensor.name,
                    dtype=str(tensor.tensor_type.name).lower(),
                    shape=shape,
                    path=snapshot_dir,
                    source_format="gguf",
                )
        for mapped_name, raw_name in archive.tensor_name_map.items():
            if mapped_name in index:
                continue
            tensor = archive.tensor_by_name.get(raw_name)
            if tensor is None:
                continue
            index[mapped_name] = WeightInfo(
                name=raw_name,
                dtype=str(tensor.tensor_type.name).lower(),
                shape=[int(x) for x in reversed(tensor.shape.tolist())],
                path=snapshot_dir,
                source_format="gguf",
            )
        return index

    index: dict[str, WeightInfo] = {}
    for path in pick_safetensors(snapshot_dir):
        header = _parse_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in index:
                continue
            dtype = str(info.get("dtype", ""))
            shape = [int(x) for x in info.get("shape", [])]
            weight = WeightInfo(name=name, dtype=dtype, shape=shape, path=path)
            index[name] = weight
            mapped = canonical_weight_name(name)
            if mapped and mapped not in index:
                index[mapped] = WeightInfo(name=name, dtype=dtype, shape=shape, path=path)
    return index


def summarize_weight_dtypes(weight_index: dict[str, WeightInfo]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for info in weight_index.values():
        key = str(info.dtype).lower()
        stats[key] = stats.get(key, 0) + 1
    return stats
