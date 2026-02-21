import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    from tokenizers import Tokenizer
except Exception:  # pragma: no cover - optional dependency
    Tokenizer = None


def find_snapshot_dir(model_dir: Path) -> Path:
    snapshot_root = model_dir / "snapshots"
    if not snapshot_root.exists():
        return model_dir
    candidates = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not candidates:
        return model_dir
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_config(snapshot_dir: Path) -> dict:
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_tokenizer_config(snapshot_dir: Path) -> dict:
    tok_cfg_path = snapshot_dir / "tokenizer_config.json"
    if not tok_cfg_path.exists():
        return {}
    try:
        return json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_tokenizer(snapshot_dir: Path) -> Optional[object]:
    if Tokenizer is None:
        return None
    tok_path = snapshot_dir / "tokenizer.json"
    if tok_path.exists():
        return Tokenizer.from_file(str(tok_path))
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
    return sorted([p for p in snapshot_dir.rglob("*.safetensors") if ".no_exist" not in str(p)])


def collect_tensor_names(snapshot_dir: Path) -> list[str]:
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


def build_weight_index(snapshot_dir: Path) -> dict[str, WeightInfo]:
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
            index[name] = WeightInfo(name=name, dtype=dtype, shape=shape, path=path)
    return index


def summarize_weight_dtypes(weight_index: dict[str, WeightInfo]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for info in weight_index.values():
        key = str(info.dtype).lower()
        stats[key] = stats.get(key, 0) + 1
    return stats
