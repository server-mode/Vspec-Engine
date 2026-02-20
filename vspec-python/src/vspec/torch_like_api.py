from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime_bridge import VspecRuntimeBridge


@dataclass
class VspecModel:
    path: str
    tensor_count: int
    _bridge: VspecRuntimeBridge

    def generate(self, prompt: str) -> str:
        return self._bridge.generate(prompt)


def load(path: str) -> VspecModel:
    bridge = VspecRuntimeBridge()
    p = Path(path)

    if p.suffix.lower() in {".vpt", ".manifest"}:
        n = bridge.load_manifest_count(str(p))
    elif p.suffix.lower() == ".safetensors":
        n = bridge.parse_safetensors_count(str(p))
    else:
        raise ValueError(f"unsupported model format: {p.suffix}")

    if n < 0:
        raise RuntimeError(f"failed to load model file: {p}")

    return VspecModel(path=str(p), tensor_count=n, _bridge=bridge)
