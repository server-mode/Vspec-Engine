from dataclasses import dataclass
from typing import Optional

from runtime_compat import resolve_runtime_target


def _first_int(value) -> Optional[int]:
    if isinstance(value, list) and value:
        return int(value[0])
    if isinstance(value, int):
        return int(value)
    return None


@dataclass
class ModelAdapter:
    name: str
    model_type: str
    vocab_size: Optional[int]
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    required_tensors: list[str]


def _make_adapter(model_type: str, config: dict) -> ModelAdapter:
    vocab = config.get("vocab_size")
    bos = _first_int(config.get("bos_token_id"))
    eos = _first_int(config.get("eos_token_id"))
    return ModelAdapter(
        name="generic",
        model_type=model_type or "generic",
        vocab_size=int(vocab) if vocab else None,
        bos_token_id=bos,
        eos_token_id=eos,
        required_tensors=[],
    )


def select_adapter(config: dict, tensor_names: list[str]) -> ModelAdapter:
    target = resolve_runtime_target(config, tensor_names)
    adapter_cfg = dict(config)
    adapter_cfg["model_type"] = target.normalized_model_type
    return _make_adapter(target.normalized_model_type, adapter_cfg)
