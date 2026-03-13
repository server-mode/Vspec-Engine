from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tokenizers import Tokenizer
except Exception:  # pragma: no cover
    Tokenizer = None


_GGUF_IMPORT_DONE = False


def _ensure_vendored_gguf() -> None:
    global _GGUF_IMPORT_DONE
    if _GGUF_IMPORT_DONE:
        return
    root = Path(__file__).resolve().parents[3]
    vendored = root / "research" / "llama.cpp-b8234" / "llama.cpp-b8234" / "gguf-py"
    if vendored.exists():
        vendored_str = str(vendored)
        if vendored_str not in sys.path:
            sys.path.insert(0, vendored_str)
    _GGUF_IMPORT_DONE = True


_ensure_vendored_gguf()

import gguf  # type: ignore  # noqa: E402


@dataclass
class GGUFTokenizerConfig:
    bos_token_id: int | None
    eos_token_id: int | None
    chat_template: str
    model: str


def is_gguf_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".gguf"


def _field_value(reader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default


class SimpleGGUFTokenizer:
    def __init__(self, tokens: list[str], bos_token_id: int | None = None, eos_token_id: int | None = None) -> None:
        self._tokens = list(tokens)
        self._bos = bos_token_id
        self._eos = eos_token_id
        self._by_token = {token: idx for idx, token in enumerate(self._tokens)}

    def encode(self, text: str):
        ids: list[int] = []
        sample = str(text or "")
        if sample in self._by_token:
            ids.append(int(self._by_token[sample]))
        else:
            for chunk in sample.split():
                if chunk in self._by_token:
                    ids.append(int(self._by_token[chunk]))
        return type("_EncodeResult", (), {"ids": ids})()

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for token_id in ids:
            idx = int(token_id)
            if 0 <= idx < len(self._tokens):
                out.append(self._tokens[idx])
        return "".join(out)

    def decode_batch(self, ids_batch: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in ids_batch]

    def get_vocab_size(self) -> int:
        return len(self._tokens)


class GGUFArchive:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.reader = gguf.GGUFReader(str(self.path))
        self.tensor_by_name = {tensor.name: tensor for tensor in self.reader.tensors}
        self.metadata = self._build_metadata()
        self.arch = str(self.metadata.get("general.architecture", "llama") or "llama")
        self.tensor_name_map = self._build_tensor_name_map()

    def _build_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for name, field in self.reader.fields.items():
            try:
                meta[name] = field.contents()
            except Exception:
                continue
        return meta

    def _build_tensor_name_map(self) -> dict[str, str]:
        mapped: dict[str, str] = {}
        for raw_name in self.tensor_by_name.keys():
            canonical = self.map_tensor_name(raw_name)
            if canonical:
                mapped[canonical] = raw_name
        if "lm_head.weight" not in mapped and "token_embd.weight" in self.tensor_by_name:
            mapped["lm_head.weight"] = "token_embd.weight"
        return mapped

    def list_tensor_names(self) -> list[str]:
        names = list(self.tensor_by_name.keys())
        for mapped in self.tensor_name_map.keys():
            if mapped not in names:
                names.append(mapped)
        return names

    def map_tensor_name(self, raw_name: str) -> str | None:
        if raw_name == "token_embd.weight":
            return "model.embed_tokens.weight"
        if raw_name == "output.weight":
            return "lm_head.weight"
        if raw_name == "output_norm.weight":
            return "model.norm.weight"

        patterns = [
            (r"^blk\.(\d+)\.attn_q\.weight$", "model.layers.{i}.self_attn.q_proj.weight"),
            (r"^blk\.(\d+)\.attn_k\.weight$", "model.layers.{i}.self_attn.k_proj.weight"),
            (r"^blk\.(\d+)\.attn_v\.weight$", "model.layers.{i}.self_attn.v_proj.weight"),
            (r"^blk\.(\d+)\.attn_output\.weight$", "model.layers.{i}.self_attn.o_proj.weight"),
            (r"^blk\.(\d+)\.attn_norm\.weight$", "model.layers.{i}.input_layernorm.weight"),
            (r"^blk\.(\d+)\.ffn_norm\.weight$", "model.layers.{i}.post_attention_layernorm.weight"),
            (r"^blk\.(\d+)\.attn_q_norm\.weight$", "model.layers.{i}.self_attn.q_norm.weight"),
            (r"^blk\.(\d+)\.attn_k_norm\.weight$", "model.layers.{i}.self_attn.k_norm.weight"),
            (r"^blk\.(\d+)\.ffn_gate\.weight$", "model.layers.{i}.mlp.gate_proj.weight"),
            (r"^blk\.(\d+)\.ffn_down\.weight$", "model.layers.{i}.mlp.down_proj.weight"),
            (r"^blk\.(\d+)\.ffn_up\.weight$", "model.layers.{i}.mlp.up_proj.weight"),
            (r"^blk\.(\d+)\.attn_q\.bias$", "model.layers.{i}.self_attn.q_proj.bias"),
            (r"^blk\.(\d+)\.attn_k\.bias$", "model.layers.{i}.self_attn.k_proj.bias"),
            (r"^blk\.(\d+)\.attn_v\.bias$", "model.layers.{i}.self_attn.v_proj.bias"),
            (r"^blk\.(\d+)\.attn_output\.bias$", "model.layers.{i}.self_attn.o_proj.bias"),
            (r"^blk\.(\d+)\.ffn_gate\.bias$", "model.layers.{i}.mlp.gate_proj.bias"),
            (r"^blk\.(\d+)\.ffn_down\.bias$", "model.layers.{i}.mlp.down_proj.bias"),
            (r"^blk\.(\d+)\.ffn_up\.bias$", "model.layers.{i}.mlp.up_proj.bias"),
        ]
        for pattern, template in patterns:
            match = re.match(pattern, raw_name)
            if match is not None:
                return template.format(i=int(match.group(1)))
        return None

    def config(self) -> dict[str, Any]:
        arch = self.arch
        tokens = _field_value(self.reader, gguf.Keys.Tokenizer.LIST, []) or []
        bos_id = _field_value(self.reader, gguf.Keys.Tokenizer.BOS_ID, None)
        eos_id = _field_value(self.reader, gguf.Keys.Tokenizer.EOS_ID, None)
        cfg = {
            "model_type": arch,
            "vocab_size": int(_field_value(self.reader, gguf.Keys.LLM.VOCAB_SIZE.format(arch=arch), len(tokens)) or len(tokens)),
            "max_position_embeddings": int(_field_value(self.reader, gguf.Keys.LLM.CONTEXT_LENGTH.format(arch=arch), 0) or 0),
            "num_hidden_layers": int(_field_value(self.reader, gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch), 0) or 0),
            "hidden_size": int(_field_value(self.reader, gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=arch), 0) or 0),
            "intermediate_size": int(_field_value(self.reader, gguf.Keys.LLM.FEED_FORWARD_LENGTH.format(arch=arch), 0) or 0),
            "num_attention_heads": int(_field_value(self.reader, gguf.Keys.Attention.HEAD_COUNT.format(arch=arch), 0) or 0),
            "num_key_value_heads": int(
                _field_value(
                    self.reader,
                    gguf.Keys.Attention.HEAD_COUNT_KV.format(arch=arch),
                    _field_value(self.reader, gguf.Keys.Attention.HEAD_COUNT.format(arch=arch), 0),
                )
                or 0
            ),
            "rms_norm_eps": float(
                _field_value(
                    self.reader,
                    gguf.Keys.Attention.LAYERNORM_RMS_EPS.format(arch=arch),
                    _field_value(self.reader, gguf.Keys.Attention.LAYERNORM_EPS.format(arch=arch), 1e-6),
                )
                or 1e-6
            ),
            "rope_theta": float(_field_value(self.reader, gguf.Keys.Rope.FREQ_BASE.format(arch=arch), 10000.0) or 10000.0),
            "bos_token_id": int(bos_id) if bos_id is not None else None,
            "eos_token_id": int(eos_id) if eos_id is not None else None,
            "tie_word_embeddings": "output.weight" not in self.tensor_by_name,
        }
        if arch == "qwen35":
            num_layers = int(cfg.get("num_hidden_layers", 0) or 0)
            ssm_group_count = int(_field_value(self.reader, f"{arch}.ssm.group_count", 0) or 0)
            ssm_time_step_rank = int(_field_value(self.reader, f"{arch}.ssm.time_step_rank", 0) or 0)
            ssm_inner_size = int(_field_value(self.reader, f"{arch}.ssm.inner_size", 0) or 0)
            layer_types: list[str] = []
            for layer_idx in range(num_layers):
                prefix = f"blk.{layer_idx}."
                if prefix + "attn_q.weight" in self.tensor_by_name:
                    layer_types.append("full_attention")
                elif prefix + "attn_qkv.weight" in self.tensor_by_name:
                    layer_types.append("linear_attention")
                else:
                    layer_types.append("unknown")
            cfg.update({
                "head_dim": int(_field_value(self.reader, f"{arch}.attention.key_length", 0) or 0),
                "rope_dimension_count": int(_field_value(self.reader, f"{arch}.rope.dimension_count", 0) or 0),
                "rope_dimension_sections": list(_field_value(self.reader, f"{arch}.rope.dimension_sections", []) or []),
                "linear_num_key_heads": ssm_group_count,
                "linear_num_value_heads": ssm_time_step_rank,
                "linear_key_head_dim": int(_field_value(self.reader, f"{arch}.ssm.state_size", 0) or 0),
                "linear_value_head_dim": (ssm_inner_size // ssm_time_step_rank) if ssm_time_step_rank > 0 else 0,
                "linear_conv_kernel_dim": int(_field_value(self.reader, f"{arch}.ssm.conv_kernel", 0) or 0),
                "layer_types": layer_types,
            })
        return cfg

    def tokenizer_config(self) -> dict[str, Any]:
        chat_template = _field_value(self.reader, gguf.Keys.Tokenizer.CHAT_TEMPLATE, "") or ""
        return {
            "chat_template": str(chat_template),
            "bos_token_id": _field_value(self.reader, gguf.Keys.Tokenizer.BOS_ID, None),
            "eos_token_id": _field_value(self.reader, gguf.Keys.Tokenizer.EOS_ID, None),
            "tokenizer_model": _field_value(self.reader, gguf.Keys.Tokenizer.MODEL, ""),
        }

    def load_tokenizer(self):
        hf_json = _field_value(self.reader, gguf.Keys.Tokenizer.HF_JSON, "") or ""
        if hf_json and Tokenizer is not None:
            try:
                return Tokenizer.from_str(str(hf_json))
            except Exception:
                pass

        tokens = _field_value(self.reader, gguf.Keys.Tokenizer.LIST, []) or []
        if tokens:
            bos = _field_value(self.reader, gguf.Keys.Tokenizer.BOS_ID, None)
            eos = _field_value(self.reader, gguf.Keys.Tokenizer.EOS_ID, None)
            return SimpleGGUFTokenizer([str(tok) for tok in tokens], bos_token_id=bos, eos_token_id=eos)
        return None

    def load_tensor(self, raw_name: str) -> np.ndarray | None:
        tensor = self.tensor_by_name.get(raw_name)
        if tensor is None:
            raw_name = self.tensor_name_map.get(raw_name, raw_name)
            tensor = self.tensor_by_name.get(raw_name)
        if tensor is None:
            return None

        data = tensor.data
        qtype = tensor.tensor_type
        if qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F64}:
            return np.asarray(data, dtype=np.float32)
        if qtype in {
            gguf.GGMLQuantizationType.Q4_0,
            gguf.GGMLQuantizationType.Q4_1,
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
            gguf.GGMLQuantizationType.Q8_0,
        }:
            return gguf.quants.dequantize(np.asarray(data), qtype).astype(np.float32, copy=False)
        if np.issubdtype(data.dtype, np.integer):
            return np.asarray(data, dtype=np.float32)
        return np.asarray(data, dtype=np.float32)


_GGUF_ARCHIVE_CACHE: dict[str, GGUFArchive] = {}


def get_gguf_archive(path: Path) -> GGUFArchive:
    key = str(Path(path).resolve())
    archive = _GGUF_ARCHIVE_CACHE.get(key)
    if archive is None:
        archive = GGUFArchive(Path(path))
        _GGUF_ARCHIVE_CACHE[key] = archive
    return archive