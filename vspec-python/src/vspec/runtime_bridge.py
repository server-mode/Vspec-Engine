from __future__ import annotations

import ctypes
from ctypes import c_char_p, c_int, c_size_t, create_string_buffer
from pathlib import Path


class VspecRuntimeBridge:
    def __init__(self, lib_path: str | None = None) -> None:
        self._lib = ctypes.CDLL(lib_path or str(self._discover_default_lib()))
        self._bind()

    def _discover_default_lib(self) -> Path:
        root = Path(__file__).resolve().parents[3]
        candidates = [
            root / "build" / "Release" / "vspec_engine_capi.dll",
            root / "build" / "vspec_engine_capi.dll",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError("vspec_engine_capi.dll not found. Build CMake target first.")

    def _bind(self) -> None:
        self._lib.vspec_py_version.restype = c_char_p

        self._lib.vspec_py_load_manifest_count.argtypes = [c_char_p]
        self._lib.vspec_py_load_manifest_count.restype = c_int

        self._lib.vspec_py_parse_safetensors_count.argtypes = [c_char_p]
        self._lib.vspec_py_parse_safetensors_count.restype = c_int

        self._lib.vspec_py_rewrite_demo_final_nodes.restype = c_int

        self._lib.vspec_py_generate.argtypes = [c_char_p, ctypes.c_char_p, c_size_t]
        self._lib.vspec_py_generate.restype = c_int

    def version(self) -> str:
        return self._lib.vspec_py_version().decode("utf-8")

    def load_manifest_count(self, path: str) -> int:
        return int(self._lib.vspec_py_load_manifest_count(path.encode("utf-8")))

    def parse_safetensors_count(self, path: str) -> int:
        return int(self._lib.vspec_py_parse_safetensors_count(path.encode("utf-8")))

    def rewrite_demo_final_nodes(self) -> int:
        return int(self._lib.vspec_py_rewrite_demo_final_nodes())

    def generate(self, prompt: str) -> str:
        out = create_string_buffer(4096)
        ok = self._lib.vspec_py_generate(prompt.encode("utf-8"), out, len(out))
        if ok != 1:
            raise RuntimeError("vspec_py_generate failed")
        return out.value.decode("utf-8")
