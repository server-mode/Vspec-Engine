from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from decode_contract import sanitize_and_validate_logits

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class StepDispatchResult:
    ok: bool
    logits: Any = None
    reason: str = ""
    masked_tail: int = 0
    used_c_dispatch: bool = False
    parity_checked: bool = False
    parity_pass: bool = True
    parity_cosine: float = 1.0
    topk_overlap: float = 1.0


class Phase3StepDispatcher:
    def __init__(self, runtime: Any, decode_optimizer: Any, expected_vocab_size: int) -> None:
        self.runtime = runtime
        self.decode_optimizer = decode_optimizer
        self.expected_vocab_size = int(max(0, expected_vocab_size))
        self.use_c_step_dispatch = str(os.getenv("VSPEC_USE_C_STEP_DISPATCH", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self.parity_shadow = str(os.getenv("VSPEC_STEP_PARITY_SHADOW", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.stats = {
            "calls": 0,
            "c_step_calls": 0,
            "python_step_calls": 0,
            "parity_checks": 0,
            "parity_failures": 0,
            "parity_fallbacks": 0,
        }

    def _as_numpy(self, logits) -> Any:
        if np is None:
            return logits
        if isinstance(logits, np.ndarray):
            arr = np.asarray(logits, dtype=np.float32)
        else:
            try:
                arr = np.asarray(list(logits), dtype=np.float32)
            except Exception:
                return logits
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=-1e9, posinf=1e9, neginf=-1e9)
        return arr

    def _python_step(self, last_token_id: int):
        return self.decode_optimizer.fetch_logits(self.runtime, int(last_token_id), self.expected_vocab_size)

    def _c_step(self, last_token_id: int):
        # C/CUDA dispatch path: call runtime direct tensor path, bypassing decode_optimizer.fetch wrapper.
        if self.runtime is None:
            return None
        if hasattr(self.runtime, "forward_logits_np"):
            return self.runtime.forward_logits_np([int(last_token_id)])
        if hasattr(self.runtime, "forward_logits"):
            return self.runtime.forward_logits([int(last_token_id)])
        return None

    def _compare_parity(self, ref_logits, test_logits) -> tuple[bool, float, float]:
        if np is None:
            return True, 1.0, 1.0
        a = self._as_numpy(ref_logits)
        b = self._as_numpy(test_logits)
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.size == 0 or b.size == 0:
            return True, 1.0, 1.0

        n = int(min(a.size, b.size))
        if n <= 0:
            return True, 1.0, 1.0
        a = a[:n]
        b = b[:n]

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = 1.0 if denom <= 1e-12 else float(np.dot(a, b) / denom)

        try:
            topk = int(os.getenv("VSPEC_STEP_PARITY_TOPK", "16") or "16")
        except Exception:
            topk = 16
        topk = max(4, min(64, topk, n))
        a_idx = set(int(i) for i in np.argpartition(a, -topk)[-topk:])
        b_idx = set(int(i) for i in np.argpartition(b, -topk)[-topk:])
        overlap = float(len(a_idx & b_idx) / max(1, topk))

        try:
            min_cos = float(os.getenv("VSPEC_STEP_PARITY_MIN_COSINE", "0.999") or "0.999")
        except Exception:
            min_cos = 0.999
        try:
            min_overlap = float(os.getenv("VSPEC_STEP_PARITY_MIN_TOPK_OVERLAP", "0.9") or "0.9")
        except Exception:
            min_overlap = 0.9

        parity_ok = bool(cosine >= min_cos and overlap >= min_overlap)
        return parity_ok, cosine, overlap

    def step(self, last_token_id: int) -> StepDispatchResult:
        self.stats["calls"] += 1
        use_c = bool(self.use_c_step_dispatch)

        if not use_c:
            self.stats["python_step_calls"] += 1
            py_logits = self._python_step(last_token_id)
            py_logits = self._as_numpy(py_logits)
            logits, contract = sanitize_and_validate_logits(py_logits, self.expected_vocab_size)
            if not contract.ok:
                return StepDispatchResult(ok=False, reason=str(contract.reason), masked_tail=int(contract.masked_tail or 0), used_c_dispatch=False)
            return StepDispatchResult(ok=True, logits=logits, masked_tail=int(contract.masked_tail or 0), used_c_dispatch=False)

        self.stats["c_step_calls"] += 1
        c_logits_raw = self._c_step(last_token_id)
        c_logits = self._as_numpy(c_logits_raw)
        logits, contract = sanitize_and_validate_logits(c_logits, self.expected_vocab_size)
        if not contract.ok:
            return StepDispatchResult(ok=False, reason=str(contract.reason), masked_tail=int(contract.masked_tail or 0), used_c_dispatch=True)

        parity_checked = False
        parity_pass = True
        parity_cos = 1.0
        parity_overlap = 1.0

        if self.parity_shadow:
            parity_checked = True
            self.stats["parity_checks"] += 1
            py_shadow = self._python_step(last_token_id)
            parity_pass, parity_cos, parity_overlap = self._compare_parity(py_shadow, logits)
            if not parity_pass:
                self.stats["parity_failures"] += 1
                self.stats["parity_fallbacks"] += 1
                self.stats["python_step_calls"] += 1
                py_logits = self._as_numpy(py_shadow)
                py_logits, py_contract = sanitize_and_validate_logits(py_logits, self.expected_vocab_size)
                if not py_contract.ok:
                    return StepDispatchResult(
                        ok=False,
                        reason=f"parity-fallback-{py_contract.reason}",
                        masked_tail=int(py_contract.masked_tail or 0),
                        used_c_dispatch=False,
                        parity_checked=parity_checked,
                        parity_pass=False,
                        parity_cosine=parity_cos,
                        topk_overlap=parity_overlap,
                    )
                return StepDispatchResult(
                    ok=True,
                    logits=py_logits,
                    reason="parity-fallback",
                    masked_tail=int(py_contract.masked_tail or 0),
                    used_c_dispatch=False,
                    parity_checked=parity_checked,
                    parity_pass=False,
                    parity_cosine=parity_cos,
                    topk_overlap=parity_overlap,
                )

        return StepDispatchResult(
            ok=True,
            logits=logits,
            masked_tail=int(contract.masked_tail or 0),
            used_c_dispatch=True,
            parity_checked=parity_checked,
            parity_pass=parity_pass,
            parity_cosine=parity_cos,
            topk_overlap=parity_overlap,
        )
