from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from runtime_core_bridge import CoreNativeDecodeLoop, CoreNativeForwardContext


@dataclass
class Phase5TurnReport:
    timed_out: bool
    contract_failed: bool
    generated_tokens: int


class SessionCoreDaemonSupervisor:
    """Phase 5: persistent C-handle supervisor for session runtime.

    Python keeps policy/guard ownership while C handles stay alive across turns.
    """

    def __init__(self, runtime: Any, max_tokens: int, native_model_file: str | None, seed: int) -> None:
        self.runtime = runtime
        self.max_tokens = int(max(1, max_tokens))
        self.native_model_file = str(native_model_file) if native_model_file else None
        self.seed = int(seed)

        self.enabled = str(os.getenv("VSPEC_ENABLE_PHASE5_DAEMON", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.native_cpp_loop_enabled = str(os.getenv("VSPEC_NATIVE_CPP_LOOP", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.native_forward_enabled = str(os.getenv("VSPEC_NATIVE_FORWARD_BLEND", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.forward_ctx_enabled = str(os.getenv("VSPEC_PHASE5_DAEMON_FORWARD_CTX", "1")).strip().lower() in {"1", "true", "yes", "on"}

        try:
            self.max_consecutive_failures = max(1, int(os.getenv("VSPEC_PHASE5_DAEMON_MAX_CONSEC_FAIL", "3") or "3"))
        except Exception:
            self.max_consecutive_failures = 3

        self.loop_handle: CoreNativeDecodeLoop | None = None
        self.forward_ctx: CoreNativeForwardContext | None = None

        self.turns_total = 0
        self.turns_timeout = 0
        self.turns_contract_failed = 0
        self.consecutive_failures = 0
        self.restarts = 0

    def start(self) -> None:
        if not self.enabled:
            return
        self._open_handles()

    def _open_handles(self) -> None:
        if self.native_cpp_loop_enabled:
            try:
                reserve = max(64, int(self.max_tokens) * 4)
                self.loop_handle = CoreNativeDecodeLoop.from_runtime(self.runtime, reserve)
            except Exception:
                self.loop_handle = None

        if self.forward_ctx_enabled and self.native_forward_enabled and self.native_model_file:
            try:
                seed_raw = os.getenv("VSPEC_NATIVE_FORWARD_SEED", str(self.seed))
                seed_val = int(seed_raw)
            except Exception:
                seed_val = self.seed
            try:
                ctx = CoreNativeForwardContext(self.native_model_file, seed=seed_val)
                self.forward_ctx = ctx if ctx.available else None
            except Exception:
                self.forward_ctx = None

    def _close_handles(self) -> None:
        if self.loop_handle is not None:
            try:
                self.loop_handle.close()
            except Exception:
                pass
            self.loop_handle = None
        if self.forward_ctx is not None:
            try:
                self.forward_ctx.close()
            except Exception:
                pass
            self.forward_ctx = None

    def restart_handles(self) -> None:
        self.restarts += 1
        self._close_handles()
        self._open_handles()

    def observe_turn(self, report: Phase5TurnReport) -> None:
        self.turns_total += 1
        self.turns_timeout += int(bool(report.timed_out))
        self.turns_contract_failed += int(bool(report.contract_failed))

        failed = bool(report.timed_out or report.contract_failed or int(report.generated_tokens) <= 0)
        if failed:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        if self.enabled and self.consecutive_failures >= self.max_consecutive_failures:
            self.consecutive_failures = 0
            self.restart_handles()

    def status(self) -> dict[str, int | bool]:
        return {
            "enabled": bool(self.enabled),
            "loop_available": bool(self.loop_handle is not None and self.loop_handle.available),
            "forward_ctx_available": bool(self.forward_ctx is not None and self.forward_ctx.available),
            "turns_total": int(self.turns_total),
            "turns_timeout": int(self.turns_timeout),
            "turns_contract_failed": int(self.turns_contract_failed),
            "restarts": int(self.restarts),
            "max_consecutive_failures": int(self.max_consecutive_failures),
        }

    def close(self) -> None:
        self._close_handles()
