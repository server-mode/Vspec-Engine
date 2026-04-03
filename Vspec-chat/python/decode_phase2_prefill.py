from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from runtime_core_bridge import CoreContinuousBatcher, CoreDecodeSession


@dataclass
class PrefillCoreScheduleResult:
    used_core_scheduler: bool
    consumed_tokens: int
    core_steps: int
    reserved_vram: int
    reason: str = ""
    native_prefill_used: bool = False
    native_prefill_calls: int = 0
    native_prefill_failures: int = 0


def run_prefill_with_core_scheduler(
    runtime_obj: Any,
    prefill_ids: list[int],
    progress_cb: Callable[[int, int], None] | None = None,
    native_forward_ctx: Any | None = None,
) -> PrefillCoreScheduleResult:
    total_prefill = int(len(prefill_ids))
    if total_prefill <= 0:
        return PrefillCoreScheduleResult(False, 0, 0, 0, "no-prefill")
    if not hasattr(runtime_obj, "prefill_tokens"):
        return PrefillCoreScheduleResult(False, 0, 0, 0, "runtime-no-prefill-tokens")

    core_prefill = None
    try:
        core_prefill = CoreContinuousBatcher.from_runtime(
            runtime_obj,
            max_batch_items=1,
            max_batch_tokens=max(64, total_prefill),
        )
        if not core_prefill.available:
            return PrefillCoreScheduleResult(False, 0, 0, 0, "core-batcher-unavailable")

        reserve_prefill = CoreDecodeSession.estimate_reserve_bytes(runtime_obj, 1)
        req_id = core_prefill.submit(
            reserve_bytes=reserve_prefill,
            prompt_tokens=total_prefill,
            max_new_tokens=1,
            priority=0,
        )
        if req_id <= 0:
            return PrefillCoreScheduleResult(False, 0, 0, 0, "core-submit-failed")

        native_prefill_requested = os.getenv("VSPEC_NATIVE_PREFILL_COMPUTE", "1").strip().lower() in {"1", "true", "yes", "on"}
        native_prefill_mode = os.getenv("VSPEC_NATIVE_PREFILL_COMPUTE_MODE", "mirror").strip().lower()
        native_prefill_replace = native_prefill_mode in {"replace", "native", "native-only", "only"}
        native_prefill_available = bool(
            native_prefill_requested
            and native_forward_ctx is not None
            and bool(getattr(native_forward_ctx, "available", False))
            and hasattr(native_forward_ctx, "prefill_tokens")
        )
        native_prefill_calls = 0
        native_prefill_failures = 0

        consumed_total = 0
        core_steps = 0
        stalled_rounds = 0
        while consumed_total < total_prefill:
            items = core_prefill.next_batch(1)
            if not items:
                stalled_rounds += 1
                if stalled_rounds >= 3:
                    core_prefill.cancel(req_id)
                    return PrefillCoreScheduleResult(False, consumed_total, core_steps, 0, "core-next-batch-stalled")
                continue

            item = items[0]
            phase = int(item.get("phase", 0))
            if phase != 1:
                core_prefill.cancel(req_id)
                return PrefillCoreScheduleResult(False, consumed_total, core_steps, 0, f"unexpected-phase-{phase}")

            cursor = max(0, int(item.get("prompt_cursor", consumed_total)))
            quota = max(1, int(item.get("token_quota", 1)))
            chunk = prefill_ids[cursor : min(total_prefill, cursor + quota)]
            if not chunk:
                stalled_rounds += 1
                if stalled_rounds >= 3:
                    core_prefill.cancel(req_id)
                    return PrefillCoreScheduleResult(
                        False,
                        consumed_total,
                        core_steps,
                        0,
                        "empty-prefill-chunk",
                        native_prefill_used=bool(native_prefill_available),
                        native_prefill_calls=int(native_prefill_calls),
                        native_prefill_failures=int(native_prefill_failures),
                    )
                continue

            stalled_rounds = 0
            consumed = len(chunk)
            native_ok = False
            if native_prefill_available:
                native_ok, _processed = native_forward_ctx.prefill_tokens(prefill_ids[: cursor + consumed])
                native_prefill_calls += int(bool(native_ok))
                if not native_ok:
                    native_prefill_failures += 1
                    if native_prefill_replace:
                        core_prefill.cancel(req_id)
                        return PrefillCoreScheduleResult(
                            False,
                            consumed_total,
                            core_steps,
                            0,
                            "native-prefill-failed",
                            native_prefill_used=True,
                            native_prefill_calls=int(native_prefill_calls),
                            native_prefill_failures=int(native_prefill_failures),
                        )

            if (not native_prefill_replace) or (not native_ok):
                runtime_obj.prefill_tokens(chunk)

            core_prefill.commit_prefill(req_id, consumed)
            consumed_total = max(consumed_total, cursor + consumed)
            core_steps += 1
            if progress_cb is not None:
                progress_cb(consumed_total, total_prefill)

        stats = core_prefill.stats() or {}
        return PrefillCoreScheduleResult(
            used_core_scheduler=True,
            consumed_tokens=consumed_total,
            core_steps=core_steps,
            reserved_vram=int(stats.get("reserved_vram", 0) or 0),
            reason="ok",
            native_prefill_used=bool(native_prefill_available),
            native_prefill_calls=int(native_prefill_calls),
            native_prefill_failures=int(native_prefill_failures),
        )
    except Exception as exc:
        return PrefillCoreScheduleResult(False, 0, 0, 0, f"exception:{exc}")
    finally:
        if core_prefill is not None:
            try:
                core_prefill.close()
            except Exception:
                pass
