from __future__ import annotations

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


def run_prefill_with_core_scheduler(
    runtime_obj: Any,
    prefill_ids: list[int],
    progress_cb: Callable[[int, int], None] | None = None,
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
                    return PrefillCoreScheduleResult(False, consumed_total, core_steps, 0, "empty-prefill-chunk")
                continue

            stalled_rounds = 0
            runtime_obj.prefill_tokens(chunk)
            consumed = len(chunk)
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
        )
    except Exception as exc:
        return PrefillCoreScheduleResult(False, 0, 0, 0, f"exception:{exc}")
    finally:
        if core_prefill is not None:
            try:
                core_prefill.close()
            except Exception:
                pass
