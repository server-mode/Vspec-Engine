from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from decode_contract import sanitize_and_validate_logits


@dataclass
class DecodeState:
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    generated_ids: list[int] = field(default_factory=list)
    prefill_done: bool = False
    contract_failed: bool = False
    finished: bool = False
    finish_reason: str = ""
    last_contract_reason: str = ""

    def mark_prefill_done(self) -> None:
        self.prefill_done = True

    def mark_contract_failed(self, reason: str) -> None:
        self.contract_failed = True
        self.finished = True
        self.finish_reason = "contract-failed"
        self.last_contract_reason = str(reason or "")

    def mark_timeout(self) -> None:
        self.finished = True
        self.finish_reason = "timeout"

    def on_token(self, token_id: int, reached_eos: bool) -> None:
        self.generated_ids.append(int(token_id))
        self.generated_tokens += 1
        if reached_eos:
            self.finished = True
            self.finish_reason = "eos"
        elif self.generated_tokens >= max(1, int(self.max_new_tokens)):
            self.finished = True
            self.finish_reason = "max-tokens"


class DecodeOrchestrationContract(Protocol):
    def prefill(self, token_ids: list[int]) -> None:
        ...

    def step(self, last_token_id: int):
        ...

    def sample(self, logits):
        ...

    def commit(self, token_id: int, reached_eos: bool) -> None:
        ...


@dataclass
class DecodeStepResult:
    ok: bool
    logits: Any = None
    reason: str = ""
    masked_tail: int = 0


class PythonDecodeOrchestrator:
    """Phase 1 orchestration contract for Python runtime decode flow."""

    def __init__(
        self,
        *,
        state: DecodeState,
        runtime: Any,
        decode_optimizer: Any,
        expected_vocab_size: int,
        scheduler_enabled: bool,
        core_decode: Any,
        step_dispatcher: Any = None,
    ) -> None:
        self.state = state
        self.runtime = runtime
        self.decode_optimizer = decode_optimizer
        self.expected_vocab_size = int(max(0, expected_vocab_size))
        self.scheduler_enabled = bool(scheduler_enabled)
        self.core_decode = core_decode
        self.step_dispatcher = step_dispatcher

    def prefill(self, token_ids: list[int]) -> None:
        _ = token_ids
        self.state.mark_prefill_done()

    def step(self, last_token_id: int) -> DecodeStepResult:
        if self.runtime is None:
            self.state.mark_contract_failed("runtime-unavailable")
            return DecodeStepResult(ok=False, reason="runtime-unavailable")

        if self.step_dispatcher is not None:
            dispatched = self.step_dispatcher.step(int(last_token_id))
            if not bool(getattr(dispatched, "ok", False)):
                reason = str(getattr(dispatched, "reason", "step-dispatch-failed") or "step-dispatch-failed")
                self.state.mark_contract_failed(reason)
                return DecodeStepResult(ok=False, reason=reason, masked_tail=int(getattr(dispatched, "masked_tail", 0) or 0))
            logits = getattr(dispatched, "logits", None)
            masked_tail = int(getattr(dispatched, "masked_tail", 0) or 0)
        else:
            logits = self.decode_optimizer.fetch_logits(self.runtime, int(last_token_id), self.expected_vocab_size)
            logits, contract = sanitize_and_validate_logits(logits, self.expected_vocab_size)
            if not contract.ok:
                self.state.mark_contract_failed(contract.reason)
                return DecodeStepResult(ok=False, reason=str(contract.reason), masked_tail=int(contract.masked_tail or 0))
            masked_tail = int(contract.masked_tail or 0)

        if self.decode_optimizer.logits_empty(logits):
            self.state.mark_contract_failed("empty-logits")
            return DecodeStepResult(ok=False, reason="empty-logits", masked_tail=masked_tail)
        return DecodeStepResult(ok=True, logits=logits, masked_tail=masked_tail)

    def sample(self, logits, sampler: Callable[..., int], *sampler_args, **sampler_kwargs) -> int:
        return int(sampler(logits, *sampler_args, **sampler_kwargs))

    def commit(self, token_id: int, reached_eos: bool) -> None:
        self.state.on_token(int(token_id), bool(reached_eos))
        if self.scheduler_enabled:
            self.core_decode.commit(1, bool(reached_eos))
