from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Optional

from decode_optimization_module import DecodeOptimizationModule
from decode_phase3_step_dispatch import Phase3StepDispatcher
from fast_output import FastOutputEngine, postprocess_output_text
from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from runtime_core_bridge import CoreContinuousBatcher, CoreDecodeSession, adaptive_step

PHASE_PREFILL = 1
PHASE_DECODE = 2


def _entropy_from_logits(logits_obj) -> float:
    try:
        vals = logits_obj.tolist() if hasattr(logits_obj, "tolist") else list(logits_obj)
        if not vals:
            return 0.0
        max_v = max(float(v) for v in vals)
        exps = [math.exp(max(-60.0, min(60.0, float(v) - max_v))) for v in vals]
        denom = max(1e-12, sum(exps))
        probs = [v / denom for v in exps]
        return float(-sum(p * math.log(max(1e-12, p)) for p in probs))
    except Exception:
        return 0.0


@dataclass
class ManagedGenerationRequest:
    prompt: str
    lang_mode: str
    token_ids: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int
    greedy: bool
    lang_top_n: int
    repetition_penalty: float
    repeat_window: int
    no_repeat_ngram: int
    stream: bool
    disable_language_guard: bool
    language_guard_strictness: float
    prioritize_english: bool
    structure_guard_strictness: float
    disable_structure_guard: bool
    decode_opt_mode: str
    request_id: int = 0
    prompt_tokens_done: int = 0
    generated: list[int] = field(default_factory=list)
    output_text: str = ""
    tokens_per_second: float = 0.0
    timed_out: bool = False
    contract_failed: bool = False
    finished: bool = False
    reached_eos: bool = False
    started_at: float = 0.0
    state_snapshot: dict | None = None
    engine: FastOutputEngine | None = field(default=None, repr=False)
    decode_optimizer: DecodeOptimizationModule | None = field(default=None, repr=False)
    step_dispatcher: Phase3StepDispatcher | None = field(default=None, repr=False)
    adaptive_entropy_prev: float = 0.0
    adaptive_latency_ms: float = 0.0
    adaptive_quality_drift: float = 0.0

    @property
    def prefill_ids(self) -> list[int]:
        if len(self.token_ids) <= 1:
            return []
        return self.token_ids[:-1]

    @property
    def history(self) -> list[int]:
        return self.token_ids


@dataclass
class ManagedGenerationBatchResult:
    requests: list[ManagedGenerationRequest]
    stats: dict[str, int]
    used_core_batcher: bool


class CoreBatchGenerationDriver:
    def __init__(
        self,
        runtime,
        tokenizer,
        adapter,
        threebit_module,
        decode_budget_seconds: float,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.threebit_module = threebit_module
        self.decode_budget_seconds = float(decode_budget_seconds)
        self.progress_cb = progress_cb
        self.vocab_size = tokenizer.get_vocab_size() if tokenizer is not None else int(getattr(adapter, "vocab_size", 0) or 32000)

    def run(self, requests: list[ManagedGenerationRequest]) -> ManagedGenerationBatchResult:
        if not requests:
            return ManagedGenerationBatchResult(requests=[], stats={}, used_core_batcher=False)

        multi_request = len(requests) > 1
        batcher = CoreContinuousBatcher.from_runtime(self.runtime, max_batch_items=max(1, len(requests)), max_batch_tokens=max(64, len(requests) * 8))
        if not batcher.available:
            return ManagedGenerationBatchResult(requests=self._run_sequential(requests), stats={}, used_core_batcher=False)

        saved_mirrors = None
        if multi_request and hasattr(self.runtime, "kv_core_mirrors"):
            try:
                saved_mirrors = list(getattr(self.runtime, "kv_core_mirrors") or [])
                self.runtime.kv_core_mirrors = []
            except Exception:
                saved_mirrors = None

        try:
            reserve_bytes = CoreDecodeSession.estimate_reserve_bytes(self.runtime, max(req.max_new_tokens for req in requests))
            request_by_id: dict[int, ManagedGenerationRequest] = {}
            active_count = 0
            for req in requests:
                self._prepare_request(req)
                req_id = batcher.submit(
                    reserve_bytes=reserve_bytes,
                    prompt_tokens=len(req.prefill_ids),
                    max_new_tokens=req.max_new_tokens,
                    priority=0,
                )
                if req_id <= 0:
                    req.contract_failed = True
                    req.finished = True
                    req.output_text = ""
                    continue
                req.request_id = req_id
                req.started_at = time.perf_counter()
                request_by_id[req_id] = req
                active_count += 1

            while active_count > 0:
                items = batcher.next_batch(max(1, len(requests)))
                if not items:
                    break

                for item in items:
                    req = request_by_id.get(int(item.get("request_id", 0)))
                    if req is None or req.finished:
                        continue
                    self._restore_runtime_state(req.state_snapshot)
                    phase = int(item.get("phase", 0))
                    quota = max(0, int(item.get("token_quota", 0)))
                    if phase == PHASE_PREFILL:
                        cursor = max(0, int(item.get("prompt_cursor", 0)))
                        consumed = self._run_prefill(req, cursor, quota)
                        req.state_snapshot = self._snapshot_runtime_state()
                        batcher.commit_prefill(req.request_id, consumed)
                        req.prompt_tokens_done = min(len(req.prefill_ids), cursor + consumed)
                        self._emit_progress("prefill", req.prompt_tokens_done, len(req.prefill_ids))
                        continue

                    generated_now = self._run_decode_quota(req, quota)
                    req.state_snapshot = self._snapshot_runtime_state()
                    batcher.commit_decode(req.request_id, generated_now, req.reached_eos or req.finished)
                    if req.finished:
                        active_count -= 1
                        self._finalize_request(req)
                        continue
                    self._emit_progress("decode", len(req.generated), req.max_new_tokens)

            for req in requests:
                if not req.finished and req.request_id > 0:
                    batcher.cancel(req.request_id)
                    req.finished = True
                    self._finalize_request(req)

            return ManagedGenerationBatchResult(requests=requests, stats=batcher.stats(), used_core_batcher=True)
        finally:
            if saved_mirrors is not None:
                try:
                    self.runtime.kv_core_mirrors = saved_mirrors
                except Exception:
                    pass
            batcher.close()

    def _run_sequential(self, requests: list[ManagedGenerationRequest]) -> list[ManagedGenerationRequest]:
        for req in requests:
            self._prepare_request(req)
            self._clear_runtime_state(reset_mirrors=True)
            if req.prefill_ids:
                self._run_prefill(req, 0, len(req.prefill_ids))
            self._run_decode_quota(req, req.max_new_tokens)
            req.finished = True
            self._finalize_request(req)
        return requests

    def _prepare_request(self, req: ManagedGenerationRequest) -> None:
        if req.engine is None:
            guard = None
            if not req.disable_language_guard:
                guard = LanguageStabilityGuard(
                    prompt=req.prompt,
                    lang_mode=req.lang_mode,
                    strictness=req.language_guard_strictness,
                    prioritize_english=req.prioritize_english,
                )
            structure_guard = None
            if not req.disable_structure_guard:
                structure_guard = LanguageStructureIntegrityManager(prompt=req.prompt, strictness=req.structure_guard_strictness)
            req.engine = FastOutputEngine(
                tokenizer=self.tokenizer,
                lang_mode=req.lang_mode,
                stream=req.stream,
                guard=guard,
                structure_guard=structure_guard,
            )
        if req.decode_optimizer is None:
            req.decode_optimizer = DecodeOptimizationModule(
                repetition_penalty=req.repetition_penalty,
                repeat_window=req.repeat_window,
                no_repeat_ngram=req.no_repeat_ngram,
                mode=req.decode_opt_mode,
            )
            req.decode_optimizer.seed_history(req.history)
        if req.step_dispatcher is None and req.decode_optimizer is not None:
            req.step_dispatcher = Phase3StepDispatcher(
                runtime=self.runtime,
                decode_optimizer=req.decode_optimizer,
                expected_vocab_size=self.vocab_size,
            )
        if req.started_at <= 0.0:
            req.started_at = time.perf_counter()

    def _run_prefill(self, req: ManagedGenerationRequest, cursor: int, quota: int) -> int:
        tokens = req.prefill_ids[cursor : cursor + quota]
        if not tokens:
            return 0
        if hasattr(self.runtime, "prefill_tokens"):
            self.runtime.prefill_tokens(tokens)
            return len(tokens)
        consumed = 0
        for token_id in tokens:
            if hasattr(self.runtime, "forward_logits"):
                self.runtime.forward_logits([int(token_id)])
                consumed += 1
        return consumed

    def _run_decode_quota(self, req: ManagedGenerationRequest, quota: int) -> int:
        if req.engine is None or req.decode_optimizer is None:
            return 0
        generated_now = 0
        req.engine.begin_stream()
        try:
            for _ in range(max(0, int(quota))):
                token_t0 = time.perf_counter()
                elapsed = time.perf_counter() - req.started_at
                if self.decode_budget_seconds > 0.0 and elapsed >= self.decode_budget_seconds:
                    req.timed_out = True
                    req.finished = True
                    break

                dispatch = req.step_dispatcher.step(req.history[-1]) if req.step_dispatcher is not None else None
                if dispatch is None or not dispatch.ok:
                    req.contract_failed = True
                    req.finished = True
                    break
                logits = dispatch.logits
                if req.decode_optimizer.logits_empty(logits):
                    req.contract_failed = True
                    req.finished = True
                    break

                logits = self.threebit_module.denoise_logits(logits, len(req.generated))
                logits = req.decode_optimizer.apply_generation_controls(logits, req.history)
                sample_temperature = self.threebit_module.auto_temperature(logits, req.temperature)

                entropy_now = _entropy_from_logits(logits)
                entropy_collapse = max(0.0, req.adaptive_entropy_prev - entropy_now)
                req.adaptive_entropy_prev = entropy_now
                token_text = ""
                try:
                    token_text = self.tokenizer.decode([int(req.history[-1])])
                except Exception:
                    token_text = str(int(req.history[-1]))
                decision = adaptive_step(
                    token_text=token_text,
                    token_entropy=entropy_now,
                    attention_entropy_collapse=entropy_collapse,
                    latency_ms=req.adaptive_latency_ms,
                    vram_pressure=0.5,
                    quality_drift=req.adaptive_quality_drift,
                    layer_type=0,
                )

                sample_top_k = int(req.top_k)
                sample_greedy = bool(req.greedy)
                if decision is not None:
                    setattr(self.runtime, "adaptive_target_bits", int(decision.target_bits))
                    setattr(self.runtime, "adaptive_routed_bits", int(decision.routed_bits))
                    setattr(self.runtime, "adaptive_reduce_attention_depth", bool(decision.reduce_attention_depth))
                    setattr(self.runtime, "adaptive_attention_depth_hint", int(decision.attention_depth_hint))
                    setattr(self.runtime, "adaptive_enable_kv_compression", bool(decision.enable_kv_compression))
                    if decision.reduce_attention_depth and decision.attention_depth_hint > 0:
                        sample_top_k = min(sample_top_k, max(8, int(decision.attention_depth_hint) * 4))
                    if decision.skip_compute:
                        sample_greedy = True
                    bit_hint = int(decision.routed_bits if decision.routed_bits > 0 else decision.target_bits)
                    if bit_hint > 0:
                        temp_scale = max(0.65, min(1.0, bit_hint / 8.0))
                        sample_temperature = max(0.05, float(sample_temperature) * temp_scale)

                next_id = req.engine.sample(logits, sample_temperature, sample_top_k, sample_greedy, req.lang_top_n)
                req.generated.append(next_id)
                req.token_ids.append(next_id)
                req.engine.stream_token(next_id)
                req.decode_optimizer.observe_token(req.history)
                req.adaptive_latency_ms = (time.perf_counter() - token_t0) * 1000.0
                req.adaptive_quality_drift = min(1.0, 0.6 * req.adaptive_quality_drift + 0.4 * entropy_collapse)
                generated_now += 1
                if self.adapter.eos_token_id is not None and next_id == self.adapter.eos_token_id:
                    req.reached_eos = True
                    req.finished = True
                    break
                if req.max_new_tokens > 0 and len(req.generated) >= req.max_new_tokens:
                    req.finished = True
                    break
        finally:
            req.engine.end_stream()
        return generated_now

    def _finalize_request(self, req: ManagedGenerationRequest) -> None:
        if req.output_text:
            return
        text = self.tokenizer.decode(req.generated) if (self.tokenizer is not None and req.generated) else ""
        req.output_text = postprocess_output_text(text, req.prompt, req.lang_mode)
        elapsed = max(1e-6, time.perf_counter() - max(req.started_at, 1e-6))
        req.tokens_per_second = float(len(req.generated)) / elapsed

    def _emit_progress(self, phase: str, current: int, total: int) -> None:
        if self.progress_cb is None or total <= 0:
            return
        try:
            self.progress_cb(phase, int(current), int(total))
        except Exception:
            pass

    def _clear_runtime_state(self, reset_mirrors: bool) -> None:
        if hasattr(self.runtime, "cache_k"):
            self.runtime.cache_k = []
        if hasattr(self.runtime, "cache_v"):
            self.runtime.cache_v = []
        if hasattr(self.runtime, "cache_len"):
            self.runtime.cache_len = []
        if hasattr(self.runtime, "position"):
            self.runtime.position = 0
        if reset_mirrors and hasattr(self.runtime, "reset_core_kv_mirrors"):
            try:
                self.runtime.reset_core_kv_mirrors()
            except Exception:
                pass

    def _snapshot_runtime_state(self) -> dict:
        state = {
            "position": int(getattr(self.runtime, "position", 0) or 0),
            "cache_k": [self._clone_entry(entry) for entry in list(getattr(self.runtime, "cache_k", []) or [])],
            "cache_v": [self._clone_entry(entry) for entry in list(getattr(self.runtime, "cache_v", []) or [])],
            "cache_len": list(getattr(self.runtime, "cache_len", []) or []),
        }
        return state

    def _restore_runtime_state(self, state: dict | None) -> None:
        if not state:
            self._clear_runtime_state(reset_mirrors=True)
            return
        self._clear_runtime_state(reset_mirrors=True)
        if hasattr(self.runtime, "cache_k"):
            self.runtime.cache_k = [self._clone_entry(entry) for entry in list(state.get("cache_k", []) or [])]
        if hasattr(self.runtime, "cache_v"):
            self.runtime.cache_v = [self._clone_entry(entry) for entry in list(state.get("cache_v", []) or [])]
        if hasattr(self.runtime, "cache_len"):
            self.runtime.cache_len = list(state.get("cache_len", []) or [])
        if hasattr(self.runtime, "position"):
            self.runtime.position = int(state.get("position", 0) or 0)

    def _clone_entry(self, entry):
        if entry is None:
            return None
        clone_fn = getattr(entry, "copy", None)
        if callable(clone_fn):
            try:
                return clone_fn()
            except Exception:
                pass
        clone_fn = getattr(entry, "clone", None)
        if callable(clone_fn):
            try:
                return clone_fn()
            except Exception:
                pass
        return entry
