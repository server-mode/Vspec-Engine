from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

from runtime_core_bridge import sample_candidate, sample_candidate_available


@dataclass
class Phase4SampleDecision:
    token_id: int
    used_c_sampler: bool
    parity_checked: bool
    parity_match: bool
    parity_fallback: bool


class Phase4SamplingCore:
    def __init__(self) -> None:
        self.use_c_sampler = str(os.getenv("VSPEC_USE_C_SAMPLER", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.parity_shadow = str(os.getenv("VSPEC_SAMPLER_PARITY_SHADOW", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.parity_fallback = str(os.getenv("VSPEC_SAMPLER_PARITY_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self.stats = {
            "calls": 0,
            "c_sampler_calls": 0,
            "python_sampler_calls": 0,
            "c_sampler_unavailable": 0,
            "parity_checks": 0,
            "parity_mismatch": 0,
            "parity_fallbacks": 0,
        }

    @staticmethod
    def _u01_from_bits(random_bits: int) -> float:
        mantissa = int(random_bits) & ((1 << 53) - 1)
        return float(mantissa) / float(1 << 53)

    @staticmethod
    def _sample_python_softmax(scored: list[tuple[int, float]], random_bits: int) -> int:
        if not scored:
            return 0
        max_logit = float(scored[0][1])
        if (not math.isfinite(max_logit)):
            return int(scored[0][0])
        exp_vals = [math.exp(float(v) - max_logit) for _, v in scored]
        total = float(sum(exp_vals))
        if (not math.isfinite(total)) or total <= 0.0:
            return int(scored[0][0])
        r = Phase4SamplingCore._u01_from_bits(random_bits)
        acc = 0.0
        for i, (tid, _) in enumerate(scored):
            acc += float(exp_vals[i] / total)
            if r <= acc:
                return int(tid)
        return int(scored[-1][0])

    def choose(self, scored: list[tuple[int, float]], greedy: bool, c_sampler_required: bool) -> Phase4SampleDecision:
        self.stats["calls"] += 1
        if not scored:
            self.stats["python_sampler_calls"] += 1
            return Phase4SampleDecision(token_id=0, used_c_sampler=False, parity_checked=False, parity_match=True, parity_fallback=False)

        if greedy:
            self.stats["python_sampler_calls"] += 1
            return Phase4SampleDecision(token_id=int(scored[0][0]), used_c_sampler=False, parity_checked=False, parity_match=True, parity_fallback=False)

        random_bits = int(random.getrandbits(63))

        if (not self.use_c_sampler) or (not sample_candidate_available()):
            if self.use_c_sampler and (not sample_candidate_available()):
                self.stats["c_sampler_unavailable"] += 1
            if c_sampler_required and (not sample_candidate_available()):
                self.stats["python_sampler_calls"] += 1
                return Phase4SampleDecision(token_id=int(scored[0][0]), used_c_sampler=False, parity_checked=False, parity_match=True, parity_fallback=False)
            self.stats["python_sampler_calls"] += 1
            py_tid = self._sample_python_softmax(scored, random_bits)
            return Phase4SampleDecision(token_id=int(py_tid), used_c_sampler=False, parity_checked=False, parity_match=True, parity_fallback=False)

        self.stats["c_sampler_calls"] += 1
        sampled = sample_candidate(
            token_ids=[int(tid) for tid, _ in scored],
            scores=[float(score) for _, score in scored],
            greedy=False,
            random_bits=random_bits,
        )
        if sampled is None:
            self.stats["c_sampler_unavailable"] += 1
            if c_sampler_required:
                self.stats["python_sampler_calls"] += 1
                return Phase4SampleDecision(token_id=int(scored[0][0]), used_c_sampler=False, parity_checked=False, parity_match=True, parity_fallback=False)
            self.stats["python_sampler_calls"] += 1
            py_tid = self._sample_python_softmax(scored, random_bits)
            return Phase4SampleDecision(token_id=int(py_tid), used_c_sampler=False, parity_checked=False, parity_match=True, parity_fallback=False)

        chosen = int(sampled)
        parity_checked = False
        parity_match = True
        parity_fallback = False

        if self.parity_shadow:
            parity_checked = True
            self.stats["parity_checks"] += 1
            py_tid = self._sample_python_softmax(scored, random_bits)
            parity_match = bool(py_tid == chosen)
            if not parity_match:
                self.stats["parity_mismatch"] += 1
                if self.parity_fallback:
                    self.stats["parity_fallbacks"] += 1
                    chosen = int(py_tid)
                    parity_fallback = True
                    self.stats["python_sampler_calls"] += 1

        return Phase4SampleDecision(
            token_id=int(chosen),
            used_c_sampler=True,
            parity_checked=parity_checked,
            parity_match=parity_match,
            parity_fallback=parity_fallback,
        )
