from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass
class DecodeOptimizationProfile:
    mode: str
    use_numpy_logits: bool
    incremental_ngram: bool


def resolve_decode_profile(mode: str) -> DecodeOptimizationProfile:
    m = (mode or "optimized").strip().lower()
    if m not in {"stable", "optimized"}:
        m = "optimized"
    if m == "stable":
        return DecodeOptimizationProfile(mode="stable", use_numpy_logits=True, incremental_ngram=False)
    return DecodeOptimizationProfile(mode="optimized", use_numpy_logits=True, incremental_ngram=True)


class DecodeOptimizationModule:
    def __init__(
        self,
        repetition_penalty: float,
        repeat_window: int,
        no_repeat_ngram: int,
        mode: str = "optimized",
    ) -> None:
        self.repetition_penalty = float(repetition_penalty)
        self.repeat_window = int(max(0, repeat_window))
        self.no_repeat_ngram = int(max(0, no_repeat_ngram))
        self.profile = resolve_decode_profile(mode)
        self._ngram_block: dict[tuple[int, ...], set[int]] = {}
        self._ngram_seeded = False

    def fetch_logits(self, runtime, last_token_id: int, vocab_size: int):
        if runtime is None:
            return [0.0 for _ in range(vocab_size)]

        if self.profile.use_numpy_logits and hasattr(runtime, "forward_logits_np"):
            logits = runtime.forward_logits_np([int(last_token_id)])
            if np is not None and isinstance(logits, np.ndarray) and logits.size > 0:
                safe = np.asarray(logits, dtype=np.float32)
                if not np.all(np.isfinite(safe)):
                    safe = np.nan_to_num(safe, nan=-1e9, posinf=1e9, neginf=-1e9)
                return safe

        if hasattr(runtime, "forward_logits"):
            logits = runtime.forward_logits([int(last_token_id)])
            if logits:
                safe_logits: list[float] = []
                for value in logits:
                    try:
                        fv = float(value)
                    except Exception:
                        fv = -1e9
                    if not math.isfinite(fv):
                        fv = -1e9 if fv < 0 else 1e9
                    safe_logits.append(fv)
                return safe_logits

        return [0.0 for _ in range(vocab_size)]

    def logits_empty(self, logits) -> bool:
        if logits is None:
            return True
        if np is not None and isinstance(logits, np.ndarray):
            return logits.size == 0
        try:
            return len(logits) == 0
        except Exception:
            return False

    def seed_history(self, history: list[int]) -> None:
        if self._ngram_seeded:
            return
        if not self.profile.incremental_ngram or self.no_repeat_ngram <= 1:
            self._ngram_seeded = True
            return
        if len(history) < self.no_repeat_ngram:
            self._ngram_seeded = True
            return

        n = self.no_repeat_ngram
        for i in range(len(history) - n + 1):
            ngram = tuple(int(v) for v in history[i : i + n])
            prefix = ngram[:-1]
            nxt = int(ngram[-1])
            bucket = self._ngram_block.get(prefix)
            if bucket is None:
                bucket = set()
                self._ngram_block[prefix] = bucket
            bucket.add(nxt)

        self._ngram_seeded = True

    def observe_token(self, history: list[int]) -> None:
        if not self.profile.incremental_ngram or self.no_repeat_ngram <= 1:
            return
        n = self.no_repeat_ngram
        if len(history) < n:
            return

        ngram = tuple(int(v) for v in history[-n:])
        prefix = ngram[:-1]
        nxt = int(ngram[-1])
        bucket = self._ngram_block.get(prefix)
        if bucket is None:
            bucket = set()
            self._ngram_block[prefix] = bucket
        bucket.add(nxt)

    def apply_generation_controls(self, logits, history: list[int]):
        if np is not None and isinstance(logits, np.ndarray):
            adjusted = logits.copy()
        else:
            adjusted = list(logits)

        if self.repetition_penalty > 1.0 and history and self.repeat_window > 0:
            for token_id in set(history[-self.repeat_window :]):
                if 0 <= token_id < len(adjusted):
                    if adjusted[token_id] > 0:
                        adjusted[token_id] /= self.repetition_penalty
                    else:
                        adjusted[token_id] *= self.repetition_penalty

        if self.no_repeat_ngram > 1 and len(history) >= self.no_repeat_ngram - 1:
            if self.profile.incremental_ngram:
                prefix = tuple(int(v) for v in history[-(self.no_repeat_ngram - 1) :])
                banned = self._ngram_block.get(prefix, set())
            else:
                prefix = tuple(int(v) for v in history[-(self.no_repeat_ngram - 1) :])
                banned = set()
                for i in range(len(history) - self.no_repeat_ngram + 1):
                    ngram = tuple(int(v) for v in history[i : i + self.no_repeat_ngram])
                    if ngram[:-1] == prefix:
                        banned.add(int(ngram[-1]))

            for token_id in banned:
                if 0 <= token_id < len(adjusted):
                    adjusted[token_id] = -1e9

        return adjusted
