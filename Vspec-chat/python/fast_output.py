import heapq
import math
import os
import re
import time
from dataclasses import dataclass

from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from meaningful_output_guard import MeaningfulOutputGuard
from decode_phase4_sampler import Phase4SamplingCore
from runtime_core_bridge import (
    native_output_guard_allow,
    native_output_guard_available,
    native_output_guard_init,
    native_output_guard_observe,
    native_output_guard_score_adjustment,
)
from runtime_meaningful_response import RuntimeMeaningfulResponseAssurance

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class SpeedPreset:
    top_k: int
    lang_top_n: int
    repetition_penalty: float
    no_repeat_ngram: int
    repeat_window: int


_SAMPLING_TIMING_STATS = {
    "sampling_ms": 0.0,
    "sampling_calls": 0,
}


def _timing_enabled() -> bool:
    return os.getenv("VSPEC_RUNTIME_TIMING", "0").strip().lower() in {"1", "true", "yes", "on"}


def sampling_timing_reset() -> None:
    _SAMPLING_TIMING_STATS["sampling_ms"] = 0.0
    _SAMPLING_TIMING_STATS["sampling_calls"] = 0


def sampling_timing_snapshot() -> dict[str, float]:
    return {
        "sampling_ms": float(_SAMPLING_TIMING_STATS["sampling_ms"]),
        "sampling_calls": float(_SAMPLING_TIMING_STATS["sampling_calls"]),
    }


def resolve_speed_preset(name: str) -> SpeedPreset:
    mode = (name or "fast").lower()
    if mode == "ultra":
        return SpeedPreset(top_k=16, lang_top_n=64, repetition_penalty=1.05, no_repeat_ngram=0, repeat_window=24)
    if mode == "normal":
        return SpeedPreset(top_k=40, lang_top_n=256, repetition_penalty=1.15, no_repeat_ngram=3, repeat_window=64)
    return SpeedPreset(top_k=24, lang_top_n=96, repetition_penalty=1.10, no_repeat_ngram=1, repeat_window=32)


class FastOutputEngine:
    def __init__(
        self,
        tokenizer,
        lang_mode: str,
        stream: bool,
        guard: LanguageStabilityGuard | None = None,
        structure_guard: LanguageStructureIntegrityManager | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.lang_mode = lang_mode
        self.stream = stream
        self.guard = guard
        self.structure_guard = structure_guard
        self.meaning_guard = MeaningfulOutputGuard(lang_mode)
        self._decoded_cache: dict[int, str] = {}
        self._allowed_cache: dict[int, bool] = {}
        self._emitted_parts: list[str] = []
        self.simple_sampling = os.getenv("VSPEC_SAMPLING_SIMPLE", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.c_sampler_required = os.getenv("VSPEC_C_SAMPLER_REQUIRED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.phase4_sampler = Phase4SamplingCore()
        self.native_output_guard_enabled = os.getenv("VSPEC_NATIVE_OUTPUT_GUARD", "1").strip().lower() in {"1", "true", "yes", "on"}
        if self.native_output_guard_enabled and native_output_guard_available():
            strictness_raw = os.getenv("VSPEC_NATIVE_OUTPUT_GUARD_STRICTNESS", "0.72")
            try:
                native_output_guard_init(float(strictness_raw))
            except Exception:
                native_output_guard_init(0.72)

    def _prime_decode_cache(self, token_ids: list[int]) -> None:
        if self.tokenizer is None or not token_ids:
            return
        missing = [int(tid) for tid in token_ids if int(tid) not in self._decoded_cache]
        if not missing:
            return

        decode_batch = getattr(self.tokenizer, "decode_batch", None)
        if callable(decode_batch):
            try:
                decoded = decode_batch([[tid] for tid in missing])
                for tid, text in zip(missing, decoded):
                    self._decoded_cache[int(tid)] = text or ""
                return
            except Exception:
                pass

        for tid in missing:
            try:
                self._decoded_cache[int(tid)] = self.tokenizer.decode([int(tid)])
            except Exception:
                self._decoded_cache[int(tid)] = ""

    def begin_stream(self) -> None:
        if self.stream:
            print("[vspec-chat] output:")

    def end_stream(self) -> None:
        if self.stream:
            print()

    def token_text(self, token_id: int) -> str:
        if self.tokenizer is None:
            return ""
        if token_id in self._decoded_cache:
            return self._decoded_cache[token_id]
        text = self.tokenizer.decode([token_id])
        self._decoded_cache[token_id] = text
        return text

    def stream_token(self, token_id: int) -> None:
        text = self.token_text(token_id)
        if self.structure_guard is not None:
            self.structure_guard.observe_text(text)
        self.meaning_guard.observe_text(text)
        if self.native_output_guard_enabled:
            native_output_guard_observe(text)
        self._emitted_parts.append(text)
        if not self.stream:
            return
        if text:
            print(text, end="", flush=True)

    def is_allowed(self, token_id: int) -> bool:
        if self.tokenizer is None:
            return True
        if token_id in self._allowed_cache:
            return self._allowed_cache[token_id]
        text = self.token_text(token_id)
        if self.guard is not None and not self.guard.allow_text(text):
            self._allowed_cache[token_id] = False
            return False
        if self.structure_guard is not None and not self.structure_guard.allow_text(text):
            self._allowed_cache[token_id] = False
            return False
        if not self.meaning_guard.allow_text(text):
            self._allowed_cache[token_id] = False
            return False
        if self.native_output_guard_enabled and (not native_output_guard_allow(text)):
            self._allowed_cache[token_id] = False
            return False
        if self.lang_mode == "auto":
            self._allowed_cache[token_id] = True
            return True
        ok = _is_allowed_text(text, self.lang_mode)
        self._allowed_cache[token_id] = ok
        return ok

    def sample(
        self,
        logits: list[float],
        temperature: float,
        top_k: int,
        greedy: bool,
        lang_top_n: int,
    ) -> int:
        timing_on = _timing_enabled()
        t0 = time.perf_counter() if timing_on else 0.0

        def _finish(token_id: int) -> int:
            if timing_on:
                _SAMPLING_TIMING_STATS["sampling_ms"] += (time.perf_counter() - t0) * 1000.0
                _SAMPLING_TIMING_STATS["sampling_calls"] += 1
            return int(token_id)

        if logits is None:
            return _finish(0)
        if np is not None and isinstance(logits, np.ndarray):
            if logits.size == 0:
                return _finish(0)
        else:
            try:
                if len(logits) == 0:
                    return _finish(0)
            except Exception:
                return _finish(0)

        if np is not None and isinstance(logits, np.ndarray):
            arr = np.asarray(logits, dtype=np.float32)
            if arr.size == 0:
                return _finish(0)
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=-1e9, posinf=1e9, neginf=-1e9)
            logits = arr.tolist()
        else:
            safe_logits: list[float] = []
            for value in logits:
                try:
                    fv = float(value)
                except Exception:
                    fv = -1e9
                if not math.isfinite(fv):
                    fv = -1e9 if fv < 0 else 1e9
                safe_logits.append(fv)
            logits = safe_logits

        if len(logits) == 0:
            return _finish(0)

        if temperature <= 0:
            temperature = 1.0

        candidate_n = max(1, min(len(logits), max(top_k, lang_top_n)))
        candidate_ids = _topn_indices(logits, candidate_n)
        self._prime_decode_cache(candidate_ids)

        allowed_ids = [tid for tid in candidate_ids if self.is_allowed(tid)]
        if not allowed_ids and self.lang_mode in {"vi", "en"}:
            expand_n = candidate_n
            while expand_n < len(logits):
                expand_n = min(len(logits), max(expand_n * 2, expand_n + 64))
                candidate_ids = _topn_indices(logits, expand_n)
                self._prime_decode_cache(candidate_ids)
                allowed_ids = [tid for tid in candidate_ids if self.is_allowed(tid)]
                if allowed_ids:
                    break
        if not allowed_ids:
            allowed_ids = candidate_ids
        if top_k > 0 and len(allowed_ids) > top_k:
            allowed_ids = allowed_ids[:top_k]

        if self.simple_sampling:
            return _finish(max(allowed_ids, key=lambda tid: logits[tid]))

        scored = []
        for tid in allowed_ids:
            scaled = logits[tid] / temperature
            if not math.isfinite(float(scaled)):
                scaled = -1e9
            text = self.token_text(tid)
            bonus = _token_quality_bonus(text, self.lang_mode)
            if self.guard is not None:
                bonus += self.guard.score_adjustment(text)
            if self.structure_guard is not None:
                bonus += self.structure_guard.score_adjustment(text)
            bonus += self.meaning_guard.score_adjustment(text)
            if self.native_output_guard_enabled:
                bonus += native_output_guard_score_adjustment(text)
            final_score = float(scaled + bonus)
            if not math.isfinite(final_score):
                final_score = -1e9
            scored.append((tid, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            return _finish(int(candidate_ids[0]))
        if greedy:
            return _finish(scored[0][0])
        decision = self.phase4_sampler.choose(scored=scored, greedy=False, c_sampler_required=self.c_sampler_required)
        return _finish(int(decision.token_id))

    def structure_report(self) -> dict | None:
        if self.structure_guard is None:
            return None
        return self.structure_guard.report()

    def phase4_sampler_report(self) -> dict:
        return dict(self.phase4_sampler.stats)


def _topn_indices(values: list[float], n: int) -> list[int]:
    if n >= len(values):
        return sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    if np is not None:
        arr = np.asarray(values, dtype=np.float32)
        idx = np.argpartition(arr, -n)[-n:]
        idx = idx[np.argsort(arr[idx])[::-1]]
        return [int(v) for v in idx]
    return [i for i, _ in heapq.nlargest(n, enumerate(values), key=lambda x: x[1])]


def _is_allowed_text(text: str, lang_mode: str) -> bool:
    if not text:
        return True
    if "�" in text:
        return False
    if any(tok in text for tok in ["http", "www", "_", "=", "\\", "/"]):
        return False
    if re.search(r"[A-Z]{3,}", text):
        return False
    if lang_mode == "vi":
        allowed_re = r"^[\sA-Za-z0-9ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ,.;:!?\-_'\"()\[\]{}]+$"
        return re.match(allowed_re, text) is not None
    if lang_mode == "en":
        return re.match(r"^[\sA-Za-z0-9,.;:!?\-_'\"()\[\]{}]+$", text) is not None
    return True


def _token_quality_bonus(text: str, lang_mode: str) -> float:
    if not text:
        return 0.0
    bonus = 0.0
    if len(text) > 20:
        bonus -= 0.8
    if re.search(r"[A-Z]{2,}", text):
        bonus -= 1.0
    if any(ch in text for ch in ["_", "=", "\\", "/", "@"]) or "http" in text:
        bonus -= 1.5
    if lang_mode == "vi" and re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ]", text.lower()):
        bonus += 0.9
    return bonus


def postprocess_output_text(text: str, prompt: str, lang_mode: str) -> str:
    assurance = RuntimeMeaningfulResponseAssurance(lang_mode)
    out = (text or "").strip()
    if not out:
        return assurance.repair(out, prompt)
    if lang_mode != "vi":
        return assurance.repair(out, prompt)
    return assurance.repair(out, prompt)
