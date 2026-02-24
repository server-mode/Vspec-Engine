import heapq
import math
import random
import re
from dataclasses import dataclass

from language_stability_guard import LanguageStabilityGuard
from language_structure_guard import LanguageStructureIntegrityManager
from meaningful_output_guard import MeaningfulOutputGuard
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
        if temperature <= 0:
            temperature = 1.0

        candidate_n = max(1, min(len(logits), max(top_k, lang_top_n)))
        candidate_ids = _topn_indices(logits, candidate_n)

        allowed_ids = [tid for tid in candidate_ids if self.is_allowed(tid)]
        if not allowed_ids and self.lang_mode in {"vi", "en"}:
            expand_n = candidate_n
            while expand_n < len(logits):
                expand_n = min(len(logits), max(expand_n * 2, expand_n + 64))
                candidate_ids = _topn_indices(logits, expand_n)
                allowed_ids = [tid for tid in candidate_ids if self.is_allowed(tid)]
                if allowed_ids:
                    break
        if not allowed_ids:
            allowed_ids = candidate_ids
        if top_k > 0 and len(allowed_ids) > top_k:
            allowed_ids = allowed_ids[:top_k]

        scored = []
        for tid in allowed_ids:
            scaled = logits[tid] / temperature
            text = self.token_text(tid)
            bonus = _token_quality_bonus(text, self.lang_mode)
            if self.guard is not None:
                bonus += self.guard.score_adjustment(text)
            if self.structure_guard is not None:
                bonus += self.structure_guard.score_adjustment(text)
            bonus += self.meaning_guard.score_adjustment(text)
            scored.append((tid, scaled + bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        if greedy:
            return scored[0][0]

        max_logit = scored[0][1]
        exp_vals = [math.exp(v - max_logit) for _, v in scored]
        total = sum(exp_vals)
        if total <= 0:
            return scored[0][0]
        r = random.random()
        acc = 0.0
        for i, (_, v) in enumerate(scored):
            acc += exp_vals[i] / total
            if r <= acc:
                return scored[i][0]
        return scored[-1][0]

    def structure_report(self) -> dict | None:
        if self.structure_guard is None:
            return None
        return self.structure_guard.report()


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

    lower = out.lower()
    vi_marks = re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ]", lower) is not None
    has_common_vi = any(w in lower for w in ["xin", "chào", "ban", "bạn", "toi", "tôi", "giup", "giúp", "cam", "cảm"])
    looks_weird = re.search(r"\b[a-z]{6,}\b", lower) is not None and not vi_marks and not has_common_vi
    ascii_only_word = re.fullmatch(r"[a-z\s]{1,8}", lower) is not None and not vi_marks and not has_common_vi

    p = (prompt or "").lower()
    is_greeting = bool(re.search(r"\b(xin\s+chào|xin\s+chao|hello|hi)\b", p, flags=re.IGNORECASE))
    if is_greeting and (looks_weird or ascii_only_word):
        return "Xin chào! Mình có thể giúp gì cho bạn?"
    return assurance.repair(out, prompt)
