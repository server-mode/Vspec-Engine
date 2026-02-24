from __future__ import annotations

import re


_VOWELS = set("aeiouyAEIOUYăâêôơưĂÂÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ")


class MeaningfulOutputGuard:
    def __init__(self, lang_mode: str) -> None:
        self.lang_mode = lang_mode
        self._recent_text = ""

    def observe_text(self, text: str) -> None:
        if not text:
            return
        self._recent_text = (self._recent_text + text)[-512:]

    def allow_text(self, text: str) -> bool:
        if not text:
            return True
        if "�" in text:
            return False
        if any((ord(ch) < 32 and ch not in ("\t", "\n", "\r")) for ch in text):
            return False

        if re.search(r"(.)\1\1\1", text):
            return False

        letters = [ch for ch in text if ch.isalpha()]
        if len(letters) >= 6:
            vowel_count = sum(1 for ch in letters if ch in _VOWELS)
            vowel_ratio = vowel_count / max(1, len(letters))
            if vowel_ratio < 0.12:
                return False

        if self._recent_text and len(text.strip()) >= 4 and text in self._recent_text[-96:]:
            return False

        return True

    def score_adjustment(self, text: str) -> float:
        if not text:
            return 0.0

        score = 0.0
        if re.search(r"([A-Za-z]{3,})\1", text):
            score -= 1.2

        letters = [ch for ch in text if ch.isalpha()]
        if len(letters) >= 5:
            vowel_count = sum(1 for ch in letters if ch in _VOWELS)
            vowel_ratio = vowel_count / max(1, len(letters))
            if vowel_ratio < 0.2:
                score -= 0.8

        if self._recent_text:
            tail = self._recent_text[-48:]
            if text and text in tail:
                score -= 0.9

        if self.lang_mode == "vi" and re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ]", text.lower()):
            score += 0.25

        return score
