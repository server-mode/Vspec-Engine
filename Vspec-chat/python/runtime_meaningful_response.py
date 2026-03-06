from __future__ import annotations

import re


class RuntimeMeaningfulResponseAssurance:
    def __init__(self, lang_mode: str, allow_semantic_rescue: bool = False) -> None:
        self.lang_mode = (lang_mode or "auto").lower()
        self.allow_semantic_rescue = bool(allow_semantic_rescue)

    def repair(self, text: str, prompt: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return self._decode_error_response()
        if self._is_decode_failure_message(cleaned):
            return self._decode_error_response()
        if self._looks_hard_corrupted(cleaned):
            return self._decode_error_response()
        return cleaned

    def _is_decode_failure_message(self, text: str) -> bool:
        lowered = text.lower()
        if "i could not confidently decode a clean response" in lowered:
            return True
        if "mình chưa giải mã được câu trả lời đủ sạch" in lowered:
            return True
        return False

    def _looks_hard_corrupted(self, text: str) -> bool:
        if "�" in text:
            return True
        if len(text) < 2:
            return True
        letters = sum(1 for ch in text if ch.isalpha())
        if len(text) >= 40 and (letters / max(1, len(text))) < 0.18:
            return True

        words = re.findall(r"[A-Za-z]{2,}", text)
        if len(words) == 1 and len(words[0]) >= 20:
            return True
        if 1 <= len(words) <= 3 and max(len(w) for w in words) >= 14:
            return True
        single_letter_words = re.findall(r"\b[A-Za-z]\b", text)
        total_word_like = len(words) + len(single_letter_words)
        if len(text) >= 80 and total_word_like >= 16:
            single_ratio = len(single_letter_words) / max(1, total_word_like)
            punct = sum(1 for ch in text if (not ch.isalnum()) and (not ch.isspace()))
            punct_ratio = punct / max(1, len(text))
            if single_ratio > 0.22 and punct_ratio > 0.08:
                return True
        if len(words) >= 8:
            long_ratio = sum(1 for w in words if len(w) >= 12) / len(words)
            if long_ratio > 0.35:
                return True

        if text.count("[") + text.count("]") >= 4:
            return True
        return False

    def _decode_error_response(self) -> str:
        if self.lang_mode == "en":
            return "[vspec-decode-error] Generation failed on this turn; no synthetic fallback answer was produced. Please retry or reduce decode pressure (tokens/layers)."
        return "[vspec-decode-error] Lượt giải mã này thất bại; hệ thống không tự tạo câu trả lời thay thế. Vui lòng thử lại hoặc giảm tải decode (tokens/layers)."
