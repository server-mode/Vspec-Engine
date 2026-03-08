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
        salvaged = self._salvage_prefix(cleaned)
        if salvaged and salvaged != cleaned:
            return salvaged
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
        if len(text) < 1:
            return True
        letters = sum(1 for ch in text if ch.isalpha())
        if letters >= 1 and len(text) <= 3:
            return False
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

    def _salvage_prefix(self, text: str) -> str:
        candidate = (text or "").strip()
        if not candidate:
            return ""

        pieces = [seg.strip() for seg in re.split(r"[\r\n]+", candidate) if seg.strip()]
        if not pieces:
            return ""

        first = pieces[0]
        if self._is_reasonable_prefix(first):
            return first

        sentence_match = re.match(r"^(.{2,160}?[.!?])(?:\s|$)", candidate)
        if sentence_match is not None:
            first_sentence = sentence_match.group(1).strip()
            if self._is_reasonable_prefix(first_sentence):
                return first_sentence

        short_clause = candidate.split(",", 1)[0].strip()
        if self._is_reasonable_prefix(short_clause):
            return short_clause

        return ""

    def _is_reasonable_prefix(self, text: str) -> bool:
        sample = (text or "").strip()
        if len(sample) < 2 or len(sample) > 160:
            return False
        if "�" in sample:
            return False
        if sample.lower().startswith("[vspec-decode-error]"):
            return False
        letters = sum(1 for ch in sample if ch.isalpha())
        if letters < 2:
            return False
        if (letters / max(1, len(sample))) < 0.35:
            return False
        words = re.findall(r"[A-Za-z]{2,}", sample)
        if not words and letters < 4:
            return False
        if len(words) == 1 and len(words[0]) > 20:
            return False
        if self._looks_hard_corrupted(sample):
            return False
        return True

    def _decode_error_response(self) -> str:
        if self.lang_mode == "en":
            return "[vspec-decode-error] Generation failed on this turn; no synthetic fallback answer was produced. Please retry or reduce decode pressure (tokens/layers)."
        return "[vspec-decode-error] Lượt giải mã này thất bại; hệ thống không tự tạo câu trả lời thay thế. Vui lòng thử lại hoặc giảm tải decode (tokens/layers)."
