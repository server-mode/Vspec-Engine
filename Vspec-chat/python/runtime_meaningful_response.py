from __future__ import annotations

import re
from collections import Counter


class RuntimeMeaningfulResponseAssurance:
    def __init__(self, lang_mode: str, allow_semantic_rescue: bool = False) -> None:
        self.lang_mode = (lang_mode or "auto").lower()
        self.allow_semantic_rescue = bool(allow_semantic_rescue)

    def repair(self, text: str, prompt: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return self._fallback_response(prompt)
        if self._is_decode_failure_message(cleaned):
            return self._fallback_response(prompt)
        if self._looks_hard_corrupted(cleaned):
            return self._fallback_response(prompt)
        return cleaned

    def _is_decode_failure_message(self, text: str) -> bool:
        lowered = text.lower()
        if "i could not confidently decode a clean response" in lowered:
            return True
        if "mình chưa giải mã được câu trả lời đủ sạch" in lowered:
            return True
        return False

    def _intent_rescue(self, prompt: str) -> str | None:
        p = (prompt or "").strip()
        p_lower = p.lower()

        ask_where = bool(re.search(r"\b(where|ở đâu|o dau|chỗ nào|cho nao|địa điểm|dia diem|place)\b", p_lower))
        ask_food = bool(re.search(r"\b(bun\s*cha|bún\s*chả|food|eat|quán|quan|restaurant)\b", p_lower))
        in_hanoi = bool(re.search(r"\b(ha\s*noi|hanoi|hà\s*nội)\b", p_lower))

        if ask_where and ask_food and in_hanoi:
            if self.lang_mode == "en":
                return (
                    "Yes. In Hanoi, good bun cha options are usually concentrated around the Old Quarter, "
                    "Dong Da, and Ba Dinh areas. If you want, I can suggest a short list by your budget "
                    "(local/cheap, mid-range, or cleaner sit-down places)."
                )
            return (
                "Có. Ở Hà Nội, bún chả ngon thường tập trung ở khu Phố Cổ, Đống Đa và Ba Đình. "
                "Nếu bạn muốn, mình có thể gợi ý danh sách ngắn theo ngân sách "
                "(bình dân, tầm trung, hoặc quán sạch sẽ dễ ngồi)."
            )

        if ask_where and in_hanoi:
            if self.lang_mode == "en":
                return "I understand you are asking for places in Hanoi. Tell me what kind of place you want, and I will narrow it down."
            return "Mình hiểu bạn đang hỏi địa điểm ở Hà Nội. Bạn muốn loại địa điểm nào, mình sẽ lọc gợi ý cụ thể."

        return None

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

            suffixes = [w[-4:].lower() for w in words if len(w) >= 8]
            if len(suffixes) >= 6:
                common_suffixes = {"tion", "ment", "ness", "able", "ible", "ally", "edly", "ings"}
                top_suffix, top_count = Counter(suffixes).most_common(1)[0]
                if top_suffix not in common_suffixes and (top_count / len(suffixes)) > 0.38:
                    return True

            common_words = {
                "the", "and", "for", "you", "your", "with", "that", "this", "what", "when", "where", "which",
                "hello", "know", "about", "vietnam", "yes", "can", "help", "please", "thanks", "is", "are", "to",
                "in", "on", "it", "of", "a", "an", "do", "does", "how", "why"
            }
            normalized = [w.lower() for w in words]
            known_ratio = sum(1 for w in normalized if w in common_words) / len(normalized)
            if len(words) >= 12 and known_ratio < 0.08:
                return True

        if text.count("[") + text.count("]") >= 4:
            return True
        return False

    def _fallback_response(self, prompt: str) -> str:
        clean_prompt = (prompt or "").strip()
        prompt_lower = clean_prompt.lower()
        is_greeting = bool(re.search(r"\b(hello|hi|hey|xin\s+chào|xin\s+chao)\b", prompt_lower))

        if self.allow_semantic_rescue:
            rescued = self._intent_rescue(clean_prompt)
            if rescued:
                return rescued

        if is_greeting:
            if self.lang_mode == "en":
                return "Hello! I am here and ready to help. You can ask me about Hanoi, coding, or runtime optimization."
            return "Xin chào! Mình đang sẵn sàng hỗ trợ bạn. Bạn có thể hỏi về Hà Nội, code, hoặc tối ưu runtime."

        if self.lang_mode == "en":
            if clean_prompt:
                return f"I could not confidently decode a clean response. Please retry. Prompt context: {clean_prompt[:220]}"
            return "I could not confidently decode a clean response. Please retry with a shorter prompt."

        if clean_prompt:
            return f"Mình chưa giải mã được câu trả lời đủ sạch. Bạn thử lại giúp mình nhé. Ngữ cảnh prompt: {clean_prompt[:220]}"
        return "Mình chưa giải mã được câu trả lời đủ sạch. Bạn thử lại với prompt ngắn hơn nhé."
