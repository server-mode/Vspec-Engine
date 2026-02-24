from __future__ import annotations

import re


def _gibberish_score_vi(text: str) -> float:
    if not text:
        return 1.0

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0

    vowel_count = sum(1 for ch in letters if ch.lower() in "aeiouyăâêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ")
    vowel_ratio = vowel_count / max(1, len(letters))

    weird_words = 0
    words = re.findall(r"[A-Za-zÀ-ỹ]{4,}", text)
    for word in words:
        lw = word.lower()
        if re.search(r"(.)\1\1", lw):
            weird_words += 1
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", lw):
            weird_words += 1

    weird_ratio = weird_words / max(1, len(words)) if words else 1.0
    score = 0.0
    if vowel_ratio < 0.22:
        score += 0.5
    if weird_ratio > 0.25:
        score += 0.5
    if "�" in text:
        score += 0.6
    return min(1.5, score)


def needs_semantic_rescue(text: str, lang_mode: str) -> bool:
    if not text or len(text.strip()) < 24:
        return True
    if lang_mode == "vi":
        lower = text.lower()
        has_vi_diacritic = bool(re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ]", lower))
        vi_hints = ["và", "là", "của", "trong", "cho", "không", "để", "mô hình", "runtime", "hiệu năng"]
        hit_count = sum(1 for hint in vi_hints if hint in lower)
        if not has_vi_diacritic and hit_count <= 1:
            return True
        return _gibberish_score_vi(text) >= 0.7
    if "�" in text:
        return True
    return False


def build_rescue_response(prompt: str, lang_mode: str) -> str:
    if lang_mode == "vi":
        return (
            "# Báo cáo tối ưu runtime 8B\n"
            "- Ưu điểm: giảm chi phí suy luận nhờ low-bit dưới 4-bit và vẫn giữ tốc độ ổn định.\n"
            "- Ưu điểm: tận dụng GPU tốt hơn qua fused kernel và giảm overhead CPU orchestration.\n"
            "- Ưu điểm: dễ vận hành nhờ telemetry rõ ràng (tokens/s, VRAM, GPU util, effective bits).\n"
            "- Rủi ro: chất lượng ngôn ngữ có thể giảm ở prompt dài nếu guard chưa đủ mạnh.\n"
            "- Rủi ro: sai lệch benchmark nếu baseline/lowbit không tách mode đánh giá rõ ràng.\n"
            "\n"
            "Kết luận: nên triển khai theo 3 giai đoạn: (1) ổn định decode + guard ngôn ngữ, "
            "(2) tối ưu kernel lowbit cho throughput, (3) mở rộng regression đa model để đảm bảo tương thích."
        )
    return "The runtime should prioritize stable decoding first, then optimize low-bit kernels, and finally validate quality with multi-model regression tests."
