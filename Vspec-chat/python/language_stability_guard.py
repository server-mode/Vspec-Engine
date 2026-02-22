from __future__ import annotations

import unicodedata
from dataclasses import dataclass


COMMON_SCRIPTS = {"Common", "Inherited"}

LANG_SCRIPT_HINTS: dict[str, set[str]] = {
    "en": {"Latin"},
    "vi": {"Latin"},
    "fr": {"Latin"},
    "de": {"Latin"},
    "es": {"Latin"},
    "it": {"Latin"},
    "pt": {"Latin"},
    "tr": {"Latin"},
    "pl": {"Latin"},
    "nl": {"Latin"},
    "sv": {"Latin"},
    "no": {"Latin"},
    "da": {"Latin"},
    "fi": {"Latin"},
    "id": {"Latin"},
    "ms": {"Latin"},
    "ro": {"Latin"},
    "cs": {"Latin"},
    "sk": {"Latin"},
    "hu": {"Latin"},
    "hr": {"Latin"},
    "sr": {"Cyrillic", "Latin"},
    "ru": {"Cyrillic"},
    "uk": {"Cyrillic"},
    "bg": {"Cyrillic"},
    "el": {"Greek"},
    "ar": {"Arabic"},
    "fa": {"Arabic"},
    "ur": {"Arabic"},
    "he": {"Hebrew"},
    "hi": {"Devanagari"},
    "bn": {"Bengali"},
    "ta": {"Tamil"},
    "te": {"Telugu"},
    "kn": {"Kannada"},
    "ml": {"Malayalam"},
    "mr": {"Devanagari"},
    "pa": {"Gurmukhi"},
    "gu": {"Gujarati"},
    "or": {"Oriya"},
    "th": {"Thai"},
    "lo": {"Lao"},
    "my": {"Myanmar"},
    "ka": {"Georgian"},
    "hy": {"Armenian"},
    "am": {"Ethiopic"},
    "zh": {"Han"},
    "ja": {"Han", "Hiragana", "Katakana"},
    "ko": {"Hangul", "Han"},
}


@dataclass
class LanguageGuardProfile:
    primary_script: str
    allowed_scripts: set[str]
    strictness: float
    prioritized_english: bool


class LanguageStabilityGuard:
    def __init__(
        self,
        prompt: str,
        lang_mode: str,
        strictness: float = 0.72,
        prioritize_english: bool = True,
    ) -> None:
        strictness = max(0.0, min(1.0, strictness))
        primary, allowed = _resolve_profile(prompt, lang_mode, prioritize_english)
        self.profile = LanguageGuardProfile(
            primary_script=primary,
            allowed_scripts=allowed,
            strictness=strictness,
            prioritized_english=prioritize_english,
        )

    def allow_text(self, text: str) -> bool:
        if not text:
            return True
        if "�" in text:
            return False
        total, mismatch, mixed = _script_mismatch(text, self.profile.allowed_scripts)
        if total == 0:
            return True

        mismatch_ratio = mismatch / total
        mixed_ratio = mixed / total
        max_mismatch = (1.0 - self.profile.strictness) + 0.08
        max_mixed = (1.0 - self.profile.strictness) + 0.12
        return mismatch_ratio <= max_mismatch and mixed_ratio <= max_mixed

    def score_adjustment(self, text: str) -> float:
        if not text:
            return 0.0
        total, mismatch, mixed = _script_mismatch(text, self.profile.allowed_scripts)
        if total == 0:
            return 0.0

        mismatch_ratio = mismatch / total
        mixed_ratio = mixed / total
        score = 0.0
        score -= mismatch_ratio * (2.0 + self.profile.strictness)
        score -= mixed_ratio * 1.6

        dominant = _dominant_script(text)
        if dominant == self.profile.primary_script:
            score += 0.25
        return score


def _resolve_profile(prompt: str, lang_mode: str, prioritize_english: bool) -> tuple[str, set[str]]:
    normalized = (lang_mode or "auto").strip().lower()
    lang_code = normalized.split("-", 1)[0]

    if lang_code in LANG_SCRIPT_HINTS:
        preferred = set(LANG_SCRIPT_HINTS[lang_code])
        primary = sorted(preferred)[0]
        return primary, preferred | COMMON_SCRIPTS

    dominant = _dominant_script(prompt)
    if dominant:
        return dominant, {dominant} | COMMON_SCRIPTS

    if prioritize_english:
        return "Latin", {"Latin"} | COMMON_SCRIPTS

    return "Latin", {"Latin"} | COMMON_SCRIPTS


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        script = _char_script(ch)
        if script in COMMON_SCRIPTS:
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _script_mismatch(text: str, allowed_scripts: set[str]) -> tuple[int, int, int]:
    total = 0
    mismatch = 0
    mixed = 0
    dominant = _dominant_script(text)
    for ch in text:
        script = _char_script(ch)
        if script in COMMON_SCRIPTS:
            continue
        total += 1
        if script not in allowed_scripts:
            mismatch += 1
        if dominant and script != dominant:
            mixed += 1
    return total, mismatch, mixed


def _char_script(ch: str) -> str:
    if not ch or ch.isspace() or ch.isdigit():
        return "Common"

    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF:
        return "Han"
    if 0x3040 <= code <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= code <= 0x30FF:
        return "Katakana"
    if 0xAC00 <= code <= 0xD7AF:
        return "Hangul"

    name = unicodedata.name(ch, "")
    if not name:
        return "Common"

    for script_name in (
        "LATIN",
        "CYRILLIC",
        "GREEK",
        "ARABIC",
        "HEBREW",
        "DEVANAGARI",
        "BENGALI",
        "TAMIL",
        "TELUGU",
        "KANNADA",
        "MALAYALAM",
        "GURMUKHI",
        "GUJARATI",
        "ORIYA",
        "THAI",
        "LAO",
        "MYANMAR",
        "GEORGIAN",
        "ARMENIAN",
        "ETHIOPIC",
    ):
        if script_name in name:
            return script_name.title()

    if "PUNCTUATION" in name or "SYMBOL" in name:
        return "Common"
    return "Common"
