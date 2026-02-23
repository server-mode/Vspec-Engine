from __future__ import annotations

import re
from dataclasses import dataclass


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = 0
    symbols = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if not ch.isalnum() and ch not in "#-_*`.,:;!?()[]{}'/\"":
            symbols += 1
    if total == 0:
        return 0.0
    return symbols / total


@dataclass
class StructureIntegrityProfile:
    expected_sections: list[int]
    strictness: float


class LanguageStructureIntegrityManager:
    def __init__(self, prompt: str, strictness: float = 0.72) -> None:
        strictness = max(0.0, min(1.0, strictness))
        self.profile = StructureIntegrityProfile(
            expected_sections=self._extract_expected_sections(prompt),
            strictness=strictness,
        )
        self._seen_sections: set[int] = set()
        self._fence_balance = 0
        self._emitted_parts: list[str] = []

    def _extract_expected_sections(self, prompt: str) -> list[int]:
        found = [int(x) for x in re.findall(r"SECTION\s+(\d+)", prompt or "", flags=re.IGNORECASE)]
        uniq = sorted(set(found))
        return uniq

    def observe_text(self, text: str) -> None:
        if not text:
            return
        self._emitted_parts.append(text)
        for section in re.findall(r"#\s*SECTION\s+(\d+)", text, flags=re.IGNORECASE):
            self._seen_sections.add(int(section))
        fence_count = text.count("```")
        if fence_count % 2 == 1:
            self._fence_balance = 1 - self._fence_balance

    def allow_text(self, text: str) -> bool:
        if not text:
            return True
        if "�" in text:
            return False
        if any((ord(ch) < 32 and ch not in ("\t", "\n", "\r")) for ch in text):
            return False

        ratio = _symbol_ratio(text)
        max_ratio = (1.0 - self.profile.strictness) + 0.45
        return ratio <= max_ratio

    def score_adjustment(self, text: str) -> float:
        if not text:
            return 0.0

        score = 0.0
        ratio = _symbol_ratio(text)
        score -= ratio * (1.2 + self.profile.strictness)

        if self.profile.expected_sections:
            remaining = [s for s in self.profile.expected_sections if s not in self._seen_sections]
            if remaining:
                next_section = remaining[0]
                if re.search(rf"#\s*SECTION\s+{next_section}\b", text, flags=re.IGNORECASE):
                    score += 0.45
                elif re.search(r"#\s*SECTION\s+\d+", text, flags=re.IGNORECASE):
                    score += 0.15

        return score

    def report(self) -> dict:
        expected = self.profile.expected_sections
        seen = sorted(self._seen_sections)
        covered = len([s for s in expected if s in self._seen_sections])
        total = len(expected)
        coverage = (covered / total) if total > 0 else 1.0

        return {
            "expected_sections": expected,
            "seen_sections": seen,
            "section_coverage": coverage,
            "code_fence_balanced": self._fence_balance == 0,
            "integrity_pass": bool(coverage >= 0.60 and self._fence_balance == 0),
        }
