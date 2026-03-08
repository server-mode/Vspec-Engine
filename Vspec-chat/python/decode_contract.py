from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class DecodeContractStatus:
    ok: bool
    logits_len: int
    expected_vocab_size: int
    masked_tail: int = 0
    reason: str = ""


def _logits_len(logits) -> int:
    if logits is None:
        return 0
    if np is not None and isinstance(logits, np.ndarray):
        return int(logits.size)
    try:
        return int(len(logits))
    except Exception:
        return 0


def sanitize_and_validate_logits(logits, expected_vocab_size: int):
    expected = int(max(0, expected_vocab_size))
    actual = _logits_len(logits)

    if actual <= 0:
        return logits, DecodeContractStatus(False, actual, expected, 0, "empty-logits")

    if expected > 0:
        min_ok = max(4096, int(expected * 0.50))
        max_ok = max(expected * 2, expected + 65536)
        if actual < min_ok:
            return logits, DecodeContractStatus(False, actual, expected, 0, "logits-too-small")
        if actual > max_ok:
            return logits, DecodeContractStatus(False, actual, expected, 0, "logits-too-large")

    masked_tail = 0
    if expected > 0 and actual > expected:
        masked_tail = actual - expected
        if np is not None and isinstance(logits, np.ndarray):
            logits = logits.copy()
            logits[expected:] = -1e9
        else:
            try:
                adjusted = list(logits)
                for idx in range(expected, len(adjusted)):
                    adjusted[idx] = -1e9
                logits = adjusted
            except Exception:
                pass

    return logits, DecodeContractStatus(True, actual, expected, masked_tail, "ok")