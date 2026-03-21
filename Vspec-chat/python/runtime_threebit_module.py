from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except Exception:
    np = None


@dataclass
class ThreeBitSamplingConfig:
    active: bool
    top_k: int
    lang_top_n: int
    repetition_penalty: float
    repeat_window: int


class ThreeBitRuntimeModule:
    def __init__(self, enabled: bool, fused_bits: int, target_bits: int) -> None:
        self.enabled = bool(enabled)
        self.fused_bits = int(fused_bits)
        self.target_bits = int(target_bits)

    def is_active(self) -> bool:
        if not self.enabled:
            return False
        # Only apply 3-bit sampling heuristics when execution is truly 3-bit.
        return self.fused_bits == 3

    def tune_sampling(
        self,
        top_k: int,
        lang_top_n: int,
        repetition_penalty: float,
        repeat_window: int,
    ) -> ThreeBitSamplingConfig:
        if not self.is_active():
            return ThreeBitSamplingConfig(
                active=False,
                top_k=int(top_k),
                lang_top_n=int(lang_top_n),
                repetition_penalty=float(repetition_penalty),
                repeat_window=int(repeat_window),
            )

        tuned_top_k = max(12, int(top_k))
        tuned_lang_top_n = max(tuned_top_k * 4, int(lang_top_n))
        tuned_penalty = max(1.04, float(repetition_penalty) - 0.06)
        tuned_window = max(24, int(repeat_window))
        return ThreeBitSamplingConfig(
            active=True,
            top_k=tuned_top_k,
            lang_top_n=tuned_lang_top_n,
            repetition_penalty=tuned_penalty,
            repeat_window=tuned_window,
        )

    def denoise_logits(self, logits, step: int):
        if not self.is_active():
            return logits
        if logits is None:
            return logits

        if np is not None and isinstance(logits, np.ndarray):
            arr = logits.astype(np.float32, copy=True)
            clip = 14.0 if step < 6 else 18.0
            arr = np.clip(arr, -clip, clip)
            arr = arr - float(np.mean(arr)) * 0.02
            return arr

        vals = list(logits)
        if not vals:
            return vals
        clip = 14.0 if step < 6 else 18.0
        mean_val = sum(vals) / max(1, len(vals))
        out = []
        for v in vals:
            c = v
            if c > clip:
                c = clip
            elif c < -clip:
                c = -clip
            out.append(c - (mean_val * 0.02))
        return out

    def auto_temperature(self, logits, base_temperature: float) -> float:
        if not self.is_active():
            return float(base_temperature)
        if logits is None:
            return float(base_temperature)

        try:
            if np is not None and isinstance(logits, np.ndarray):
                vec = logits.astype(np.float32, copy=False)
                if vec.size == 0:
                    return float(base_temperature)
                top = np.partition(vec, -32)[-32:]
                top = top - float(np.max(top))
                p = np.exp(top)
                p_sum = float(np.sum(p))
                if p_sum <= 0:
                    return float(base_temperature)
                p = p / p_sum
                entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-8))))
            else:
                vals = list(logits)
                if not vals:
                    return float(base_temperature)
                top = sorted(vals)[-32:]
                m = max(top)
                exps = [float(np.exp(v - m)) if np is not None else (2.718281828 ** (v - m)) for v in top]
                s = sum(exps)
                if s <= 0:
                    return float(base_temperature)
                probs = [v / s for v in exps]
                entropy = 0.0
                for p in probs:
                    if p > 1e-8:
                        entropy += -p * (float(np.log(p)) if np is not None else __import__("math").log(p))
        except Exception:
            return float(base_temperature)

        # entropy in ~[0, ln(32)=3.46], map to temperature in [0.58, base]
        ratio = max(0.0, min(1.0, entropy / 3.4657))
        floor_t = 0.58
        target_t = floor_t + (float(base_temperature) - floor_t) * ratio
        return max(0.45, min(float(base_temperature), target_t))
