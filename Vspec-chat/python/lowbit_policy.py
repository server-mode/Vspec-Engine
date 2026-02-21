from __future__ import annotations


def build_layer_bits(num_layers: int, target_bits: int) -> list[int]:
    if num_layers <= 0 or target_bits <= 0:
        return []

    target = max(2, min(4, int(target_bits)))
    bits = [target] * num_layers

    if num_layers >= 1:
        bits[0] = min(4, target + 1)
    if num_layers >= 2:
        bits[-1] = min(4, target + 1)

    if target == 2 and num_layers >= 4:
        mid = num_layers // 2
        bits[mid] = 3

    return bits


def effective_bits(bits: list[int]) -> float:
    if not bits:
        return 0.0
    return float(sum(bits)) / float(len(bits))


def summarize_layer_bits(bits: list[int]) -> str:
    if not bits:
        return "none"
    if len(bits) <= 8:
        return ",".join(str(v) for v in bits)
    head = ",".join(str(v) for v in bits[:4])
    tail = ",".join(str(v) for v in bits[-4:])
    return f"{head},...,{tail}"
