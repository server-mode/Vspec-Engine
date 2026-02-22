from __future__ import annotations

import subprocess
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _cuda_mem_info() -> tuple[int, int] | None:
    try:
        from vspec_cuda_bridge import cuda_mem_info
    except Exception:
        return None
    try:
        return cuda_mem_info()
    except Exception:
        return None


def _query_nvidia_smi() -> dict[str, Any] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2)
    except Exception:
        return None

    line = ""
    for row in out.splitlines():
        row = row.strip()
        if row:
            line = row
            break
    if not line:
        return None

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None

    return {
        "device_name": parts[0],
        "gpu_utilization_pct": _safe_float(parts[1]),
        "memory_utilization_pct": _safe_float(parts[2]),
        "memory_used_mib": _safe_float(parts[3]),
        "memory_total_mib": _safe_float(parts[4]),
        "source": "nvidia-smi",
    }


def capture_hardware_snapshot(runtime: Any = None, backend_hint: str | None = None) -> dict[str, Any]:
    backend = backend_hint or "unknown"
    if runtime is not None and hasattr(runtime, "use_native_cuda_norm"):
        backend = "cuda-native" if bool(getattr(runtime, "use_native_cuda_norm", False)) else "cpu"

    info: dict[str, Any] = {
        "backend": backend,
        "device_name": "unknown",
        "gpu_utilization_pct": None,
        "memory_utilization_pct": None,
        "vram_used_bytes": None,
        "vram_total_bytes": None,
        "vram_utilization_pct": None,
        "fused_bits": int(getattr(runtime, "fused_bits", 0)) if runtime is not None and hasattr(runtime, "fused_bits") else None,
    }

    smi = _query_nvidia_smi()
    if smi is not None:
        info.update({
            "device_name": smi.get("device_name", "unknown"),
            "gpu_utilization_pct": smi.get("gpu_utilization_pct"),
            "memory_utilization_pct": smi.get("memory_utilization_pct"),
            "source": smi.get("source"),
        })

    mem = _cuda_mem_info()
    if mem is not None:
        free_b, total_b = mem
        used_b = int(total_b - free_b)
        info["vram_used_bytes"] = used_b
        info["vram_total_bytes"] = int(total_b)
        info["vram_utilization_pct"] = (float(used_b) / float(total_b) * 100.0) if total_b > 0 else None
        if info.get("memory_utilization_pct") is None and info["vram_utilization_pct"] is not None:
            info["memory_utilization_pct"] = info["vram_utilization_pct"]

    if info["device_name"] == "unknown" and info["backend"] == "cuda-native":
        info["device_name"] = "NVIDIA CUDA"

    return info


def summarize_hardware_usage(snapshot: dict[str, Any]) -> str:
    backend = snapshot.get("backend", "unknown")
    name = snapshot.get("device_name", "unknown")
    gpu = snapshot.get("gpu_utilization_pct")
    mem = snapshot.get("memory_utilization_pct")

    gpu_str = f"{gpu:.1f}%" if isinstance(gpu, (int, float)) else "n/a"
    mem_str = f"{mem:.1f}%" if isinstance(mem, (int, float)) else "n/a"
    return f"backend={backend}; device={name}; gpu_util={gpu_str}; mem_util={mem_str}"


def build_hardware_report(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    report = {
        "before": before,
        "after": after,
        "vram_delta_bytes": None,
        "gpu_utilization_after_pct": after.get("gpu_utilization_pct"),
        "memory_utilization_after_pct": after.get("memory_utilization_pct"),
    }

    b = before.get("vram_used_bytes")
    a = after.get("vram_used_bytes")
    if isinstance(b, int) and isinstance(a, int):
        report["vram_delta_bytes"] = int(a - b)

    return report
