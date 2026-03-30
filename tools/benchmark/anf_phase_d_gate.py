from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _run_exe(exe_path: Path, timeout_s: int) -> tuple[int, str]:
    proc = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    merged = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return int(proc.returncode), merged.strip()


def _extract_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def _extract_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1))


def _parse_rollback(output: str) -> dict:
    return {
        "pre_mode": _extract_int(r"pre_mode=(\d+)", output),
        "pre_deescalate_count": _extract_int(r"pre_deescalate_count=(\d+)", output),
        "pre_forced_fallback_count": _extract_int(r"pre_forced_fallback_count=(\d+)", output),
        "rollback_ms": _extract_float(r"rollback_ms=([0-9]+\.[0-9]+)", output),
        "final_mode": _extract_int(r"final_mode=(\d+)", output),
        "silent_stop_count": _extract_int(r"silent_stop_count=(\d+)", output),
        "status": "pass" if "status=pass" in output else "fail",
    }


def _parse_soak(output: str) -> dict:
    turns = _extract_int(r"turns=(\d+)", output)
    forced_fallback_count = _extract_int(r"forced_fallback_count=(\d+)", output)
    forced_fallback_limit = _extract_int(r"limit=(\d+)", output)
    return {
        "turns": turns,
        "crash_count": _extract_int(r"crash_count=(\d+)", output),
        "silent_stop_count": _extract_int(r"silent_stop_count=(\d+)", output),
        "forced_fallback_count": forced_fallback_count,
        "forced_fallback_limit": forced_fallback_limit,
        "deescalate_count": _extract_int(r"deescalate_count=(\d+)", output),
        "final_mode": _extract_int(r"final_mode=(\d+)", output),
        "status": "pass" if "status=pass" in output else "fail",
    }


def _compute_go_nogo(rollback: dict, soak: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    rollback_pass = (
        rollback.get("status") == "pass"
        and rollback.get("rollback_ms") is not None
        and rollback.get("rollback_ms") < 60000.0
        and rollback.get("final_mode") == 0
        and rollback.get("silent_stop_count") == 0
    )
    if not rollback_pass:
        reasons.append("rollback gate failed")

    soak_pass = (
        soak.get("status") == "pass"
        and soak.get("turns") == 1000
        and soak.get("crash_count") == 0
        and soak.get("silent_stop_count") == 0
        and soak.get("forced_fallback_count") is not None
        and soak.get("forced_fallback_limit") is not None
        and soak.get("forced_fallback_count") <= soak.get("forced_fallback_limit")
    )
    if not soak_pass:
        reasons.append("soak gate failed")

    return bool(rollback_pass and soak_pass), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="ANF Phase D go/no-go gate runner")
    parser.add_argument("--build-dir", default="build/Release")
    parser.add_argument("--output", default="logs/anf_phase_d_gate.json")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    rollback_exe = build_dir / "vspec_anf_phase_d_rollback_smoke.exe"
    soak_exe = build_dir / "vspec_anf_phase_d_soak_1000.exe"

    gate_report: dict = {
        "phase": "D",
        "build_dir": str(build_dir),
        "executables": {
            "rollback": str(rollback_exe),
            "soak": str(soak_exe),
        },
        "rollback": {},
        "soak": {},
        "go": False,
        "reasons": [],
    }

    missing = []
    if not rollback_exe.exists():
        missing.append(str(rollback_exe))
    if not soak_exe.exists():
        missing.append(str(soak_exe))

    if missing:
        gate_report["reasons"] = ["missing executable"] + missing
    else:
        rb_code, rb_out = _run_exe(rollback_exe, args.timeout)
        sk_code, sk_out = _run_exe(soak_exe, args.timeout)

        gate_report["rollback"] = {
            "exit_code": rb_code,
            "raw_output": rb_out,
            "parsed": _parse_rollback(rb_out),
        }
        gate_report["soak"] = {
            "exit_code": sk_code,
            "raw_output": sk_out,
            "parsed": _parse_soak(sk_out),
        }

        go, reasons = _compute_go_nogo(
            gate_report["rollback"]["parsed"],
            gate_report["soak"]["parsed"],
        )
        gate_report["go"] = go
        gate_report["reasons"] = reasons

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(gate_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
