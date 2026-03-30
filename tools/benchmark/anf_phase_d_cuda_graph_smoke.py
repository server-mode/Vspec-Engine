from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
sys.path.insert(0, str(CHAT_PY))

CoreNativeDecodeLoop = importlib.import_module("runtime_core_bridge").CoreNativeDecodeLoop


def _run_case(capture_enabled: bool, rounds: int, unique_signatures: int) -> dict:
    os.environ["VSPEC_CUDA_GRAPH_CAPTURE"] = "1" if capture_enabled else "0"
    loop = CoreNativeDecodeLoop(
        total_vram_bytes=2 * 1024 * 1024 * 1024,
        max_active=2,
        max_batch_tokens=16,
        token_quantum=1,
    )
    if not loop.available:
        return {
            "available": False,
            "elapsed_ms": 0.0,
            "stats": {},
        }

    begin = time.perf_counter()
    ok_steps = 0
    for i in range(rounds):
        graph_signature = 0xD0000000 + (i % max(1, unique_signatures))
        if not loop.begin(prompt_tokens=64, max_new_tokens=1, graph_signature=graph_signature, priority=1):
            continue
        quota = loop.next_quota()
        if quota <= 0:
            loop.cancel()
            continue
        if not loop.commit(generated_tokens=1, reached_eos=True):
            loop.cancel()
            continue
        ok_steps += 1
    elapsed_ms = (time.perf_counter() - begin) * 1000.0

    stats = loop.stats()
    loop.close()
    return {
        "available": True,
        "elapsed_ms": elapsed_ms,
        "ok_steps": ok_steps,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ANF Phase D CUDA graph capture/replay smoke")
    parser.add_argument("--rounds", type=int, default=320)
    parser.add_argument("--unique-signatures", type=int, default=8)
    parser.add_argument("--output", default="logs/anf_phase_d_cuda_graph_smoke.json")
    args = parser.parse_args()

    run_on = _run_case(True, args.rounds, args.unique_signatures)
    run_off = _run_case(False, args.rounds, args.unique_signatures)

    report: dict = {
        "phase": "D",
        "benchmark": "cuda_graph_smoke",
        "rounds": int(args.rounds),
        "unique_signatures": int(args.unique_signatures),
        "capture_on": run_on,
        "capture_off": run_off,
        "kpi_cuda_graph_replay_pass": False,
        "kpi_cuda_graph_latency_gain_proxy_pass": False,
        "status": "fail",
    }

    if run_on.get("available") and run_off.get("available"):
        stats_on = run_on.get("stats", {})
        stats_off = run_off.get("stats", {})

        steps_on = max(1, int(stats_on.get("steps", 0)))
        replay_on = int(stats_on.get("graph_replays", 0))
        captures_on = int(stats_on.get("graph_captures", 0))
        cached_on = int(stats_on.get("graph_cached_signatures", 0))
        enabled_on = int(stats_on.get("graph_capture_enabled", 0))

        enabled_off = int(stats_off.get("graph_capture_enabled", 0))
        replay_off = int(stats_off.get("graph_replays", 0))

        replay_coverage_on = float(replay_on) / float(steps_on)
        replay_coverage_off = float(replay_off) / float(max(1, int(stats_off.get("steps", 0))))
        replay_gain_proxy_pct = (replay_coverage_on - replay_coverage_off) * 100.0

        report["metrics"] = {
            "replay_coverage_on": replay_coverage_on,
            "replay_coverage_off": replay_coverage_off,
            "replay_gain_proxy_pct": replay_gain_proxy_pct,
            "elapsed_per_step_on_ms": float(run_on.get("elapsed_ms", 0.0)) / float(max(1, int(run_on.get("ok_steps", 0)))),
            "elapsed_per_step_off_ms": float(run_off.get("elapsed_ms", 0.0)) / float(max(1, int(run_off.get("ok_steps", 0)))),
        }

        replay_ok = (
            enabled_on == 1
            and enabled_off == 0
            and captures_on >= int(args.unique_signatures)
            and cached_on >= int(args.unique_signatures)
            and replay_on >= int(args.rounds)
            and replay_off == 0
        )
        gain_proxy_ok = replay_gain_proxy_pct >= 90.0

        report["kpi_cuda_graph_replay_pass"] = bool(replay_ok)
        report["kpi_cuda_graph_latency_gain_proxy_pass"] = bool(gain_proxy_ok)
        report["status"] = "pass" if (replay_ok and gain_proxy_ok) else "fail"
    else:
        report["reasons"] = ["native decode loop bridge unavailable"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
