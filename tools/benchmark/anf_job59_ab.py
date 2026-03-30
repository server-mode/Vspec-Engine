from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_lane(repo_root: Path, lane: str) -> dict:
    env = os.environ.copy()
    if lane == "prototype":
        env["VSPEC_CHAT_PROTOTYPE"] = "1"
        env["VSPEC_ENABLE_ANF"] = "1"
        env.setdefault("VSPEC_ANF_MODE", "active")
    else:
        env["VSPEC_CHAT_PROTOTYPE"] = "0"
        env["VSPEC_ENABLE_ANF"] = "0"

    code = (
        "import json,sys,numpy as np;"
        "sys.path.insert(0,r'" + str(repo_root / "Vspec-chat" / "python").replace("\\", "\\\\") + "');"
        "from runtime_core_bridge import native_anf_prototype_enabled,native_anf_observe_activations,native_anf_observe_quality,native_anf_report;"
        "enabled=native_anf_prototype_enabled();"
        "a=np.random.uniform(-1,1,(1,2048)).astype('float32');"
        "native_anf_observe_activations(a) if enabled else None;"
        "native_anf_observe_activations(a) if enabled else None;"
        "native_anf_observe_quality(0.12,0.05,0.08) if enabled else None;"
        "rep=native_anf_report() or {};"
        "print(json.dumps({'lane':'" + lane + "','prototype_enabled':bool(enabled),'report':rep},ensure_ascii=False))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"lane={lane} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
    return payload


def _evaluate(default_payload: dict, prototype_payload: dict) -> dict:
    d = default_payload.get("report", {}) or {}
    p = prototype_payload.get("report", {}) or {}
    checks = {
        "default_gate_off": not bool(default_payload.get("prototype_enabled", False)),
        "prototype_gate_on": bool(prototype_payload.get("prototype_enabled", False)),
        "default_tokens_zero": int(d.get("tokens_observed", 0)) == 0,
        "prototype_tokens_positive": int(p.get("tokens_observed", 0)) > 0,
        "prototype_skip_ratio_positive": float(p.get("skip_ratio_avg", 0.0)) > 0.0,
        "prototype_hot_ratio_target": float(p.get("hot_ratio_avg", 1.0)) <= 0.15,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Job59 ANF A/B sanity: default lane vs prototype lane")
    parser.add_argument("--out", default="logs/anf_job59_ab.json", help="Output JSON path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default_payload = _run_lane(repo_root, "default")
    prototype_payload = _run_lane(repo_root, "prototype")
    summary = _evaluate(default_payload, prototype_payload)

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "default": default_payload,
        "prototype": prototype_payload,
        "summary": summary,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
