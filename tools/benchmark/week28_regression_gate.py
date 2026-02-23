from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "week25": ("week25_scheduler_stress_week28.json", "kpi_week25_pass"),
    "week26": ("week26_quality_perplexity_gate_week28.json", "kpi_week26_pass"),
    "week27": ("week27_performance_consolidation_week28.json", "kpi_week27_pass"),
}


def _load_json(path: Path) -> dict:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = path.read_text(encoding=encoding, errors="strict")
            text = text.strip("\ufeff\x00 \t\r\n")
            if text:
                return json.loads(text)
        except Exception:
            continue
    raise ValueError(f"cannot parse json artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 28 regression gate aggregator")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output", default="logs/week28_regression_gate.json")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    checks: dict[str, dict] = {}
    pass_all = True

    for name, (filename, key) in REQUIRED.items():
        fp = logs_dir / filename
        status = {
            "file": str(fp),
            "exists": fp.exists(),
            "pass_key": key,
            "pass": False,
        }
        if fp.exists():
            data = _load_json(fp)
            status["pass"] = bool(data.get(key, False))
        else:
            status["pass"] = False
        checks[name] = status
        pass_all = pass_all and status["pass"]

    report = {
        "week": 28,
        "checks": checks,
        "kpi_regression_all_pass": bool(pass_all),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
