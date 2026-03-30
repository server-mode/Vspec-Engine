from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_markdown(path: Path, report: dict) -> None:
    lines = []
    lines.append("# ANF Phase D Dashboard")
    lines.append("")
    lines.append(f"- go_no_go: `{str(report.get('go', False)).lower()}`")
    lines.append(f"- phase_d_all_pass: `{str(report.get('kpi_phase_d_all_pass', False)).lower()}`")
    lines.append("")
    lines.append("## Gate Summary")
    lines.append("")
    lines.append(f"- rollback_gate_pass: `{str(report.get('rollback_gate_pass', False)).lower()}`")
    lines.append(f"- soak_gate_pass: `{str(report.get('soak_gate_pass', False)).lower()}`")
    lines.append(f"- cuda_graph_replay_pass: `{str(report.get('cuda_graph_replay_pass', False)).lower()}`")
    lines.append(f"- cuda_graph_latency_gain_proxy_pass: `{str(report.get('cuda_graph_latency_gain_proxy_pass', False)).lower()}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- gate_json: `{report.get('artifacts', {}).get('gate_json', '')}`")
    lines.append(f"- cuda_graph_json: `{report.get('artifacts', {}).get('cuda_graph_json', '')}`")
    lines.append(f"- dashboard_json: `{report.get('artifacts', {}).get('dashboard_json', '')}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANF Phase D dashboard aggregator")
    parser.add_argument("--gate-json", default="logs/anf_phase_d_gate.json")
    parser.add_argument("--cuda-graph-json", default="logs/anf_phase_d_cuda_graph_smoke.json")
    parser.add_argument("--output", default="logs/anf_phase_d_dashboard.json")
    parser.add_argument("--markdown", default="logs/anf_phase_d_dashboard.md")
    args = parser.parse_args()

    gate = _load_json(Path(args.gate_json))
    cuda_graph = _load_json(Path(args.cuda_graph_json))

    rollback_gate_pass = bool(gate.get("rollback", {}).get("parsed", {}).get("status") == "pass")
    soak_gate_pass = bool(gate.get("soak", {}).get("parsed", {}).get("status") == "pass")
    cuda_graph_replay_pass = bool(cuda_graph.get("kpi_cuda_graph_replay_pass", False))
    cuda_graph_latency_gain_proxy_pass = bool(cuda_graph.get("kpi_cuda_graph_latency_gain_proxy_pass", False))

    all_pass = rollback_gate_pass and soak_gate_pass and cuda_graph_replay_pass and cuda_graph_latency_gain_proxy_pass

    report = {
        "phase": "D",
        "go": bool(gate.get("go", False) and all_pass),
        "rollback_gate_pass": rollback_gate_pass,
        "soak_gate_pass": soak_gate_pass,
        "cuda_graph_replay_pass": cuda_graph_replay_pass,
        "cuda_graph_latency_gain_proxy_pass": cuda_graph_latency_gain_proxy_pass,
        "kpi_phase_d_all_pass": bool(all_pass),
        "artifacts": {
            "gate_json": str(Path(args.gate_json)),
            "cuda_graph_json": str(Path(args.cuda_graph_json)),
            "dashboard_json": str(Path(args.output)),
            "dashboard_md": str(Path(args.markdown)),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_markdown(Path(args.markdown), report)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
