import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "Release" / "vspec_benchmark.exe"
STRESS = ROOT / "build" / "Release" / "vspec_stress_test.exe"


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout.strip()


def main() -> None:
    out = {
        "benchmark": run([str(BIN)]),
        "stress": run([str(STRESS)]),
    }
    report = ROOT / "tools" / "benchmark" / "report.json"
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
