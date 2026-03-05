from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _find_benchmark_binary(root: Path) -> Path | None:
    candidates = [
        root / "build" / "Release" / "vspec_unified_runtime_benchmark.exe",
        root / "build" / "Release" / "vspec_benchmark.exe",
        root / "build" / "vspec_unified_runtime_benchmark",
        root / "build" / "vspec_benchmark",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convenience wrapper for Vspec benchmark binary")
    parser.add_argument("--bin", default="", help="Custom benchmark binary path")
    parser.add_argument("args", nargs="*", help="Arguments passed to benchmark binary")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    exe = Path(args.bin) if args.bin else _find_benchmark_binary(root)
    if exe is None or not exe.exists():
        raise SystemExit("Benchmark binary not found. Build project first.")

    cmd = [str(exe), *args.args]
    print(f"[vspec-benchmark] bin={exe}")
    proc = subprocess.run(cmd)
    raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
