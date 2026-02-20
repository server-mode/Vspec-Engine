import argparse
from pathlib import Path


def write_log(
    path: Path,
    tokens: int,
    seconds: float,
    perplexity_drift: float,
    sm_occupancy: float,
    memory_bandwidth: float,
    warp_stall_reason: str,
    sequence_scaling: str
) -> None:
    path.write_text(
        "\n".join(
            [
                f"tokens={tokens}",
                f"seconds={seconds}",
                f"perplexity_drift={perplexity_drift}",
                f"sm_occupancy={sm_occupancy}",
                f"memory_bandwidth={memory_bandwidth}",
                f"warp_stall_reason={warp_stall_reason}",
                f"sequence_scaling={sequence_scaling}",
                "",
            ]
        ),
        encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample log files for custom bench")
    parser.add_argument("--out-dir", default="logs", help="Output directory for logs")
    parser.add_argument("--vspec-tokens", type=int, default=512)
    parser.add_argument("--vspec-seconds", type=float, default=6.4)
    parser.add_argument("--baseline-tokens", type=int, default=512)
    parser.add_argument("--baseline-seconds", type=float, default=8.8)
    parser.add_argument("--vspec-perplexity-drift", type=float, default=0.012)
    parser.add_argument("--vspec-sm-occupancy", type=float, default=62.5)
    parser.add_argument("--vspec-memory-bandwidth", type=float, default=71.2)
    parser.add_argument("--vspec-warp-stall-reason", default="memory_dependency")
    parser.add_argument("--vspec-sequence-scaling", default="pass")
    parser.add_argument("--baseline-perplexity-drift", type=float, default=0.021)
    parser.add_argument("--baseline-sm-occupancy", type=float, default=58.1)
    parser.add_argument("--baseline-memory-bandwidth", type=float, default=64.3)
    parser.add_argument("--baseline-warp-stall-reason", default="execution_dependency")
    parser.add_argument("--baseline-sequence-scaling", default="pass")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_log(
        out_dir / "vspec_run.txt",
        args.vspec_tokens,
        args.vspec_seconds,
        args.vspec_perplexity_drift,
        args.vspec_sm_occupancy,
        args.vspec_memory_bandwidth,
        args.vspec_warp_stall_reason,
        args.vspec_sequence_scaling
    )
    write_log(
        out_dir / "baseline_run.txt",
        args.baseline_tokens,
        args.baseline_seconds,
        args.baseline_perplexity_drift,
        args.baseline_sm_occupancy,
        args.baseline_memory_bandwidth,
        args.baseline_warp_stall_reason,
        args.baseline_sequence_scaling
    )

    print(f"wrote {out_dir / 'vspec_run.txt'}")
    print(f"wrote {out_dir / 'baseline_run.txt'}")


if __name__ == "__main__":
    main()
