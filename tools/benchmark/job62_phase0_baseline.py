from __future__ import annotations

import argparse
import json
import os
import traceback
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI_DIR = ROOT / "tools" / "cli"

import sys

sys.path.insert(0, str(CLI_DIR))

from vspec_runner import VspecRunArgs, run_once  # type: ignore[import-not-found]


def _load_models_from_cache() -> list[str]:
    cache = ROOT / "logs" / "model_scan_cache.json"
    if not cache.exists():
        return []
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in payload.get("candidates", []) or []:
        p = str(item.get("path", "")).strip()
        if not p:
            continue
        rp = str(Path(p).resolve())
        if rp in seen:
            continue
        if Path(rp).exists():
            seen.add(rp)
            out.append(rp)
    return out


def _split_csv_models(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        p = str(part).strip().strip('"').strip("'")
        if not p:
            continue
        rp = str(Path(p).resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _resolve_models(model_csv: str, models_file: str, use_cache: bool) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for p in _split_csv_models(model_csv):
        if p in seen:
            continue
        if Path(p).exists():
            seen.add(p)
            merged.append(p)

    mf = str(models_file or "").strip()
    if mf:
        mfp = Path(mf)
        if mfp.exists():
            for line in mfp.read_text(encoding="utf-8", errors="replace").splitlines():
                p = str(line).strip().strip('"').strip("'")
                if not p:
                    continue
                rp = str(Path(p).resolve())
                if rp in seen:
                    continue
                if Path(rp).exists():
                    seen.add(rp)
                    merged.append(rp)

    if use_cache:
        for p in _load_models_from_cache():
            if p in seen:
                continue
            seen.add(p)
            merged.append(p)

    return merged


def _parse_float(metrics: dict[str, str], key: str) -> float:
    raw = str((metrics or {}).get(key, "")).strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except Exception:
        return 0.0


def _parse_int(metrics: dict[str, str], key: str) -> int:
    raw = str((metrics or {}).get(key, "")).strip()
    if not raw:
        return 0
    try:
        return int(float(raw))
    except Exception:
        return 0


def _run_profile(model: str, prompt: str, max_tokens: int, speed: str) -> dict:
    args = VspecRunArgs(
        model=model,
        prompt=prompt,
        prompts_file="",
        batch_output_file="",
        device=None,
        fused_bits=None,
        target_bits=None,
        max_layers=0,
        max_tokens=max_tokens,
        max_decode_seconds=40.0,
        max_retry_seconds=-1.0,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.15,
        repeat_window=64,
        no_repeat_ngram=3,
        speed_preset=speed,
        lang="auto",
        stream=False,
        unsafe_low_layers=False,
    )

    t0 = time.perf_counter()
    result = run_once(args)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics = result.get("metrics", {}) or {}
    out_text = str(result.get("text", "") or "")
    tokens_generated = _parse_int(metrics, "tokens_generated")
    if tokens_generated <= 0:
        tokens_generated = _parse_int(metrics, "output_tokens")
    decode_contract_ok = str(metrics.get("decode_contract_ok", "")).strip().lower()
    contract_pass = not decode_contract_ok.startswith("false")
    profile_pass = bool(result.get("ok", False)) and int(result.get("returncode", 1)) == 0 and tokens_generated > 0 and contract_pass

    return {
        "ok": bool(result.get("ok", False)),
        "returncode": int(result.get("returncode", 1)),
        "profile_pass": profile_pass,
        "tokens_generated": tokens_generated,
        "decode_contract_ok": str(metrics.get("decode_contract_ok", "")),
        "output_chars": len(out_text),
        "output_preview": out_text[:160],
        "wall_ms": round(wall_ms, 2),
        "metric_elapsed_ms": _parse_float(metrics, "elapsed_ms"),
        "metric_tokens_per_sec": _parse_float(metrics, "tokens_per_sec"),
        "metric_output_tokens": _parse_float(metrics, "output_tokens"),
        "metrics": metrics,
        "stderr_preview": str(result.get("stderr", "") or "")[:240],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Job62 Phase 0 multi-model baseline (Python launcher + Python guards)")
    parser.add_argument("--models", default="", help="Comma-separated model paths")
    parser.add_argument("--models-file", default="", help="Text file with one model path per line")
    parser.add_argument("--use-cache-models", action="store_true", help="Include all models from logs/model_scan_cache.json")
    parser.add_argument("--output", default="logs/job62_phase0_baseline_multimodel.json", help="Output JSON artifact path")
    parser.add_argument("--short-tokens", type=int, default=24)
    parser.add_argument("--long-tokens", type=int, default=64)
    args = parser.parse_args()

    use_cache = bool(args.use_cache_models)
    if not str(args.models or "").strip() and not str(args.models_file or "").strip():
        use_cache = True

    models = _resolve_models(
        model_csv=str(args.models or ""),
        models_file=str(args.models_file or ""),
        use_cache=use_cache,
    )
    if not models:
        raise SystemExit("No models found. Pass --models/--models-file or enable cache discovery")

    # Hard rule for Phase 0: launcher and guard stack remain Python-owned.
    os.environ["VSPEC_CHAT_MODE"] = "python"
    os.environ["VSPEC_NATIVE_BACKEND"] = "python"
    os.environ["VSPEC_REQUIRE_PY_QUALITY_GUARDS"] = "1"
    os.environ.setdefault("VSPEC_C_SAMPLER_REQUIRED", "0")

    profiles: list[tuple[str, str, int]] = [
        ("short_chat_latency", "Xin chao, tom tat 3 y chinh ve Vspec Engine.", int(args.short_tokens)),
        ("long_decode_stability", "Viet doan phan tich ky thuat ngan gon ve chuyen doi Python sang C theo tung phase.", int(args.long_tokens)),
        ("quality_sanity_vi", "Hay viet 5 gach dau dong ve nguyen tac an toan khi migrate runtime.", 96),
    ]

    model_runs: dict[str, dict] = {}
    failed_models: list[str] = []
    all_models_ok = True

    for model in models:
        runs: dict[str, dict] = {}
        all_ok = True
        failed_profiles: list[str] = []
        fatal_error = ""

        for name, prompt, max_tokens in profiles:
            try:
                run = _run_profile(model=model, prompt=prompt, max_tokens=max_tokens, speed="fast")
            except Exception as exc:
                fatal_error = f"{type(exc).__name__}: {exc}"
                run = {
                    "ok": False,
                    "returncode": 99,
                    "profile_pass": False,
                    "tokens_generated": 0,
                    "decode_contract_ok": "",
                    "output_chars": 0,
                    "output_preview": "",
                    "wall_ms": 0.0,
                    "metric_elapsed_ms": 0.0,
                    "metric_tokens_per_sec": 0.0,
                    "metric_output_tokens": 0.0,
                    "metrics": {},
                    "stderr_preview": traceback.format_exc(limit=1)[:240],
                }
            runs[name] = run
            passed = bool(run.get("profile_pass", False))
            all_ok = all_ok and passed
            if not passed:
                failed_profiles.append(name)

        model_summary = {
            "model_path": model,
            "all_profiles_ok": bool(all_ok),
            "failed_profiles": failed_profiles,
            "short_wall_ms": float(runs.get("short_chat_latency", {}).get("wall_ms", 0.0)),
            "long_wall_ms": float(runs.get("long_decode_stability", {}).get("wall_ms", 0.0)),
            "fatal_error": fatal_error,
        }
        model_runs[model] = {
            "profiles": runs,
            "summary": model_summary,
        }
        if not all_ok:
            failed_models.append(model)
            all_models_ok = False

    summary = {
        "phase": 0,
        "launcher_python": os.environ.get("VSPEC_CHAT_MODE") == "python",
        "backend_python": os.environ.get("VSPEC_NATIVE_BACKEND") == "python",
        "require_python_guards": os.environ.get("VSPEC_REQUIRE_PY_QUALITY_GUARDS") == "1",
        "all_models_ok": bool(all_models_ok),
        "model_count": len(models),
        "passed_model_count": int(len(models) - len(failed_models)),
        "failed_model_count": len(failed_models),
        "failed_models": failed_models,
    }

    artifact = {
        "job": "job62",
        "date_unix": time.time(),
        "models": models,
        "rules": {
            "launcher_python": True,
            "guards_python": True,
            "others_move_gradually_to_c": True,
        },
        "model_runs": model_runs,
        "summary": summary,
    }

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(artifact, indent=2, ensure_ascii=False))

    if not summary["all_models_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
