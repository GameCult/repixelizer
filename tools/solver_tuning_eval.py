from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repixelizer.benchmark import run_roundtrip_benchmark
from repixelizer.params import SolverHyperParams, load_solver_params


DEFAULT_CASES = [
    "9rpgenemies-06",
    "andromalius",
    "dagrons5-05",
    "dagrons5-06",
    "gnu",
    "mage-2",
    "more-rpg-enemies-05",
    "more-rpg-enemies-06",
    "shadow",
    "threeformspj2-01",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-6))


def row_score(row: dict[str, Any]) -> float:
    error_ratio = safe_ratio(float(row["optimized_error_to_original"]), float(row["naive_error_to_original"]))
    adjacency_ratio = safe_ratio(
        float(row["optimized_adjacency_error_to_original"]),
        float(row["naive_adjacency_error_to_original"]),
    )
    motif_ratio = safe_ratio(float(row["optimized_motif_error_to_original"]), float(row["naive_motif_error_to_original"]))
    return error_ratio * 0.20 + adjacency_ratio * 0.45 + motif_ratio * 0.35


def score_summary(summary: dict[str, Any]) -> dict[str, Any]:
    scores = [row_score(row) for row in summary["rows"]]
    mean_score = sum(scores) / max(1, len(scores))
    worst_score = max(scores) if scores else math.inf
    sorted_scores = sorted(scores)
    p90_score = sorted_scores[min(len(sorted_scores) - 1, int(0.90 * (len(sorted_scores) - 1)))] if scores else math.inf
    return {
        "score": mean_score * 0.65 + worst_score * 0.25 + p90_score * 0.10,
        "mean_row_score": mean_score,
        "worst_row_score": worst_score,
        "p90_row_score": p90_score,
        "row_count": len(scores),
    }


def patch_params(base: SolverHyperParams, patch: dict[str, float | int]) -> SolverHyperParams:
    values = base.to_dict()
    values.update(patch)
    for field in fields(SolverHyperParams):
        if isinstance(getattr(base, field.name), int):
            values[field.name] = int(round(float(values[field.name])))
        else:
            values[field.name] = float(values[field.name])
    return SolverHyperParams(**values)


def compact_cases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "profile": case["profile"],
            "optimized_error_mean": case["optimized_error_mean"],
            "optimized_adjacency_error_mean": case["optimized_adjacency_error_mean"],
            "optimized_motif_error_mean": case["optimized_motif_error_mean"],
            "optimized_beats_naive_rate": case["optimized_beats_naive_rate"],
            "optimized_beats_diffusion_rate": case["optimized_beats_diffusion_rate"],
        }
        for case in summary.get("cases", [])
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_patch(raw: str, patch_file: str | None = None) -> dict[str, float | int]:
    if patch_file:
        raw = Path(patch_file).read_text(encoding="utf-8-sig")
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--patch-json must be a JSON object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one deliberate solver tuning hypothesis.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--patch-json", default="{}")
    parser.add_argument("--patch-file", default=None)
    parser.add_argument("--interpretation", default="")
    parser.add_argument("--corpus-dir", default=str(ROOT / "examples" / "corpus"))
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--profiles", nargs="*", default=["ai", "soft", "crisp"])
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 1)))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    patch = parse_patch(args.patch_json, args.patch_file)
    base = load_solver_params()
    params = patch_params(base, patch)
    bench_dir = out_dir / "benchmarks" / args.run_id
    status_path = out_dir / "status.json"
    ledger_path = out_dir / "ledger.jsonl"

    write_json(
        status_path,
        {
            "updated_at": utc_now(),
            "state": "running",
            "run_id": args.run_id,
            "hypothesis": args.hypothesis,
            "patch": patch,
            "pid": os.getpid(),
        },
    )
    start = time.time()
    summary = run_roundtrip_benchmark(
        args.corpus_dir,
        bench_dir,
        variants=args.variants,
        profiles=args.profiles,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        infer_size=False,
        include_cases=args.cases,
        keep_existing=False,
        solver_params=params,
        workers=args.workers,
    )
    metrics = score_summary(summary)
    record = {
        "at": utc_now(),
        "run_id": args.run_id,
        "hypothesis": args.hypothesis,
        "interpretation": args.interpretation,
        "patch": patch,
        "metrics": metrics,
        "duration_seconds": round(time.time() - start, 3),
        "profiles": args.profiles,
        "cases": args.cases,
        "variants": args.variants,
        "steps": args.steps,
        "seed": args.seed,
        "workers": args.workers,
        "benchmark_dir": str(bench_dir),
        "case_summary": compact_cases(summary),
    }
    append_jsonl(ledger_path, record)
    write_json(out_dir / f"{args.run_id}.json", record)
    write_json(out_dir / "status.json", {"updated_at": utc_now(), "state": "complete", "latest": record})
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
