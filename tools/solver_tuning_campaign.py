from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import fields, replace
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

DEFAULT_PROFILES = ["ai", "soft", "crisp"]


PARAM_VALUES: dict[str, list[float | int]] = {
    "phase_field_grid_alignment_weight": [0.35, 0.70, 1.25, 2.25, 4.00],
    "phase_field_local_search_radius_ratio": [0.35, 0.50, 0.65, 0.85, 1.10],
    "phase_field_local_search_blend": [0.04, 0.08, 0.14, 0.24, 0.36],
    "phase_field_local_search_move_weight": [0.0, 0.015, 0.04, 0.09, 0.18],
    "phase_field_signal_weight": [0.05, 0.12, 0.25, 0.55, 1.00, 1.60],
    "phase_field_signal_pyramid_levels": [1, 2, 3, 5, 7],
    "phase_field_signal_level_decay": [0.0, 0.15, 0.35, 0.60, 0.85],
    "phase_field_signal_gradient_weight": [0.0, 0.08, 0.20, 0.45, 0.90],
    "phase_field_signal_curvature_weight": [0.10, 0.30, 0.65, 1.10, 1.75],
    "phase_field_signal_energy_power": [0.60, 1.00, 1.40, 2.00, 3.00],
    "phase_field_signal_peak_power": [1.00, 1.50, 2.25, 3.25, 4.75],
    "phase_field_local_search_grid_weight": [0.0, 0.08, 0.22, 0.50],
    "phase_field_local_search_interval": [1, 2, 4],
    "candidate_rerank_preview_steps": [4, 8, 16],
    "candidate_rerank_support_weight": [0.10, 0.45, 0.90],
    "candidate_rerank_edge_position_weight": [0.0, 0.20, 0.55],
    "candidate_rerank_wobble_weight": [0.0, 0.20, 0.55],
    "candidate_rerank_edge_concentration_weight": [0.0, 0.10, 0.35],
    "candidate_rerank_size_penalty_weight": [0.0, 0.18, 0.45],
    "candidate_rerank_inference_penalty_weight": [0.0, 0.05, 0.20],
    "candidate_rerank_confidence_threshold": [0.02, 0.12, 0.28],
    "candidate_rerank_max_size_delta_ratio": [0.15, 0.40, 0.75],
    "candidate_rerank_margin": [0.0, 0.004, 0.02],
}


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


def compact_case_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        compact.append(
            {
                "case_id": case["case_id"],
                "profile": case["profile"],
                "optimized_error_mean": case["optimized_error_mean"],
                "optimized_adjacency_error_mean": case["optimized_adjacency_error_mean"],
                "optimized_motif_error_mean": case["optimized_motif_error_mean"],
                "optimized_beats_naive_rate": case["optimized_beats_naive_rate"],
                "optimized_beats_diffusion_rate": case["optimized_beats_diffusion_rate"],
            }
        )
    return compact


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_status(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, {"updated_at": utc_now(), **payload})


def patch_params(base: SolverHyperParams, patch: dict[str, float | int]) -> SolverHyperParams:
    values = base.to_dict()
    values.update(patch)
    for field in fields(SolverHyperParams):
        if isinstance(getattr(base, field.name), int):
            values[field.name] = int(round(float(values[field.name])))
        else:
            values[field.name] = float(values[field.name])
    return SolverHyperParams(**values)


def different_values(name: str, base: SolverHyperParams) -> list[float | int]:
    current = getattr(base, name)
    values = PARAM_VALUES.get(name, [])
    result: list[float | int] = []
    for value in values:
        if isinstance(current, int):
            normalized: float | int = int(round(float(value)))
            is_same = normalized == current
        else:
            normalized = float(value)
            is_same = math.isclose(normalized, float(current), rel_tol=1e-7, abs_tol=1e-7)
        if not is_same and normalized not in result:
            result.append(normalized)
    return result


def candidate_id(index: int, stage: str) -> str:
    return f"{index:04d}-{stage}"


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.ledger_path = self.out_dir / "ledger.jsonl"
        self.status_path = self.out_dir / "status.json"
        self.summary_path = self.out_dir / "summary.json"
        self.param_effects_path = self.out_dir / "parameter-effects.json"
        self.benchmark_root = self.out_dir / "benchmarks"
        self.base_params = load_solver_params(args.params)
        self.baseline_metrics: dict[str, Any] | None = None
        self.best_record: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.counter = 0

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.out_dir / "campaign-config.json", self.campaign_config())
        write_status(
            self.status_path,
            {
                "state": "running",
                "stage": "setup",
                "message": "Campaign initialized.",
                "pid": os.getpid(),
                "out_dir": str(self.out_dir),
            },
        )

        baseline = self.evaluate(
            "baseline",
            {},
            "Measure the current config before touching anything. This is the control specimen.",
            keep_benchmark=True,
        )
        self.baseline_metrics = baseline["metrics"]
        self.best_record = baseline

        single_records = self.single_knob_sweep()
        promising = self.promising_single_knobs(single_records)
        pair_records = self.pair_sweep(promising)
        greedy_records = self.greedy_combine(promising, pair_records)
        self.local_refine(greedy_records or pair_records or single_records)
        self.write_parameter_effects()
        self.final_full_corpus()
        self.write_summary(final_state="complete")
        write_status(
            self.status_path,
            {
                "state": "complete",
                "stage": "done",
                "message": "Campaign complete. The machine has survived another night in the woods.",
                "pid": os.getpid(),
                "best": self.best_record,
            },
        )

    def campaign_config(self) -> dict[str, Any]:
        return {
            "started_at": utc_now(),
            "corpus_dir": str(Path(self.args.corpus_dir)),
            "focused_cases": self.args.cases,
            "focused_profiles": self.args.profiles,
            "variants": self.args.variants,
            "steps": self.args.steps,
            "device": self.args.device,
            "workers": self.args.workers,
            "seed": self.args.seed,
            "final_variants": self.args.final_variants,
            "score": "0.65 mean + 0.25 worst + 0.10 p90 over row scores; row score is 0.20 error + 0.45 adjacency + 0.35 motif ratios to naive",
            "base_params": self.base_params.to_dict(),
            "parameter_values": PARAM_VALUES,
        }

    def evaluate(
        self,
        stage: str,
        patch: dict[str, float | int],
        hypothesis: str,
        *,
        keep_benchmark: bool = False,
        full_corpus: bool = False,
    ) -> dict[str, Any]:
        self.counter += 1
        run_id = candidate_id(self.counter, stage)
        params = patch_params(self.base_params, patch)
        bench_dir = self.benchmark_root / run_id
        start = time.time()
        write_status(
            self.status_path,
            {
                "state": "running",
                "stage": stage,
                "candidate": run_id,
                "hypothesis": hypothesis,
                "patch": patch,
                "completed_candidates": len(self.records),
                "pid": os.getpid(),
            },
        )
        try:
            summary = run_roundtrip_benchmark(
                self.args.corpus_dir,
                bench_dir,
                variants=self.args.final_variants if full_corpus else self.args.variants,
                profiles=self.args.profiles,
                seed=self.args.seed,
                steps=self.args.steps,
                device=self.args.device,
                infer_size=False,
                include_cases=None if full_corpus else self.args.cases,
                keep_existing=False,
                solver_params=params,
                workers=self.args.workers,
            )
            metrics = score_summary(summary)
            baseline_delta = None
            if self.baseline_metrics is not None:
                baseline_delta = float(metrics["score"] - self.baseline_metrics["score"])
            best_before = self.best_record["metrics"]["score"] if self.best_record else math.inf
            improved_best = metrics["score"] < best_before
            record = {
                "at": utc_now(),
                "run_id": run_id,
                "stage": stage,
                "hypothesis": hypothesis,
                "patch": patch,
                "metrics": metrics,
                "baseline_delta": baseline_delta,
                "improved_best": improved_best,
                "duration_seconds": round(time.time() - start, 3),
                "benchmark_dir": str(bench_dir) if keep_benchmark or full_corpus or improved_best else None,
                "case_summary": compact_case_summary(summary),
            }
            append_jsonl(self.ledger_path, record)
            self.records.append(record)
            if improved_best:
                self.best_record = record
                write_json(self.out_dir / "best-focused-params.json", params.to_dict())
                if not keep_benchmark and not full_corpus:
                    self.keep_best_benchmark(bench_dir)
                    record["benchmark_dir"] = str(bench_dir)
            elif not keep_benchmark and not full_corpus:
                shutil.rmtree(bench_dir, ignore_errors=True)
            self.write_summary(final_state="running")
            return record
        except Exception as exc:
            record = {
                "at": utc_now(),
                "run_id": run_id,
                "stage": stage,
                "hypothesis": hypothesis,
                "patch": patch,
                "error": repr(exc),
                "duration_seconds": round(time.time() - start, 3),
            }
            append_jsonl(self.ledger_path, record)
            write_status(
                self.status_path,
                {
                    "state": "failed",
                    "stage": stage,
                    "candidate": run_id,
                    "error": repr(exc),
                    "pid": os.getpid(),
                },
            )
            raise

    def keep_best_benchmark(self, bench_dir: Path) -> None:
        keep_dir = self.out_dir / "best-focused-benchmark"
        if keep_dir.exists():
            shutil.rmtree(keep_dir)
        if bench_dir.exists():
            shutil.copytree(bench_dir, keep_dir)

    def single_knob_sweep(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        param_names = [field.name for field in fields(SolverHyperParams)]
        for name in param_names:
            values = different_values(name, self.base_params)
            if not values:
                append_jsonl(
                    self.ledger_path,
                    {
                        "at": utc_now(),
                        "stage": "single-skip",
                        "parameter": name,
                        "hypothesis": "No alternate values were configured for this knob.",
                    },
                )
                continue
            for value in values:
                hypothesis = (
                    f"Single-knob isolation: set {name}={value!r} and leave every other knob fixed, "
                    "so any score movement can be attributed to this parameter in the focused artificial set."
                )
                records.append(self.evaluate("single", {name: value}, hypothesis))
        return records

    def promising_single_knobs(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        baseline_score = self.baseline_metrics["score"] if self.baseline_metrics else math.inf
        by_param: dict[str, dict[str, Any]] = {}
        for record in records:
            if "metrics" not in record or not record.get("patch"):
                continue
            name = next(iter(record["patch"]))
            current = by_param.get(name)
            if current is None or record["metrics"]["score"] < current["metrics"]["score"]:
                by_param[name] = record
        ranked = sorted(by_param.values(), key=lambda item: item["metrics"]["score"])
        winners = [record for record in ranked if record["metrics"]["score"] <= baseline_score * self.args.single_keep_ratio]
        if len(winners) < self.args.min_promising:
            winners = ranked[: self.args.min_promising]
        winners = winners[: self.args.max_promising]
        write_json(
            self.out_dir / "promising-single-knobs.json",
            {
                "baseline_score": baseline_score,
                "selected": winners,
                "all_best_by_param": ranked,
            },
        )
        return winners

    def pair_sweep(self, promising: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        pairs_tested = 0
        for left_index, left in enumerate(promising):
            for right in promising[left_index + 1 :]:
                if pairs_tested >= self.args.max_pairs:
                    return records
                patch = {**left["patch"], **right["patch"]}
                if len(patch) < 2:
                    continue
                left_name = next(iter(left["patch"]))
                right_name = next(iter(right["patch"]))
                hypothesis = (
                    f"Pair interaction: combine the best isolated moves for {left_name} and {right_name}. "
                    "If the effect stacks, they are probably steering different failure modes; if it collapses, "
                    "they are fighting over the same bit of the animal."
                )
                records.append(self.evaluate("pair", patch, hypothesis))
                pairs_tested += 1
        return records

    def greedy_combine(self, promising: list[dict[str, Any]], pair_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seeds = sorted(pair_records or promising, key=lambda item: item["metrics"]["score"])
        if not seeds:
            return records
        current_patch = dict(seeds[0]["patch"])
        current_score = float(seeds[0]["metrics"]["score"])
        for record in promising:
            next_patch = {**current_patch, **record["patch"]}
            if next_patch == current_patch:
                continue
            added = [name for name in record["patch"] if name not in current_patch]
            if not added:
                continue
            hypothesis = (
                f"Greedy combination: add {', '.join(added)} to the current best patch. "
                "Only keep the mental model if the combined score beats the running patch."
            )
            tested = self.evaluate("greedy", next_patch, hypothesis)
            records.append(tested)
            if tested["metrics"]["score"] < current_score:
                current_patch = next_patch
                current_score = float(tested["metrics"]["score"])
        return records

    def local_refine(self, source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not source_records:
            return records
        best = min(source_records, key=lambda item: item["metrics"]["score"])
        patch = dict(best["patch"])
        for name, value in list(patch.items()):
            current = getattr(self.base_params, name)
            if isinstance(current, int):
                candidates = sorted({max(0, int(value) - 1), int(value) + 1})
            else:
                value_f = float(value)
                current_f = float(current)
                midpoint = (value_f + current_f) / 2.0
                away = value_f + (value_f - current_f) * 0.5
                candidates = [midpoint, away]
            for candidate in candidates:
                refined = {**patch, name: candidate}
                hypothesis = (
                    f"Local refinement: nudge {name} around the best combined patch to see whether the isolated "
                    "winner was a real optimum or just a signpost."
                )
                records.append(self.evaluate("refine", refined, hypothesis))
        return records

    def final_full_corpus(self) -> None:
        if not self.args.final_full_corpus:
            return
        if self.best_record is None:
            return
        baseline_hypothesis = (
            "Final full-corpus control: run the original config across every corpus case once, after focused tuning is done."
        )
        best_hypothesis = (
            "Final full-corpus candidate: run the best focused configuration across every corpus case. "
            "This is the only stage allowed to touch the full corpus."
        )
        self.evaluate("final-baseline-full", {}, baseline_hypothesis, keep_benchmark=True, full_corpus=True)
        self.evaluate(
            "final-best-full",
            dict(self.best_record["patch"]),
            best_hypothesis,
            keep_benchmark=True,
            full_corpus=True,
        )

    def write_parameter_effects(self) -> None:
        baseline_score = self.baseline_metrics["score"] if self.baseline_metrics else math.inf
        effects: dict[str, Any] = {}
        for name in [field.name for field in fields(SolverHyperParams)]:
            records = [
                record
                for record in self.records
                if record.get("stage") == "single" and set(record.get("patch", {}).keys()) == {name}
            ]
            if not records:
                continue
            ranked = sorted(records, key=lambda item: item["metrics"]["score"])
            effects[name] = {
                "base_value": getattr(self.base_params, name),
                "best_value": ranked[0]["patch"][name],
                "best_score": ranked[0]["metrics"]["score"],
                "best_delta_from_baseline": ranked[0]["metrics"]["score"] - baseline_score,
                "worst_value": ranked[-1]["patch"][name],
                "worst_score": ranked[-1]["metrics"]["score"],
                "observations": [
                    {
                        "value": record["patch"][name],
                        "score": record["metrics"]["score"],
                        "delta_from_baseline": record["metrics"]["score"] - baseline_score,
                    }
                    for record in ranked
                ],
            }
        write_json(self.param_effects_path, effects)

    def write_summary(self, *, final_state: str) -> None:
        write_json(
            self.summary_path,
            {
                "state": final_state,
                "updated_at": utc_now(),
                "candidate_count": len(self.records),
                "baseline": self.records[0] if self.records else None,
                "best": self.best_record,
                "ledger": str(self.ledger_path),
                "parameter_effects": str(self.param_effects_path),
                "status": str(self.status_path),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ordered solver tuning campaign with a hypothesis ledger.")
    parser.add_argument("--corpus-dir", default=str(ROOT / "examples" / "corpus"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--params", default=None)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--final-variants", type=int, default=2)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--single-keep-ratio", type=float, default=0.995)
    parser.add_argument("--min-promising", type=int, default=10)
    parser.add_argument("--max-promising", type=int, default=18)
    parser.add_argument("--max-pairs", type=int, default=48)
    parser.add_argument("--final-full-corpus", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    Campaign(parse_args()).run()


if __name__ == "__main__":
    main()
