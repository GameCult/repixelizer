from __future__ import annotations

import argparse
import sys

from .benchmark import run_roundtrip_benchmark
from .compare import run_compare
from .corpus import prepare_corpus
from .pipeline import run_pipeline
from .tuning import tune_solver_hyperparams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repixelize", description="Optimization-based repixelizer for fake pixel art.")
    subparsers = parser.add_subparsers(dest="command")

    def add_shared_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("input", help="Source image path")
        target.add_argument("--out", required=True, help="Output image path")
        target.add_argument("--target-size", type=int, default=None, help="Override the inferred target max dimension")
        target.add_argument("--target-width", type=int, default=None, help="Pin the output lattice width directly")
        target.add_argument("--target-height", type=int, default=None, help="Pin the output lattice height directly")
        target.add_argument("--phase-x", type=float, default=None, help="Pin the lattice phase offset on the x axis")
        target.add_argument("--phase-y", type=float, default=None, help="Pin the lattice phase offset on the y axis")
        target.add_argument("--palette", default=None, help="Optional palette file (.gpl, .txt, .json)")
        target.add_argument(
            "--palette-mode",
            choices=("off", "fit", "strict"),
            default="off",
            help="Palette handling mode",
        )
        target.add_argument("--diagnostics-dir", default=None, help="Directory for debug artifacts")
        target.add_argument("--seed", type=int, default=7, help="Random seed")
        target.add_argument("--steps", type=int, default=200, help="Number of optimizer steps")
        target.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Torch device")
        target.add_argument(
            "--reconstruction-mode",
            default="continuous",
            choices=("continuous", "tile-graph"),
            help="Reconstruction engine to run after lattice estimation",
        )
        target.add_argument(
            "--strip-background",
            action="store_true",
            help="Remove light neutral edge-connected backgrounds such as fake transparency checkerboards",
        )
        target.add_argument(
            "--skip-phase-rerank",
            action="store_true",
            help="Skip low-confidence phase reranking and run the selected or pinned lattice directly",
        )

    run_parser = subparsers.add_parser("run", help="Run the optimizer.")
    add_shared_arguments(run_parser)
    compare_parser = subparsers.add_parser("compare", help="Run the optimizer and comparison baselines.")
    add_shared_arguments(compare_parser)
    benchmark_parser = subparsers.add_parser("benchmark", help="Run the round-trip corpus benchmark.")
    benchmark_parser.add_argument("--corpus-dir", default="examples/corpus", help="Corpus root containing originals/")
    benchmark_parser.add_argument("--out-dir", default="artifacts/benchmark", help="Directory for benchmark outputs")
    benchmark_parser.add_argument("--variants", type=int, default=3, help="Facsimile variants per original")
    benchmark_parser.add_argument(
        "--profile",
        action="append",
        choices=("soft", "crisp", "ai"),
        default=None,
        help="Corruption profile to include; may be repeated. Defaults to both soft and crisp.",
    )
    benchmark_parser.add_argument("--seed", type=int, default=7, help="Random seed")
    benchmark_parser.add_argument("--steps", type=int, default=200, help="Number of optimizer steps")
    benchmark_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Torch device")
    benchmark_parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Restrict benchmark to a specific case id or sprite basename; may be repeated",
    )
    benchmark_parser.add_argument("--limit", type=int, default=None, help="Only benchmark the first N matching cases")
    benchmark_parser.add_argument(
        "--infer-size",
        action="store_true",
        help="Let the pipeline infer output size instead of locking it to the original ground-truth size",
    )
    benchmark_parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep any existing files in the benchmark output directory instead of clearing it first",
    )
    tune_parser = subparsers.add_parser("tune", help="Run a black-box hyperparameter search on a benchmark slice.")
    tune_parser.add_argument("--corpus-dir", default="examples/corpus", help="Corpus root containing originals/")
    tune_parser.add_argument("--out-dir", default="artifacts/tuning", help="Directory for tuning outputs")
    tune_parser.add_argument("--trials", type=int, default=8, help="Number of parameter sets to evaluate")
    tune_parser.add_argument("--variants", type=int, default=1, help="Facsimile variants per original")
    tune_parser.add_argument(
        "--profile",
        action="append",
        choices=("soft", "crisp", "ai"),
        default=None,
        help="Corruption profile to include; may be repeated. Defaults to soft only for tuning.",
    )
    tune_parser.add_argument("--seed", type=int, default=7, help="Random seed")
    tune_parser.add_argument("--steps", type=int, default=48, help="Number of optimizer steps")
    tune_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Torch device")
    tune_parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Restrict tuning to a specific case id or sprite basename; may be repeated",
    )
    tune_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only tune on the first N matching cases. Defaults to 8 when no cases are specified.",
    )
    tune_parser.add_argument(
        "--infer-size",
        action="store_true",
        help="Let the pipeline infer output size instead of locking it to the original ground-truth size",
    )
    prepare_parser = subparsers.add_parser("prepare-corpus", help="Normalize imported corpus sheets into single sprites.")
    prepare_parser.add_argument("--corpus-dir", default="examples/corpus", help="Corpus root containing originals/")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0] not in {"run", "compare", "benchmark", "tune", "prepare-corpus", "-h", "--help"}:
        raw_argv = ["run", *raw_argv]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    command = args.command or "run"
    if command == "compare":
        run_compare(
            args.input,
            args.out,
            target_size=args.target_size,
            target_width=args.target_width,
            target_height=args.target_height,
            phase_x=args.phase_x,
            phase_y=args.phase_y,
            palette_path=args.palette,
            palette_mode=args.palette_mode,
            diagnostics_dir=args.diagnostics_dir,
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            reconstruction_mode=args.reconstruction_mode,
            strip_background=args.strip_background,
            enable_phase_rerank=not args.skip_phase_rerank,
        )
        return 0
    if command == "benchmark":
        run_roundtrip_benchmark(
            args.corpus_dir,
            args.out_dir,
            variants=args.variants,
            profiles=args.profile,
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            infer_size=args.infer_size,
            include_cases=args.case,
            limit_cases=args.limit,
            keep_existing=args.keep_existing,
        )
        return 0
    if command == "tune":
        tune_solver_hyperparams(
            args.corpus_dir,
            args.out_dir,
            trials=args.trials,
            variants=args.variants,
            profiles=args.profile,
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            infer_size=args.infer_size,
            include_cases=args.case,
            limit_cases=args.limit,
        )
        return 0
    if command == "prepare-corpus":
        prepare_corpus(args.corpus_dir)
        return 0

    run_pipeline(
        args.input,
        args.out,
        target_size=args.target_size,
        target_width=args.target_width,
        target_height=args.target_height,
        phase_x=args.phase_x,
        phase_y=args.phase_y,
        palette_path=args.palette,
        palette_mode=args.palette_mode,
        diagnostics_dir=args.diagnostics_dir,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        reconstruction_mode=args.reconstruction_mode,
        strip_background=args.strip_background,
        enable_phase_rerank=not args.skip_phase_rerank,
    )
    return 0
