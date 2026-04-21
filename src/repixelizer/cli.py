from __future__ import annotations

import argparse
import sys

from .compare import run_compare
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repixelize", description="Optimization-based repixelizer for fake pixel art.")
    subparsers = parser.add_subparsers(dest="command")

    def add_shared_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("input", help="Source image path")
        target.add_argument("--out", required=True, help="Output image path")
        target.add_argument("--target-size", type=int, default=None, help="Override the inferred target max dimension")
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
        target.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Torch device")

    run_parser = subparsers.add_parser("run", help="Run the optimizer.")
    add_shared_arguments(run_parser)
    compare_parser = subparsers.add_parser("compare", help="Run the optimizer and comparison baselines.")
    add_shared_arguments(compare_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0] not in {"run", "compare", "-h", "--help"}:
        raw_argv = ["run", *raw_argv]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    command = args.command or "run"
    if command == "compare":
        run_compare(
            args.input,
            args.out,
            target_size=args.target_size,
            palette_path=args.palette,
            palette_mode=args.palette_mode,
            diagnostics_dir=args.diagnostics_dir,
            seed=args.seed,
            steps=args.steps,
            device=args.device,
        )
        return 0

    run_pipeline(
        args.input,
        args.out,
        target_size=args.target_size,
        palette_path=args.palette,
        palette_mode=args.palette_mode,
        diagnostics_dir=args.diagnostics_dir,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
    )
    return 0
