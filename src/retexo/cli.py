"""Command line entry point: ``retexo export`` turns a run's raw predictions into the released format."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from retexo import __version__
from retexo.export import export_folds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retexo", description="Word-level explanations of Latin text reuse."
    )
    parser.add_argument("--version", action="version", version=f"retexo {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser(
        "export",
        help="convert raw predictions.jsonl files (one per test fold) to the released format",
        description="Reads the fold of every record, so the order of the files does not matter.",
    )
    export.add_argument(
        "predictions", nargs="+", help="raw predictions.jsonl files, one per test fold"
    )
    export.add_argument(
        "--out", required=True, help="output folder; writes fold_<k>.jsonl or fold_<k>.parquet"
    )
    export.add_argument("--format", choices=("jsonl", "parquet"), default="jsonl")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        counts = export_folds(args.predictions, args.out, fmt=args.format)
        for fold, n in counts.items():
            print(f"fold {fold}: {n} pairs")
        print(f"{sum(counts.values())} pairs written to {args.out}")
        return 0
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
