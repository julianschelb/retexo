# retexo/baselines/tables.py
"""The paper's table rows out of the per-fold run JSONs.

``run_baseline.py`` writes one ``runs/<exp>.json`` per fold. A table row is
the mean over the five folds (the definition, section 1.2: one seed per fold,
the spread across folds is the reported spread). ``FoldRow`` reads one run
prefix, ``Table1`` knows which keys the columns of Table 1 come from, and the
module prints a Markdown table:

    python -m retexo.baselines.tables sim_latin_bert sim_ugarit
    python -m retexo.baselines.tables --level V1 --folds 0,1,2,3,4 sim_latin_bert
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: The columns of Table 1 in the definition (section 4.1) and where each lives in a run JSON.
TABLE1_COLUMNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Token acc.", ("token_accuracy",)),
    ("Link P", ("link", "precision")),
    ("Link R", ("link", "recall")),
    ("Link F1", ("link", "f1")),
    ("Op. macro F1", ("ops", "{level}", "macro_f1")),
    ("COPY F1", ("ops", "{level}", "per_class", "COPY", "F1")),
    ("MORPH F1", ("ops", "{level}", "per_class", "MORPH", "F1")),
    ("SUBST F1", ("ops", "{level}", "per_class", "SUBST", "F1")),
    ("INS F1", ("ops", "{level}", "per_class", "INS", "F1")),
    ("DEL F1", ("ops", "{level}", "per_class", "DEL", "F1")),
    ("FRAME span F1", ("frame_span", "F1")),
)


@dataclass
class FoldRow:
    """One run prefix over its folds: ``runs/<prefix>_f<k>.json``."""

    prefix: str
    scores: Dict[int, dict]

    @classmethod
    def load(
        cls, prefix: str, *, runs: Path = Path("runs"), folds: Sequence[int] = (0, 1, 2, 3, 4)
    ) -> FoldRow:
        scores = {}
        for k in folds:
            path = runs / f"{prefix}_f{k}.json"
            if path.exists():
                scores[k] = json.loads(path.read_text())
        return cls(prefix, scores)

    @property
    def folds(self) -> List[int]:
        return sorted(self.scores)

    def value(self, k: int, keys: Sequence[str], *, level: str) -> Optional[float]:
        node = self.scores[k]
        for key in keys:
            key = key.format(level=level)
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return float(node) if isinstance(node, (int, float)) else None

    def cell(
        self, keys: Sequence[str], *, level: str
    ) -> Tuple[Optional[float], Optional[float], int]:
        """``(mean, std, n)`` over the folds that have the value."""
        values = [
            v for v in (self.value(k, keys, level=level) for k in self.folds) if v is not None
        ]
        if not values:
            return None, None, 0
        return (
            statistics.fmean(values),
            (statistics.stdev(values) if len(values) > 1 else 0.0),
            len(values),
        )


class Table1:
    """Table 1's columns for a list of run prefixes.

    Example:
        ```python
        print(Table1(level="V1").markdown(["sim_latin_bert", "sim_ugarit"]))
        ```
    """

    def __init__(
        self,
        *,
        level: str = "V1",
        runs: Path = Path("runs"),
        folds: Sequence[int] = (0, 1, 2, 3, 4),
        with_std: bool = True,
    ):
        self.level = level
        self.runs = runs
        self.folds = tuple(folds)
        self.with_std = with_std

    def rows(self, prefixes: Sequence[str]) -> List[FoldRow]:
        return [FoldRow.load(p, runs=self.runs, folds=self.folds) for p in prefixes]

    def markdown(self, prefixes: Sequence[str]) -> str:
        header = "| Method | folds | " + " | ".join(name for name, _ in TABLE1_COLUMNS) + " |"
        lines = [header, "|" + "---|" * (len(TABLE1_COLUMNS) + 2)]
        for row in self.rows(prefixes):
            cells = []
            for _, keys in TABLE1_COLUMNS:
                mean, std, n = row.cell(keys, level=self.level)
                if mean is None:
                    cells.append("–")
                elif self.with_std and n > 1:
                    cells.append(f"{mean:.3f} ± {std:.3f}")
                else:
                    cells.append(f"{mean:.3f}")
            lines.append(
                f"| {row.prefix} | {','.join(map(str, row.folds)) or '–'} | "
                + " | ".join(cells)
                + " |"
            )
        return "\n".join(lines)

    def per_fold(self, prefix: str) -> str:
        """The same columns fold by fold, for the appendix and for spotting an outlier."""
        row = FoldRow.load(prefix, runs=self.runs, folds=self.folds)
        lines = [
            "| fold | " + " | ".join(name for name, _ in TABLE1_COLUMNS) + " |",
            "|" + "---|" * (len(TABLE1_COLUMNS) + 1),
        ]
        for k in row.folds:
            cells = [
                ("–" if (v := row.value(k, keys, level=self.level)) is None else f"{v:.3f}")
                for _, keys in TABLE1_COLUMNS
            ]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Table 1 rows from per-fold run JSONs")
    parser.add_argument(
        "prefixes",
        nargs="+",
        help="run prefixes, e.g. sim_latin_bert (reads runs/sim_latin_bert_f<k>.json)",
    )
    parser.add_argument(
        "--level", default="V1", help="operation level for the op columns (V0, V1, V3, mode, group)"
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--per-fold", action="store_true", help="also print every prefix fold by fold"
    )
    parser.add_argument("--no-std", action="store_true")
    args = parser.parse_args(argv)
    folds = [int(k) for k in args.folds.split(",")]
    table = Table1(level=args.level, runs=Path(args.runs), folds=folds, with_std=not args.no_std)
    print(table.markdown(args.prefixes))
    if args.per_fold:
        for prefix in args.prefixes:
            print(f"\n{prefix}\n{table.per_fold(prefix)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
