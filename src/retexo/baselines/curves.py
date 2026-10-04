# retexo/baselines/curves.py
"""Training curves out of the driver's log, and their plots.

Every trainer in ``retexo/baselines`` reports its progress through the
``log`` callable the driver hands to ``fit`` -- lines such as
``    epoch 3/8  loss 0.4123`` (``ChangeDetector``), ``[span_aligner] epoch
2/3: loss 0.51`` or ``[span_pair] pass 4/6: loss 0.20`` -- and nothing keeps
them. ``CurveRecorder`` sits between the driver and its log, parses those
lines and holds the points, so that the run JSON can carry the curve
(``result["curve"]``) and a note can show it:

    python -m retexo.baselines.curves runs/dry_typed_pointer_f4.json --png assets/typed_pointer_f4.png
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

#: ``epoch 3/8``, ``pass 2/6``, ``update 500/2000``, ``typer refine 1/3``: the stage words and the step.
_STEP = re.compile(r"(?P<stage>[A-Za-z][A-Za-z _-]*?)\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)")
_LOSS = re.compile(r"loss\s*[:=]?\s*(?P<loss>-?\d+(?:\.\d+)?(?:e-?\d+)?)", re.IGNORECASE)
_TAG = re.compile(r"^\s*\[(?P<tag>[^\]]+)\]")


@dataclass
class CurvePoint:
    """One logged loss: which trainer (``tag``), which stage (``epoch``, ``pass``,
    ``typer refine``), the step within it, its total, and the loss."""

    tag: str
    stage: str
    step: int
    total: int
    loss: float


class CurveRecorder:
    """A ``log`` wrapper that keeps every loss line as a ``CurvePoint``.

    Example:
        ```python
        recorder = CurveRecorder(print)
        baseline.fit(train, dev, log=recorder)
        result["curve"] = recorder.points_as_dicts()
        ```
    """

    def __init__(self, inner: Optional[Callable[[str], None]] = None):
        self._inner = inner
        self.points: List[CurvePoint] = []

    def __call__(self, message: str) -> None:
        if self._inner is not None:
            self._inner(message)
        point = self.parse(message)
        if point is not None:
            self.points.append(point)

    @staticmethod
    def parse(line: str) -> Optional[CurvePoint]:
        """The point a log line carries, or ``None`` when it has no loss or no step."""
        loss = _LOSS.search(str(line))
        if loss is None:
            return None
        tag = _TAG.match(line)
        body = line[tag.end():] if tag else line
        step = _STEP.search(body)
        if step is None:
            return None
        stage = step.group("stage").strip().lower().strip(":")
        return CurvePoint(tag=tag.group("tag") if tag else "", stage=stage or "epoch",
                          step=int(step.group("step")), total=int(step.group("total")),
                          loss=float(loss.group("loss")))

    def points_as_dicts(self) -> List[Dict[str, object]]:
        return [asdict(p) for p in self.points]


class CurvePlot:
    """One line per (tag, stage) over its steps: the PNG a method note embeds."""

    @staticmethod
    def series(points: Sequence[Dict[str, object]]) -> Dict[str, List[tuple]]:
        out: Dict[str, List[tuple]] = {}
        for p in points:
            key = " ".join(str(x) for x in (p.get("tag", ""), p.get("stage", "")) if x).strip() or "loss"
            out.setdefault(key, []).append((int(p["step"]), float(p["loss"])))
        return out

    @classmethod
    def save(cls, points: Sequence[Dict[str, object]], png: Path, *, title: str = "") -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series = cls.series(points)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for name, values in series.items():
            xs = list(range(1, len(values) + 1)) if len({s for s, _ in values}) < len(values) else [s for s, _ in values]
            ax.plot(xs, [l for _, l in values], marker="o", markersize=3, label=name)
        ax.set_xlabel("step (epoch / pass / update, in log order)")
        ax.set_ylabel("loss")
        if title:
            ax.set_title(title, fontsize=9)
        if series:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no loss lines logged", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        png = Path(png)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=130)
        plt.close(fig)
        return png


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 1
    run_json = Path(args[0])
    png = Path(args[args.index("--png") + 1]) if "--png" in args else run_json.with_suffix(".curve.png")
    result = json.loads(run_json.read_text())
    points = result.get("curve", [])
    title = f"{result.get('method', run_json.stem)} ({result.get('where', '')}; fit {result.get('fit_seconds', '?')} s)"
    CurvePlot.save(points, png, title=title)
    print(f"{len(points)} points -> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
