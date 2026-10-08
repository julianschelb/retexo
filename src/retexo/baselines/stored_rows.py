# retexo/baselines/stored_rows.py
"""Score rows read back from dumps, decoded and typed by the shared decoder and typer.

The row for the merged views of the failure-mode dry run (chain 1b): ``tools/merge_views.py``
writes dumps whose ``pred.top`` / ``pred.rev_top`` hold the merged rows of two methods, and
this method hands them to the driver as if a model had produced them, so theta is tuned on
the dev rows, the test rows are decoded at that theta, the rule typer names the links and the
random-pairing negatives give the invented-link rate -- the whole scorer, no GPU.

    python run_baseline.py --method stored_rows --fold 4 --hold-out-dev --gold-records <records> \\
        --negatives data/gold_full/negatives_random_f4.jsonl --typer rule \\
        --extra test=runs/x/predictions.jsonl,dev=runs/x/dev_predictions.jsonl,negatives=runs/x/negatives.jsonl

A record with no stored row gets the all-null row (no link), never a crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, BaselineConfig, Prediction
from retexo.baselines.record import Record


@BaselineRegistry.register
class StoredRowsBaseline(Baseline):
    """Rows from dumps in, the shared decode and typing out.

    Example:
        ```python
        cfg = BaselineConfig(fold=4, dev_fold=0, extra={"test": "runs/x/predictions.jsonl"})
        preds = StoredRowsBaseline(cfg).predict(records)   # each pred carries the dump's rows
        ```
    """

    name = "stored_rows"
    emits = "scores"
    trainable = False

    #: The ``--extra`` keys that name a dump, in the order the driver needs them; ``also`` holds the rows of
    #: records the driver predicts through ``--predict-also`` (the real pairs a teacher labels).
    DUMP_KEYS = ("test", "dev", "negatives", "also")

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.rows: Dict[str, Prediction] = {}
        for key in self.DUMP_KEYS:
            path = cfg.extra.get(key)
            if path:
                self.load_rows(Path(str(path)))

    def load_rows(self, path: Path) -> int:
        """Read one dump's rows, keyed by record id; returns how many carried rows."""
        from retexo.baselines.adapters import read_dump

        n = 0
        for record, pred in read_dump(path):
            if pred.scores:
                self.rows[record.id] = pred
                n += 1
        return n

    def tune(self, dev: List[Record], *, log=None) -> Dict[str, float]:
        """The decoder dials given on the command line (``frag_scale``, ``gap`` of the fragment decoder)."""
        return {
            k: float(self.cfg.extra[k])
            for k in ("frag_scale", "gap", "split_repair")
            if k in self.cfg.extra
        }

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            stored = self.rows.get(record.id)
            pred = Prediction.empty(record.n_reuse)
            if stored is not None and stored.scores and len(stored.scores) == record.n_reuse:
                pred.scores = [list(row) for row in stored.scores]
                pred.rev_scores = (
                    [list(row) for row in stored.rev_scores] if stored.rev_scores else None
                )
            else:
                pred.scores = [[(-1, 1.0)] for _ in range(record.n_reuse)]
            out.append(pred)
        return out
