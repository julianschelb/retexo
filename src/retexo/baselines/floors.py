# retexo/baselines/floors.py
"""The two degenerate floors Table 1 reports in the text.

*Do nothing* (Gu et al. 2019): every reuse word is an insertion, every source
word a deletion; its token accuracy is the share of unlinked reuse words,
about 0.84 on the gold, which is why token accuracy alone flatters. *Copy the
input* (Raheja et al. 2023): identical forms are linked left to right, one to
one and monotone, so a repeated *et* goes to the next unused *et*; everything
else is an insertion. Both return their prediction from ``postprocess``
unchanged: there is nothing to decode and the tags are already known.
"""

from __future__ import annotations

from typing import Dict, List

from retexo.baselines import BaselineRegistry
from retexo.baselines.base import Baseline, Prediction
from retexo.baselines.record import Record
from retexo.core.normalize import normalize


@BaselineRegistry.register
class DoNothing(Baseline):
    """Every reuse token NOMATCH; the floor every row must beat.

    Example:
        ```python
        preds = DoNothing(cfg).predict(records)   # links all -1, tags all INS
        # python run_baseline.py --method do_nothing --fold 4 --smoke 20 --device cpu
        ```
    """

    name = "do_nothing"
    emits = "edges"
    trainable = False

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            pred = Prediction.empty(record.n_reuse)
            pred.tags = ["INS"] * record.n_reuse
            out.append(pred)
        return out

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        return pred


@BaselineRegistry.register
class CopyInput(Baseline):
    """Identical forms linked one to one, left to right; the copy floor.

    Example:
        ```python
        preds = CopyInput(cfg).predict(records)   # COPY on identical forms, INS elsewhere
        # python run_baseline.py --method copy_input --fold 4 --smoke 20 --device cpu
        ```
    """

    name = "copy_input"
    emits = "edges"
    trainable = False

    def predict(self, records: List[Record]) -> List[Prediction]:
        out = []
        for record in records:
            source = [normalize(w) for w in record.source_tokens]
            pred = Prediction.empty(record.n_reuse)
            pred.tags = ["INS"] * record.n_reuse
            cursor = 0
            for t, word in enumerate(record.reuse_tokens):
                form = normalize(word)
                for s in range(cursor, len(source)):
                    if source[s] == form:
                        pred.links[t] = s
                        pred.tags[t] = "COPY"
                        cursor = s + 1
                        break
            out.append(pred)
        return out

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        return pred
