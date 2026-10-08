# baselines/span_pair/__init__.py
"""Note 21: the span-pair scorer, Seq2Edits' idea for our setting.

Instead of a label per word, the model decides at the level of *runs*: "this
stretch of the reuse comes from that stretch of the source, by this kind of
change", then names the words inside. Seq2Edits (Stahlberg & Kumar 2020) emits
(tag, span end, replacement) triples autoregressively; here the replacement is
a *reference* to a source span, so the cheap and faithful form is a scorer
over candidate span pairs:

1. a **substrate** gives one vector per word of the jointly encoded pair (the
   row's own encoder, trained jointly) and a word grid for pruning;
2. **candidate pairs**: reuse spans of length 1 to ``L_max`` against source spans
   of length 1 to ``L_max + 2`` whose rectangle holds a grid cell above ``eps``
   or an equal normalised form, capped at ``K_pairs`` per record; every reuse
   span also pairs with the two nulls INS and FRAME;
3. a **span-pair head** scores each pair for the span tags QUOTE and ADAPT and
   each reuse span for the two nulls, one softmax per reuse span over
   ``{INS, FRAME} u {(source span, tag)}``, the pointer's cell softmax lifted
   to spans; a **word-type head** names the word pairs inside ADAPT spans;
4. **targets** are the record's maximal monotone runs (QUOTE when every edge is
   COPY, ADAPT otherwise, INS and FRAME for sourceless runs; a crossing or a
   gap starts a new run);
5. **decoding** is a segmentation dynamic programme over reuse positions (every
   position covered by exactly one chosen span, scores are log probabilities,
   null spans of length one always available) with a greedy one-to-one repair
   on the source side, then the expansion of each chosen pair to word links
   (position-wise on equal lengths, the Hungarian inside the rectangle
   otherwise, an enclitic SPLIT/MERGE on a length mismatch of one).

ADAPT is an internal segmentation label: it never reaches an edge. The class
overrides ``postprocess`` with its own segmentation; the word rows it emits
serve the dump and the ``word_decoder=1`` ablation.

    python run_baseline.py --method span_pair --fold 4 --smoke 20 --device cpu --extra gold_passes=1
    python run_baseline.py --method span_pair --set multimwa_mtref --base-model bert-base-cased \\
        --extra gold_passes=6,swap=double,L_max=6

The single module is now a package; each concern lives in its own submodule and
``__init__`` re-exports every public name, so ``from
retexo.baselines.span_pair import X`` keeps working unchanged:

- ``constants`` — the span tags ``QUOTE``, ``ADAPT``, ``INS``, ``FRAME``,
  ``NONE``, ``PAIR_TAGS``, ``NULL_TAGS``, the word types ``WORD_TYPES``, the
  dials ``SPAN_DEFAULTS``, the ``Span`` alias and ``ENCLITICS``;
- ``runs`` — ``Run`` and ``RunReader`` (the record's gold as runs);
- ``candidates`` — ``SpanEnumerator`` (the spans and the pruned pairs);
- ``head`` — ``SpanPairHead`` (the pair, null and word-type scorers);
- ``substrate`` — ``Substrate`` (the joint encoder and the pruning grid);
- ``decoding`` — ``SpanChoice``, ``Segmenter`` and ``Expander``;
- ``scorer`` — ``SpanPairScorer`` (the method row itself).
"""

from retexo.baselines import BaselineRegistry, labels
from retexo.baselines.base import Baseline, BaselineConfig, Prediction, Rows
from retexo.baselines.record import Edge, Record, RecordInterface
from retexo.baselines.span_pair.candidates import SpanEnumerator
from retexo.baselines.span_pair.constants import (
    ADAPT,
    ENCLITICS,
    FRAME,
    INS,
    NONE,
    NULL_TAGS,
    PAIR_TAGS,
    QUOTE,
    SPAN_DEFAULTS,
    WORD_TYPES,
    Span,
)
from retexo.baselines.span_pair.decoding import Expander, Segmenter, SpanChoice
from retexo.baselines.span_pair.head import SpanPairHead
from retexo.baselines.span_pair.runs import Run, RunReader
from retexo.baselines.span_pair.scorer import SpanPairScorer
from retexo.baselines.span_pair.substrate import Substrate
from retexo.baselines.typed_pointer import TypedPointerBaseline
from retexo.core.normalize import normalize

# The names the flat module imported at its top level stay attributes of the
# package, so ``from retexo.baselines.span_pair import X`` resolves as before.

__all__ = [
    "ADAPT",
    "ENCLITICS",
    "Expander",
    "FRAME",
    "INS",
    "NONE",
    "NULL_TAGS",
    "PAIR_TAGS",
    "QUOTE",
    "Run",
    "RunReader",
    "Segmenter",
    "Span",
    "SPAN_DEFAULTS",
    "SpanChoice",
    "SpanEnumerator",
    "SpanPairHead",
    "SpanPairScorer",
    "Substrate",
    "WORD_TYPES",
]
