# retexo/baselines/typed_pointer.py
"""Note 15: the typed pointer as a method row, architecture only.

Our own method with everything the full system adds switched off: for every
reuse word one softmax over every (source word, operation type) cell plus the
two nulls (INS, FRAME) on the joint encoding ``[CLS] source [SEP] reuse [SEP]``,
trained on gold in both orientations (and on a typed synthetic pool first, when
the corpus that seeds it is present), scored on its own predicted types through
the shared default decoder. No evidence features, no identity bonus, no lemma
re-ranking, no null scale, no repairs, no ensemble, no rater.

The model is ``formulations.typed_pointer.TypedPointer`` (E25 v7) or, for the
"untyped pointer" row of Table 2, ``formulations.change_detector.ChangeDetector``
with ``pointer=True`` and no operation types. The training recipe is the one
``attic/scripts/run_e28_score.train_aligner`` ran, ported: pool, eight gold
passes with 600 pool pairs each and mixed negatives at half the gold count,
both orientations (``swap = double``), encoder at 2e-5 and heads at 1e-3.

The class overrides ``postprocess``: decode with the shared decoder, then name
the links with the model's own typer at the decoded cells, then the frame head
and the deletions; in untyped mode the shared typer names the links instead.

    python run_baseline.py --method typed_pointer --fold 4 --smoke 20 --device cpu --extra size=0,gold_passes=1
    python run_baseline.py --method typed_pointer --set multimwa_mtref --base-model bert-base-cased \\
        --extra typed=0,evidence=0,size=0,gold_passes=6,negatives=none,swap=double
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from retexo.baselines.base import Baseline, BaselineConfig, Prediction
from retexo.baselines.record import Record
from retexo.baselines import BaselineRegistry

#: The recipe's published defaults (``attic/scripts/run_e36.py`` 35-70, ``run_e28_score.py`` 61-246).
RECIPE_DEFAULTS: Dict[str, Any] = {
    "typed": 1, "evidence": 0, "directions": "both", "swap": "double", "negatives": "mixed",
    "neg_ratio": 0.5, "null_weight": 0.2, "size": 25000, "gold_passes": 8, "dense_rate": 0.0,
    "train_on": "all", "refine": "none", "load_base": "", "warm_start": "", "pool_per_pass": 600, "rare_min": 1200,
    "typer_lr": 1e-3, "frame_positive_weight": 3.0, "frame_null_weight": 1.75,
    "mlm_subst": 0, "mlm_model": "bowphs/LaBerta", "fine_operations": "fine", "gold_fine": 1,
    "reorder_inter": 0.0, "reorder_intra": 0.0, "gold_reorder": 0.0, "channels": "", "dep": 0, "channel_lr": 0.0,
    "curriculum": 0, "curriculum_floor": 0.5, "curriculum_candidates": 6000,
    "span_view": 0, "span_weight": 0.5, "vectors": "",
    "norel_share": 0.0, "stem_subst": 0.0, "ins_slot": 0.0, "lexical_share": "",
    "stretch_tower": 0, "stretch_head": 0, "null_fuse": 0.0, "stretch_weight": 1.0,
    "slot_conv": 0, "slot_kernel": 3, "slot_anchor_only": 0,
    "copy_variants": 0.0, "rel_share": "", "frameless_fill": "", "frame_channel": 0, "pair_head": 0.0,
}

#: E34's pair labels: the head's three classes by the record's ``pair_label``.
PAIR_CLASSES = {"no_match": 0, "cit": 1, "cf": 2}

#: Class weights of the name loss: NOP 1, MORPH 2, every lexical class 4.
NAME_CLASS_WEIGHTS = {"NOP": 1.0, "MORPH": 2.0}

#: The Mesham row's small inventory (note 33): train at V1, score at V1.
V1_TYPES = ("NOP", "MORPH", "SUBST")

#: Seed passages for the synthetic pool are kept in this length band (``run_e28_score.py:154``).
SEED_LENGTHS = (8, 40)


# =============================================================================
# The recipe
# =============================================================================


@dataclass(frozen=True)
class Recipe:
    """The dials of note 15 read from ``cfg.extra`` with the published defaults.

    Example:
        ```python
        recipe = Recipe.from_config(cfg)          # cfg.extra = {"typed": 0, "size": 0}
        recipe.typed, recipe.size                 # False, 0
        ```
    """

    typed: bool = True
    evidence: bool = False
    directions: str = "both"
    swap: str = "double"
    negatives: str = "mixed"
    neg_ratio: float = 0.5
    null_weight: float = 0.2
    size: int = 20000
    gold_passes: int = 8
    dense_rate: float = 0.0
    train_on: str = "all"
    refine: str = "none"
    load_base: str = ""
    #: a saved model directory whose weights start the gold passes (the active-learning dry run: the synthetic
    #: stage trained once, every round continues from it on the pairs acquired so far; ``size=0`` beside it)
    warm_start: str = ""
    pool_per_pass: int = 600
    rare_min: int = 1200
    typer_lr: float = 1e-3
    frame_positive_weight: float = 3.0
    frame_null_weight: float = 1.75
    #: note 16's aligner B: dense rewrites with masked-LM proposals in context
    mlm_subst: bool = False
    mlm_model: str = "bowphs/LaBerta"
    #: note 33's Mesham row: ``v1`` trains the typer on (NOP, MORPH, SUBST) instead of the twelve fine types
    fine_operations: str = "fine"
    #: the annotated V3 operations supervise the typer where the gold is annotated at V3 (``0``: the
    #: old path, lexical typing learned from the synthetic pairs only; a Table 4 line)
    gold_fine: bool = True
    #: the failure-mode dry run (chain 1): the generator's inversions (the second fragment first; a
    #: two-word fragment turned around) at these rates, and a reordered copy (halves of the reuse
    #: side swapped, links remapped) of this share of the gold records
    reorder_inter: float = 0.0
    reorder_intra: float = 0.0
    gold_reorder: float = 0.0
    #: chain 2 row 6: word-level lemma / POS / morphology channels in the encoder input
    channels: str = ""
    #: chain 2 row 7b: E41's dependency cells (same UPOS, same relation, heads the same word, both
    #: content) appended to every evidence cell; needs ``evidence=1`` and Stanza's Latin model
    dep: bool = False
    #: the channel tables' learning rate (0 = the fresh heads' 1e-3; the v3 rows use the encoder's 2e-5)
    channel_lr: float = 0.0
    #: failure-mode row 8, version 1: the pool pairs of every gold pass drawn by the model's current
    #: errors (a random floor share, the rest error-weighted from a scored candidate slice)
    curriculum: bool = False
    curriculum_floor: float = 0.5
    curriculum_candidates: int = 6000
    #: failure-mode row 9: the span view on the same encoder, averaged with the pointer at span_weight
    span_view: bool = False
    span_weight: float = 0.5
    #: row 14, evidence coverage: a gensim Word2Vec model over Latin lemmas behind the grid's ``cos`` feature
    #: ("" = the resources' default path, absent on the training server so far -- the feature was dead in every row)
    vectors: str = ""
    #: failure-mode row 15 (the whether levers, 2026-09-19): a share of the slot fillers replaced by the
    #: relation-free residual; derivational relatives by surface stem, tagged POS; the later author's own
    #: word spliced into a reused stretch as INS
    norel_share: float = 0.0
    stem_subst: float = 0.0
    ins_slot: float = 0.0
    #: the share of lexical tags among the generator's substitutions (error analysis 2026-09-26): "" = the generator's
    #: own weights, 0 = inflections only (no synthetic substitution), 1 = no synthetic inflection
    lexical_share: Optional[float] = None
    #: chain 15, the unified test: the span view and a stretch head on their own top layers, the
    #: head's NOMATCH belief fused into the pointer's null logit
    stretch_tower: int = 0
    stretch_head: bool = False
    null_fuse: float = 0.0
    stretch_weight: float = 1.0
    #: chain 16: the slot convolution over the pair's score matrix (hidden channels; 0 = off) and its window
    slot_conv: int = 0
    slot_kernel: int = 3
    slot_anchor_only: bool = False
    #: 2026-09-27 (Synthetic Stage rows, Frame Support dry run): enclitics and spelling variants of copied words as
    #: COPY at this share; the realised share of relation-bearing substitutions among the lexical ones ("" = the
    #: generator's own); ``nolink``: unrelated slot fillers without a kept word nearby emitted as insertions; one more
    #: evidence channel, the in-order anchor support of each cell (``FrameChannel``)
    copy_variants: float = 0.0
    rel_share: Optional[float] = None
    frameless_fill: str = ""
    frame_channel: bool = False
    #: E34, the Downstream pair-head row: the weight of a three-way pair head (no match / cit. / cf.) on the [CLS]
    #: vector, trained jointly on the annotated pairs and the negatives (0 = off); its probabilities go into the dumps
    pair_head: float = 0.0

    @classmethod
    def from_config(cls, cfg: BaselineConfig) -> "Recipe":
        values = {**RECIPE_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in RECIPE_DEFAULTS}}
        if cfg.train_on and "train_on" not in cfg.extra:
            values["train_on"] = cfg.train_on
        return cls(
            typed=bool(int(values["typed"])), evidence=bool(int(values["evidence"])),
            directions=str(values["directions"]), swap=str(values["swap"]), negatives=str(values["negatives"]),
            neg_ratio=float(values["neg_ratio"]), null_weight=float(values["null_weight"]),
            size=int(values["size"]), gold_passes=int(values["gold_passes"]), dense_rate=float(values["dense_rate"]),
            train_on=str(values["train_on"]), refine=str(values["refine"]), load_base=str(values["load_base"]),
            warm_start=str(values.get("warm_start", "")),
            pool_per_pass=int(values["pool_per_pass"]), rare_min=int(values["rare_min"]),
            typer_lr=float(values["typer_lr"]), frame_positive_weight=float(values["frame_positive_weight"]),
            frame_null_weight=float(values["frame_null_weight"]),
            mlm_subst=bool(int(values["mlm_subst"])), mlm_model=str(values["mlm_model"]),
            fine_operations=str(values["fine_operations"]), gold_fine=bool(int(values["gold_fine"])),
            reorder_inter=float(values["reorder_inter"]), reorder_intra=float(values["reorder_intra"]),
            gold_reorder=float(values["gold_reorder"]), channels=str(values["channels"]).replace("+", ","),
            dep=bool(int(values["dep"])), channel_lr=float(values["channel_lr"]),
            curriculum=bool(int(values["curriculum"])), curriculum_floor=float(values["curriculum_floor"]),
            curriculum_candidates=int(values["curriculum_candidates"]),
            span_view=bool(int(values["span_view"])), span_weight=float(values["span_weight"]),
            vectors=str(values["vectors"]),
            norel_share=float(values["norel_share"]), stem_subst=float(values["stem_subst"]),
            ins_slot=float(values["ins_slot"]),
            lexical_share=(None if str(values.get("lexical_share", "")).strip() in ("", "none", "None") else float(values["lexical_share"])),
            stretch_tower=int(values["stretch_tower"]), stretch_head=bool(int(values["stretch_head"])),
            null_fuse=float(values["null_fuse"]), stretch_weight=float(values["stretch_weight"]),
            slot_conv=int(values["slot_conv"]), slot_kernel=int(values["slot_kernel"]),
            slot_anchor_only=bool(int(values["slot_anchor_only"])),
            copy_variants=float(values["copy_variants"]),
            rel_share=(None if str(values.get("rel_share", "")).strip() in ("", "none", "None") else float(values["rel_share"])),
            frameless_fill=str(values["frameless_fill"]), frame_channel=bool(int(values["frame_channel"])),
            pair_head=float(values["pair_head"]))

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# =============================================================================
# The model
# =============================================================================


class PointerFactory:
    """Builds the untrained model a recipe asks for.

    Example:
        ```python
        model = PointerFactory.build(recipe, cfg)      # TypedPointer or ChangeDetector
        ```
    """

    @staticmethod
    def feature_dim(recipe: Recipe) -> int:
        if not recipe.evidence:
            return 0
        from retexo.edit_typing.dep_features import DEP_FEATURES
        from retexo.edit_typing.link_features import N_FEATURES

        return int(N_FEATURES) + (len(DEP_FEATURES) if recipe.dep else 0) + (1 if recipe.frame_channel else 0)

    @classmethod
    def build(cls, recipe: Recipe, cfg: BaselineConfig):
        from retexo.datasets import synthetic as syn
        from retexo.formulations.change_detector import ChangeDetector, ChangeDetectorConfig
        from retexo.formulations.typed_pointer import TypedPointer

        fine_ops = (tuple(syn.FINE_OPERATIONS) if recipe.fine_operations == "fine" else V1_TYPES) if recipe.typed else ()
        fine_weights = tuple((op, NAME_CLASS_WEIGHTS.get(op, 4.0)) for op in fine_ops)
        config = ChangeDetectorConfig(
            base_model=cfg.base_model, pooling="mean", device=cfg.device, epochs=1,
            batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, seed=42 + cfg.seed,
            max_length=cfg.max_length, source_head=False, operations=(), pointer=True, pointer_style="dot",
            null_pointer_weight=recipe.null_weight, fine_operations=fine_ops,
            feature_dim=cls.feature_dim(recipe), use_link_features=recipe.evidence,
            fine_class_weights=fine_weights, frame_head=True,
            frame_positive_weight=recipe.frame_positive_weight, typer_lr=recipe.typer_lr,
            typed_pointer=recipe.typed, channels=recipe.channels, channel_lr=recipe.channel_lr,
            span_view=recipe.span_view, span_weight=recipe.span_weight,
            stretch_tower=recipe.stretch_tower, stretch_head=recipe.stretch_head,
            null_fuse=recipe.null_fuse, stretch_weight=recipe.stretch_weight,
            slot_conv=recipe.slot_conv, slot_kernel=recipe.slot_kernel, slot_anchor_only=recipe.slot_anchor_only)
        if not recipe.typed:
            return ChangeDetector(config)
        model = TypedPointer(config)
        model.frame_null_weight = recipe.frame_null_weight
        model.name_loss_weight = 0.0                       # v7
        model.factorized = True
        model.refine_mode = recipe.refine
        model.pair_head_weight = recipe.pair_head
        return model


# =============================================================================
# The training data
# =============================================================================


class TrainingData:
    """Gold examples, their swapped twins, the synthetic pool and the negatives
    of the recipe, each stage skipped with a log line when its material is
    absent rather than silently.

    Example:
        ```python
        data = TrainingData(recipe, cfg, featurizer)
        gold = data.gold_examples(train_records)             # both orientations
        pool = data.synthetic_pool(train_records, log=print)  # [] when nothing seeds it
        ```
    """

    def __init__(self, recipe: Recipe, cfg: BaselineConfig, featurizer=None, *, held_texts: Sequence[str] = ()):
        self.recipe = recipe
        self.cfg = cfg
        self.featurizer = featurizer
        #: passages no synthetic seed and no negative may use: the validation sample's (and any
        #: other held record's); the test and dev folds' come from the benchmark labels by fold
        self.held_texts = list(held_texts)

    @property
    def held_folds(self) -> Tuple[int, ...]:
        """The test fold and the dev fold, never training material."""
        return tuple(sorted({f for f in (self.cfg.fold, self.cfg.dev_fold) if f is not None and f >= 0}))

    def held_passages(self, data) -> Set[str]:
        """Every normalised passage of the held-out folds and of ``held_texts``."""
        from retexo.core.normalize import normalize

        out = {normalize(t) for t in self.held_texts}
        if data is not None and self.held_folds:
            labels = data.labels[data.labels.fold_id.isin(self.held_folds)]
            for column in ("text_query_cleaned", "text_corpus_cleaned"):
                out |= {normalize(t) for t in labels[column].astype(str)}
        return out

    # ---------- gold ----------

    @staticmethod
    def sure_only(record: Record) -> Record:
        """The record with its possible links dropped (they become nulls)."""
        return replace(record, links=[e for e in record.links if e.sure])

    def gold_examples(self, records: Sequence[Record], *, swap: Optional[str] = None) -> List:
        """The records as typed examples, in the orientations ``swap`` asks for."""
        from retexo.aligners.agreement import PairSwap
        from retexo.baselines.adapters import PredictionAdapter
        from retexo.baselines.record import RecordCodec

        swap = self.recipe.swap if swap is None else swap
        rng = random.Random(1234 + self.cfg.seed)
        # always through fine_from_gold: without a featurizer to_example returns the plain
        # GoldPair view, which carries no fine operations and no frame labels, and the
        # typer and the frame head would then train on nothing
        featurizer = self.featurizer or PredictionAdapter._stub_featurizer()
        out = []
        for record in records:
            if self.recipe.train_on == "sure":
                record = self.sure_only(record)
            example = RecordCodec.to_example(record, featurizer, supervise_v3=None if self.recipe.gold_fine else False)
            if self.recipe.pair_head > 0:
                # E34: the pair's class for the pair head; swapped copies inherit it below (the relation is symmetric)
                object.__setattr__(example, "pair_label", PAIR_CLASSES.get(str(record.pair_label).rstrip("."), 0))
            if self.recipe.fine_operations == "v1" and example.fine_operations:
                # the Mesham row's typer knows NOP / MORPH / SUBST only: an annotated lexical operation collapses
                example = replace(example, fine_operations=[op if op in V1_TYPES or op in ("?", "INS") else "SUBST"
                                                            for op in example.fine_operations])
            if swap == "double":
                out.append(example); out.append(self._inherit(PairSwap.labelled(example), example))
            elif swap == "random":
                out.append(self._inherit(PairSwap.labelled(example), example) if rng.random() < 0.5 else example)
            elif swap == "dup":
                out.append(example); out.append(example)
            else:
                out.append(example)
        return out

    @staticmethod
    def _inherit(copy, example):
        """A swapped copy keeps the pair head's label."""
        if hasattr(example, "pair_label"):
            object.__setattr__(copy, "pair_label", example.pair_label)
        return copy

    # ---------- the corpus behind the pool and the negatives ----------

    @staticmethod
    def benchmark():
        """``BenchmarkData`` when its tables are present, else ``None``."""
        try:
            from retexo.datasets.dataset import BenchmarkData

            return BenchmarkData.load()
        except (FileNotFoundError, ImportError):
            return None

    @staticmethod
    def seeds_from(data, records: Sequence[Record]) -> List[List[str]]:
        """Seed passages: the corpus when present, else the training records'
        own passages (the source side of every pair is Latin text too)."""
        if data is not None:
            texts = [t.split() for t in data.corpus_texts()]
        else:
            texts = [list(r.source_tokens) for r in records] + [list(r.reuse_tokens) for r in records]
        lo, hi = SEED_LENGTHS
        seeds = [s for s in texts if lo <= len(s) <= hi]
        random.Random(0).shuffle(seeds)
        return seeds

    def synthetic_pool(self, records: Sequence[Record], *, log=None) -> List:
        """The typed synthetic pool of ``recipe.size`` pairs, or ``[]`` when
        ``size`` is 0 or nothing can seed the generator."""
        if self.recipe.size <= 0:
            return []
        from retexo.datasets import synthetic as syn

        data = self.benchmark()
        seeds = self.seeds_from(data, records)
        held = self.held_passages(data)
        if held:
            from retexo.core.normalize import normalize

            before = len(seeds)
            seeds = [s for s in seeds if normalize(" ".join(s)) not in held]
            if log:
                log(f"[typed_pointer] synthetic seeds: {before - len(seeds)} held-out passages excluded "
                    f"(test fold, dev fold, validation sample), {len(seeds)} left")
        if len(seeds) < 20:
            if log:
                log("[typed_pointer] synthetic pool skipped: fewer than 20 seed passages")
            return []
        frames = self._frames(records)
        # generate over as many seeds as the pool needs (three candidates per kept pair, for the rare-op floor of
        # ``select_pool``), not over the whole corpus: 64,195 seeds x 5 pairs took 14 minutes per run, and with the
        # masked-LM rewrites of aligner B the per-word precompute over every seed ran for hours (dry run, 2026-09-16)
        attested = syn.attested_forms(seeds, min_count=3)
        wanted = max(2000, 3 * self.recipe.size // 5)
        if len(seeds) > wanted:
            seeds = random.Random(11 + self.cfg.seed).sample(seeds, wanted)
        try:
            pool, report = syn.generate_typed(
                seeds, seeds[:8000], per_seed=5, seed=self.cfg.seed, workers=self.cfg.workers,
                attested=attested, frames=frames,
                frame_rate=float(self.cfg.extra.get("frame_rate", syn.FRAME_RATE)), enclitic_rate=syn.ENCLITIC_RATE,
                reorder_inter=self.recipe.reorder_inter, reorder_intra=self.recipe.reorder_intra,
                dense_rate=self.recipe.dense_rate,
                mlm_subst=self.recipe.mlm_subst, mlm_model=self.recipe.mlm_model, log=log,
                norel_share=self.recipe.norel_share, stem_subst=self.recipe.stem_subst, ins_slot=self.recipe.ins_slot,
                lexical_share=self.recipe.lexical_share, rel_share=self.recipe.rel_share,
                copy_variants=self.recipe.copy_variants, frameless_fill=self.recipe.frameless_fill)
        except Exception as error:                      # the generator's resources are optional here
            if log:
                log(f"[typed_pointer] synthetic pool skipped: {type(error).__name__}: {error}")
            return []
        if log:
            fine = report.fine_counts
            related = sum(fine.get(t, 0) for t in syn.RELATED_TAGS)
            lexical = related + fine.get("SUBST", 0)
            log(f"[typed_pointer] generator: related among lexical substitutions {related}/{lexical} "
                f"({related / max(lexical, 1):.3f}); spelling variants as COPY {report.spelling} of "
                f"{fine.get('NOP', 0)} copies ({report.spelling / max(fine.get('NOP', 0), 1):.3f}); enclitic events "
                f"{report.enclitics}; frameless fillers unlinked {report.frameless}")
        pool = syn.select_pool(pool, self.recipe.size, random.Random(7 + self.cfg.seed), rare_min=self.recipe.rare_min)
        if log:
            log(f"[typed_pointer] synthetic pool: {len(pool)} pairs from {len(seeds)} seeds "
                f"({'corpus' if data is not None else 'training passages'})")
        return pool

    @staticmethod
    def _frames(records: Sequence[Record]) -> List[List[str]]:
        """Attribution formulas from the training records' FRAME spans."""
        out = []
        for record in records:
            for span in record.spans:
                if span.label == "FRAME":
                    words = list(record.reuse_tokens[span.start:span.end])
                    if words:
                        out.append(words)
        return out

    def negatives(self, n: int, *, log=None) -> List:
        """``n`` mixed negatives from the corpus, or ``[]`` when the corpus is absent."""
        if self.recipe.negatives == "none" or n <= 0:
            return []
        data = self.benchmark()
        if data is None:
            if log:
                log("[typed_pointer] negatives skipped: BenchmarkData not available")
            return []
        from retexo.datasets.negatives import KINDS, NegativeBuilder

        kinds = KINDS if self.recipe.negatives == "mixed" else (self.recipe.negatives,)
        out = NegativeBuilder(data, held_out=self.held_folds or self.cfg.fold, exclude_texts=self.held_texts
                              ).build(n, kinds=kinds, seed=self.cfg.seed, for_training=True)
        if log:
            log(f"[typed_pointer] negatives: {len(out)} ({self.recipe.negatives})")
        return out

    # ---------- evidence ----------

    def featurize(self, examples: Sequence, *, log=None) -> None:
        """The evidence grid at every cell, only when the recipe reads it."""
        if not self.recipe.evidence or not examples:
            return
        from retexo.datasets import synthetic as syn

        syn.DEP_FEATURES_ON = bool(self.recipe.dep)      # E41's four cells, parsed once per passage (Stanza la)
        syn.FRAME_CHANNEL_ON = bool(self.recipe.frame_channel)
        syn.featurize_pairs(list(examples), workers=self.cfg.workers, log=log,
                            vectors_path=self.recipe.vectors or None)


# =============================================================================
# The method row
# =============================================================================


@BaselineRegistry.register
class TypedPointerBaseline(Baseline):
    """"Typed pointer, ours, architecture only": Table 1's pointer row and the
    end of Table 2's strip-back.

    ``cfg.extra`` dials (``RECIPE_DEFAULTS``): ``typed`` (0 = the untyped
    pointer, named by the shared typer), ``evidence`` (1 in note 16),
    ``directions`` (``both`` | ``one``), ``swap`` (``double`` | ``random`` |
    ``dup`` | ``none``), ``negatives`` (``mixed`` | ``none`` | one kind),
    ``neg_ratio``, ``null_weight``, ``size`` (0 = no synthetic stage),
    ``gold_passes``, ``dense_rate``, ``train_on`` (``all`` | ``sure``),
    ``refine`` (note 22), ``load_base`` (a saved model directory).

    Example:
        ```python
        cfg = BaselineConfig(fold=4, dev_fold=0, device="cpu", extra={"size": 0, "gold_passes": 1})
        method = TypedPointerBaseline(cfg).fit(train, dev)
        preds = [method.postprocess(r, p, {"theta": 0.45}) for r, p in zip(test, method.predict(test))]
        # python run_baseline.py --method typed_pointer --fold 4 --extra size=0
        ```
    """

    name = "typed_pointer"
    early_stopping_capable = True
    emits = "scores"
    trainable = True
    typer = "own"
    decoder = "default"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.recipe = Recipe.from_config(cfg)
        self.model = None
        self._evidence = None                # the LinkFeaturizer behind the evidence grid (note 16)
        self._examples: Dict[str, Any] = {}  # record id -> its example, for postprocess
        if not self.recipe.typed:
            self.typer = "rule"

    # ---------- the evidence source ----------

    def evidence_featurizer(self):
        """The Latin resources behind the evidence vector, built once; ``None``
        when the recipe does not read evidence (``self.featurizer`` is the
        driver's, used by the shared typer in untyped mode only)."""
        if not self.recipe.evidence:
            return None
        if self._evidence is None:
            from pathlib import Path

            from retexo.edit_typing.link_features import LinkFeaturizer
            from retexo.resources import Resources

            kw = {"vectors_path": Path(self.recipe.vectors)} if self.recipe.vectors else {}
            self._evidence = LinkFeaturizer(Resources(offline=True, **kw))
        return self._evidence

    def data(self) -> TrainingData:
        return TrainingData(self.recipe, self.cfg, self.evidence_featurizer(), held_texts=self.held_texts)

    @property
    def held_texts(self) -> List[str]:
        """The validation sample's passages (both sides), kept out of the pool and the negatives."""
        return [" ".join(tokens) for r in self.validation for tokens in (r.source_tokens, r.reuse_tokens)]

    # ---------- training ----------

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()
           ) -> "TypedPointerBaseline":
        if self.recipe.load_base:
            return self.load_into(Path(self.recipe.load_base), log=log)
        data = self.data()
        if self.recipe.gold_reorder > 0:
            from retexo.baselines.augment import GoldReorder

            before = len(train)
            train = GoldReorder.augment(train, rate=self.recipe.gold_reorder, seed=self.cfg.seed)
            if log:
                log(f"[typed_pointer] gold reorder {self.recipe.gold_reorder}: {len(train) - before} reordered copies of {before} records")
        gold = data.gold_examples(train)
        if not gold:
            return self
        n_gold_for_negatives = len(gold)                 # after swapping: run_e28_score's n_for_neg
        shared = bool(self.shared_synthetic) and not int(self.cfg.extra.get("own_pool", 0))
        if shared:
            # the shared training set's pool (the paper's rows read one pool per fold); ``own_pool=1``
            # generates the recipe's own instead, for rows whose generator settings differ
            pool = data.gold_examples(self.shared_synthetic[: self.recipe.size], swap="none") if self.recipe.size > 0 else []
            if log:
                log(f"[typed_pointer] synthetic pool: {len(pool)} pairs from the shared training set")
        else:
            pool = data.synthetic_pool(train, log=log)
        if pool and self.recipe.swap == "double":
            from retexo.aligners.agreement import PairSwap

            pool = pool + [PairSwap.labelled(ex) for ex in pool]
        n_negatives = int(round(self.recipe.neg_ratio * n_gold_for_negatives))
        if self.shared_negatives and self.recipe.negatives != "none":
            negatives = data.gold_examples(self.shared_negatives[:n_negatives], swap="none")
            if log:
                log(f"[typed_pointer] negatives: {len(negatives)} from the shared training set")
        else:
            negatives = data.negatives(n_negatives, log=log)
        if self.recipe.pair_head > 0:
            for ex in negatives:                           # E34: every negative is a no-match pair
                object.__setattr__(ex, "pair_label", 0)
        for items in (pool, gold, negatives):
            data.featurize(items, log=log)
        if self.recipe.warm_start:
            self.load_into(Path(self.recipe.warm_start), log=log)     # continue from a saved model
        else:
            self.model = PointerFactory.build(self.recipe, self.cfg)
        if log:
            log(f"[typed_pointer] {'typed' if self.recipe.typed else 'untyped'} pointer on {self.cfg.base_model}: "
                f"{len(gold)} gold examples ({self.recipe.swap}), pool {len(pool)}, negatives {len(negatives)}, "
                f"{self.recipe.gold_passes} gold passes")
        from retexo.baselines.early_stopping import TrainingMonitor

        monitor = TrainingMonitor.for_method(self, log=log)       # the note's metrics over training time
        if pool:
            self.model.fit(pool, log=log, on_batch=monitor.progress("synthetic", int(self.cfg.extra.get("synthetic_evals", 4))) if monitor else None)
            if monitor is not None:
                monitor.evaluate("synthetic", fraction=1.0)
        rng = random.Random(99 + self.cfg.seed)
        n_neg = int(round(self.recipe.neg_ratio * n_gold_for_negatives))
        passes = self.recipe.gold_passes                    # 0 = the synthetic-only regime of note 33
        if pool == [] and passes <= 0:
            passes = 1                                      # nothing else to train on
        if self.cfg.smoke:
            passes = min(passes, 2)
        from retexo.baselines.early_stopping import EarlyStopping

        stopper = EarlyStopping.for_method(self, log=log) if self.recipe.gold_passes > 0 else None
        if stopper is not None:
            passes = stopper.max_epochs                     # the gold passes become early-stopped epochs
        curriculum = None
        if self.recipe.curriculum and pool:
            from retexo.baselines.curriculum import ErrorCurriculum

            curriculum = ErrorCurriculum(floor=self.recipe.curriculum_floor, candidates=self.recipe.curriculum_candidates,
                                         seed=self.cfg.seed)
        for pass_no in range(1, passes + 1):
            batch = list(gold)
            if negatives:
                batch += rng.sample(negatives, min(len(negatives), n_neg))
            if pool:
                k = min(self.recipe.pool_per_pass, len(pool))
                batch += curriculum.draw(self.model, pool, k, log=log) if curriculum is not None else rng.sample(pool, k)
            # the pass's loss line goes through ``log`` so the run's curve keeps it (CurveRecorder)
            self.model.fit(batch, log=(lambda m, k=pass_no: log(f"[typed_pointer] gold pass {k}/{passes}: {m.strip()}")) if log else None)
            if log:
                log(f"[typed_pointer] gold pass {pass_no}/{passes} ({len(batch)} examples)")
            if stopper is not None and not stopper.step(pass_no, self.modules()):
                break
        if stopper is not None:
            stopper.restore(self.modules())
            stopper.release()
        return self

    def modules(self) -> Dict[str, Any]:
        """Every trainable module of the model, by attribute name (what ``save`` writes)."""
        import torch

        if self.model is None:
            return {}
        return {name: mod for name, mod in vars(self.model).items() if isinstance(mod, torch.nn.Module)}

    # ---------- inference ----------

    def validation_loss(self, records: Sequence[Record]) -> Optional[float]:
        """The model's training loss on ``records`` (one orientation, no gradient)."""
        return self.model.evaluation_loss(self.examples_of(records)) if self.model is not None else None

    def examples_of(self, records: Sequence[Record]) -> List:
        """The records' examples, cached by id so ``postprocess`` finds them."""
        data = self.data()
        out = []
        for record in records:
            if record.id not in self._examples:
                self._examples[record.id] = data.gold_examples([record], swap="none")[0]
            out.append(self._examples[record.id])
        if self.recipe.evidence:
            data.featurize([e for e in out if getattr(e, "pair_features", None) is None])
        return out

    def predict(self, records: List[Record]) -> List[Prediction]:
        from retexo.aligners.agreement import PairSwap

        preds = [Prediction.empty(r.n_reuse) for r in records]
        if self.model is None:
            return preds
        live = [(i, r) for i, r in enumerate(records) if r.source_tokens and r.reuse_tokens]
        if not live:
            return preds
        examples = self.examples_of([r for _, r in live])
        forward = self.model.predict_alignment_scores(examples)
        reverse = None
        if self.recipe.directions == "both":
            swapped = [PairSwap.example(e) for e in examples]
            if self.recipe.evidence:
                self.data().featurize(swapped)
            reverse = self.model.predict_alignment_scores(swapped)
        head = self.model.predict_pair(examples) if self.recipe.pair_head > 0 and self.recipe.typed else None
        for k, (i, record) in enumerate(live):
            if head is not None:
                preds[i].meta["pair_head"] = [float(p) for p in head[k]]     # [p(no match), p(cit.), p(cf.)]
            preds[i].scores = [[(int(s), float(p)) for s, p in row] for row in forward[k]]
            if reverse is not None:
                preds[i].rev_scores = [[(int(s), float(p)) for s, p in row] for row in reverse[k]]
        return preds

    def postprocess(self, record: Record, pred: Prediction, dials: Dict[str, float]) -> Prediction:
        """Decode, then name at the decoded cells (typed) or through the shared
        typer (untyped); then the frame head and the deletions."""
        from retexo.baselines.adapters import PredictionAdapter

        pred = PredictionAdapter.decode_prediction(pred, self.decoder, dials, record)
        if pred.scores is None:
            return pred
        if self.model is None:
            # a re-decode from a dump (``--from-dump``): no model to type with, the shared typer names the links
            return PredictionAdapter.type_prediction(pred, record, "rule" if self.typer == "own" else self.typer,
                                                     self.featurizer, frame_rule=str(dials.get("frame_rule", "keyword")),
                                                     head=getattr(self, "typer_head", None))
        example = self.examples_of([record])[0]
        # ``--typer rule`` / ``trained`` on a typed model: the model's links, the shared typer's names
        # (Table 2's "own typer vs rule typer" line); ``own`` (the class default) is the model's typer
        if self.recipe.typed and self.typer == "own":
            pred.tags = [tag if s >= 0 else "" for tag, s in
                         zip(self.model.predict_typed([example], [pred.links])[0], pred.links)]
            pred.frame = self.model.predict_frames([example], [pred.links])[0]
            pred.frame_p = self.frame_probabilities(example)
            pred.dels = self.model.predict_source([example], [pred.links])[0]
        else:
            pred = PredictionAdapter.type_prediction(pred, record, self.typer, self.featurizer,
                                                     frame_rule=str(dials.get("frame_rule", "keyword")),
                                                     head=getattr(self, "typer_head", None))
            head_frames = self.model.predict_frames([example])[0]
            pred.frame = [int(f) if s < 0 else 0 for f, s in zip(head_frames, pred.links)]
            used = {s for s in pred.links if s >= 0}
            pred.dels = [0 if s in used else 1 for s in range(record.n_source)]
        return pred

    def frame_probabilities(self, example) -> List[Optional[float]]:
        """The frame head's probability per reuse word (``dump_predictions._frame_probs``)."""
        import torch

        words = self.model.predict_cells([example])[0]
        out = []
        for w in words:
            head = None if w is None else w[4]
            out.append(None if head is None else round(float(torch.softmax(head.float(), dim=0)[1]), 4))
        return out

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        """Every module's weights plus the recipe (``run_e28_score.save_aligner``)."""
        import json

        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            return
        modules = {name: mod.state_dict() for name, mod in vars(self.model).items() if isinstance(mod, torch.nn.Module)}
        torch.save(modules, path / "modules.pt")
        (path / "recipe.json").write_text(json.dumps({"recipe": self.recipe.as_dict(), "base_model": self.cfg.base_model},
                                                     indent=1))

    def load_into(self, path: Path, *, log=None) -> "TypedPointerBaseline":
        import torch

        self.model = PointerFactory.build(self.recipe, self.cfg)
        modules = torch.load(Path(path) / "modules.pt", map_location=self.cfg.device)
        for name, state in modules.items():
            getattr(self.model, name).load_state_dict(state)
        if log:
            log(f"[typed_pointer] weights loaded from {path} ({len(modules)} modules)")
        return self

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> "TypedPointerBaseline":
        return cls(cfg).load_into(Path(path))
