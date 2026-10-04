# retexo/datasets/generation.py
"""
Manufacturing training data: sample operations, apply them, keep what is hard.

Two properties of the recipe are easy to lose. A generated script must be the
*cheapest* script relating its pair, not merely one that relates them —
deleting a word and inserting another produces the same pair as substituting a
synonym, at four times the cost, and training on the expensive analysis teaches
the model to over-price the relation the metric is meant to capture. And an
example is worth keeping only if a trivial baseline cannot already recover it,
which is the analogue of retaining only the proofs that needed an auxiliary
construction.

Every stage reports its yield. A generator whose losses are invisible cannot be
debugged from its output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from retexo.core.builder import VariantBuilder
from retexo.core.normalize import normalize
from retexo.operations import EditOperation, OperationRegistry
from retexo.core.scriba import Scriba
from retexo.core.script import EditScript

# =============================================================================
# Substitution content
# =============================================================================


class SubstitutionSource:
    """Supplies the replacement token for a substitution.

    Choosing *that* a synonym is substituted is free; choosing *which* needs a
    lexicon. Implementations differ only in where the replacement comes from.
    """

    #: Tags this source can actually realise. A source asked for a tag outside
    #: this set would return something, but the label would not describe what
    #: it returned, and the generated example would train a distinction the
    #: data does not contain. ``None`` means "no restriction", for sources that
    #: genuinely handle every tag.
    supported_tags: Optional[frozenset] = None

    def realises(self, tag: str) -> bool:
        """Whether a substitution under ``tag`` would mean what it says."""
        return self.supported_tags is None or tag in self.supported_tags

    def replacement(self, token: str, tag: str, rng: random.Random) -> Optional[str]:
        """A replacement for ``token`` under ``tag``, or ``None`` if none exists."""
        raise NotImplementedError

    def replacement_in_context(
        self,
        words: Sequence[str],
        position: int,
        tag: str,
        rng: random.Random,
    ) -> Optional[str]:
        """A replacement that may consult the surrounding words.

        Sources that ignore context fall back to :meth:`replacement`, so a
        context-free source needs no changes to be used here.
        """
        return self.replacement(words[position], tag, rng)


class MockSubstitutionSource(SubstitutionSource):
    """Deterministic pseudo-replacements, for tests and smoke runs.

    Realises every tag, because its "substitutions" are recognisable manglings
    that encode the tag in the token itself. That is exactly why it is unfit
    for training and fit for exercising a pipeline.

    Produces a recognisable, reversible mangling rather than a real synonym, so
    a pipeline can be exercised end to end before the lexical resources are in
    place. Not suitable for training a model that will be believed.
    """

    def __init__(self, suffixes: Optional[Dict[str, str]] = None):
        self.suffixes = suffixes or {
            "SYN": "-syn", "SYN-DIST": "-sim", "HYPER": "-gen",
            "HYPO": "-spec", "ANT": "-ant", "NE-SUB": "-ne",
            "POS": "-der", "MORPH": "-infl",
        }

    def replacement(self, token: str, tag: str, rng: random.Random) -> Optional[str]:
        """A recognisable mangling of the token, not a real synonym."""
        suffix = self.suffixes.get(tag)
        return f"{token}{suffix}" if suffix else None


class LexicalSubstitutionSource(SubstitutionSource):
    """Draws replacements from the resources rather than inventing them.

    Realises the WordNet relations, since it reads them from WordNet -- which
    is also the circularity described below.

    Returns ``None`` when the resources cannot supply a replacement for a tag,
    so the sampler keeps the token unchanged instead of fabricating one. A
    generator that quietly invents Latin is worse than one that declines.

    Note the circularity this creates: drawing substitutions from the same
    WordNet the oracle detects with makes every synthetic example
    oracle-recoverable by construction, so the corpus teaches a model nothing
    the oracle does not already know. Breaking that needs a source wider than
    the detector.
    """

    def __init__(self, resources, fallback: Optional[SubstitutionSource] = None, cohypo: bool = False):
        self.resources = resources
        self.fallback = fallback
        #: E38b: SYN-DIST draws a WordNet co-hyponym first (Moritz et al. 2016's
        #: repl_co-hypo: caelum/polus, pontus/aequor), the vector neighbour second
        self.cohypo = cohypo

    _RELATION = {"SYN": "synonyms", "HYPER": "hypernyms", "HYPO": "hyponyms",
                 "ANT": "antonyms"}

    def _cohyponym(self, token: str, rng: random.Random) -> Optional[str]:
        if not self.resources.has("wordnet"):
            return None
        lemma = self.resources.lemma_or_surface(token)
        for pos in self.resources.pos_candidates(token):
            options = []
            for hyper in self.resources.wordnet.lookup(lemma, pos).get("hypernyms", [])[:6]:
                options += [h for h in self.resources.wordnet.lookup(hyper, pos).get("hyponyms", []) if h != lemma]
            options = sorted(set(options))
            if options:
                return rng.choice(options)
        return None

    def replacement(self, token: str, tag: str, rng: random.Random) -> Optional[str]:
        """A replacement drawn from the resources, or ``None`` if none exists."""
        if tag == "MORPH":
            return self._inflection(token, rng)
        if tag == "SYN-DIST" and self.cohypo:
            co = self._cohyponym(token, rng)
            if co:
                return co
        relation = self._RELATION.get(tag)
        if relation and self.resources.has("wordnet"):
            lemma = self.resources.lemma_or_surface(token)
            for pos in self.resources.pos_candidates(token):
                options = self.resources.wordnet.lookup(lemma, pos).get(relation, [])
                if options:
                    return rng.choice(options)
        if tag == "SYN-DIST" and self.resources.has("vectors"):
            return self._distributional(token, rng)
        return self.fallback.replacement(token, tag, rng) if self.fallback else None

    def _inflection(self, token: str, rng: random.Random) -> Optional[str]:
        """A different attested inflection of the same lemma."""
        if not self.resources.has("morphology"):
            return None
        from retexo.core.normalize import normalize

        try:
            forms = self.resources.morphology._get_decliner().decline(
                self.resources.morphology.lemma(token)
            )
        except Exception:
            return None
        others = [f for f, _ in forms if normalize(f) != normalize(token)]
        return rng.choice(others) if others else None

    def _distributional(self, token: str, rng: random.Random) -> Optional[str]:
        """A near neighbour in the vector space."""
        lemma = self.resources.lemma_or_surface(token)
        try:
            neighbours = self.resources.vectors._get().most_similar(lemma, topn=10)
        except Exception:
            return None
        return rng.choice([w for w, _ in neighbours]) if neighbours else None


# =============================================================================
# Configuration
# =============================================================================

#: Default operation mix. Anchored to what the oracle finds on real pairs
#: rather than uniform, since a uniform draw over the inventory produces
#: variants dominated by rare operations.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "MORPH": 0.30,
    "SYN": 0.20,
    "SYN-DIST": 0.10,
    "HYPER": 0.06,
    "HYPO": 0.04,
    "DEL": 0.12,
    "INS": 0.12,
    "NE-SUB": 0.03,
    "ANT": 0.03,
}


@dataclass(frozen=True)
class GenerationConfig:
    """Knobs for the sampler."""

    #: Operation counts to draw from, and their weights: the difficulty dial.
    op_counts: Tuple[int, ...] = (1, 2, 3, 4, 5)
    op_count_weights: Optional[Tuple[float, ...]] = None

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    variants_per_seed: int = 4
    seed: int = 42

    #: Discard examples a trivial baseline recovers. Off by default: the
    #: threshold is uncalibrated and discarding is irreversible, so difficulty
    #: is recorded per example instead (see ``Difficulty``).
    filter_trivial: bool = False

    #: Re-plan each generated pair with the oracle and train on *its* script.
    #: Requires an oracle; without one the applied script is used unchanged.
    canonicalize: bool = True

    #: Probability of embedding the variant in unrelated context, framed. Real
    #: reuse is a fragment inside the citing author's own prose, and a
    #: generator that returns same-length copies teaches the opposite
    #: phenomenon (DIAGNOSIS.md §4). Below 1.0 so bare pairs stay represented.
    frame_probability: float = 0.85

    #: Bounds on the framing length as a multiple of the fragment length. The
    #: lower bound keeps an applied frame from being a single token; the upper
    #: matches the long-context tail seen in the benchmark.
    frame_share: Tuple[float, float] = (0.3, 2.0)


@dataclass(frozen=True)
class Difficulty:
    """How hard an example is, recorded rather than thresholded.

    Filtering on any one of these would be irreversible and the thresholds are
    uncalibrated, so every signal is kept on the example and left for a
    curriculum, a loss weighting, or a filter applied later.

    Attributes:
        n_operations: Length of the gold script; the difficulty dial.
        n_substitutions: Operations that are neither copies nor structural.
        lexical_overlap: Share of reuse tokens present verbatim in the source.
        positional_recoverable: Whether the trivial positional baseline
            reproduces the gold operation profile.
        oracle_recoverable: Whether the oracle recovers the gold script. ``None``
            until the oracle's resources are available. This is the signal that
            corresponds to AlphaGeometry's auxiliary-construction filter.
    """

    n_operations: int
    n_substitutions: int
    lexical_overlap: float
    positional_recoverable: bool
    oracle_recoverable: Optional[bool] = None

    @staticmethod
    def positional_baseline(
        source_tokens: Sequence[str], target_tokens: Sequence[str]
    ) -> EditScript:
        """Align position *i* with position *i*; copy where equal, else delete and insert.

        Deliberately the weakest analysis that is still well formed. An example a
        model could learn nothing from is one this recovers exactly.
        """
        operations: List[EditOperation] = []
        for i, token in enumerate(source_tokens):
            if i >= len(target_tokens) or normalize(token) != normalize(target_tokens[i]):
                operations.append(
                    EditOperation("DEL", (i,), (), (token,), ())
                )
        for j, token in enumerate(target_tokens):
            if j < len(source_tokens) and normalize(source_tokens[j]) == normalize(token):
                operations.append(
                    EditOperation("NOP", (j,), (j,), (source_tokens[j],), (token,))
                )
            else:
                operations.append(EditOperation("INS", (), (j,), (), (token,)))
        return EditScript(list(source_tokens), list(target_tokens), operations)

    @staticmethod
    def is_trivially_recoverable(script: EditScript, baseline: EditScript) -> bool:
        """Whether the baseline reproduces the script's operation profile exactly."""
        return script.op_counts() == baseline.op_counts()

    @classmethod
    def score(
        cls,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        script: EditScript,
    ) -> "Difficulty":
        """Record every difficulty signal available without the oracle."""
        structural = {"NOP", "INS", "DEL", "REORDER", "ADAPT", "DISPERSE", "QUOTE"}
        counts = script.op_counts()
        substitutions = sum(n for tag, n in counts.items() if tag not in structural)

        source_forms = {normalize(t) for t in source_tokens}
        shared = sum(1 for t in target_tokens if normalize(t) in source_forms)
        overlap = shared / len(target_tokens) if target_tokens else 0.0

        baseline = cls.positional_baseline(source_tokens, target_tokens)
        return cls(
            n_operations=len(script.operations),
            n_substitutions=substitutions,
            lexical_overlap=overlap,
            positional_recoverable=cls.is_trivially_recoverable(script, baseline),
        )


@dataclass
class GenerationReport:
    """What the generator produced, and what it discarded on the way."""

    seeds: int = 0
    attempted: int = 0
    failed_to_build: int = 0
    failed_verification: int = 0
    filtered_trivial: int = 0
    kept: int = 0
    op_mix: Dict[str, int] = field(default_factory=dict)
    positional_recoverable: int = 0
    canonicalized: int = 0
    canonical_disagreements: int = 0
    framed: int = 0

    @property
    def retained_fraction(self) -> float:
        """Share of attempts that survived. Near 1.0 means the filter is not
        discriminating; near 0.0 means the sampler is producing trivia."""
        return self.kept / self.attempted if self.attempted else 0.0

    def summary(self) -> str:
        """One line for a run log: what was produced and what was lost on the way."""
        return (
            f"seeds={self.seeds} attempted={self.attempted} kept={self.kept} "
            f"({self.retained_fraction:.1%})  "
            f"build_fail={self.failed_to_build} verify_fail={self.failed_verification} "
            f"trivial_dropped={self.filtered_trivial} "
            f"trivially_recoverable={self.positional_recoverable} "
            f"canonicalized={self.canonicalized} "
            f"disagreed={self.canonical_disagreements}"
        )


# =============================================================================
# Generator
# =============================================================================


class SyntheticGenerator:
    """Produces (source, variant, script) triples by applying chosen operations.

    Example:
        ```python
        generator = SyntheticGenerator(MockSubstitutionSource())
        examples, report = generator.generate(["arma uirumque cano Troiae"])
        print(report.summary())
        ```
    """

    def __init__(
        self,
        substitutions: Optional[SubstitutionSource] = None,
        config: Optional[GenerationConfig] = None,
        registry: Optional[OperationRegistry] = None,
        oracle=None,
    ):
        self.substitutions = substitutions or MockSubstitutionSource()
        self.config = config or GenerationConfig()
        self.registry = registry or OperationRegistry.default()
        self.scriba = Scriba(self.registry)
        self.oracle = oracle

    # ---------- Public API ----------

    def generate(
        self,
        seeds: Sequence[str],
        *,
        context_pool: Optional[Sequence[str]] = None,
        progress: Optional[Callable[[int, int, GenerationReport], None]] = None,
        progress_every: int = 50,
    ) -> Tuple[List[dict], GenerationReport]:
        """Generate examples from seed passages, with a report of the yield.

        Args:
            progress: Called with ``(seeds_done, seeds_total, report)`` every
                ``progress_every`` seeds. Generation over the full benchmark
                takes long enough that a run with no output is indistinguishable
                from one that has hung, which is worth avoiding.
        """
        rng = random.Random(self.config.seed)
        report = GenerationReport(seeds=len(seeds))
        kept: List[dict] = []

        for index, seed in enumerate(seeds):
            if progress and index and index % progress_every == 0:
                # Kept is otherwise only totalled after the loop, which would
                # make every progress line report zero.
                report.kept = len(kept)
                progress(index, len(seeds), report)
            tokens = seed.split() if isinstance(seed, str) else list(seed)
            if len(tokens) < 2:
                continue
            for _ in range(self.config.variants_per_seed):
                report.attempted += 1
                built = self._build_one(tokens, rng)
                if built is None:
                    report.failed_to_build += 1
                    continue
                variant, script = built

                fragment_span = (0, len(variant))
                framed = (
                    context_pool
                    and rng.random() < self.config.frame_probability
                )
                if framed:
                    variant, script, fragment_span = self._embed_in_frame(
                        tokens, variant, script, context_pool, rng
                    )
                    report.framed += 1

                if not self.scriba.verify(script, tokens, variant):
                    report.failed_verification += 1
                    continue

                applied = script
                # A framed pair is never re-planned: the oracle has no way to
                # emit FRAME, so its account is strictly more expensive and the
                # applied script always wins; skipping saves the compute.
                if self.config.canonicalize and self.oracle is not None and not framed:
                    script = self._canonicalize(tokens, variant, applied, report)

                difficulty = Difficulty.score(tokens, variant, script)
                if difficulty.positional_recoverable:
                    report.positional_recoverable += 1
                    if self.config.filter_trivial:
                        report.filtered_trivial += 1
                        continue

                for tag, n in script.op_counts().items():
                    report.op_mix[tag] = report.op_mix.get(tag, 0) + n
                kept.append(
                    {
                        "source_tokens": list(tokens),
                        "target_tokens": list(variant),
                        "script": script,
                        "origin": "synthetic",
                        "difficulty": difficulty,
                        "applied_script": applied,
                        "framed": bool(framed),
                        "fragment_span": fragment_span,
                    }
                )
        report.kept = len(kept)
        return kept, report

    # ---------- Internals ----------

    def _embed_in_frame(self, source_tokens, variant, script, context_pool, rng):
        """Splice the variant into unrelated prose, describing the prose as FRAME.

        Real reuse is a fragment inside the citing author's own text — twelve
        words of Virgil inside sixteen of Ambrose — and the surrounding prose
        is a single act of framing, not a run of independent insertions. One
        context passage is split around the fragment so prefix and suffix read
        as one interrupted sentence, and each side becomes one ``FRAME``
        operation writing its span, priced per act rather than per token.

        Returns the framed variant, its script, and the fragment's span inside
        it, which later stages use as span supervision.
        """
        context = rng.choice(context_pool)
        context_tokens = (
            context.split() if isinstance(context, str) else list(context)
        )
        low, high = self.config.frame_share
        want = max(1, int(len(variant) * rng.uniform(low, high)))
        want = min(want, len(context_tokens))
        start = rng.randrange(0, len(context_tokens) - want + 1)
        window = context_tokens[start:start + want]
        cut = rng.randint(0, len(window))
        prefix, suffix = window[:cut], window[cut:]

        offset = len(prefix)
        shifted = [
            EditOperation(
                op.tag,
                op.source_indices,
                tuple(i + offset for i in op.target_indices),
                op.source_tokens,
                op.target_tokens,
            )
            for op in script.operations
        ]
        framed_variant = prefix + list(variant) + suffix
        operations = []
        if prefix:
            operations.append(EditOperation(
                "FRAME", (), tuple(range(len(prefix))), (), tuple(prefix)
            ))
        operations.extend(shifted)
        if suffix:
            begin = offset + len(variant)
            operations.append(EditOperation(
                "FRAME", (), tuple(range(begin, begin + len(suffix))), (),
                tuple(suffix)
            ))
        framed_script = EditScript(
            list(source_tokens), framed_variant, operations, script.registry
        )
        return framed_variant, framed_script, (offset, offset + len(variant))

    def _canonicalize(
        self,
        source_tokens: Sequence[str],
        target_tokens: Sequence[str],
        applied: EditScript,
        report: "GenerationReport",
    ) -> EditScript:
        """Replace the applied script with the oracle's analysis of the pair.

        Choosing operations in advance yields *a* script relating the pair, not
        the cheapest one: deleting a word and inserting another produces the
        same pair as substituting a synonym, at four times the cost. Training
        on the expensive analysis teaches the model to over-price exactly the
        relation the distance is meant to capture.

        Disagreement between the applied and the recovered script is recorded
        rather than discarded. It measures the oracle — a pair the oracle
        analyses differently is one where several analyses compete, which is
        the interesting case, not a defective sample.
        """
        try:
            recovered = self.oracle.plan_tokens(source_tokens, target_tokens)
        except Exception:
            return applied
        if not self.scriba.verify(recovered, source_tokens, target_tokens):
            return applied
        report.canonicalized += 1
        if recovered.op_counts() != applied.op_counts():
            report.canonical_disagreements += 1
        # The cheaper analysis is the canonical one by definition.
        return recovered if recovered.cost() <= applied.cost() else applied

    def _build_one(self, tokens: Sequence[str], rng: random.Random):
        """Sample operations for one variant and apply them.

        The sampler emits one operation per changed position and always a
        single operation for a substitution, never a deletion plus an
        insertion, so the script it produces is canonical by construction.
        """
        k = self._sample_count(rng)
        changed = set(rng.sample(range(len(tokens)), min(k, len(tokens))))
        builder = VariantBuilder(list(tokens), self.registry)
        inserts: List[str] = []

        for i, token in enumerate(tokens):
            if i not in changed:
                builder.keep(i)
                continue
            tag = self._sample_tag(rng)
            if tag == "DEL":
                builder.delete(i, detail="sampled")
            elif tag == "INS":
                builder.keep(i)
                inserts.append(f"{token}-ins")
            else:
                replacement = self.substitutions.replacement_in_context(
                    tokens, i, tag, rng
                )
                if replacement is None:
                    builder.keep(i)
                else:
                    builder.substitute(i, replacement, tag=tag, detail="sampled")
        for token in inserts:
            builder.insert(token, detail="sampled")

        try:
            return builder.build()
        except ValueError:
            return None

    def _usable_weights(self) -> Dict[str, float]:
        """Tag weights restricted to what the substitution source can realise.

        Sampling a tag the source cannot honour produces an example whose label
        is not supported by the change that was made -- a hypernym substitution
        that is merely a distributionally similar word, say. Such an example
        teaches a distinction the data does not contain, so the tag is dropped
        and the remaining weights are renormalised rather than silently
        mislabelled. Structural tags are always available: they need no lexicon.
        """
        structural = {"INS", "DEL", "REORDER"}
        keep_substitution = {
            tag: weight for tag, weight in self.config.weights.items()
            if tag not in structural and self.substitutions.realises(tag)
        }
        if not keep_substitution:
            return dict(self.config.weights)

        # The weight freed by a dropped tag is redistributed among the
        # substitutions that remain, never to insertion and deletion. Dropping
        # SYN because the source cannot attest one is a statement about the
        # lexicon; letting it raise the insertion rate would turn that into a
        # claim about how much of this reuse is unexplained, which it is not.
        substitution_total = sum(
            weight for tag, weight in self.config.weights.items()
            if tag not in structural
        )
        scale = substitution_total / sum(keep_substitution.values())
        usable = {tag: weight * scale for tag, weight in keep_substitution.items()}
        usable.update({
            tag: weight for tag, weight in self.config.weights.items()
            if tag in structural
        })
        total = sum(usable.values())
        return {tag: weight / total for tag, weight in usable.items()}

    def dropped_tags(self) -> List[str]:
        """Tags the configuration asks for that the source cannot realise."""
        return sorted(set(self.config.weights) - set(self._usable_weights()))

    def _sample_count(self, rng: random.Random) -> int:
        counts = self.config.op_counts
        weights = self.config.op_count_weights or [1.0] * len(counts)
        return rng.choices(counts, weights=weights, k=1)[0]

    def _sample_tag(self, rng: random.Random) -> str:
        usable = self._usable_weights()
        tags = list(usable)
        weights = [usable[t] for t in tags]
        return rng.choices(tags, weights=weights, k=1)[0]
