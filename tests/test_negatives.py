"""Negatives never leak: no labelled pair, no held-out query or cited passage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.datasets.negatives import KINDS, NegativeBuilder  # noqa: E402


@pytest.fixture(scope="module")
def data():
    from retexo.datasets.dataset import BenchmarkData
    try:
        return BenchmarkData.load()
    except FileNotFoundError:
        pytest.skip("benchmark data not present")


def test_training_negatives_are_clean(data):
    b = NegativeBuilder(data, held_out=4)
    negs = b.build(90, kinds=KINDS, seed=0, for_training=True)
    assert len(negs) >= 60
    kinds = {getattr(e, "negative_kind") for e in negs}
    assert kinds == set(KINDS)
    for e in negs:
        src, tgt = " ".join(e.source_tokens), " ".join(e.target_tokens)
        assert (src, tgt) not in b.attested
        assert tgt not in b.held_queries and src not in b.held_sources
        assert all(a == -1 for a in e.alignments) and e.frame_labels is None


def test_held_out_negatives_use_held_out_queries(data):
    b = NegativeBuilder(data, held_out=4)
    negs = b.build(30, kinds=("easy",), seed=1, for_training=False)
    from retexo.core.normalize import normalize

    assert negs and all(normalize(" ".join(e.target_tokens)) in b.held_queries for e in negs)
    for e in negs:
        assert (" ".join(e.source_tokens), " ".join(e.target_tokens)) not in b.attested


def test_adjacent_is_a_neighbour_of_the_cited_passage(data):
    import random
    b = NegativeBuilder(data, held_out=4)
    rng = random.Random(3)
    hits = 0
    for row in b.train_rows[:200]:
        src = b.adjacent(row, rng, for_training=True)
        if src is None:
            continue
        i = b.by_position[str(row.position_corpus)]
        j = b.by_position[str(b.rows.position.iloc[[k for k in (i - 1, i + 1)
                                                    if 0 <= k < len(b.rows) and b._text(k) == src][0]])]
        assert abs(i - j) == 1
        hits += 1
    assert hits > 50


def test_training_negatives_avoid_every_held_fold_and_excluded_passage(data):
    from retexo.core.normalize import normalize

    labels = data.labels
    extra = str(labels[labels.fold_id == 1].text_query_cleaned.iloc[0])      # a "validation" passage in a training fold
    b = NegativeBuilder(data, held_out=(4, 0), exclude_texts=[extra])
    held = {normalize(t) for col in ("text_query_cleaned", "text_corpus_cleaned")
            for t in labels[labels.fold_id.isin((4, 0))][col].astype(str)} | {normalize(extra)}
    negs = b.build(60, seed=2, for_training=True)
    used = {normalize(" ".join(e.source_tokens)) for e in negs} | {normalize(" ".join(e.target_tokens)) for e in negs}
    assert negs and not used & held

