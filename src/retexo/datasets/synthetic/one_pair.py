# datasets/synthetic/one_pair.py
"""One typed pair: the report dataclass, the spelling and enclitic events, and the pair constructor."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from retexo.core.normalize import normalize
from retexo.datasets.synthetic.inventory import ENCLITIC_RATE, ENCLITICS, FRAME_RATE
from retexo.datasets.synthetic.shape import (
    COPY_VARIANT_SHARE,
    FRAGMENT_LENGTHS,
    FRAGMENTS,
    _draw,
    _variant,
)
from retexo.formulations.change_detector import ChangeExample

# =============================================================================
# One example
# =============================================================================


@dataclass
class TypedReport:
    attempted: int = 0
    kept: int = 0
    no_substitute: int = 0
    op_counts: Dict[str, int] = field(default_factory=dict)
    fine_counts: Dict[str, int] = field(default_factory=dict)
    frames: int = 0
    reorders: int = 0
    enclitics: int = 0
    spelling: int = 0
    frameless: int = 0


# ---------- Orthographic and enclitic events ----------

#: Orthographic alternations the COPY gate folds (``Repairer.spelling_key``), in the direction that writes a
#: variant: (pattern, replacement) on the lowercased body; the result must keep the spelling key and change the form.
_SPELLING_RULES = (
    ("ae", "e"),
    ("e", "ae"),
    ("oe", "e"),
    ("i", "y"),
    ("nt", "mpt"),
    ("mpt", "nt"),
    ("f", "ph"),
    ("ph", "f"),
    (r"^([aeiou])", r"h\1"),
    (r"^h", ""),
    (r"([bcdfglmnprst])\1", r"\1"),
    (r"([aeiou])([lmnrst])([aeiou])", r"\1\2\2\3"),
    (r"ii$", "i"),
    (r"i$", "ii"),
    (r"d$", "t"),
    (r"t$", "d"),
    (r"ies$", "iens"),
    (r"iens$", "ies"),
    (r"m$", "n"),
)


def spelling_variant(token: str, rng, attested=None) -> Optional[str]:
    """A spelling variant of ``token`` the gold counts as COPY (*temptare* / *tentare*, *Grai* / *Graii*,
    *haud* / *haut*): the same ``Repairer.spelling_key``, a different ``normalize`` form, and attested in the
    corpus vocabulary -- the rules alone write non-words (*temptarre*), so without ``attested`` there is no variant.
    None when no rule applies."""
    from retexo.edit_typing.repair import Repairer

    if not attested:
        return None
    bare = re.sub(r"[^A-Za-z]+$", "", token)
    trail = token[len(bare) :]
    if len(bare) < 3:
        return None
    lower = bare.lower()
    key = Repairer.spelling_key(lower)
    for pattern, repl in rng.sample(_SPELLING_RULES, len(_SPELLING_RULES)):
        cand = re.sub(pattern, repl, lower, count=1)
        if (
            cand == lower
            or Repairer.spelling_key(cand) != key
            or normalize(cand) == normalize(lower)
        ):
            continue
        if normalize(cand) not in attested:
            continue
        if bare[:1].isupper():
            cand = cand[:1].upper() + cand[1:]
        return cand + trail
    return None


def _attach_enclitic(token: str, rng, attested) -> Optional[str]:
    """``uirum`` -> ``uirumque``, if that is a form somebody wrote."""
    bare = re.sub(r"[^A-Za-z]+$", "", token)
    trail = token[len(bare) :]
    for enclitic in rng.sample(ENCLITICS, len(ENCLITICS)):
        merged = bare + enclitic
        if attested is None or normalize(merged) in attested:
            return merged + trail
    return None


# ---------- The pair constructor ----------


def _make_one_typed(
    seed_tokens,
    context_pool,
    substitute,
    featurizer,
    rng,
    *,
    frames: Sequence[Sequence[str]] = (),
    frame_rate=FRAME_RATE,
    enclitic_rate=ENCLITIC_RATE,
    reorder_inter=0.0,
    reorder_intra=0.0,
    attested=None,
    dense_rate=0.0,
    ins_slot=0.0,
    slot_words: Optional[List[str]] = None,
    copy_variants=0.0,
    frameless_fill="",
):
    """One pair with typed links, optional frame, enclitic events, reorderings.

    ``copy_variants`` (2026-09-27, the convention wall of the error analysis): enclitic events on copied words are
    COPY, not SPLIT/MERGE (the gold counts *aristisque* <- *aristis* as a copy), and this share of the copied words
    is written in a spelling variant the gold also counts as COPY (``spelling_variant``). ``frameless_fill``
    (``nolink``, the Frame Support dry run): an unrelated slot filler (tag SUBST) with no copied or inflected word
    within two positions of its fragment is emitted as an insertion, its source word deleted -- a filler without a
    kept frame is not a substitution."""
    source = list(seed_tokens)
    if len(source) < 6:
        return None, None
    target = list(rng.choice(context_pool))
    if len(target) < 6:
        return None, None

    stats = {"frame": 0, "reorder": 0, "enclitic": 0, "spelling": 0, "frameless": 0}
    n_fragments = _draw(FRAGMENTS, rng)
    source_del = [1] * len(source)
    pieces = []
    taken = set()

    for _ in range(n_fragments):
        length = min(_draw(FRAGMENT_LENGTHS, rng), len(source))
        starts = [
            i
            for i in range(len(source) - length + 1)
            if not any(j in taken for j in range(i, i + length))
        ]
        if not starts:
            break
        start = rng.choice(starts)
        core = source[start : start + length]
        for offset in range(len(core)):
            taken.add(start + offset)
            source_del[start + offset] = 0
        # E38c: an allusion-shaped fragment -- every content word replaced in context,
        # only the skeleton (short words) kept, so the substitutions are the stretch
        dense = dense_rate > 0 and rng.random() < dense_rate

        reworked, ops, fine, links = [], [], [], []
        for offset, token in enumerate(core):
            links.append(start + offset)
            drawn = substitute(
                token, rng, (source, start + offset), force=(dense and len(normalize(token)) >= 4)
            )
            if drawn:
                replacement, tag = drawn
                reworked.append(replacement)
                ops.append("SUBST")
                fine.append(tag)
                continue
            # cardinality: detach or attach an enclitic on a copied word
            if rng.random() < enclitic_rate:
                split = featurizer.enclitic(token)
                if split and normalize(split[0]) != normalize(token):
                    if copy_variants > 0:
                        reworked.append(split[0])
                        ops.append("COPY")
                        fine.append("NOP")
                    else:
                        reworked.append(split[0])
                        ops.append("SUBST")
                        fine.append("SPLIT")
                    stats["enclitic"] += 1
                    continue
                merged = _attach_enclitic(token, rng, attested)
                if merged and featurizer.enclitic(merged) and not featurizer.enclitic(token):
                    if copy_variants > 0:
                        reworked.append(merged)
                        ops.append("COPY")
                        fine.append("NOP")
                    else:
                        reworked.append(merged)
                        ops.append("SUBST")
                        fine.append("MERGE")
                    stats["enclitic"] += 1
                    continue
            if copy_variants > 0 and rng.random() < copy_variants:
                variant = spelling_variant(token, rng, attested)
                if variant:
                    reworked.append(variant)
                    ops.append("COPY")
                    fine.append("NOP")
                    stats["spelling"] += 1
                    continue
            if rng.random() < COPY_VARIANT_SHARE:
                token = _variant(token, rng)
            reworked.append(token)
            ops.append("COPY")
            fine.append("NOP")
        if frameless_fill == "nolink":
            kept = [o == "COPY" or f == "MORPH" for o, f in zip(ops, fine)]
            for i, f in enumerate(fine):
                if f == "SUBST" and not any(
                    kept[j] for j in range(max(0, i - 2), min(len(fine), i + 3)) if j != i
                ):
                    source_del[links[i]] = 1
                    ops[i], fine[i], links[i] = "INS", "INS", -1
                    stats["frameless"] += 1
        # failure-mode row 15 (``ins_slot``): the later author's own word inside the reused
        # stretch -- an attested content word absent from the source, spliced between two
        # reused words as INS, so that "inside a stretch" is not itself the reason to link
        if ins_slot > 0 and slot_words and len(reworked) >= 3 and rng.random() < ins_slot:
            source_norm = {normalize(w) for w in source}
            for _ in range(4):
                word = rng.choice(slot_words)
                if normalize(word) not in source_norm:
                    at = rng.randrange(1, len(reworked))
                    reworked.insert(at, word)
                    ops.insert(at, "INS")
                    fine.insert(at, "INS")
                    links.insert(at, -1)
                    break
        # intra-fragment inversion: a two-word fragment turned around, which is
        # what *iter durum* -> *durum iter* is. Only ever the whole fragment.
        if len(reworked) == 2 and rng.random() < reorder_intra:
            for row in (reworked, ops, fine, links):
                row.reverse()
            stats["reorder"] += 1
        pieces.append((reworked, ops, fine, links))

    # inter-fragment inversion: the second fragment comes out first
    if len(pieces) >= 2 and rng.random() < reorder_inter:
        pieces.reverse()
        stats["reorder"] += 1

    operations = ["INS"] * len(target)
    fine_ops = ["INS"] * len(target)
    alignments = [-1] * len(target)
    frame_mask = [0] * len(target)
    cuts = sorted(rng.sample(range(len(target) + 1), min(len(pieces), len(target))))
    grown = 0
    for offset, (position, (reworked, ops, fine, links)) in enumerate(zip(cuts, pieces)):
        at = position + grown
        # an attribution formula immediately before the first fragment, in the
        # citing author's own words, drawn from the training-fold hand labels
        if offset == 0 and frames and rng.random() < frame_rate:
            formula = list(rng.choice(frames))
            target[at:at] = formula
            operations[at:at] = ["INS"] * len(formula)
            fine_ops[at:at] = ["INS"] * len(formula)
            alignments[at:at] = [-1] * len(formula)
            frame_mask[at:at] = [1] * len(formula)
            grown += len(formula)
            at += len(formula)
            stats["frame"] = 1
        target[at:at] = reworked
        operations[at:at] = ops
        fine_ops[at:at] = fine
        alignments[at:at] = links
        frame_mask[at:at] = [0] * len(reworked)
        grown += len(reworked)

    # evidence for every link, computed where the resources live
    features = [None] * len(target)
    for t, s in enumerate(alignments):
        if s >= 0:
            features[t] = featurizer(source[s], target[t], s, t, len(source), len(target))

    example = ChangeExample(
        source_tokens=source,
        target_tokens=target,
        labels=[0 if o == "COPY" else 1 for o in operations],
        operations=operations,
        n_operations=len(taken),
        source_labels=source_del,
        source_operations=["DEL" if d else "COPY" for d in source_del],
        alignments=alignments,
        fine_operations=fine_ops,
        frame_labels=frame_mask,
        link_features=features,
    )
    return example, stats
