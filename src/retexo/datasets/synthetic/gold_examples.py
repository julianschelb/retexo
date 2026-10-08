# datasets/synthetic/gold_examples.py
"""Hand-labelled pairs as typed examples: ``fine_from_gold`` and its link-only coarse view."""

from __future__ import annotations

from retexo.datasets.synthetic.inventory import FINE_OPERATIONS
from retexo.datasets.synthetic.shape import coarse
from retexo.edit_typing.link_features import LinkFeaturizer
from retexo.formulations.change_detector import ChangeExample

# =============================================================================
# Gold with fine labels and evidence
# =============================================================================


def _link_only_coarse(pair, t: int) -> str:
    """The coarse view of a link-only edge: ``NOP`` for the same normalised form, ``SUBST`` for any change."""
    from retexo.core.normalize import normalize

    s = pair.target_align[t]
    return (
        "NOP" if normalize(pair.source_tokens[s]) == normalize(pair.target_tokens[t]) else "SUBST"
    )


def fine_from_gold(pair, featurizer: LinkFeaturizer, *, gold_fine: bool = False):
    """A hand-labelled pair as a typed example.

    The silver labels carry NOP / MORPH / SUBST and FRAME spans. NOP and MORPH
    are exact fine tags; a silver SUBST is a *lexical* change whose relation
    the labeller did not name, so it is left unsupervised (-100 at training)
    rather than forced into a class -- the typer learns the fine relations from
    the constructed data and the coarse ones from the real data. With
    ``gold_fine`` (a V3 annotation, definition section 3.1: SYN, POS, NE-SUB,
    SUBST, SPLIT, MERGE named on the link) every operation in
    ``FINE_OPERATIONS`` is the fine target itself; the ``detail`` of a link
    (HYPER, ANT, the MORPH features) is never a label.
    """
    target_ops = list(pair.target_ops)  # fine: NOP / MORPH / SUBST / FRAME / INS
    fine, frame, features = [], [], []
    for t, op in enumerate(target_ops):
        s = pair.target_align[t] if pair.target_align and t < len(pair.target_align) else -1
        if op == "FRAME":
            fine.append("INS")
            frame.append(1)
            features.append(None)
            continue
        frame.append(0)
        if s < 0:
            fine.append("INS")
            features.append(None)
            continue
        if op == "NOP":
            fine.append("NOP")
        elif op == "MORPH":
            fine.append("MORPH")
        elif op == "LINK":
            fine.append(
                "LINK"
            )  # a link-only edge (self-training): the link trains, no type head does
        elif gold_fine and op in FINE_OPERATIONS:
            fine.append(op)  # the annotated V3 operation
        else:
            fine.append("?")  # unsupervised lexical change
        features.append(
            featurizer(
                pair.source_tokens[s],
                pair.target_tokens[t],
                s,
                t,
                len(pair.source_tokens),
                len(pair.target_tokens),
            )
        )
    # The coarse head and the pointer see exactly what E4-era training saw:
    # COPY / SUBST / INS. Only the typer and the frame head see the fine view.
    # A link-only edge says nothing about its kind: the coarse head gets COPY where the two words are the same
    # form and SUBST (a change, INFLECT included) otherwise, never a lexical class it was not shown.
    coarse_ops = [
        coarse(_link_only_coarse(pair, t) if op == "LINK" else op)
        for t, op in enumerate(target_ops)
    ]
    return ChangeExample(
        source_tokens=list(pair.source_tokens),
        target_tokens=list(pair.target_tokens),
        labels=[0 if op == "COPY" else 1 for op in coarse_ops],
        operations=coarse_ops,
        n_operations=sum(1 for op in coarse_ops if op != "COPY"),
        source_labels=list(pair.source_del),
        source_operations=["DEL" if d else "COPY" for d in pair.source_del],
        alignments=list(pair.target_align) if pair.target_align else None,
        fine_operations=fine,
        frame_labels=frame,
        link_features=features,
    )
