# retexo/unified/gate.py
"""The mode gate: the whether-decision separated from the which-decision.

The typed pointer decides "which source word, or none" in one flat softmax over
``[INS, FRAME, s_1 .. s_n]``, so the null has to out-score every source column
one by one -- a decision whose difficulty grows with the source length, which
is what ``null_pointer_weight`` was hand-set to counteract. The gate answers
the coarse question once, from the reuse word's own vector, as a three-way
softmax ``(reused, not reused, frame)``, and factorises the location softmax
with it:

    p(INS | t)   = p_not
    p(FRAME | t) = p_frame
    p(s_i | t)   = p_reused . softmax_i(location logits over the source columns)

Same probabilities, same shape going into the decoder; the null mass is decided
by the gate and the source columns only divide up what the gate gives them. Soft
by construction: a hard gate would commit early and lose what the flat softmax
would have recovered (the Hierarchical Typer's regression, E31).
"""

from __future__ import annotations

from typing import List, Sequence

#: The gate's three classes, in the order of its logits.
REUSED, NOT_REUSED, IS_FRAME = 0, 1, 2

#: Where the two nulls sit in the pointer's location row (``typed_pointer.NULL_INS`` / ``NULL_FRAME``).
_NULL_INS, _NULL_FRAME, _CELLS_FROM = 0, 1, 2


def gate_log_probs(loc_flat, gate_logits):
    """Factorise the pointer's location logits ``[T, 2 + width]`` with the gate
    logits ``[T, 3]``; returns log probabilities of the same shape, usable
    wherever the logits were (a softmax of log probabilities is the same
    distribution)."""
    import torch

    log_gate = torch.log_softmax(gate_logits, dim=-1)                      # [T, 3]
    columns = loc_flat[:, _CELLS_FROM:]
    # a row whose every source column is -inf (no candidates) keeps all its mass on the nulls;
    # the softmax runs on a zeroed copy of those rows so no NaN reaches the backward pass
    has_columns = torch.isfinite(columns).any(dim=-1, keepdim=True)
    safe = torch.where(has_columns, columns, torch.zeros_like(columns))
    log_columns = torch.where(has_columns, torch.log_softmax(safe, dim=-1), columns)
    log_reused = torch.where(has_columns.squeeze(-1), log_gate[:, REUSED], torch.full_like(log_gate[:, REUSED], float("-inf")))
    log_null = torch.stack([log_gate[:, NOT_REUSED], log_gate[:, IS_FRAME]], dim=-1)
    if bool((~has_columns).any()):
        # renormalise the two nulls where the reused class has nowhere to go
        only_nulls = torch.log_softmax(log_null, dim=-1)
        log_null = torch.where(has_columns, log_null, only_nulls)
    return torch.cat([log_null, log_reused.unsqueeze(-1) + log_columns], dim=-1)


def gate_targets(align: Sequence[int], frame: Sequence[int]) -> List[List[bool]]:
    """Which gate classes the gold allows per reuse word: a linked word is
    ``reused``; an unlinked word is ``not reused`` or ``frame`` by its frame
    flag, or either when the flag is unknown (-100); an unknown link (-100)
    allows nothing and is skipped by the loss."""
    out = []
    for a, f in zip(align, frame):
        if a == -100:
            out.append([False, False, False])
        elif a >= 0:
            out.append([True, False, False])
        elif f == 1:
            out.append([False, False, True])
        elif f == 0:
            out.append([False, True, False])
        else:
            out.append([False, True, True])
    return out


def gate_loss(gate_logits, allowed: Sequence[Sequence[bool]], class_weights: Sequence[float] = (1.0, 1.0, 1.0)):
    """Set cross-entropy: ``-log sum_{allowed} p`` per word, weighted by the
    weight of the word's first allowed class, averaged over the words that
    allow anything."""
    import torch

    ok = torch.tensor(allowed, dtype=torch.bool, device=gate_logits.device)
    keep = ok.any(dim=-1)
    if not bool(keep.any()):
        return gate_logits.new_zeros(())
    logits = gate_logits[keep]
    ok = ok[keep]
    log_all = torch.logsumexp(logits, dim=-1)
    log_ok = torch.logsumexp(logits.masked_fill(~ok, float("-inf")), dim=-1)
    w = torch.tensor(class_weights, device=gate_logits.device)
    first = ok.float().argmax(dim=-1)
    weights = w[first]
    return ((log_all - log_ok) * weights).sum() / weights.sum().clamp(min=1e-6)
