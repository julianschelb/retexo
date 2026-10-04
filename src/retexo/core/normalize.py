# retexo/core/normalize.py
"""Orthographic conventions used whenever two Latin tokens are compared."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w]", re.UNICODE)

# =============================================================================
# Normalization
# =============================================================================


def normalize(token: str, *, lowercase: bool = True, fold_ij_uv: bool = True) -> str:
    """Normalize a token for comparison.

    Lowercases, folds ``u``/``v`` and ``j``/``i``, strips diacritics and
    punctuation. These are the conventions of the underlying benchmark, so an
    orthographic variant is not mistaken for a transformation.

    Example:
        ```python
        normalize("Uox,")   # -> 'vox'
        normalize("iam")    # -> 'iam'
        ```
    """
    text = unicodedata.normalize("NFD", token)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if lowercase:
        text = text.lower()
    if fold_ij_uv:
        text = text.replace("j", "i").replace("v", "u")
    text = _fold_variants(text)
    return _PUNCT.sub("", text)


#: Orthographic variation beyond i/j and u/v, found empirically: the Stage-1
#: alignment diagnostic showed the residual's strongest conversions were pairs
#: like *ture/Thure*, *Chalybes/Calybes*, *tinguere/tingere* and
#: *inpare/impare* — the same word under classical spelling variation, falling
#: through as DEL+INS. Both sides of a comparison are folded, so equality is
#: preserved; each fold is kept narrow enough not to merge distinct lemmata
#: (gemination is left alone — *anus* is not *annus* — and archaic superlative
#: *-umus* is not folded, since a general u/i fold would wreck the lexicon).
_VARIANT_FOLDS = (
    ("th", "t"),   # aspiration in Greek loans: Thure / ture
    ("ph", "p"),   # Phoebus / Poebus
    ("ch", "c"),   # Chalybes / Calybes
    ("rh", "r"),   # Rhenus / Renus
    ("y", "i"),    # Greek upsilon: Calybes / calibes
    ("ngue", "nge"),  # variant verb stems: tinguere / tingere -- kept this
                      # narrow so *anguis* (snake) is not merged with *angis*
    ("np", "mp"),  # unassimilated nasals: inpare / impare
    ("nb", "mb"),  # inbellis / imbellis
)


def _fold_variants(text: str) -> str:
    for variant, folded in _VARIANT_FOLDS:
        text = text.replace(variant, folded)
    return text
