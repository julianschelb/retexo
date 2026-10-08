# formulations/typed_pointer/cells.py
"""The flattened cell layout constants and the stretch head's modes."""

# =============================================================================
# The cell layout
# =============================================================================

#: chain 15: the stretch head's modes (the paper's modes of reuse) and the one the null fusion reads
STRETCH_MODES = ("VERBATIM", "ALLUSION", "FRAME", "NOMATCH")
STRETCH_NOMATCH = STRETCH_MODES.index("NOMATCH")

# cell layout of the flattened logits, per reuse word:
#   0            (null, INS)
#   1            (null, FRAME)
#   2 + c*K + k  (source column c, type k)
NULL_INS, NULL_FRAME, CELLS_FROM = 0, 1, 2
