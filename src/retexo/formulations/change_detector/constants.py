# formulations/change_detector/constants.py
"""Constants shared by the change detector, its typer heads and their consumers."""

#: E24: fine tags a hand-labelled SUBST may resolve to, and the sentinel used
#: as the typer target for such a link.
LEXICAL_TAGS = frozenset({"SYN", "HYPER", "HYPO", "ANT", "SYN-DIST", "NE-SUB", "POS", "SUBST"})
GROUP_TARGET = -2
