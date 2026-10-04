"""Tests for orthographic folding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.core.normalize import normalize  # noqa: E402

PASSED = FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}")


def same(a, b):
    return normalize(a) == normalize(b)


# Variants the Stage-1 diagnostic found falling through as DEL+INS.
check("aspirated t", same("ture", "Thure"))
check("greek upsilon + ch", same("Chalybes", "Calybes"))
check("variant stem ngue", same("tinguere", "tingere"))
check("nasal assimilation np", same("inpare", "impare"))
check("nasal assimilation nb", same("inbellis", "imbellis"))
check("rh", same("Rhenus", "renus"))
check("existing u/v", same("Uox,", "vox"))
check("existing i/j", same("iam", "jam"))

# Distinctions that must survive the folds.
check("gemination kept", not same("anus", "annus"))
check("anguis not angis", not same("anguis", "angis"))
check("archaic -umus not folded", not same("optuma", "optima"))
check("different words stay different", not same("arma", "amor"))

print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
