"""Round-trip: a tiny tagger saves, loads, and predicts a valid script."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.core.scriba import Scriba  # noqa: E402
from retexo.formulations.token_classifier import (  # noqa: E402
    TokenClassifierConfig,
    TokenClassifierModel,
)

PASSED = FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}")


TINY = "hf-internal-testing/tiny-random-bert"


def test_save_load_predict():
    model = TokenClassifierModel(TokenClassifierConfig(base_model=TINY, device="cpu"))
    model._build()
    source = "arma virumque cano".split()
    target = "arma virumque cecini".split()
    with tempfile.TemporaryDirectory() as d:
        model.save(Path(d))
        loaded = TokenClassifierModel.load(Path(d))
    check("labels preserved", loaded.labels == model.labels)
    check("INS/DEL absent from head", "INS" not in loaded.labels and "DEL" not in loaded.labels)
    script = loaded.predict(source, target)
    check("predicts a script", script is not None)
    check("script is valid", Scriba().validate(script, len(source)).ok)


test_save_load_predict()
print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
