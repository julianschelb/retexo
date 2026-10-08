"""Tests for embedding a variant in framed context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.core.scriba import Scriba  # noqa: E402
from retexo.datasets.generation import (  # noqa: E402
    GenerationConfig,
    MockSubstitutionSource,
    SyntheticGenerator,
)
from retexo.formulations.encoding import NULL_SOURCE, ScriptEncoder  # noqa: E402

PASSED = FAILED = 0


def check(name, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}")


SEED = "arma uirumque cano Troiae qui primus ab oris".split()
CONTEXT = [
    "sic ait et dicto citius tumida aequora placat collectasque fugat nubes solemque reducit"
]


def generate(frame_probability=1.0):
    generator = SyntheticGenerator(
        MockSubstitutionSource(),
        GenerationConfig(variants_per_seed=4, seed=7, frame_probability=frame_probability),
    )
    records, report = generator.generate([" ".join(SEED)], context_pool=CONTEXT)
    return records, report


def test_framed_scripts_verify_and_replay():
    records, report = generate()
    check("everything kept", report.kept == len(records) and report.kept > 0)
    check("framing happened", report.framed > 0)
    scriba = Scriba()
    for record in records:
        script = record["script"]
        check("verifies", scriba.verify(script, record["source_tokens"], record["target_tokens"]))
        replay = scriba.execute(script, record["source_tokens"])
        check("replay length", len(replay) == len(record["target_tokens"]))


def test_frame_is_priced_per_act():
    records, _ = generate()
    framed = [r for r in records if r["framed"]]
    check("framed records exist", bool(framed))
    for record in framed:
        counts = record["script"].op_counts()
        check("at most two FRAME ops", 0 < counts.get("FRAME", 0) <= 2)
        frame_tokens = sum(
            len(op.target_indices) for op in record["script"].operations if op.tag == "FRAME"
        )
        a, b = record["fragment_span"]
        check(
            "frame covers exactly the context",
            frame_tokens == len(record["target_tokens"]) - (b - a),
        )


def test_fragment_span_is_recorded():
    records, _ = generate()
    for record in records:
        a, b = record["fragment_span"]
        check("span inside target", 0 <= a <= b <= len(record["target_tokens"]))
        if record["framed"]:
            inside = record["target_tokens"][a:b]
            check(
                "span holds the fragment",
                any(tok.startswith("arma") or "uirumque" in tok for tok in inside),
            )


def test_unframed_when_disabled():
    records, report = generate(frame_probability=0.0)
    check("no framing at probability zero", report.framed == 0)
    check("no FRAME ops", all("FRAME" not in r["script"].op_counts() for r in records))


def test_encoding_labels_frames():
    records, _ = generate()
    framed = next(r for r in records if r["framed"])
    encoded = ScriptEncoder.to_token_labels(framed["script"])
    a, b = framed["fragment_span"]
    outside = [i for i in range(len(framed["target_tokens"])) if i < a or i >= b]
    check("context labelled FRAME", all(encoded["op_labels"][i] == "FRAME" for i in outside))
    check(
        "context points nowhere", all(encoded["source_indices"][i] == NULL_SOURCE for i in outside)
    )


def test_decode_merges_frame_runs():
    from retexo.formulations.token_classifier import (
        TokenClassifierConfig,
        TokenClassifierModel,
    )

    model = TokenClassifierModel(TokenClassifierConfig(base_model="unused", device="cpu"))
    source = ["alpha", "beta"]
    target = ["x", "y", "alpha", "z"]
    tags = ["FRAME", "FRAME", "NOP", "FRAME"]
    sources = [NULL_SOURCE, NULL_SOURCE, 0, NULL_SOURCE]
    script = model._decode(source, target, tags, sources)
    frames = [op for op in script.operations if op.tag == "FRAME"]
    check("two runs, two ops", len(frames) == 2)
    check("first run spans two tokens", frames[0].target_indices == (0, 1))
    check("replay works", Scriba().verify(script, source, target))


for fn in [
    test_framed_scripts_verify_and_replay,
    test_frame_is_priced_per_act,
    test_fragment_span_is_recorded,
    test_unframed_when_disabled,
    test_encoding_labels_frames,
    test_decode_merges_frame_runs,
]:
    fn()

print(f"{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
