"""The demo's data step: released records in, the viewer's payload out."""

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from retexo.export import read_predictions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "raw_predictions.jsonl"


@pytest.fixture(scope="module")
def prepare():
    spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "webapp" / "scripts" / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frame():
    return pd.DataFrame(list(read_predictions(FIXTURE)))


def test_every_pair_becomes_one_script(prepare, frame):
    payload = prepare.build_scripts(frame)
    assert payload["model"]["pairs"] == len(frame)
    assert [p["id"] for p in payload["pairs"]] == list(frame.sort_values(["benchmark_id", "id"])["id"])


def test_script_lists_have_one_entry_per_later_word(prepare, frame):
    for pair in prepare.build_scripts(frame)["pairs"]:
        n = len(pair["later"])
        assert all(len(pair["pred"][key]) == n for key in ("links", "ops", "relations", "confidence"))


def test_ops_follow_the_records(prepare, frame):
    payload = {p["id"]: p for p in prepare.build_scripts(frame)["pairs"]}
    for record in frame.itertuples(index=False):
        script = payload[record.id]["pred"]
        for link in record.links:
            assert script["links"][link["reuse"]] == link["source"]
            assert script["ops"][link["reuse"]] == link["label"]
        for span in record.frame:
            assert all(script["ops"][r] == "FRAME" for r in range(span["start"], span["end"]))
        for r in record.insertions:
            assert script["ops"][r] == "INS" and script["links"][r] == -1


def test_the_payload_is_json(prepare, frame):
    json.dumps(prepare.build_scripts(frame))


def test_citations_lose_their_angle_brackets(prepare):
    assert prepare._cite("<verg. aen. 6.847.1>") == "verg. aen. 6.847.1"
    assert prepare._cite(None) == ""
