"""The review app's data step: released records and the labels dataset in, the app's records out."""

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
def scripts():
    return pd.DataFrame(list(read_predictions(FIXTURE)))


@pytest.fixture(scope="module")
def labels(scripts):
    """A labels frame with the columns the app reads, one row per reference."""
    rows = []
    for record in scripts.itertuples(index=False):
        row = {"id": record.benchmark_id}
        for prefix, side in (("corpus", record.source), ("query", record.reuse)):
            text = " ".join(side["tokens"])
            row.update({f"{prefix}_text": text, f"{prefix}_text_original": text.upper(), f"{prefix}_text_english": None})
        rows.append(row)
    return pd.DataFrame(rows)


def test_every_script_becomes_one_record(prepare, scripts, labels):
    records = prepare.build_records(scripts, labels)
    assert [r["id"] for r in records] == list(scripts.sort_values(["benchmark_id", "id"])["id"])


def test_a_record_has_what_the_app_reads(prepare, scripts, labels):
    for record in prepare.build_records(scripts, labels):
        assert record["pair_label"] in ("cit", "cf")
        for side in (record["source"], record["reuse"]):
            assert set(side) == {"author", "work", "citation", "tokens", "text", "text_original", "text_english"}
        for link in record["pred"]["links"]:
            assert set(link) == {"r", "s", "op", "p", "relation"}
            assert 0 <= link["r"] < len(record["reuse"]["tokens"]) and 0 <= link["s"] < len(record["source"]["tokens"])


def test_links_and_spans_follow_the_released_record(prepare, scripts, labels):
    payload = {r["id"]: r for r in prepare.build_records(scripts, labels)}
    for released in scripts.itertuples(index=False):
        pred = payload[released.id]["pred"]
        assert [(l["r"], l["s"], l["op"]) for l in pred["links"]] == [
            (l["reuse"], l["source"], l["label"]) for l in released.links]
        assert [(s["start"], s["end"]) for s in pred["frame_spans"]] == [(s["start"], s["end"]) for s in released.frame]


def test_texts_come_from_the_labels_and_missing_ones_are_null(prepare, scripts, labels):
    record = prepare.build_records(scripts, labels)[0]
    assert record["source"]["text_original"] == record["source"]["text"].upper()
    assert record["source"]["text_english"] is None


def test_the_records_are_json(prepare, scripts, labels):
    json.dumps(prepare.build_records(scripts, labels), ensure_ascii=False)
