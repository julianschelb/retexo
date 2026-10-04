"""The released format: labels of the paper, no gold, and a record that cannot contradict itself."""

import json
from pathlib import Path

import pytest

from retexo.cli import main
from retexo.export import LABELS, convert_record, export_folds, label_of, label_shares, read_predictions

FIXTURE = Path(__file__).parent / "fixtures" / "raw_predictions.jsonl"


@pytest.fixture(scope="module")
def raw():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def records():
    return list(read_predictions(FIXTURE))


def test_code_tags_become_the_papers_labels():
    assert label_of("NOP") == ("COPY", None)
    assert label_of("MORPH") == ("INFLECT", None)
    assert label_of("SYN") == ("SUBST", "SYN")
    assert label_of("SUBST") == ("SUBST", None)
    assert label_of("SPLIT") == ("SPLIT", None)
    with pytest.raises(ValueError):
        label_of("???")


def test_every_reuse_word_is_linked_inserted_or_in_a_citing_formula(records):
    for record in records:
        n = len(record["reuse"]["tokens"])
        linked = {link["reuse"] for link in record["links"]}
        framed = {r for span in record["frame"] for r in range(span["start"], span["end"])}
        inserted = set(record["insertions"])
        assert linked | framed | inserted == set(range(n))
        assert not (linked & framed) and not (linked & inserted) and not (framed & inserted)


def test_deletions_are_the_source_words_no_link_claims(records):
    for record in records:
        claimed = {link["source"] for link in record["links"]}
        assert record["deletions"] == [s for s in range(len(record["source"]["tokens"])) if s not in claimed]


def test_labels_are_the_five_link_labels_and_a_relation_only_on_substitutions(records):
    for record in records:
        for link in record["links"]:
            assert link["label"] in LABELS
            assert link["relation"] is None or link["label"] == "SUBST"
            assert 0.0 <= link["confidence"] <= 1.0


def test_fixture_covers_split_merge_frame_and_a_named_relation(records):
    labels = {link["label"] for r in records for link in r["links"]}
    assert {"COPY", "SPLIT", "MERGE"} <= labels
    assert any(r["frame"] for r in records)
    assert any(link["relation"] == "SYN" for r in records for link in r["links"])


def test_no_gold_leaks_into_a_record(records):
    for record in records:
        assert set(record) == {"id", "benchmark_id", "fold", "reference_type", "source", "reuse", "links", "frame",
                               "insertions", "deletions"}
        assert set(record["source"]) == {"author", "work", "citation", "tokens"}


def test_frame_ranges_are_half_open_like_the_raw_spans(raw):
    for r in raw:
        mine = [(s["start"], s["end"]) for s in convert_record(r)["frame"]]
        assert mine == [(s["start"], s["end"]) for s in r["pred"]["frame_spans"]]


def test_a_record_that_contradicts_itself_is_refused(raw):
    broken = json.loads(json.dumps(raw[0]))
    broken["dels"] = [1 - d for d in broken["dels"]]
    with pytest.raises(ValueError, match="deletions"):
        convert_record(broken)
    short = json.loads(json.dumps(raw[0]))
    short["frames"] = short["frames"][:-1]
    with pytest.raises(ValueError, match="do not match"):
        convert_record(short)


def test_label_shares_sum_to_one(records):
    assert sum(label_shares(records).values()) == pytest.approx(1.0)


def _per_fold_files(raw, folder):
    """The fixture's raw records regrouped into one raw file per fold, as a run writes them."""
    paths = {}
    for r in raw:
        fold = r["record"]["fold"]
        paths.setdefault(fold, []).append(r)
    out = []
    for fold, rows in sorted(paths.items()):
        path = folder / f"raw_f{fold}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        out.append(path)
    return out


def test_export_writes_one_file_per_fold_and_the_cli_does_the_same(tmp_path, raw, records):
    sources = _per_fold_files(raw, tmp_path)
    counts = export_folds(sources, tmp_path / "a")
    assert counts == {f: sum(1 for r in records if r["fold"] == f) for f in sorted({r["fold"] for r in records})}
    fold = next(iter(counts))
    lines = (tmp_path / "a" / f"fold_{fold}.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [r["id"] for r in records if r["fold"] == fold]
    assert main(["export", *map(str, sources), "--out", str(tmp_path / "b")]) == 0
    assert (tmp_path / "b" / f"fold_{fold}.jsonl").exists()


def test_a_file_that_mixes_folds_is_refused(tmp_path):
    with pytest.raises(ValueError, match="exactly one fold"):
        export_folds([FIXTURE], tmp_path)


def test_parquet_round_trips(tmp_path, raw, records):
    pq = pytest.importorskip("pyarrow.parquet")
    sources = _per_fold_files(raw, tmp_path)
    counts = export_folds(sources, tmp_path / "pq", fmt="parquet")
    fold = next(iter(counts))
    table = pq.read_table(tmp_path / "pq" / f"fold_{fold}.parquet")
    first = next(r for r in records if r["fold"] == fold)
    assert table.num_rows == counts[fold]
    assert table.to_pylist()[0]["links"] == first["links"]
