"""Predicted edit scripts in the released format.

A run of the driver writes one ``predictions.jsonl`` per test fold, with the model's raw scores, the gold it was scored
against, and the code's internal tags (``NOP`` for a copy, ``MORPH`` for an inflection). This module turns such a file into
one self-contained record per pair, with the labels of the paper and no gold, so the predictions can be shared and read
without the code that produced them.

One record, in ``SCHEMA_VERSION`` 1::

    {
      "id": "p0001", "benchmark_id": 1, "fold": 4, "reference_type": "cf.",
      "source": {"author": "verg", "work": "verg. aen.", "citation": "<verg. aen. 6.847.1>", "tokens": [...]},
      "reuse":  {"author": "hier", "work": "hier. epist.", "citation": "<hier. epist. 117.7.1.1>", "tokens": [...]},
      "links": [{"reuse": 5, "source": 2, "label": "COPY", "relation": null, "confidence": 0.9994}, ...],
      "frame": [{"start": 0, "end": 6}],
      "insertions": [0, 1, 2, ...],
      "deletions": [0, 1, 3, ...]
    }

Indices are word positions in ``tokens``. ``links`` has one entry per linked reuse word; ``label`` is one of ``COPY``,
``INFLECT``, ``SUBST``, ``SPLIT`` and ``MERGE``, and a ``SUBST`` link may name its ``relation`` (``SYN``, ``POS``, ...).
``frame`` lists the citing formulas as half-open ranges of reuse words, ``insertions`` the unlinked reuse words that are
not part of one, and ``deletions`` the source words no link claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

SCHEMA_VERSION = 1

#: The code's tag for a link, and the label of the paper it stands for.
LABEL_OF_TAG = {
    "NOP": "COPY", "COPY": "COPY",
    "MORPH": "INFLECT", "INFLECT": "INFLECT",
    "SUBST": "SUBST", "SPLIT": "SPLIT", "MERGE": "MERGE",
}

#: A former lexical tag is a substitution that names its relation.
RELATION_TAGS = ("SYN", "SYN-DIST", "HYPER", "HYPO", "CO-HYPO", "ANT", "POS", "NE-SUB", "FUNC", "META", "CONSTR", "PRON", "NUM")

LABELS = ("COPY", "INFLECT", "SUBST", "SPLIT", "MERGE")


def label_of(tag: str) -> tuple[str, Optional[str]]:
    """The label and the relation a code tag stands for; ``("SUBST", "SYN")`` for ``SYN``."""
    if tag in LABEL_OF_TAG:
        return LABEL_OF_TAG[tag], None
    if tag in RELATION_TAGS:
        return "SUBST", tag
    raise ValueError(f"unknown link tag {tag!r}")


def _side(raw: Dict[str, Any], key: str, tokens: Sequence[str]) -> Dict[str, Any]:
    meta = (raw.get("record") or {}).get(key) or {}
    return {
        "author": meta.get("author"),
        "work": meta.get("work"),
        "citation": meta.get("citation"),
        "tokens": list(tokens),
    }


def _runs(flags: Sequence[int]) -> List[Dict[str, int]]:
    """Consecutive flagged positions as half-open ranges."""
    spans: List[Dict[str, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            spans.append({"start": start, "end": i})
            start = None
    if start is not None:
        spans.append({"start": start, "end": len(flags)})
    return spans


def convert_record(raw: Dict[str, Any], *, fold: Optional[int] = None) -> Dict[str, Any]:
    """One raw prediction record as a released record.

    Raises ``ValueError`` when the record contradicts itself (a link without a tag, a deletion that a link claims), so a
    corrupt file fails here and not in somebody's analysis.
    """
    source, reuse = raw["source"], raw["target"]
    links, tags, frames, dels = raw["links"], raw["gated_tags"], raw["frames"], raw["dels"]
    if not (len(links) == len(tags) == len(frames) == len(reuse)):
        raise ValueError(f"{raw['id']}: per-word lists do not match the reuse passage's {len(reuse)} words")
    if len(dels) != len(source):
        raise ValueError(f"{raw['id']}: deletions do not match the source passage's {len(source)} words")

    confidence = {(x["r"], x["s"]): x.get("p") for x in raw.get("pred", {}).get("links", [])}
    out_links = []
    for r, s in enumerate(links):
        if s < 0:
            continue
        label, relation = label_of(tags[r])
        p = confidence.get((r, s))
        out_links.append({"reuse": r, "source": s, "label": label, "relation": relation,
                          "confidence": None if p is None else round(float(p), 5)})
    claimed = {link["source"] for link in out_links}
    deletions = [s for s in range(len(source)) if s not in claimed]
    if deletions != [s for s, flag in enumerate(dels) if flag]:
        raise ValueError(f"{raw['id']}: the deletions disagree with the links")

    frame_words = [1 if (frames[r] and links[r] < 0) else 0 for r in range(len(reuse))]
    meta = raw.get("record") or {}
    return {
        "id": raw["id"],
        "benchmark_id": meta.get("benchmark_id"),
        "fold": meta.get("fold", fold),
        "reference_type": raw.get("ref_type"),
        "source": _side(raw, "source", source),
        "reuse": _side(raw, "reuse", reuse),
        "links": out_links,
        "frame": _runs(frame_words),
        "insertions": [r for r in range(len(reuse)) if links[r] < 0 and not frame_words[r]],
        "deletions": deletions,
    }


def read_predictions(path: Path | str, *, fold: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """The released records of one raw ``predictions.jsonl``."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield convert_record(json.loads(line), fold=fold)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path | str) -> int:
    """Write records, one per line; returns how many."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def write_parquet(records: Sequence[Dict[str, Any]], path: Path | str) -> int:
    """Write records as Parquet (needs ``pyarrow``); returns how many."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as err:  # pragma: no cover - environment
        raise ImportError("Parquet output needs pyarrow: pip install pyarrow") from err
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(records)), path)
    return len(records)


def export_folds(sources: Sequence[Path | str], out_dir: Path | str, *, fmt: str = "jsonl") -> Dict[int, int]:
    """Convert one raw file per test fold into ``out_dir/fold_<k>.<fmt>``; returns pairs per fold.

    The fold is read from the records themselves, so the order of ``sources`` does not matter.
    """
    if fmt not in ("jsonl", "parquet"):
        raise ValueError("fmt must be 'jsonl' or 'parquet'")
    out_dir = Path(out_dir)
    counts: Dict[int, int] = {}
    for src in sources:
        records = list(read_predictions(src))
        folds = {record["fold"] for record in records}
        if len(folds) != 1 or None in folds:
            raise ValueError(f"{src}: expected the pairs of exactly one fold, found {sorted(map(str, folds))}")
        fold = folds.pop()
        target = out_dir / f"fold_{fold}.{fmt}"
        counts[fold] = write_jsonl(records, target) if fmt == "jsonl" else write_parquet(records, target)
    return dict(sorted(counts.items()))


def label_shares(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """The share of each label among all links, as a sanity check against a paper table."""
    totals = {label: 0 for label in LABELS}
    for record in records:
        for link in record["links"]:
            totals[link["label"]] += 1
    n = sum(totals.values()) or 1
    return {label: count / n for label, count in totals.items()}


__all__ = ["SCHEMA_VERSION", "LABELS", "convert_record", "read_predictions", "export_folds", "label_shares", "label_of",
           "write_jsonl", "write_parquet"]
