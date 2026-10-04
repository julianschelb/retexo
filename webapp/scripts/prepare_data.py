"""Build the review app's input from the downloaded datasets.

Writes public/data/records.jsonl, one pair per line:

    {"id", "pair_label" ("cit" | "cf"), "fold",
     "source": {"author", "work", "citation", "tokens", "text", "text_original", "text_english"},
     "reuse":  {... the same ...},
     "pred": {"links": [{"r", "s", "op", "p", "relation"}], "frame_spans": [{"start", "end", "label"}]}}

The links, labels and citing formulas come from the edit_scripts dataset; the Latin texts and English translations
from the labels dataset, joined on the reference's id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RECORDS_FILE = "records.jsonl"


def _none(value):
    """NaN and missing values as None, so that they are written as null."""
    if value is None:
        return None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def build_records(scripts: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    """The app's records, one per predicted script, ordered by the benchmark's reference id."""
    by_id = labels.set_index("id")
    records = []
    for row in scripts.sort_values(["benchmark_id", "id"]).itertuples(index=False):
        label = by_id.loc[row.benchmark_id]

        def side(meta, prefix):
            return {
                "author": meta["author"], "work": meta["work"], "citation": meta["citation"],
                "tokens": list(meta["tokens"]),
                "text": _none(label[f"{prefix}_text"]),
                "text_original": _none(label[f"{prefix}_text_original"]),
                "text_english": _none(label[f"{prefix}_text_english"]),
            }

        links = [{"r": int(link["reuse"]), "s": int(link["source"]), "op": link["label"],
                  "p": None if _none(link["confidence"]) is None else float(link["confidence"]),
                  "relation": _none(link["relation"])} for link in row.links]
        spans = [{"start": int(s["start"]), "end": int(s["end"]), "label": "FRAME"} for s in row.frame]
        records.append({
            "id": row.id,
            "pair_label": str(row.reference_type).rstrip("."),
            "fold": int(row.fold),
            "source": side(row.source, "corpus"),
            "reuse": side(row.reuse, "query"),
            "pred": {"links": links, "frame_spans": spans},
        })
    return records


def prepare(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts = pd.read_parquet(data_dir / "edit_scripts.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet")
    records = build_records(scripts, labels)
    with open(out_dir / RECORDS_FILE, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {out_dir / RECORDS_FILE}: {len(records)} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("public/data"))
    args = parser.parse_args()
    prepare(args.data_dir, args.out_dir)
