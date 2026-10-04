"""Convert the downloaded parquet datasets into JSON files for the web app.

Output goes to public/data/ and is bundled by Vite:
  stats.json       - row counts and columns of the datasets
  graph.json       - document/author graph of the references (nodes, hulls, edges)
  references.json  - all references with both segments (for the document browser)
  scripts.json     - retexo's predicted edit script of every reference (from the edit_scripts dataset)
  docs/<side>/<work>.json - all segments of one work (loaded on demand)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

AUTHOR_NAMES = {
    "hier": "Jerome",
    "lact": "Lactantius",
    "val": "Valerius Flaccus",
    "verg": "Virgil",
    "ov": "Ovid",
    "lucan": "Lucan",
    "stat": "Statius",
    "cic": "Cicero",
    "hor": "Horace",
    "lucr": "Lucretius",
    "prop": "Propertius",
    "mart": "Martial",
    "catull": "Catullus",
    "tib": "Tibullus",
}


def author_name(abbr: str) -> str:
    return AUTHOR_NAMES.get(abbr, abbr)


def build_graph(labels: pd.DataFrame, corpus: pd.DataFrame, queries: pd.DataFrame) -> dict:
    seg_counts = {
        "query": queries.groupby("work").size().to_dict(),
        "source": corpus.groupby("work").size().to_dict(),
    }
    nodes: dict[str, dict] = {}

    def node(work: str, author: str, side: str) -> dict:
        key = f"{side}:{work}"
        if key not in nodes:
            nodes[key] = {
                "id": key,
                "work": work,
                "author": author,
                "side": side,
                "segments": int(seg_counts[side].get(work, 0)),
                "refs": 0,
                "cit": 0,
                "cf": 0,
            }
        return nodes[key]

    edges: dict[tuple[str, str], dict] = {}
    for row in labels.itertuples(index=False):
        q = node(row.query_work, row.query_author, "query")
        s = node(row.corpus_work, row.corpus_author, "source")
        kind = "cit" if row.reference_type.startswith("cit") else "cf"
        for n in (q, s):
            n["refs"] += 1
            n[kind] += 1
        e = edges.setdefault((q["id"], s["id"]), {"source": q["id"], "target": s["id"], "refs": 0, "cit": 0, "cf": 0})
        e["refs"] += 1
        e[kind] += 1

    authors: dict[str, dict] = {}
    for n in nodes.values():
        a = authors.setdefault(
            n["author"],
            {"id": n["author"], "name": author_name(n["author"]), "side": n["side"], "works": [], "refs": 0, "cit": 0, "cf": 0},
        )
        a["works"].append(n["id"])
        a["refs"] += n["refs"]
        a["cit"] += n["cit"]
        a["cf"] += n["cf"]

    return {
        "nodes": sorted(nodes.values(), key=lambda n: (n["side"], -n["refs"])),
        "authors": sorted(authors.values(), key=lambda a: (a["side"], -a["refs"])),
        "edges": sorted(edges.values(), key=lambda e: -e["refs"]),
        "totals": {
            "refs": int(len(labels)),
            "cit": int((labels.reference_type.str.startswith("cit")).sum()),
            "cf": int((labels.reference_type.str.startswith("cf")).sum()),
        },
    }


def slug(work: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", work.lower()).strip("-")


PROVENANCE = {
    "burns": "Burns et al. (2021)",
    "ngram_pipeline": "Schropp et al. (2024, DHQ)",
    "schropp_goldstandard": "Schropp et al. (2024, DCO)",
}


def write_references(labels: pd.DataFrame, out_dir: Path) -> None:
    refs = []
    for r in labels.itertuples(index=False):
        refs.append({
            "id": int(r.id),
            "type": "cit" if r.reference_type.startswith("cit") else "cf",
            "q": {"author": r.query_author, "work": r.query_work, "cit": r.query_citation, "text": r.query_text, "en": r.query_text_english},
            "s": {"author": r.corpus_author, "work": r.corpus_work, "cit": r.corpus_citation, "text": r.corpus_text, "en": r.corpus_text_english},
            "prov": {"key": r.provenance_dataset, "label": PROVENANCE.get(r.provenance_dataset, r.provenance_dataset), "title": r.provenance_title, "url": r.provenance_url},
        })
    (out_dir / "references.json").write_text(json.dumps(refs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_dir / 'references.json'}: {len(refs)} references")


def write_docs(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    index = {}
    for side, name in (("query", "queries"), ("source", "corpus")):
        df = frames[name]
        d = out_dir / "docs" / side
        d.mkdir(parents=True, exist_ok=True)
        for work, g in df.groupby("work", sort=True):
            segs = [{"id": int(x.id), "cit": x.citation, "text": x.text} for x in g.itertuples(index=False)]
            file = f"{slug(work)}.json"
            (d / file).write_text(json.dumps(segs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            index[f"{side}:{work}"] = {"side": side, "work": work, "author": g.author.iloc[0], "author_name": author_name(g.author.iloc[0]), "file": f"docs/{side}/{file}", "segments": len(segs)}
    (out_dir / "docs.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_dir / 'docs.json'}: {len(index)} documents")


def prepare(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {p.stem: pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))}

    stats = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    for name, df in frames.items():
        stats[name] = {"rows": int(len(df)), "columns": list(df.columns)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'stats.json'}")

    graph = build_graph(frames["labels"], frames["corpus"], frames["queries"])
    (out_dir / "graph.json").write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_dir / 'graph.json'}: {len(graph['nodes'])} nodes, {len(graph['authors'])} authors, {len(graph['edges'])} edges")

    write_references(frames["labels"], out_dir)
    write_docs(frames, out_dir)
    write_scripts(frames, out_dir)


MODEL_NOTE = (
    "A pointer network over LatinBERT that chooses a source word, or none, for each word of the later passage and "
    "names the operation in the same step, with lexical evidence from Latin resources, a stretch head, and a copy "
    "rule at decoding. The predictions of each fold come from a model that was not trained on that fold."
)


def _cite(citation: str | None) -> str:
    """``<verg. aen. 6.847.1>`` as ``verg. aen. 6.847.1``."""
    return (citation or "").strip().strip("<>")


def build_scripts(frame: pd.DataFrame) -> dict:
    """The viewer's payload from the released records: per later word its source word, label, relation and confidence."""
    pairs = []
    for row in frame.sort_values(["benchmark_id", "id"]).itertuples(index=False):
        source, reuse = row.source, row.reuse
        n = len(reuse["tokens"])
        links, ops, relations, confidence = [-1] * n, ["INS"] * n, [None] * n, [None] * n
        for link in row.links:
            r = int(link["reuse"])
            links[r], ops[r] = int(link["source"]), link["label"]
            relations[r] = link["relation"]
            confidence[r] = link["confidence"]
        for span in row.frame:
            for r in range(int(span["start"]), int(span["end"])):
                ops[r] = "FRAME"
        pairs.append({
            "id": row.id,
            "ref_type": row.reference_type,
            "fold": int(row.fold),
            "earlier": list(source["tokens"]),
            "later": list(reuse["tokens"]),
            "earlier_cite": _cite(source["citation"]),
            "later_cite": _cite(reuse["citation"]),
            "pred": {"links": links, "ops": ops, "relations": relations, "confidence": confidence},
        })
    return {
        "model": {"name": "Stretch-and-link pointer", "note": MODEL_NOTE, "pairs": len(pairs),
                  "folds": int(frame["fold"].nunique())},
        "pairs": pairs,
    }


def write_scripts(frames: dict, out_dir: Path) -> None:
    """Write scripts.json from the edit_scripts dataset; the section stays empty when the dataset is missing."""
    frame = frames.get("edit_scripts")
    if frame is None:
        print("No edit_scripts dataset; the edit-script section will stay empty")
        return
    payload = build_scripts(frame)
    (out_dir / "scripts.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_dir / 'scripts.json'}: {len(payload['pairs'])} scripts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("public/data"))
    args = parser.parse_args()
    prepare(args.data_dir, args.out_dir)
