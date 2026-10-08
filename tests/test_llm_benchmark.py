"""E32 probe: item building from the adjudicated verdicts, tag parsing, scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retexo.llm.llm_benchmark import TagParser, TypeJob  # noqa: E402
from retexo.llm.llm_prompts import PROPOSE, TAGS, TYPE  # noqa: E402


def test_parsers():
    assert TagParser.tag("SYN-DIST") == "SYN-DIST"  # the longer tag wins
    assert TagParser.tag("The answer is SYN.") == "SYN"
    assert TagParser.tag("no tag here") is None
    assert TagParser.last_tag("they share a lemma\nMORPH") == "MORPH"
    got = TagParser.json_relations('{"SYN": ["ensis", ["ferrum", "iron"]], "ANT": []}')
    assert got["SYN"] == ["ensis", "ferrum"] and got["ANT"] == [] and got["HYPO"] == []


def test_items_and_scoring(tmp_path):
    dump = tmp_path / "d.jsonl"
    verdicts = tmp_path / "v.json"
    rows = [
        {
            "pair": "p1",
            "t": 3,
            "source": "pavor,",
            "target": "gemitus",
            "link_found": True,
            "ref_type": "cf.",
            "hybrid": "SYN-DIST",
            "gated": "SYN-DIST",
            "symbolic": "SUBST",
            "context_source": "ubique pavor,",
            "context_target": "ubique gemitus",
        },
        {"pair": "p2", "t": 1, "source": "a", "target": "b", "link_found": False},
        {
            "pair": "p3",
            "t": 2,
            "source": "x",
            "target": "y",
            "link_found": True,
            "ref_type": "cit.",
            "hybrid": "MORPH",
            "gated": "MORPH",
            "symbolic": "MORPH",
            "context_source": "x",
            "context_target": "y",
        },
    ]
    dump.write_text("\n".join(json.dumps(r) for r in rows))
    verdicts.write_text(json.dumps({"p1:pavor,>gemitus": "SYN-DIST", "p3:x>y": "?"}))
    job = TypeJob.from_dump(dump, verdicts)
    assert len(job) == 1 and job.items[0]["gold"] == "SYN-DIST"  # unlinked and "?" dropped
    sc = job.score(["SYN"])
    assert sc["llm_strict"] == 0.0 and sc["llm_lenient"] == 1.0  # SYN-DIST folds into SYN
    assert sc["gated_strict"] == 1.0 and sc["symbolic_strict"] == 0.0
    assert sc["confusion"]["SYN-DIST->SYN"] == 1


def test_every_prompt_renders_and_names_a_parser():
    item = {
        "source": "pavor,",
        "target": "gemitus",
        "context_source": "a b",
        "context_target": "c d",
    }
    for name in TYPE:
        system, user, parse = TYPE.render(name, **item)
        assert system and "pavor," in user and "gemitus" in user and parse in TagParser.PARSERS
    for name in PROPOSE:
        system, user, parse = PROPOSE.render(name, lemma="gladius", pos="noun")
        assert "gladius" in user and parse in TagParser.PARSERS and '{"SYN"' in user
    assert len(TAGS) == 12
