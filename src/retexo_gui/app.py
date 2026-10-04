# demo/app.py
"""
Gradio demo: two Latin sentences in, an edit plan and an alignment out.

Backends:

  1. ``--pointer`` — E7's pointer model. For each reuse word it names the source
     word it came from, or declines. On the hand labels it lands on the right
     source word for 94.1% of aligned tokens and correctly declines for 97.6%
     of the rest. This is the alignment half of the problem.
  2. ``--model`` — a trained tagger checkpoint;
  3. ``--stepwise`` — a trained step-wise seq2seq checkpoint;
  4. the symbolic teacher (no model needed), which always works and is the only
     backend that names *which relation* licensed a substitution.

**The two halves are shown side by side on purpose.** E2 established that the
model and the oracle are good at opposite things: the model aligns and cannot
name lexical relations (SYN 0.233, ANT 0.171), the oracle names them by lookup
and cannot align. Neither panel is the whole answer.

Launch:
    python demo/app.py --typed-pointer runs/e36b_base --device cuda    # the champion's aligner A
    python demo/app.py --pointer attic/runs/e7_models
    python demo/app.py --model runs/night_best
    python demo/app.py                              # teacher-only
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

from retexo.datasets.detection import DetectionScore  # noqa: E402
from retexo.core.oracle import EditPlanOracle  # noqa: E402
from retexo.resources import Resources  # noqa: E402
from retexo.core.scriba import Scriba  # noqa: E402
from retexo.datasets.teacher import RelabellingTeacher  # noqa: E402

def load_examples():
    """Every pair from the labelled corpus, hand-read ones first.

    The three hand-written examples this demo shipped with were chosen to look
    good. These are the actual corpus, in the order the labelling went, so what
    the app shows is what the data is.
    """
    from retexo.paths import home

    root = home()
    try:
        pairs = json.loads((root / "attic" / "labels_auto" / "pairs.json").read_text())
    except (OSError, ValueError):
        return [["arma virumque cano Troiae qui primus ab oris",
                 "arma virumque cano qui primus ab oris Italiam venit"]]
    try:
        labelled = set(json.loads(
            (root / "data" / "gold_full" / "labels.json").read_text()))
    except (OSError, ValueError):
        labelled = set()
    # Which pairs the served checkpoint has never seen. Most of this list was
    # its training data, so without the marker a reader would be judging the
    # model on examples it was fitted to -- flattering and misleading.
    held = set()
    for directory in sorted((root / "runs").glob("*/held_out.json")):
        try:
            held |= set(json.loads(directory.read_text())["held_out_ids"])
        except (OSError, ValueError, KeyError):
            pass
    pairs.sort(key=lambda pair: (pair["id"] not in held,
                                 pair["id"] not in labelled, pair["id"]))
    return [[pair["source"], pair["target"],
             "held out" if pair["id"] in held else "seen in training"]
            for pair in pairs]


COLOR = {
    "NOP": "#8a8a8a", "MORPH": "#22aa77", "SYN": "#2277aa",
    "SYN-DIST": "#2277aa", "HYPER": "#aa7722", "HYPO": "#aa7722",
    "ANT": "#cc3333", "REORDER": "#9966cc", "INS": "#cc7733",
    "DEL": "#cc7733", "FRAME": "#aaaaaa", "QUOTE": "#22aa77",
    "COPY": "#8a8a8a", "SUBST": "#2277aa",
}


from retexo.aligners.assignment import AssignmentPolicy, Reranker  # noqa: E402


class Pointer:
    """E7's pointer, wrapped for the demo.

    Loaded through ``load_model`` rather than reconstructed here, so the
    checkpoint cannot come up under a configuration it was not trained under --
    which for the pointer would silently change what it means.
    """

    def __init__(self, directory, device="cpu", assignment="hungarian",
                 identity_bonus=2.0):
        from retexo.formulations.checkpoint import PointerCheckpoint

        self.assignment = assignment
        self.identity_bonus = identity_bonus
        self.model, self.classes = PointerCheckpoint.load(Path(directory), device=device)
        if not self.model.config.pointer:
            raise ValueError(f"{directory} has no pointer head")
        self.name = f"pointer — {directory}"

    def align(self, source, target):
        """Everything the model says about one pair.

        Returns the chosen source word per reuse word, the operation derived
        from it, the *tagging head's* own opinion for comparison, the full
        candidate distribution, and the deletion head's verdict per source word.
        """
        from retexo.formulations.change_detector import ChangeExample

        example = ChangeExample(
            source_tokens=list(source), target_tokens=list(target),
            labels=[0] * len(target), operations=["INS"] * len(target),
            n_operations=0, source_labels=[1] * len(source),
        )
        # The pointer scores each reuse word independently, so nothing stops two
        # of them claiming the same source word -- it happened in 14.6% of pairs,
        # and cost twice over, since the word that needed that source found it
        # gone. E8 settled the fix: a global one-to-one assignment, Hungarian
        # rather than greedy, worth +0.076 macro and +0.123 SUBST over three
        # seeds for no retraining. ``argmax`` is what the model did before, kept
        # so the two can be shown side by side.
        # E9: the score is a dot product of two contextual vectors and nothing
        # else, so it cannot see that two words are spelled alike -- and when a
        # word moves, its representation moves with it while whatever takes the
        # vacated slot drifts towards it. Adding a bonus for surface identity is
        # worth +0.022 macro and +0.050 SUBST on top of the assignment, and it
        # is what lets the model follow a word across a reordering.
        raw = self.model.predict_alignment_scores([example])[0]
        scores = Reranker.rerank([raw], [example], self.identity_bonus)[0]
        links = (AssignmentPolicy.links_hungarian([scores])[0] if self.assignment == "hungarian"
                 else [(c[0][0] if c else -1) for c in scores])
        return {
            "links": links,
            "assignment": self.assignment,
            "argmax_links": self.model.predict_alignment([example])[0],
            "tags": self.model.operations_from_alignment([example], [links])[0],
            "head": self.model.predict_operations([example])[0],
            "scores": scores,
            "deleted": self.model.predict_source([example])[0],
        }


class TypedPointerBackend:
    """The champion's base aligner (E36, ``runs/e36b_base``: the typed pointer
    with the evidence vector), loaded through the harness adapter of note 15
    and decoded with the shared decoder; its own fine types name the links.

    The full champion adds a second aligner, the Llama rater and three repairs
    (``retexo/baselines/full_system.py``); this backend is its first and
    largest part, fold 4, four-way macro 0.807 on its own.
    """

    def __init__(self, directory, device="cpu", *, evidence=True, theta=0.45):
        from retexo.baselines import labels
        from retexo.baselines.base import BaselineConfig
        from retexo.baselines.typed_pointer import TypedPointerBaseline

        self.labels = labels
        self.theta = theta
        cfg = BaselineConfig(device=device, extra={"typed": 1, "evidence": int(evidence), "size": 0, "negatives": "none"})
        self.baseline = TypedPointerBaseline.load(Path(directory), cfg)
        self.assignment = "hungarian"
        self.name = f"typed pointer with evidence (the champion's aligner A) — {directory}"

    def align(self, source, target):
        from retexo.baselines.base import Prediction
        from retexo.baselines.record import Record

        # the adapter caches examples by record id: one id per pair, and no cache across requests
        pair_id = "demo/" + hashlib.sha1((" ".join(source) + "\n" + " ".join(target)).encode("utf-8")).hexdigest()[:12]
        self.baseline._examples.clear()
        record = Record(id=pair_id, level="demo", fold=-1, source_work="", source_tokens=list(source), reuse_work="",
                        reuse_tokens=list(target), pair_label="cit")
        pred = self.baseline.predict([record])[0]
        pred = self.baseline.postprocess(record, pred, {"theta": self.theta})
        rows = pred.scores or [[(-1, 1.0)] for _ in target]
        links = list(pred.links)
        tags = []
        for t, s in enumerate(links):
            if s < 0:
                tags.append("FRAME" if pred.frame and pred.frame[t] else "INS")
            else:
                canon, detail = self.labels.canonical(pred.tags[t] or "SUBST")
                tags.append(self.labels.TO_LINK_TAG.get(canon, canon) if not detail else detail)
        argmax = [(row[0][0] if row else -1) for row in rows]
        return {
            "links": links,
            "assignment": self.assignment,
            "argmax_links": argmax,
            "tags": tags,
            "head": tags,
            "scores": rows,
            "deleted": pred.dels or [0] * len(source),
        }


class Engine:
    def __init__(self, model_dir=None, stepwise_dir=None, pointer_dir=None,
                 device="cpu", assignment="hungarian", identity_bonus=2.0, typed_pointer_dir=None):
        self.resources = Resources(offline=True)
        self.oracle = EditPlanOracle(resources=self.resources)
        self.scriba = Scriba()
        self.teacher = RelabellingTeacher()
        self.model = None
        self.pointer = (Pointer(pointer_dir, device, assignment, identity_bonus)
                        if pointer_dir else None)
        if typed_pointer_dir:
            self.pointer = TypedPointerBackend(typed_pointer_dir, device)
        self.kind = "teacher (symbolic)"
        if model_dir:
            from retexo.formulations.token_classifier import (
                TokenClassifierModel,
            )
            self.model = TokenClassifierModel.load(Path(model_dir))
            self.kind = f"tagger — {model_dir}"
        elif stepwise_dir:
            from retexo.formulations.seq2seq_stepwise import (
                Seq2SeqStepwiseModel,
            )
            self.model = Seq2SeqStepwiseModel.load(Path(stepwise_dir))
            self.kind = f"step-wise seq2seq — {stepwise_dir}"

    def analyse(self, source, target):
        src, tgt = source.split(), target.split()
        if not src or not tgt:
            return None, None, None
        note = self.kind
        script = None
        if self.model is not None:
            try:
                script = self.model.predict(src, tgt)
            except Exception:
                script = None
            if script is None or not self.scriba.verify(script, src, tgt):
                script, _ = self.teacher.relabel_pair(src, tgt, self.oracle)
                note = f"{self.kind} → teacher fallback (model script invalid)"
        else:
            script, _ = self.teacher.relabel_pair(src, tgt, self.oracle)
        verified = self.scriba.verify(script, src, tgt)
        return script, verified, note


def _morph(resources, word):
    """Lemma and morphological features, where Collatinus can supply them."""
    if not resources.has("morphology"):
        return "", ""
    morphology = resources.morphology
    return morphology.lemma(word) or "", morphology.morpho_features(word) or ""


def _gloss(resources, source_word, target_word):
    """Whether two aligned words share a lemma, and what moved if so."""
    lemma_s, feat_s = _morph(resources, source_word)
    lemma_t, feat_t = _morph(resources, target_word)
    if lemma_s and lemma_s == lemma_t:
        moved = (f": {feat_s} → {feat_t}"
                 if feat_s and feat_t and feat_s != feat_t else "")
        return f"same lemma {lemma_s}{moved}", "#22aa77"
    if lemma_s and lemma_t:
        return f"{lemma_s} → {lemma_t}", "#2277aa"
    return "no morphology", "#999"


ROW, TOP, LEFT_X, RIGHT_X = 26, 48, 300, 560


def _svg_open(n_left, n_right):
    height = max(n_left, n_right, 1) * ROW + TOP + 30
    return (f"<svg width='100%' viewBox='0 0 1000 {height}' "
            f"style='font-family:system-ui;font-size:13px'>"
            f"<text x='{LEFT_X}' y='24' text-anchor='end' fill='#888' "
            f"font-size='11'>SOURCE</text>"
            f"<text x='{RIGHT_X}' y='24' fill='#888' font-size='11'>REUSE</text>")


def _curve(j, i, colour, weight, opacity, tooltip):
    y1, y2 = TOP + j * ROW - 4, TOP + i * ROW - 4
    mid = (LEFT_X + RIGHT_X) / 2
    return (f"<path d='M {LEFT_X + 8} {y1} C {mid} {y1}, {mid} {y2}, "
            f"{RIGHT_X - 8} {y2}' fill='none' stroke='{colour}' "
            f"stroke-width='{weight:.2f}' opacity='{opacity:.2f}'>"
            f"<title>{tooltip}</title></path>")


def render_diagram(resources, source, target, out):
    """The pointer: one curved line per decision, thickness = confidence."""
    links, tags, head, scores = (out["links"], out["tags"], out["head"],
                                 out["scores"])
    claimed = {j for j in links if j >= 0}
    parts = [_svg_open(len(source), len(target))]

    for i, token in enumerate(target):
        j = links[i] if i < len(links) else -1
        if j < 0:
            continue
        confidence = scores[i][0][1] if i < len(scores) and scores[i] else 0.0
        gloss, _ = _gloss(resources, source[j], token)
        parts.append(_curve(
            j, i, COLOR.get(tags[i], "#555"), 1 + 3 * confidence,
            0.3 + 0.6 * confidence,
            f"{html.escape(source[j])} &#8594; {html.escape(token)}  ·  "
            f"{tags[i]} {confidence:.2f}  ·  {html.escape(gloss)}"))

    for j, token in enumerate(source):
        y, kept = TOP + j * ROW, j in claimed
        parts.append(
            f"<text x='{LEFT_X}' y='{y}' text-anchor='end' "
            f"fill='{'#333' if kept else COLOR['DEL']}' "
            f"font-weight='{600 if kept else 400}'>{html.escape(token)}</text>")
        if not kept:
            parts.append(f"<text x='{LEFT_X + 14}' y='{y}' fill='{COLOR['DEL']}'"
                         f" font-size='10'>DEL</text>")

    for i, token in enumerate(target):
        y = TOP + i * ROW
        j = links[i] if i < len(links) else -1
        tag = tags[i]
        confidence = scores[i][0][1] if i < len(scores) and scores[i] else 0.0
        parts.append(
            f"<text x='{RIGHT_X}' y='{y}' fill='{'#333' if j >= 0 else COLOR['INS']}'"
            f" font-weight='{600 if j >= 0 else 400}'>{html.escape(token)}</text>"
            f"<text x='{RIGHT_X + 150}' y='{y}' fill='{COLOR.get(tag, '#555')}' "
            f"font-size='11' font-weight='600'>{tag}</text>"
            f"<text x='{RIGHT_X + 205}' y='{y}' fill='#999' font-size='11'>"
            f"{confidence:.2f}</text>")
        if i < len(head) and head[i] != tag:
            parts.append(f"<text x='{RIGHT_X + 250}' y='{y}' fill='#cc7733' "
                         f"font-size='10'>head says {html.escape(head[i])}</text>")

    parts.append("</svg>")
    return "".join(parts) + _legend()


def _legend():
    return ("<div style='font-family:system-ui;font-size:11px;color:#888;"
            "margin-top:2px'>"
            f"<b style='color:{COLOR['COPY']}'>━ COPY</b> the same word &nbsp; "
            f"<b style='color:{COLOR['SUBST']}'>━ SUBST</b> a different word in "
            f"the same slot &nbsp; "
            f"<b style='color:{COLOR['INS']}'>INS</b> no line — the citing "
            f"author's own &nbsp; "
            f"<b style='color:{COLOR['DEL']}'>DEL</b> nothing points at it"
            "&nbsp;·&nbsp; thicker line = more confident &nbsp;·&nbsp; "
            "hover a line for the morphology</div>")


def render_head_view(source, target, out):
    """The tagging head alone: a label per reuse word, with no source link.

    This is the older formulation, kept in the same model as a control. It sees
    the pair but never names a source word, so it cannot tell a substitution
    from an insertion except by guessing from context — which is the whole
    reason the pointer exists.
    """
    head, tags = out["head"], out["tags"]
    pills = []
    for i, token in enumerate(target):
        tag = head[i] if i < len(head) else "INS"
        differs = i < len(tags) and tags[i] != tag
        colour = COLOR.get(tag, "#555")
        border = f"2px solid {COLOR['INS']}" if differs else "1px solid #ddd"
        note = (f"<div style='font-size:9px;color:{COLOR['INS']}'>"
                f"pointer: {tags[i]}</div>" if differs else "")
        pills.append(
            f"<span style='display:inline-block;margin:3px;padding:4px 9px;"
            f"border:{border};border-radius:7px;text-align:center'>"
            f"<div style='font-weight:600'>{html.escape(token)}</div>"
            f"<div style='font-size:10px;color:{colour};font-weight:600'>{tag}"
            f"</div>{note}</span>")
    disagreements = sum(1 for i, t in enumerate(tags)
                        if i < len(head) and head[i] != t)
    note = (f"<div style='font-size:11px;color:{COLOR['INS']};margin-top:6px'>"
            f"{disagreements} word(s) where the two heads of the same model "
            f"contradict each other — outlined above</div>"
            if disagreements else
            "<div style='font-size:11px;color:#22aa77;margin-top:6px'>"
            "the two heads agree everywhere</div>")
    return (f"<div style='font-family:system-ui'>{''.join(pills)}{note}</div>")


def render_oracle_diagram(resources, source, target, script):
    """The symbolic oracle's typed plan, drawn the same way as the pointer's.

    The oracle has the full 18-tag inventory and says *why* — ``wn:n`` for a
    WordNet synonym, ``lemma=arma sg.nom.→pl.abl.`` for an inflection. It cannot
    align: it decides one pair of words at a time by lookup.
    """
    parts = [_svg_open(len(source), len(target))]
    linked_source, linked_target = set(), set()

    for op in script.operations:
        if not op.source_indices or not op.target_indices:
            continue
        for j in op.source_indices:
            for i in op.target_indices:
                if j >= len(source) or i >= len(target):
                    continue
                linked_source.add(j); linked_target.add(i)
                detail = f"  ·  {op.detail}" if op.detail else ""
                parts.append(_curve(
                    j, i, COLOR.get(op.tag, "#555"), 2.0, 0.7,
                    f"{html.escape(source[j])} &#8594; {html.escape(target[i])}"
                    f"  ·  {op.tag}{html.escape(detail)}"))

    tag_for_target = {}
    for op in script.operations:
        for i in op.target_indices:
            tag_for_target[i] = op

    for j, token in enumerate(source):
        y, kept = TOP + j * ROW, j in linked_source
        parts.append(
            f"<text x='{LEFT_X}' y='{y}' text-anchor='end' "
            f"fill='{'#333' if kept else COLOR['DEL']}' "
            f"font-weight='{600 if kept else 400}'>{html.escape(token)}</text>")
        if not kept:
            parts.append(f"<text x='{LEFT_X + 14}' y='{y}' fill='{COLOR['DEL']}'"
                         f" font-size='10'>DEL</text>")

    for i, token in enumerate(target):
        y = TOP + i * ROW
        op = tag_for_target.get(i)
        tag = op.tag if op else "INS"
        parts.append(
            f"<text x='{RIGHT_X}' y='{y}' "
            f"fill='{'#333' if i in linked_target else COLOR['INS']}' "
            f"font-weight='{600 if i in linked_target else 400}'>"
            f"{html.escape(token)}</text>"
            f"<text x='{RIGHT_X + 150}' y='{y}' fill='{COLOR.get(tag, '#555')}' "
            f"font-size='11' font-weight='600'>{tag}</text>")
        if op is not None and op.detail:
            parts.append(f"<text x='{RIGHT_X + 215}' y='{y}' fill='#999' "
                         f"font-size='10'>{html.escape(op.detail[:34])}</text>")

    parts.append("</svg>")
    return "".join(parts)


def render_summary(source, target, out):
    """What to look at, and the checks that can actually fail."""
    from collections import Counter
    from retexo.core.normalize import normalize

    links, tags, head, scores = (out["links"], out["tags"], out["head"],
                                 out["scores"])
    argmax_links = out.get("argmax_links", links)
    aligned = sum(1 for j in links if j >= 0)
    counts = Counter(tags)
    confident = [s[0][1] for s in scores if s]
    mean_c = sum(confident) / len(confident) if confident else 0.0
    low = [(i, target[i], s[0][1]) for i, s in enumerate(scores)
           if s and s[0][1] < 0.9]

    claimed = Counter(j for j in links if j >= 0)
    twice = [j for j, n in claimed.items() if n > 1]
    rescued = sum(1 for a, b in zip(links, argmax_links) if a != b)
    bad_copy = [i for i, tag in enumerate(tags)
                if tag == "COPY" and links[i] >= 0
                and normalize(target[i]) != normalize(source[links[i]])]
    disagree = [i for i, t in enumerate(tags)
                if i < len(head) and head[i] != t]

    def card(label, value, colour="#333"):
        return (f"<div style='display:inline-block;min-width:104px;padding:8px 12px;"
                f"margin:0 8px 8px 0;border:1px solid #eee;border-radius:8px'>"
                f"<div style='font-size:10px;color:#999;text-transform:uppercase;"
                f"letter-spacing:.4px'>{label}</div>"
                f"<div style='font-size:17px;font-weight:600;color:{colour}'>"
                f"{value}</div></div>")

    cards = (card("aligned", f"{aligned}/{len(target)}")
             + card("COPY", counts.get("COPY", 0), COLOR["COPY"])
             + card("SUBST", counts.get("SUBST", 0), COLOR["SUBST"])
             + card("INS", counts.get("INS", 0), COLOR["INS"])
             + card("DEL", len(source) - len(set(j for j in links if j >= 0)),
                    COLOR["DEL"])
             + card("mean confidence", f"{mean_c:.2f}"))

    flags = []
    if twice:
        flags.append(f"<b style='color:#cc3333'>{len(twice)} source word(s) "
                     f"claimed by more than one reuse word</b> — the alignment "
                     f"is meant to be one-to-one, so this is a contradiction")
    elif rescued:
        # With the one-to-one assignment on, the contradiction cannot occur, so
        # the panel reports what it repaired instead of a warning that can no
        # longer fire. These are the links the raw argmax would have got wrong.
        flags.append(f"<b style='color:#2a7'>{rescued} link(s) moved by the "
                     f"one-to-one assignment</b> — the raw pointer spent a "
                     f"source word twice here and this reclaimed it")
    if bad_copy:
        flags.append(f"<b style='color:#cc3333'>{len(bad_copy)} COPY link(s) "
                     f"whose two forms differ</b> even after normalization")
    if disagree:
        flags.append(f"<b style='color:{COLOR['INS']}'>{len(disagree)} word(s) "
                     f"where the model's two heads contradict each other</b> — "
                     f"see the next panel")
    if low:
        worst = ", ".join(f"<i>{html.escape(w)}</i> {c:.2f}" for _, w, c in low[:4])
        flags.append(f"{len(low)} decision(s) below 0.90 confidence: {worst}")
    if not flags:
        flags.append("no contradictions; every decision above 0.90 confidence")

    caveat = ("<div style='font-size:11px;color:#999;margin-top:10px;"
              "border-top:1px solid #eee;padding-top:8px;text-align:left'>"
              "No “verified” tick here on purpose: replaying “write this token” "
              "reproduces the reuse whatever the alignment says, so a tick "
              "would mean nothing. The flags above are the things that can "
              "actually be wrong. On the hand-labelled pairs the pointer picks "
              "the right source word for <b>94.1%</b> of aligned tokens and "
              "correctly declines for <b>97.6%</b> of the rest — and its "
              "confidence is <b>not</b> a reliable guide to which are which."
              "</div>")
    return ("<div style='font-family:system-ui;text-align:center'>"
            + "<div style='display:inline-block;text-align:center'>" + cards
            + "</div>"
            + "<div style='max-width:760px;margin:6px auto 0;text-align:left'>"
            + "<ul style='margin:0 0 0 18px;padding:0;font-size:12px;"
            "color:#555;line-height:1.7'>"
            + "".join(f"<li>{f}</li>" for f in flags) + "</ul>"
            + caveat + "</div></div>")


def render(script, verified):
    rows = []
    for op in script.operations:
        colour = COLOR.get(op.tag, "#555")
        src = " ".join(op.source_tokens) or "∅"
        tgt = " ".join(op.target_tokens) or "∅"
        arrow = "→" if op.target_tokens else "⊗"
        rows.append(
            f"<tr><td style='color:{colour};font-weight:600;padding:2px 12px'>"
            f"{html.escape(op.tag)}</td>"
            f"<td style='padding:2px 12px'>{html.escape(src)}</td>"
            f"<td style='color:#999'>{arrow}</td>"
            f"<td style='padding:2px 12px'>{html.escape(tgt)}</td></tr>"
        )
    ok = "✓ verified — replaying the plan reproduces the reuse" if verified \
         else "⚠ not reproducible"
    colour = "#22aa77" if verified else "#cc3333"
    return (f"<table style='border-collapse:collapse;font-family:system-ui'>"
            f"{''.join(rows)}</table>"
            f"<p style='color:{colour};font-weight:600;margin-top:8px'>{ok}</p>")


def build(engine):
    import gradio as gr

    def run(source_text, target_text):
        src, tgt = source_text.split(), target_text.split()
        if not src or not tgt:
            return "", "", "", ""

        if engine.pointer is None:
            script, verified, note = engine.analyse(source_text, target_text)
            return render(script, verified), note, "", ""

        out = engine.pointer.align(src, tgt)
        try:
            script, _, _ = engine.analyse(source_text, target_text)
            oracle_html = render_oracle_diagram(engine.resources, src, tgt, script)
        except Exception as error:  # noqa: BLE001 - the demo must not die here
            oracle_html = (f"<div style='color:#cc3333;font-family:system-ui'>"
                           f"oracle failed: {html.escape(str(error))}</div>")

        return (render_diagram(engine.resources, src, tgt, out),
                render_summary(src, tgt, out),
                render_head_view(src, tgt, out),
                oracle_html)

    examples = load_examples()

    with gr.Blocks(title="Latin Edit Plans", theme=gr.themes.Soft()) as app:
        if engine.pointer:
            gr.Markdown(
                "# Latin intertextual reuse\n"
                "Three views of the same pair of passages, from three different "
                "things: a pointer model that says where each reuse word came "
                "from, the tagging head sitting beside it in the same network, "
                "and a symbolic oracle that uses no model at all. They disagree "
                "with each other, and where they disagree is usually the "
                f"interesting part. Model: {engine.pointer.name}.")
        else:
            gr.Markdown(f"# Latin intertextual reuse\nBackend: {engine.kind}.")

        with gr.Row():
            source = gr.Textbox(label="Source — the earlier author", lines=3,
                                scale=1)
            target = gr.Textbox(label="Reuse — the citing author", lines=3,
                                scale=1)
        go = gr.Button("Analyse", variant="primary")

        # The served checkpoint was fine-tuned on most of this corpus, so a
        # reader clicking through it is mostly seeing training data. The column
        # says which, and the held-out pairs are listed first.
        held = sum(1 for e in examples if e[2] == "held out")
        provenance = gr.Textbox(label="This pair, for the served model",
                                interactive=False, lines=1)
        with gr.Accordion(f"Examples — all {len(examples):,} pairs of the "
                          f"corpus, the {held} held out from training first",
                          open=False):
            gr.Examples(examples, [source, target, provenance],
                        examples_per_page=8, cache_examples=False, label="")

        if engine.pointer:
            gr.Markdown(
                "### The pointer\n"
                "For every word of the reuse the model names the source word it "
                "came from, or declines to point at anything. The operation "
                "follows from that choice rather than being predicted "
                "separately: pointing nowhere is an insertion, pointing at a "
                "word with the same normalised form is a copy, and pointing "
                "anywhere else is a substitution. Line thickness is the "
                "model's confidence, and crossing lines mean the reuse has "
                "reordered its source.")
        plan = gr.HTML()
        summary = gr.HTML()

        if engine.pointer:
            gr.Markdown(
                "### The tagging head\n"
                "The same network has a second output that labels each reuse "
                "word directly, as a copy, a substitution or an insertion, "
                "without ever naming a source word. It is the older "
                "formulation, kept as a control, and when it disagrees with the "
                "pointer the model is contradicting itself about the same word. "
                "That happens because a head with no source word to point at "
                "cannot separate a substitution from an insertion except by "
                "guessing from context, which is the reason the pointer was "
                "added: on the held-out fold the pointer scores 0.719 on "
                "substitutions and this head 0.453.")
        head_view = gr.HTML()

        if engine.pointer:
            gr.Markdown(
                "### The symbolic oracle\n"
                "No model at all, just dictionary and WordNet lookups over the "
                "full eighteen-tag inventory. It can say why a substitution is "
                "one, naming an attested synonym or the exact inflection that "
                "moved, which the model cannot do at all. What it cannot do is "
                "align, because it judges one pair of words at a time with no "
                "view of the passage. Hover a line to see its reason.")
        oracle_view = gr.HTML()

        go.click(run, [source, target],
                 [plan, summary, head_view, oracle_view])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="")
    parser.add_argument("--stepwise", default="")
    parser.add_argument("--pointer", default="",
                        help="E7 checkpoint directory, e.g. attic/runs/e7_models")
    parser.add_argument("--typed-pointer", default="",
                        help="a typed-pointer checkpoint (modules.pt), e.g. runs/e36b_base: "
                             "the champion's base aligner, loaded through the harness")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--identity-bonus", type=float, default=2.0,
                        help="E9 surface-identity bonus added to any candidate "
                             "spelled like the reuse word; 0.0 disables it")
    parser.add_argument("--assignment", default="hungarian",
                        choices=("hungarian", "argmax"),
                        help="how the pointer's scores become one source per "
                             "reuse word. 'hungarian' is the E8 one-to-one "
                             "assignment (+0.076 macro); 'argmax' is the "
                             "unconstrained behaviour it replaced")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on; 0.0.0.0 exposes the demo to the network")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    engine = Engine(args.model or None, args.stepwise or None,
                    args.pointer or None, args.device,
                    assignment=args.assignment,
                    identity_bonus=args.identity_bonus,
                    typed_pointer_dir=args.typed_pointer or None)
    print(f"backend: {engine.kind}", flush=True)
    if engine.pointer:
        print(f"pointer: {engine.pointer.name}", flush=True)
    build(engine).launch(share=args.share, server_port=args.port,
                         server_name=args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
