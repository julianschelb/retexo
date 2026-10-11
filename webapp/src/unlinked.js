// The label of a word without a link, in the paper's operations (2026-10-09). The reused stretch runs from the
// first to the last linked word of a passage: inside it, an unlinked reuse word is an insertion (INS) and an
// unclaimed source word a deletion (DEL); outside it they are the later author's new text (NEW) and source text not
// taken up (OUT). A reuse word in a citing formula is MARK. The same rule writes the data file
// as the published edit-scripts dataset (columns insertions, new_text, deletions, not_taken_up), so the page and the
// dataset agree word for word.

function stretch(indices) {
  let lo = Infinity, hi = -Infinity;
  for (const i of indices) { if (i < lo) lo = i; if (i > hi) hi = i; }
  return [lo, hi];
}

// One tag per reuse word: the link's operation, MARK, INS or NEW.
export function reuseTags(n, links, spans = []) {
  const mark = new Set();
  for (const span of spans) for (let r = span.start; r < span.end; r += 1) mark.add(r);
  const op = new Map();
  for (const e of [...links].sort((a, b) => a.r - b.r || a.s - b.s)) if (!op.has(e.r)) op.set(e.r, e.op);
  const [lo, hi] = stretch(op.keys());
  return Array.from({ length: n }, (_, r) =>
    op.get(r) ?? (mark.has(r) ? "MARK" : lo < r && r < hi ? "INS" : "NEW"));
}

// One tag per source word: the operation of the first link that claims it, DEL or OUT.
export function sourceTags(n, links) {
  const op = new Map();
  for (const e of links) if (!op.has(e.s)) op.set(e.s, e.op);
  const [lo, hi] = stretch(op.keys());
  return Array.from({ length: n }, (_, s) => op.get(s) ?? (lo < s && s < hi ? "DEL" : "OUT"));
}

// Tags of words without a source or without a reuse: shown muted.
export const UNLINKED = new Set(["MARK", "INS", "NEW", "DEL", "OUT"]);
