// Filter model shared by the graph, the filter bar and the document browser.
export const EMPTY_FILTERS = { qAuthor: null, qWork: null, sAuthor: null, sWork: null, type: null };

/** Convert a graph selection (author / work / edge) into filters. */
export function filtersFromSelection(sel, graph) {
  if (!sel) return EMPTY_FILTERS;
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  if (sel.kind === "author") {
    return sel.side === "query" ? { ...EMPTY_FILTERS, qAuthor: sel.id } : { ...EMPTY_FILTERS, sAuthor: sel.id };
  }
  if (sel.kind === "work") {
    const n = byId[sel.id];
    return n.side === "query"
      ? { ...EMPTY_FILTERS, qAuthor: n.author, qWork: n.work }
      : { ...EMPTY_FILTERS, sAuthor: n.author, sWork: n.work };
  }
  const q = byId[sel.source];
  const s = byId[sel.target];
  return { ...EMPTY_FILTERS, qAuthor: q.author, qWork: q.work, sAuthor: s.author, sWork: s.work };
}

/** Derive the graph highlight from the filters. */
export function selectionFromFilters(f, graph) {
  if (!graph) return null;
  const name = (id) => graph.authors.find((a) => a.id === id)?.name ?? id;
  if (f.qWork && f.sWork) return { kind: "edge", source: `query:${f.qWork}`, target: `source:${f.sWork}`, label: `${f.qWork} → ${f.sWork}` };
  if (f.qWork) return { kind: "work", id: `query:${f.qWork}`, label: f.qWork };
  if (f.sWork) return { kind: "work", id: `source:${f.sWork}`, label: f.sWork };
  if (f.qAuthor) return { kind: "author", id: f.qAuthor, side: "query", label: name(f.qAuthor) };
  if (f.sAuthor) return { kind: "author", id: f.sAuthor, side: "source", label: name(f.sAuthor) };
  return null;
}

export function applyFilters(refs, f) {
  return refs.filter(
    (r) =>
      (!f.qAuthor || r.q.author === f.qAuthor) &&
      (!f.qWork || r.q.work === f.qWork) &&
      (!f.sAuthor || r.s.author === f.sAuthor) &&
      (!f.sWork || r.s.work === f.sWork) &&
      (!f.type || r.type === f.type)
  );
}

// URL hash <-> filters, so filtered views can be shared.
const KEYS = ["qAuthor", "qWork", "sAuthor", "sWork", "type"];

export function filtersToHash(f) {
  const p = new URLSearchParams();
  KEYS.forEach((k) => f[k] && p.set(k, f[k]));
  const s = p.toString();
  return s ? `#${s}` : "";
}

export function filtersFromHash(hash) {
  const p = new URLSearchParams(hash.replace(/^#/, ""));
  const f = { ...EMPTY_FILTERS };
  KEYS.forEach((k) => { if (p.get(k)) f[k] = p.get(k); });
  return f;
}
