// Filtering and sorting of the pair list.
export const EMPTY_FILTERS = {
  query: "",
  label: "all",          // all | cit | cf
  sourceAuthor: "", sourceWork: "", reuseAuthor: "", reuseWork: "",
  hasSubst: false, hasInflect: false,   // in the predicted script
  sort: "id",
};

export const SORTS = [
  ["id", "Pair id"],
  ["confidence", "Least confident prediction first"],
  ["source", "Source citation"],
  ["reuse", "Reuse citation"],
  ["length", "Longest reuse passage first"],
];

export function activeCount(f) {
  return ["label", "sourceAuthor", "sourceWork", "reuseAuthor", "reuseWork", "hasSubst", "hasInflect"]
    .filter((k) => (k === "label" ? f.label !== "all" : Boolean(f[k]))).length;
}

// Sorted distinct values of one side's author, or its works (under an author, when one is chosen).
export function options(records, side, field, author = "") {
  const values = new Set();
  for (const r of records) {
    if (field === "work" && author && r[side].author !== author) continue;
    if (r[side][field]) values.add(r[side][field]);
  }
  return [...values].sort((a, b) => a.localeCompare(b));
}

const hasOp = (record, op) => (record.pred?.links ?? []).some((e) => e.op === op);

function matchesQuery(record, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  return [record.id, record.source.citation, record.reuse.citation, record.source.text, record.reuse.text]
    .some((s) => s?.toLowerCase().includes(q));
}

const minConfidence = (r) => Math.min(1, ...(r.pred?.links ?? []).map((e) => e.p ?? 1));

const COMPARE = {
  id: (a, b) => a.id.localeCompare(b.id),
  confidence: (a, b) => minConfidence(a.record) - minConfidence(b.record),
  source: (a, b) => a.record.source.citation.localeCompare(b.record.source.citation, undefined, { numeric: true }),
  reuse: (a, b) => a.record.reuse.citation.localeCompare(b.record.reuse.citation, undefined, { numeric: true }),
  length: (a, b) => b.record.reuse.tokens.length - a.record.reuse.tokens.length,
};

export function applyFilters(records, f) {
  const kept = records.filter((r) =>
    matchesQuery(r, f.query) &&
    (f.label === "all" || r.pair_label === f.label) &&
    (!f.sourceAuthor || r.source.author === f.sourceAuthor) &&
    (!f.sourceWork || r.source.work === f.sourceWork) &&
    (!f.reuseAuthor || r.reuse.author === f.reuseAuthor) &&
    (!f.reuseWork || r.reuse.work === f.reuseWork) &&
    (!f.hasSubst || hasOp(r, "SUBST")) &&
    (!f.hasInflect || hasOp(r, "INFLECT")));
  const compare = COMPARE[f.sort] ?? COMPARE.id;
  // ties fall back to the id, so the order is stable
  return kept.map((record) => ({ id: record.id, record }))
    .sort((a, b) => compare(a, b) || a.id.localeCompare(b.id))
    .map((x) => x.record);
}
