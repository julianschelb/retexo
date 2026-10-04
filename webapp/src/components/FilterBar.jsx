import { EMPTY_FILTERS } from "../filters.js";

function Select({ label, value, options, onChange, placeholder = "any" }) {
  return (
    <label className="flex min-w-0 flex-col gap-1">
      <span className="text-[.65rem] font-extrabold uppercase tracking-[.1em] text-muted">{label}</span>
      <span className="relative">
        <select
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value || null)}
          className={`w-full appearance-none rounded-xl border bg-surface py-2 pl-3 pr-8 text-[.9rem] font-bold transition focus-visible:outline-2 focus-visible:outline-accent ${
            value ? "border-accent text-accent" : "border-line text-ink-2 hover:border-accent"
          }`}
        >
          <option value="">{placeholder}</option>
          {options.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[.7rem] text-muted">▼</span>
      </span>
    </label>
  );
}

export default function FilterBar({ graph, filters, onChange, shown, total }) {
  if (!graph) return null;
  const authors = (side) => graph.authors.filter((a) => a.side === side).map((a) => ({ value: a.id, label: a.name }));
  const works = (side, author) =>
    graph.nodes
      .filter((n) => n.side === side && (!author || n.author === author))
      .sort((a, b) => a.work.localeCompare(b.work))
      .map((n) => ({ value: n.work, label: `${n.work} (${n.refs})` }));
  const set = (patch) => onChange({ ...filters, ...patch });
  const active = Object.values(filters).some(Boolean);

  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-card">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[1fr_1fr_1fr_1fr_.7fr]">
        <Select label="Citing author" value={filters.qAuthor} options={authors("query")}
          onChange={(v) => set({ qAuthor: v, qWork: null })} />
        <Select label="Citing work" value={filters.qWork} options={works("query", filters.qAuthor)}
          onChange={(v) => set({ qWork: v, qAuthor: v ? graph.nodes.find((n) => n.side === "query" && n.work === v).author : filters.qAuthor })} />
        <Select label="Source author" value={filters.sAuthor} options={authors("source")}
          onChange={(v) => set({ sAuthor: v, sWork: null })} />
        <Select label="Source work" value={filters.sWork} options={works("source", filters.sAuthor)}
          onChange={(v) => set({ sWork: v, sAuthor: v ? graph.nodes.find((n) => n.side === "source" && n.work === v).author : filters.sAuthor })} />
        <Select label="Type" value={filters.type} placeholder="both"
          options={[{ value: "cit", label: "verbatim (cit.)" }, { value: "cf", label: "allusion (cf.)" }]}
          onChange={(v) => set({ type: v })} />
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[.85rem] font-semibold text-muted">
        <span>
          Showing <span className="font-mono text-ink">{shown.toLocaleString()}</span> of <span className="font-mono">{total.toLocaleString()}</span> references
          {active ? "" : " · select an author, work or edge in the graph, or use the filters"}
        </span>
        {active && (
          <button type="button" onClick={() => onChange(EMPTY_FILTERS)} className="rounded-full border border-line px-3 py-1 font-bold text-ink-2 transition hover:border-pop hover:text-pop">
            Clear filters ×
          </button>
        )}
      </div>
    </div>
  );
}
