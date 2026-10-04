import { EMPTY_FILTERS, SORTS, activeCount, options } from "../filters.js";

function Segmented({ value, onChange, items, label }) {
  return (
    <div className="inline-flex rounded-lg border border-line p-0.5 text-xs font-bold" role="group" aria-label={label}>
      {items.map(([key, name]) => (
        <button key={key} type="button" onClick={() => onChange(key)} aria-pressed={value === key}
                className={`rounded-md px-2 py-0.5 ${value === key ? "bg-accent text-white" : "text-ink-2 hover:bg-bg"}`}>
          {name}
        </button>
      ))}
    </div>
  );
}

function Select({ label, value, onChange, values, all }) {
  return (
    <label className="block min-w-0">
      <span className="text-[.65rem] font-semibold uppercase tracking-[.08em] text-muted">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
              className="mt-0.5 w-full truncate rounded-md border border-line bg-surface px-2 py-1 text-sm outline-none focus:border-accent">
        <option value="">{all}</option>
        {values.map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
    </label>
  );
}

function Toggle({ checked, onChange, children }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-1.5 text-sm">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-accent" />
      {children}
    </label>
  );
}

export default function FilterPanel({ records, filters, onChange, open, onToggle }) {
  const set = (patch) => onChange({ ...filters, ...patch });
  const active = activeCount(filters);
  return (
    <div className="space-y-2">
      <input type="search" value={filters.query} onChange={(e) => set({ query: e.target.value })}
             placeholder="Search id, citation or text"
             className="w-full rounded-lg border border-line bg-surface px-3 py-1.5 text-sm outline-none placeholder:text-muted focus:border-accent" />

      <div className="flex items-center gap-2">
        <button type="button" onClick={onToggle} aria-expanded={open}
                className="text-xs font-bold text-accent hover:underline">
          {open ? "Hide filters" : "Filters"}{active ? ` (${active})` : ""}
        </button>
        {active > 0 && (
          <button type="button" onClick={() => onChange({ ...EMPTY_FILTERS, query: filters.query, sort: filters.sort })}
                  className="text-xs font-bold text-muted hover:text-pop">Reset</button>
        )}
        <label className="ml-auto flex items-center gap-1 text-xs text-muted">
          Sort
          <select value={filters.sort} onChange={(e) => set({ sort: e.target.value })}
                  className="max-w-[11rem] rounded-md border border-line bg-surface px-1.5 py-0.5 text-xs text-ink outline-none focus:border-accent">
            {SORTS.map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select>
        </label>
      </div>

      {open && (
        <div className="space-y-3 rounded-md border border-line bg-surface/60 p-3">
          <div className="flex items-center gap-2">
            <span className="text-[.65rem] font-semibold uppercase tracking-[.08em] text-muted">Type</span>
            <Segmented label="Reference type" value={filters.label} onChange={(label) => set({ label })}
                       items={[["all", "All"], ["cit", "cit."], ["cf", "cf."]]} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Select label="Source author" value={filters.sourceAuthor} all="All authors"
                    values={options(records, "source", "author")}
                    onChange={(sourceAuthor) => set({ sourceAuthor, sourceWork: "" })} />
            <Select label="Source work" value={filters.sourceWork} all="All works"
                    values={options(records, "source", "work", filters.sourceAuthor)}
                    onChange={(sourceWork) => set({ sourceWork })} />
            <Select label="Reuse author" value={filters.reuseAuthor} all="All authors"
                    values={options(records, "reuse", "author")}
                    onChange={(reuseAuthor) => set({ reuseAuthor, reuseWork: "" })} />
            <Select label="Reuse work" value={filters.reuseWork} all="All works"
                    values={options(records, "reuse", "work", filters.reuseAuthor)}
                    onChange={(reuseWork) => set({ reuseWork })} />
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Toggle checked={filters.hasSubst} onChange={(hasSubst) => set({ hasSubst })}>has SUBST</Toggle>
            <Toggle checked={filters.hasInflect} onChange={(hasInflect) => set({ hasInflect })}>has INFLECT</Toggle>
          </div>
        </div>
      )}
    </div>
  );
}
