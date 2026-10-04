import { useEffect, useMemo, useRef, useState } from "react";
import FilterPanel from "./FilterPanel.jsx";
import { EMPTY_FILTERS, applyFilters } from "../filters.js";
import { LABEL_NAME } from "../labels.js";

const PAGE_SIZES = [25, 50, 100];

function Pager({ page, pages, onPage, total, from, count, pageSize, onPageSize }) {
  const button = "rounded-md px-2 py-0.5 text-xs font-bold text-accent hover:bg-accent-soft disabled:text-muted disabled:hover:bg-transparent";
  return (
    <nav className="flex items-center gap-1 border-t border-line px-3 py-2 text-xs text-muted" aria-label="Pages">
      <button type="button" className={button} onClick={() => onPage(0)} disabled={page === 0} aria-label="First page">«</button>
      <button type="button" className={button} onClick={() => onPage(page - 1)} disabled={page === 0}>Prev</button>
      <span className="px-1 tabular-nums">
        {total ? `${from + 1}–${from + count} of ${total}` : "0 of 0"} · page {page + 1}/{pages}
      </span>
      <button type="button" className={button} onClick={() => onPage(page + 1)} disabled={page >= pages - 1}>Next</button>
      <button type="button" className={button} onClick={() => onPage(pages - 1)} disabled={page >= pages - 1} aria-label="Last page">»</button>
      <select value={pageSize} onChange={(e) => onPageSize(Number(e.target.value))} aria-label="Pairs per page"
              className="ml-auto rounded-md border border-line bg-surface px-1 py-0.5 text-xs text-ink outline-none focus:border-accent">
        {PAGE_SIZES.map((n) => <option key={n} value={n}>{n} per page</option>)}
      </select>
    </nav>
  );
}

export default function PairList({ records, selectedId, onSelect }) {
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [open, setOpen] = useState(true);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(PAGE_SIZES[1]);
  const shown = useMemo(() => applyFilters(records, filters), [records, filters]);
  const pages = Math.max(1, Math.ceil(shown.length / pageSize));
  const current = Math.min(page, pages - 1);
  const onPage = shown.slice(current * pageSize, (current + 1) * pageSize);
  const listRef = useRef(null);

  // new filters or a new page size start on the first page
  const changeFilters = (next) => { setFilters(next); setPage(0); };
  const changePageSize = (size) => { setPageSize(size); setPage(0); };

  // a selection from outside (keyboard, URL) opens its page and scrolls into view
  useEffect(() => {
    const at = shown.findIndex((r) => r.id === selectedId);
    if (at >= 0 && Math.floor(at / pageSize) !== current) setPage(Math.floor(at / pageSize));
  }, [selectedId]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    listRef.current?.querySelector(`[data-id="${selectedId}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selectedId, current]);
  // a new page starts at its top
  useEffect(() => { listRef.current?.scrollTo({ top: 0 }); }, [current, pageSize]);

  function onKeyDown(event) {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const at = shown.findIndex((r) => r.id === selectedId);
    const next = shown[Math.min(shown.length - 1, Math.max(0, at + (event.key === "ArrowDown" ? 1 : -1)))];
    if (next) onSelect(next.id);          // crossing a page edge turns the page (effect above)
  }

  return (
    <aside className="relative z-10 flex min-h-0 flex-col border-black/10 bg-sidebar shadow-[3px_0_12px_-3px_rgba(0,0,0,0.14)] md:border-r">
      <div className="border-b border-line p-3">
        <FilterPanel records={records} filters={filters} onChange={changeFilters} open={open} onToggle={() => setOpen(!open)} />
        <p className="mt-2 text-xs text-muted">{shown.length} of {records.length} pairs</p>
      </div>
      <ul ref={listRef} tabIndex={0} onKeyDown={onKeyDown} aria-label="Pairs"
          className="min-h-0 flex-1 overflow-y-auto outline-none focus-visible:ring-2 focus-visible:ring-accent/40">
        {shown.length === 0 && <li className="p-4 text-sm text-muted">No pair matches these filters.</li>}
        {onPage.map((record, i) => {
          const index = current * pageSize + i;
          const active = record.id === selectedId;
          const cit = record.pair_label === "cit";
          return (
            <li key={record.id} data-id={record.id}>
              <button
                type="button"
                onClick={() => onSelect(record.id)}
                aria-current={active ? "true" : undefined}
                className={`flex w-full gap-3 border-b border-line-soft px-4 py-2.5 text-left transition-colors ${
                  active ? "bg-select" : "hover:bg-line-soft"
                }`}
              >
                <span className="w-6 shrink-0 pt-0.5 text-right text-[.75rem] font-bold tabular-nums text-muted">{index + 1}</span>

                <span className="min-w-0 flex-1">
                  <span className="flex items-start gap-3">
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[.95rem] font-bold text-ink">
                        {record.source.citation.replace(/[<>]/g, "")} → {record.reuse.citation.replace(/[<>]/g, "")}
                      </span>
                    </span>
                    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[.65rem] font-semibold uppercase tracking-wide ${
                      cit ? "bg-accent-soft text-accent" : "bg-pop-soft text-pop"
                    }`}>{LABEL_NAME[record.pair_label] ?? record.pair_label}</span>
                  </span>
                  <span className="mt-1 line-clamp-3 font-serif text-[.95rem] leading-snug text-ink-2">{record.reuse.text}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <Pager page={current} pages={pages} onPage={setPage} total={shown.length} from={current * pageSize}
             count={onPage.length} pageSize={pageSize} onPageSize={changePageSize} />
    </aside>
  );
}
