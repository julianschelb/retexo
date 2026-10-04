import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

const CIT = "#5b4cb0";
const CF = "#c05f21";
const PAGE = 150;

const segKey = (side, cit) => `${side}:${cit}`;

/** Group references into documents (works) per side, keeping only linked segments. */
function groupDocuments(refs, side) {
  const docs = new Map();
  refs.forEach((r) => {
    const seg = r[side === "query" ? "q" : "s"];
    let d = docs.get(seg.work);
    if (!d) {
      d = { work: seg.work, author: seg.author, segments: new Map(), refs: 0 };
      docs.set(seg.work, d);
    }
    d.refs += 1;
    let s = d.segments.get(seg.cit);
    if (!s) {
      s = { cit: seg.cit, text: seg.text, en: seg.en, refs: [] };
      d.segments.set(seg.cit, s);
    }
    s.refs.push(r);
  });
  const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  return [...docs.values()]
    .map((d) => ({ ...d, segments: new Map([...d.segments.entries()].sort((a, b) => natural.compare(a[0], b[0]))) }))
    .sort((a, b) => b.refs - a.refs);
}

function Segment({ side, seg, linked, active, dim, onHover, onClick, register }) {
  const key = segKey(side, seg.cit);
  return (
    <li
      ref={(el) => register(key, el)}
      onMouseEnter={() => linked && onHover(key)}
      onMouseLeave={() => linked && onHover(null)}
      onClick={() => linked && onClick(key)}
      className={`rounded-lg px-2.5 py-1.5 transition-colors ${
        linked ? "cursor-pointer" : ""
      } ${active ? "bg-accent-soft" : linked ? "hover:bg-line-soft" : ""} ${dim ? "opacity-40" : ""}`}
    >
      <div className="flex items-baseline gap-2">
        <span className={`shrink-0 font-mono text-[.7rem] ${linked ? "font-bold text-accent" : "text-muted"}`}>{seg.cit.replace(/^<|>$/g, "")}</span>
        {linked && seg.refs.length > 1 && (
          <span className="shrink-0 rounded-full bg-accent-soft px-1.5 font-mono text-[.65rem] font-bold text-accent">{seg.refs.length}</span>
        )}
      </div>
      <p className={`text-[.88rem] leading-[1.5] ${linked ? "text-ink" : "text-ink-2"}`}>{seg.text}</p>
      {active && seg.en && <p className="mt-1 text-[.8rem] italic leading-[1.45] text-muted">{seg.en}</p>}
    </li>
  );
}

function DocumentCard({ side, doc, docsIndex, activeKeys, dimKeys, onHover, onClick, register, onLayout }) {
  const meta = docsIndex?.[`${side}:${doc.work}`];
  const [whole, setWhole] = useState(null); // all segments of the work, when loaded
  const [loading, setLoading] = useState(false);

  const toggleWhole = async () => {
    if (whole) { setWhole(null); return; }
    if (!meta) return;
    setLoading(true);
    try {
      const r = await fetch(`${import.meta.env.BASE_URL}data/${meta.file}`);
      setWhole(await r.json());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { onLayout(); }, [whole, onLayout]);

  const rows = whole
    ? whole.map((s) => doc.segments.get(s.cit) ?? { cit: s.cit, text: s.text, refs: [] })
    : [...doc.segments.values()];

  return (
    <article className="rounded-2xl border border-line bg-surface shadow-card">
      <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-line-soft px-4 py-3">
        <div>
          <p className="text-[.65rem] font-extrabold uppercase tracking-[.1em] text-pop">{meta?.author_name ?? doc.author}</p>
          <h3 className="font-mono text-[1rem] font-bold text-ink">{doc.work}</h3>
        </div>
        <div className="flex items-center gap-3 text-[.78rem] font-bold">
          <span className="text-muted">{doc.refs} ref{doc.refs === 1 ? "" : "s"} · {doc.segments.size} seg{doc.segments.size === 1 ? "" : "s"}{meta ? ` of ${meta.segments.toLocaleString()}` : ""}</span>
          <button type="button" onClick={toggleWhole} disabled={loading || !meta}
            className="rounded-full border border-line px-2.5 py-0.5 text-ink-2 transition hover:border-accent hover:text-accent disabled:opacity-50">
            {loading ? "Loading…" : whole ? "Linked only" : "Whole document"}
          </button>
        </div>
      </header>
      {whole && whole.length > 3000 && (
        <p className="border-b border-line-soft bg-pop-soft px-4 py-1.5 text-[.75rem] font-semibold text-pop">Long document: {whole.length.toLocaleString()} segments.</p>
      )}
      <ul className="flex flex-col gap-0.5 p-2">
        {rows.map((seg) => {
          const key = segKey(side, seg.cit);
          return (
            <Segment key={seg.cit} side={side} seg={seg} linked={seg.refs.length > 0}
              active={activeKeys.has(key)} dim={dimKeys.size > 0 && !dimKeys.has(key) && seg.refs.length > 0}
              onHover={onHover} onClick={onClick} register={register} />
          );
        })}
      </ul>
    </article>
  );
}

export default function DocumentBrowser({ refs, docsIndex }) {
  const [limit, setLimit] = useState(PAGE);
  const [hoverKey, setHoverKey] = useState(null);
  const [pinKey, setPinKey] = useState(null);
  const [lineTip, setLineTip] = useState(null);
  const [lines, setLines] = useState([]);
  const containerRef = useRef(null);
  const nodeMap = useRef(new Map());

  useEffect(() => { setLimit(PAGE); setPinKey(null); setHoverKey(null); }, [refs]);

  const shown = useMemo(() => refs.slice(0, limit), [refs, limit]);
  const left = useMemo(() => groupDocuments(shown, "query"), [shown]);
  const right = useMemo(() => groupDocuments(shown, "source"), [shown]);

  const register = useCallback((key, el) => {
    if (el) nodeMap.current.set(key, el); else nodeMap.current.delete(key);
  }, []);

  const [layoutTick, setLayoutTick] = useState(0);
  const bump = useCallback(() => setLayoutTick((t) => t + 1), []);

  // Compute connection lines from the rendered segment rows.
  useLayoutEffect(() => {
    const box = containerRef.current?.getBoundingClientRect();
    if (!box) return;
    const next = [];
    shown.forEach((r) => {
      const a = nodeMap.current.get(segKey("query", r.q.cit));
      const b = nodeMap.current.get(segKey("source", r.s.cit));
      if (!a || !b) return;
      const ra = a.getBoundingClientRect();
      const rb = b.getBoundingClientRect();
      next.push({ ref: r, x1: ra.right - box.left, y1: ra.top + ra.height / 2 - box.top, x2: rb.left - box.left, y2: rb.top + rb.height / 2 - box.top });
    });
    setLines(next);
  }, [shown, layoutTick]);

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const ro = new ResizeObserver(() => bump());
    ro.observe(containerRef.current);
    window.addEventListener("resize", bump);
    return () => { ro.disconnect(); window.removeEventListener("resize", bump); };
  }, [bump]);

  // Active segment (pinned wins over hover) and everything linked to it.
  const focusKey = pinKey ?? hoverKey;
  const { activeKeys, dimKeys } = useMemo(() => {
    const active = new Set();
    const keep = new Set();
    if (focusKey) {
      active.add(focusKey);
      shown.forEach((r) => {
        const qk = segKey("query", r.q.cit);
        const sk = segKey("source", r.s.cit);
        if (qk === focusKey || sk === focusKey) { active.add(qk); active.add(sk); }
      });
      active.forEach((k) => keep.add(k));
    }
    return { activeKeys: active, dimKeys: keep };
  }, [focusKey, shown]);

  const lineActive = (l) => !focusKey || activeKeys.has(segKey("query", l.ref.q.cit)) && activeKeys.has(segKey("source", l.ref.s.cit)) && (segKey("query", l.ref.q.cit) === focusKey || segKey("source", l.ref.s.cit) === focusKey);

  if (refs.length === 0) {
    return <p className="rounded-2xl border-2 border-dashed border-line p-8 text-center text-muted">No references match the current filters.</p>;
  }

  const path = (l) => {
    const mx = (l.x1 + l.x2) / 2;
    return `M${l.x1},${l.y1} C${mx},${l.y1} ${mx},${l.y2} ${l.x2},${l.y2}`;
  };

  return (
    <div>
      <div ref={containerRef} className="relative">
        <div className="grid grid-cols-[1fr_96px_1fr] items-start">
          <div className="flex flex-col gap-4">{left.map((d) => (
            <DocumentCard key={d.work} side="query" doc={d} docsIndex={docsIndex} activeKeys={activeKeys} dimKeys={dimKeys}
              onHover={setHoverKey} onClick={(k) => setPinKey((p) => (p === k ? null : k))} register={register} onLayout={bump} />
          ))}</div>
          <div />
          <div className="flex flex-col gap-4">{right.map((d) => (
            <DocumentCard key={d.work} side="source" doc={d} docsIndex={docsIndex} activeKeys={activeKeys} dimKeys={dimKeys}
              onHover={setHoverKey} onClick={(k) => setPinKey((p) => (p === k ? null : k))} register={register} onLayout={bump} />
          ))}</div>
        </div>

        <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" aria-hidden="true">
          {lines.map((l) => {
            const on = lineActive(l);
            return (
              <g key={l.ref.id} className="pointer-events-auto">
                <path d={path(l)} fill="none" stroke={l.ref.type === "cit" ? CIT : CF} strokeWidth={on && focusKey ? 2.5 : 1.5}
                  opacity={focusKey ? (on ? 0.9 : 0.08) : 0.45} strokeLinecap="round" />
                <path d={path(l)} fill="none" stroke="transparent" strokeWidth="10" className="cursor-pointer"
                  onMouseEnter={(e) => { const b = containerRef.current.getBoundingClientRect(); setLineTip({ ref: l.ref, x: e.clientX - b.left, y: e.clientY - b.top }); setHoverKey(segKey("query", l.ref.q.cit)); }}
                  onMouseMove={(e) => { const b = containerRef.current.getBoundingClientRect(); setLineTip((t) => t && { ...t, x: e.clientX - b.left, y: e.clientY - b.top }); }}
                  onMouseLeave={() => { setLineTip(null); setHoverKey(null); }} />
              </g>
            );
          })}
        </svg>

        {lineTip && (
          <div className="pointer-events-none absolute z-10 w-64 rounded-xl border border-line bg-surface px-3 py-2 text-[.78rem] shadow-card-lg" style={{ left: lineTip.x + 12, top: lineTip.y + 12 }}>
            <div className="mb-1 flex items-center gap-2">
              <span className="rounded-full px-2 py-0.5 font-extrabold text-white" style={{ background: lineTip.ref.type === "cit" ? CIT : CF }}>
                {lineTip.ref.type === "cit" ? "verbatim (cit.)" : "allusion (cf.)"}
              </span>
              <span className="font-mono text-muted">#{lineTip.ref.id}</span>
            </div>
            <div className="font-mono text-[.72rem] text-ink-2">{lineTip.ref.q.cit.replace(/^<|>$/g, "")} → {lineTip.ref.s.cit.replace(/^<|>$/g, "")}</div>
            <div className="mt-1 text-muted">Source: {lineTip.ref.prov.label}</div>
          </div>
        )}
      </div>

      {refs.length > shown.length && (
        <div className="mt-6 text-center">
          <button type="button" onClick={() => setLimit((l) => l + PAGE)}
            className="rounded-full border border-line bg-surface px-5 py-2 text-[.9rem] font-extrabold text-ink-2 shadow-card transition hover:-translate-y-0.5 hover:border-accent hover:text-accent">
            Show {Math.min(PAGE, refs.length - shown.length)} more of {(refs.length - shown.length).toLocaleString()} remaining
          </button>
        </div>
      )}
    </div>
  );
}
