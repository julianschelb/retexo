import { useLayoutEffect, useMemo, useRef, useState } from "react";

/** Operation vocabulary of the edit script, with the colours used everywhere in this view. */
const OPS = {
  COPY: { label: "copied", stroke: "#8f8899", chip: "bg-line-soft text-ink-2 border-line" },
  INFLECT: { label: "re-inflected", stroke: "#5b4cb0", chip: "bg-accent-soft text-accent border-accent/25" },
  SUBST: { label: "replaced", stroke: "#c05f21", chip: "bg-pop-soft text-pop border-pop/25" },
  SPLIT: { label: "split", stroke: "#c05f21", chip: "bg-pop-soft text-pop border-pop/25" },
  MERGE: { label: "merged", stroke: "#c05f21", chip: "bg-pop-soft text-pop border-pop/25" },
  FRAME: { label: "citing formula", stroke: "#837c8e", chip: "bg-surface text-muted border-dashed border-muted/40" },
  INS: { label: "the later author's own", stroke: "#d8d3c9", chip: "bg-surface text-muted border-line" },
};
const ORDER = ["COPY", "INFLECT", "SUBST", "SPLIT", "MERGE", "FRAME", "INS"];

const cite = (id) => id.replace(/^p0*/, "pair ");

function Chip({ op, children }) {
  const o = OPS[op] ?? OPS.INS;
  return <span className={`rounded-md border px-1.5 py-0.5 text-[.68rem] font-extrabold ${o.chip}`}>{children ?? o.label}</span>;
}

/** One passage as wrapped word spans; every word registers its element so links can be drawn. */
function Passage({ words, ops, links, side, register, hover, onHover, dimUnlinked, notes }) {
  return (
    <p className="flex flex-wrap gap-x-1.5 gap-y-1 leading-[2.1]">
      {words.map((w, i) => {
        const op = side === "later" ? ops?.[i] ?? "INS" : links?.includes(i) ? "COPY" : "INS";
        const linked = side === "later" ? (links?.[i] ?? -1) >= 0 || ops?.[i] === "FRAME" : links?.includes(i);
        const active = hover === `${side}:${i}` || (side === "later" && hover === `earlier:${links?.[i]}`) ||
          (side === "earlier" && hover?.startsWith("later:") && links?.[Number(hover.slice(6))] === i);
        const style = side === "later" ? OPS[op] ?? OPS.INS : OPS[linked ? "COPY" : "INS"];
        return (
          <span
            key={i}
            ref={(el) => register(`${side}:${i}`, el)}
            title={side === "later" ? notes?.[i] : undefined}
            onMouseEnter={() => onHover(`${side}:${i}`)}
            onMouseLeave={() => onHover(null)}
            className={`rounded-md px-1 transition-colors ${style.chip.replace(/border-\S+/g, "")} ${
              active ? "outline outline-2 outline-accent" : ""
            } ${dimUnlinked && !linked ? "opacity-45" : ""}`}
          >
            {w}
          </span>
        );
      })}
    </p>
  );
}

/** SVG arcs from every linked later word up to the earlier word it came from. */
function Links({ box, rects, links, ops, hover }) {
  if (!box) return null;
  const dense = links.filter((s) => s >= 0).length > 14;
  const paths = [];
  links.forEach((s, t) => {
    if (s < 0) return;
    const a = rects[`earlier:${s}`];
    const b = rects[`later:${t}`];
    if (!a || !b) return;
    const x1 = a.left + a.width / 2 - box.left;
    const y1 = a.top + a.height - box.top;
    const x2 = b.left + b.width / 2 - box.left;
    const y2 = b.top - box.top;
    const mid = (y1 + y2) / 2;
    const op = ops[t] ?? "COPY";
    const on = hover === `later:${t}` || hover === `earlier:${s}`;
    paths.push(
      <path
        key={t}
        d={`M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`}
        fill="none"
        stroke={OPS[op]?.stroke ?? "#8f8899"}
        strokeWidth={on ? 2.8 : 1.1}
        strokeOpacity={hover ? (on ? 0.95 : 0.12) : dense ? 0.3 : 0.5}
      />
    );
  });
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
      {paths}
    </svg>
  );
}

function Metric({ name, value }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-3 py-2">
      <div className="font-mono text-[1.05rem] font-extrabold text-accent">{value}</div>
      <div className="text-[.68rem] font-bold uppercase tracking-[.08em] text-muted">{name}</div>
    </div>
  );
}

export default function EditScripts({ data }) {
  const [i, setI] = useState(0);
  const [source, setSource] = useState("pred");
  const [kind, setKind] = useState(null);
  const [hover, setHover] = useState(null);
  const [rects, setRects] = useState({});
  const [box, setBox] = useState(null);
  const els = useRef(new Map());
  const wrap = useRef(null);

  const pairs = useMemo(() => {
    const all = data?.pairs ?? [];
    return kind ? all.filter((p) => p.ref_type === kind) : all;
  }, [data, kind]);
  const pair = pairs[Math.min(i, pairs.length - 1)];

  const register = (key, el) => {
    if (el) els.current.set(key, el);
    else els.current.delete(key);
  };

  useLayoutEffect(() => {
    const measure = () => {
      if (!wrap.current) return;
      const b = wrap.current.getBoundingClientRect();
      const next = {};
      els.current.forEach((el, key) => {
        const r = el.getBoundingClientRect();
        next[key] = { left: r.left, top: r.top, width: r.width, height: r.height };
      });
      setBox({ left: b.left, top: b.top });
      setRects(next);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [pair, source]);

  if (!data) return <p className="text-muted">Loading edit scripts…</p>;
  if (!pair) return <p className="text-muted">No pairs.</p>;

  const hasGold = Boolean(pair.gold);
  const script = hasGold ? pair[source] : pair.pred;
  const gold = pair.gold;
  const found = hasGold ? script.links.filter((s, t) => s >= 0 && s === gold.links[t]).length : 0;
  const goldLinks = hasGold ? gold.links.filter((s) => s >= 0).length : 0;
  const invented = hasGold ? script.links.filter((s, t) => s >= 0 && gold.links[t] < 0).length : 0;
  const notes = script.ops.map((op, t) => {
    const parts = [OPS[op]?.label ?? op];
    if (script.relations?.[t]) parts.push(script.relations[t]);
    if (script.confidence?.[t] != null) parts.push(`p = ${script.confidence[t].toFixed(2)}`);
    return parts.join(" · ");
  });
  const counts = ORDER.map((op) => [op, script.ops.filter((o) => o === op).length]).filter(([, n]) => n > 0);

  return (
    <div className="flex flex-col gap-5">
      {/* model card */}
      <div className="rounded-2xl border border-line bg-surface p-5 shadow-card">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h3 className="text-[1.05rem] font-extrabold text-accent">{data.model.name}</h3>
          <span className="font-mono text-[.72rem] font-bold text-muted">
            {data.model.pairs} pairs · {data.model.folds} test folds{data.model.date ? ` · ${data.model.date}` : ""}
          </span>
        </div>
        {data.model.note && <p className="mb-4 max-w-[70ch] text-[.92rem] text-ink-2">{data.model.note}</p>}
        {data.model.metrics && (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
            {Object.entries(data.model.metrics).map(([k, v]) => (
              <Metric key={k} name={k.replace(/ \(.*\)/, "")} value={v.toFixed(3)} />
            ))}
          </div>
        )}
      </div>

      {/* controls */}
      <div className="flex flex-wrap items-center gap-3">
        {hasGold && <div className="flex overflow-hidden rounded-xl border border-line">
          {[["pred", "Model"], ["gold", "Hand labels"]].map(([v, l]) => (
            <button
              key={v}
              onClick={() => setSource(v)}
              className={`px-3 py-1.5 text-[.85rem] font-extrabold transition ${
                source === v ? "bg-accent text-white" : "bg-surface text-ink-2 hover:bg-line-soft"
              }`}
            >
              {l}
            </button>
          ))}
        </div>}
        <div className="flex overflow-hidden rounded-xl border border-line">
          {[[null, "all"], ["cit.", "citation"], ["cf.", "allusion"]].map(([v, l]) => (
            <button
              key={l}
              onClick={() => { setKind(v); setI(0); }}
              className={`px-3 py-1.5 text-[.85rem] font-extrabold transition ${
                kind === v ? "bg-accent-soft text-accent" : "bg-surface text-ink-2 hover:bg-line-soft"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setI((k) => (k - 1 + pairs.length) % pairs.length)}
            className="rounded-xl border border-line bg-surface px-3 py-1.5 text-[.85rem] font-extrabold text-ink-2 hover:border-accent hover:text-accent"
          >
            ←
          </button>
          <span className="font-mono text-[.8rem] font-bold text-muted">
            {cite(pair.id)} · fold {pair.fold} · {Math.min(i, pairs.length - 1) + 1}/{pairs.length}
          </span>
          <button
            onClick={() => setI((k) => (k + 1) % pairs.length)}
            className="rounded-xl border border-line bg-surface px-3 py-1.5 text-[.85rem] font-extrabold text-ink-2 hover:border-accent hover:text-accent"
          >
            →
          </button>
        </div>
      </div>

      {/* the script */}
      <div ref={wrap} className="relative overflow-hidden rounded-2xl border border-line bg-surface p-5 shadow-card">
        <div className="mb-2 flex items-baseline gap-2">
          <span className="text-[.65rem] font-extrabold uppercase tracking-[.1em] text-muted">Earlier passage</span>
          <span className="font-mono text-[.7rem] text-muted">{pair.earlier_cite} · {pair.ref_type === "cit." ? "quoted" : "alluded to"}</span>
        </div>
        <Passage words={pair.earlier} links={script.links} side="earlier" register={register} hover={hover} onHover={setHover} dimUnlinked />
        <div className="my-14 h-0" />
        <div className="mb-2 flex items-baseline gap-2">
          <span className="text-[.65rem] font-extrabold uppercase tracking-[.1em] text-muted">Later passage</span>
          <span className="font-mono text-[.7rem] text-muted">{pair.later_cite}</span>
        </div>
        <Passage words={pair.later} ops={script.ops} links={script.links} side="later" register={register} hover={hover} onHover={setHover} notes={notes} />
        <Links box={box} rects={rects} links={script.links} ops={script.ops} hover={hover} />
      </div>

      {/* legend and per-pair counts */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[.85rem] text-ink-2">
        <span className="flex flex-wrap items-center gap-2">
          {ORDER.map((op) => (
            <Chip key={op} op={op} />
          ))}
        </span>
        <span className="ml-auto font-mono text-[.8rem] text-muted">
          {hasGold
            ? source === "pred"
              ? `found ${found}/${goldLinks} hand-labelled links · invented ${invented} · `
              : `${goldLinks} hand-labelled links · `
            : ""}
          {counts.map(([op, n]) => `${n} ${OPS[op].label}`).join(" · ")}
        </span>
      </div>
      {pair.note && <p className="max-w-[80ch] text-[.88rem] text-muted">{pair.note}</p>}
    </div>
  );
}
