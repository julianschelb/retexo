import { useLayoutEffect, useRef, useState } from "react";
import { OP_STYLE } from "../labels.js";

// The mapping, laid out across the width: the source passage on top, the reuse passage below, each on one line
// (the pane scrolls sideways for long passages), and one curve per link from the source word down to the reuse word,
// coloured by operation and thicker and more opaque the more confident (as in the Gradio demo's diagram).

const HEAD_LENGTH = 7, HEAD_WIDTH = 7;

const STROKE = {
  COPY: "var(--color-op-copy)",
  INFLECT: "var(--color-op-inflect)",
  SUBST: "var(--color-op-subst)",
  SPLIT: "var(--color-op-split)",
  MERGE: "var(--color-op-split)",
};

function Words({ tokens, side, tags, linked, register, hover, onHover }) {
  return (
    <p className="flex flex-nowrap items-start gap-x-1.5 whitespace-nowrap font-serif text-[1.05rem] leading-snug">
      {tokens.map((token, i) => {
        const key = `${side}:${i}`;
        const tag = tags[i];
        const style = OP_STYLE[tag] ?? OP_STYLE.INS;
        const on = hover?.has(key);
        const plain = !linked.has(i) && tag !== "FRAME";
        const word = (
          <span ref={(el) => register(key, el)} onMouseEnter={() => onHover(key)} onMouseLeave={() => onHover(null)}
                className={`rounded-sm px-1 transition-colors ${plain ? "text-muted" : `${style.bg} text-ink`} ${
                  on ? `outline outline-2 ${plain ? "outline-muted" : style.outline}` : ""}`}
                title={side === "reuse" ? `${token}: ${tag}` : plain ? `${token}: DEL` : token}>
            {token}
          </span>
        );
        if (side !== "reuse") return <span key={i}>{word}</span>;
        // the reuse word's label sits below it, clear of the arrows arriving from above
        return (
          <span key={i} className="inline-flex flex-col items-center">
            {word}
            <span className={`mt-0.5 font-sans text-[.6rem] font-semibold uppercase tracking-wide ${
              tag === "INS" || tag === "FRAME" ? "text-muted/70" : style.text}`}>{tag}</span>
          </span>
        );
      })}
    </p>
  );
}

export default function MappingDiagram({ source, reuse, links, spans = [] }) {
  const [rects, setRects] = useState({});
  const [hoverKey, setHoverKey] = useState(null);
  const els = useRef(new Map());
  const wrap = useRef(null);       // the full-width content, which the curves are measured against
  const scroller = useRef(null);   // the visible window, scrolled sideways

  const register = (key, el) => (el ? els.current.set(key, el) : els.current.delete(key));

  // measure every word once laid out, again when the width changes (resize, fonts arriving)
  useLayoutEffect(() => {
    const measure = () => {
      if (!wrap.current) return;
      const box = wrap.current.getBoundingClientRect();
      const next = { box: { width: box.width, height: box.height } };
      els.current.forEach((el, key) => {
        const r = el.getBoundingClientRect();
        next[key] = { x: r.left - box.left + r.width / 2, top: r.top - box.top, bottom: r.bottom - box.top };
      });
      setRects(next);
    };
    measure();
    // open each pair scrolled to its first link, with a little context before it
    const first = Math.min(...links.map((e) => Math.min(
      els.current.get(`source:${e.s}`)?.offsetLeft ?? Infinity, els.current.get(`reuse:${e.r}`)?.offsetLeft ?? Infinity)));
    if (scroller.current) scroller.current.scrollLeft = Number.isFinite(first) ? Math.max(0, first - 80) : 0;
    const observer = new ResizeObserver(measure);
    observer.observe(wrap.current);
    document.fonts?.ready.then(measure);
    return () => observer.disconnect();
  }, [source, reuse, links]);

  const frame = new Set();
  for (const span of spans) for (let r = span.start; r < span.end; r += 1) frame.add(r);
  const reuseTags = reuse.map((_, r) => links.find((e) => e.r === r)?.op ?? (frame.has(r) ? "FRAME" : "INS"));
  const sourceTags = source.map((_, s) => links.find((e) => e.s === s)?.op ?? "INS");
  const linkedSource = new Set(links.map((e) => e.s));
  const linkedReuse = new Set(links.map((e) => e.r));

  // hovering a word lights up its links and the words at their other ends
  const hot = links.filter((e) => hoverKey === `source:${e.s}` || hoverKey === `reuse:${e.r}`);
  const hover = hoverKey ? new Set([hoverKey, ...hot.flatMap((e) => [`source:${e.s}`, `reuse:${e.r}`])]) : null;

  return (
    <div>
      <div ref={scroller} className="overflow-x-auto pb-2">
      <div ref={wrap} className="relative w-max min-w-full pr-4">
        <p className="mb-1 text-[.65rem] font-semibold uppercase tracking-[.08em] text-muted">Source</p>
        <Words tokens={source} side="source" tags={sourceTags} linked={linkedSource} register={register}
               hover={hover} onHover={setHoverKey} />
        <div className="h-24" aria-hidden="true" />
        <Words tokens={reuse} side="reuse" tags={reuseTags} linked={linkedReuse} register={register}
               hover={hover} onHover={setHoverKey} />
        <p className="mt-1 text-[.65rem] font-semibold uppercase tracking-[.08em] text-muted">Reuse</p>

        {rects.box && (
          <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" aria-hidden="true">
            {links.map((e) => {
              const a = rects[`source:${e.s}`], b = rects[`reuse:${e.r}`];
              if (!a || !b) return null;
              const confidence = e.p ?? 1;
              const on = hot.includes(e);
              // the curve ends vertically, so the head is a downward triangle whose tip touches the reuse word;
              // the line stops at the head's base and both share one opacity
              const colour = STROKE[e.op] ?? "var(--color-ink-2)";
              const tip = b.top - 1, base = tip - HEAD_LENGTH, mid = (a.bottom + base) / 2;
              return (
                <g key={`${e.r}-${e.s}`} opacity={hoverKey ? (on ? 0.9 : 0.08) : 0.25 + 0.4 * confidence}>
                  <path d={`M ${a.x} ${a.bottom + 1} C ${a.x} ${mid}, ${b.x} ${mid}, ${b.x} ${base}`}
                        fill="none" stroke={colour} strokeWidth={(on ? 1.5 : 1) * (0.8 + 1.4 * confidence)} />
                  <path d={`M ${b.x} ${tip} L ${b.x - HEAD_WIDTH / 2} ${base} L ${b.x + HEAD_WIDTH / 2} ${base} Z`} fill={colour} />
                </g>
              );
            })}
          </svg>
        )}
      </div>
      </div>

      <p className="mt-2 min-h-5 text-xs text-ink-2">
        {hot.length
          ? hot.map((e) => `${source[e.s]} → ${reuse[e.r]} · ${e.op}${e.p != null ? ` ${e.p.toFixed(2)}` : ""}`).join("   ")
          : <span className="text-muted">Hover a word to see its link.</span>}
      </p>
    </div>
  );
}
