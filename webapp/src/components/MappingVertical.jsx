// The mapping diagram of the Gradio demo (notebooks/80_Edit_Script_Generation/demo/app.py, render_diagram):
// source words right-aligned on the left, reuse words on the right, one cubic curve per link, thicker and more
// opaque the more confident; each reuse word carries its operation and probability, unlinked source words DEL.
// The demo's geometry, scaled up for reading (an 860-wide viewBox, 36-unit rows, 18-unit words); lighter lines;
// colours from this app's operation palette.

const ROW = 36, TOP = 56, LEFT_X = 290, RIGHT_X = 520, WIDTH = 860, HEAD_LENGTH = 9, HEAD_WIDTH = 8;

const COLOUR = {
  COPY: "var(--color-op-copy)",
  INFLECT: "var(--color-op-inflect)",
  SUBST: "var(--color-op-subst)",
  SPLIT: "var(--color-op-split)",
  MERGE: "var(--color-op-split)",
  FRAME: "var(--color-muted)",
  INS: "var(--color-muted)",
  DEL: "var(--color-muted)",
};

// The curve ends horizontally, so the head is a right-pointing triangle whose tip stops just short of the reuse
// word; the line ends at the head's base and both share one opacity.
function Curve({ s, r, colour, weight, opacity, tooltip }) {
  const y1 = TOP + s * ROW - 6, y2 = TOP + r * ROW - 6;
  const tip = RIGHT_X - 8, base = tip - HEAD_LENGTH;
  const mid = (LEFT_X + base) / 2;
  return (
    <g opacity={opacity.toFixed(2)}>
      <title>{tooltip}</title>
      <path d={`M ${LEFT_X + 10} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${base} ${y2}`}
            fill="none" stroke={colour} strokeWidth={weight.toFixed(2)} />
      <path d={`M ${tip} ${y2} L ${base} ${y2 - HEAD_WIDTH / 2} L ${base} ${y2 + HEAD_WIDTH / 2} Z`} fill={colour} />
    </g>
  );
}

// links: [{r, s, op, p?}]; spans: FRAME spans on the reuse side.
export default function MappingVertical({ source, reuse, links, spans = [] }) {
  const height = Math.max(source.length, reuse.length, 1) * ROW + TOP + 24;
  const byR = new Map(links.map((e) => [e.r, e]));
  const claimed = new Set(links.map((e) => e.s));
  const frame = new Set();
  for (const span of spans) for (let r = span.start; r < span.end; r += 1) frame.add(r);

  return (
    <div className="overflow-x-auto">
      <svg width="100%" viewBox={`0 0 ${WIDTH} ${height}`} className="min-w-[640px] font-serif text-[18px]" role="img"
           aria-label="Mapping of reuse words to source words">
        <text x={LEFT_X} y={26} textAnchor="end" fill="var(--color-muted)" fontSize={13} letterSpacing=".08em" className="font-sans">SOURCE</text>
        <text x={RIGHT_X} y={26} fill="var(--color-muted)" fontSize={13} letterSpacing=".08em" className="font-sans">REUSE</text>

        {links.map((e) => {
          const confidence = e.p ?? 1;
          return (
            <Curve key={`${e.r}-${e.s}`} s={e.s} r={e.r} colour={COLOUR[e.op] ?? "var(--color-ink-2)"}
                   weight={0.8 + 1.4 * confidence} opacity={0.25 + 0.4 * confidence}
                   tooltip={`${source[e.s]} → ${reuse[e.r]}  ·  ${e.op}${e.p != null ? ` ${e.p.toFixed(2)}` : ""}`} />
          );
        })}

        {source.map((token, s) => {
          const y = TOP + s * ROW, kept = claimed.has(s);
          return (
            <g key={`s${s}`}>
              <text x={LEFT_X} y={y} textAnchor="end" fill={kept ? "var(--color-ink)" : COLOUR.DEL} fontWeight={kept ? 600 : 400}>{token}</text>
              {!kept && <text x={LEFT_X + 14} y={y} fill={COLOUR.DEL} fontSize={12} className="font-sans">DEL</text>}
            </g>
          );
        })}

        {reuse.map((token, r) => {
          const y = TOP + r * ROW, e = byR.get(r);
          const tag = e ? e.op : frame.has(r) ? "FRAME" : "INS";
          return (
            <g key={`r${r}`}>
              <text x={RIGHT_X} y={y} fill={e ? "var(--color-ink)" : COLOUR.INS} fontWeight={e ? 600 : 400}>{token}</text>
              <text x={RIGHT_X + 190} y={y} fill={COLOUR[tag] ?? "var(--color-ink-2)"} fontSize={13} fontWeight={600} className="font-sans">{tag}</text>
              {e?.p != null && <text x={RIGHT_X + 262} y={y} fill="var(--color-muted)" fontSize={13} className="font-sans">{e.p.toFixed(2)}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
