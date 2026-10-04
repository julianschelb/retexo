import { useMemo, useState } from "react";
import { forceSimulation, forceX, forceY, forceCollide, forceLink } from "d3-force";
import { polygonHull } from "d3-polygon";

const W = 1000;
const H = 880;
const CIT = "#5b4cb0";
const CF = "#c05f21";

// One hue per author (modern, evenly spaced); nodes use the hue, hulls a light tint of it.
const AUTHOR_COLORS = [
  "#6366f1", // indigo
  "#f97316", // orange
  "#10b981", // emerald
  "#ec4899", // pink
  "#0ea5e9", // sky
  "#eab308", // yellow
  "#14b8a6", // teal
  "#a855f7", // violet
  "#84cc16", // lime
  "#f43f5e", // rose
  "#06b6d4", // cyan
  "#8b5cf6", // purple
];

const radius = (refs) => 5 + Math.sqrt(refs) * 0.85;
const edgeWidth = (refs) => 1 + Math.sqrt(refs) * 0.55;
const edgeColor = (e) => (e.cit >= e.cf ? CIT : CF);

/** Deterministic force layout: query authors left, source authors right. */
function useLayout(graph) {
  return useMemo(() => {
    if (!graph) return null;
    const bySide = { query: [], source: [] };
    graph.authors.forEach((a) => bySide[a.side].push(a));
    const nodeRefs = Object.fromEntries(graph.nodes.map((n) => [n.id, n.refs]));
    const anchors = {};
    for (const side of ["query", "source"]) {
      const list = bySide[side];
      // Slot height from the cluster footprint: circles of radius r+7 packed together, plus label room.
      const weights = list.map((a) => {
        const area = a.works.reduce((s, id) => s + Math.PI * (radius(nodeRefs[id]) + 7) ** 2, 0);
        return 2.4 * Math.sqrt(area / Math.PI) + 80;
      });
      const total = weights.reduce((s, w) => s + w, 0);
      let y = 30;
      list.forEach((a, i) => {
        const h = ((H - 60) * weights[i]) / total;
        anchors[a.id] = { x: side === "query" ? W * 0.22 : W * 0.78, y: y + h / 2 };
        y += h;
      });
    }
    // Deterministic initial positions (golden-angle spiral around the author anchor).
    const nodes = graph.nodes.map((n, i) => ({
      ...n,
      r: radius(n.refs),
      x: anchors[n.author].x + Math.cos(i * 2.39996) * (8 + i * 1.5),
      y: anchors[n.author].y + Math.sin(i * 2.39996) * (8 + i * 1.5),
    }));
    const index = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const links = graph.edges.map((e) => ({ ...e, source: index[e.source], target: index[e.target] }));

    const sim = forceSimulation(nodes)
      .force("x", forceX((n) => anchors[n.author].x).strength(0.35))
      .force("y", forceY((n) => anchors[n.author].y).strength(0.45))
      .force("collide", forceCollide((n) => n.r + 7).iterations(3))
      .force("link", forceLink(links).strength(0.01).distance(380))
      .stop();
    for (let i = 0; i < 300; i += 1) sim.tick();

    const hulls = graph.authors.map((a, i) => {
      const pts = [];
      a.works.forEach((id) => {
        const n = index[id];
        const R = n.r + 16;
        for (let k = 0; k < 12; k += 1) {
          const t = (k / 12) * Math.PI * 2;
          pts.push([n.x + Math.cos(t) * R, n.y + Math.sin(t) * R]);
        }
      });
      const hull = polygonHull(pts) || pts;
      // Label at the outer top corner of the hull, away from the edges in the middle.
      const minY = Math.min(...hull.map((p) => p[1]));
      const outerX = a.side === "query" ? Math.min(...hull.map((p) => p[0])) : Math.max(...hull.map((p) => p[0]));
      return { ...a, hull, color: AUTHOR_COLORS[i % AUTHOR_COLORS.length], labelX: outerX, labelY: minY - 8 };
    });
    const authorColor = Object.fromEntries(hulls.map((h) => [h.id, h.color]));
    return { nodes, links, hulls, index, authorColor };
  }, [graph]);
}

const pathFor = (l) => {
  const mx = (l.source.x + l.target.x) / 2;
  return `M${l.source.x},${l.source.y} C${mx},${l.source.y} ${mx},${l.target.y} ${l.target.x},${l.target.y}`;
};

function isRelated(sel, item, kind) {
  if (!sel) return true;
  if (kind === "edge") {
    if (sel.kind === "author") return item.source.author === sel.id || item.target.author === sel.id;
    if (sel.kind === "work") return item.source.id === sel.id || item.target.id === sel.id;
    if (sel.kind === "edge") return item.source.id === sel.source && item.target.id === sel.target;
  }
  if (kind === "node") {
    if (sel.kind === "author") return item.author === sel.id;
    if (sel.kind === "work") return item.id === sel.id;
    if (sel.kind === "edge") return item.id === sel.source || item.id === sel.target;
  }
  return true;
}

export default function ReferenceGraph({ graph, selection, onSelect }) {
  const layout = useLayout(graph);
  const [hover, setHover] = useState(null); // { x, y, title, lines }

  if (!layout) {
    return <div className="flex h-[880px] items-center justify-center text-muted">Loading graph…</div>;
  }
  const { nodes, links, hulls, authorColor } = layout;

  // Nodes connected to the current selection (for highlighting neighbours of an author/work).
  const connected = new Set();
  if (selection) {
    links.forEach((l) => {
      if (isRelated(selection, l, "edge")) {
        connected.add(l.source.id);
        connected.add(l.target.id);
      }
    });
  }
  const nodeActive = (n) => !selection || isRelated(selection, n, "node") || connected.has(n.id);
  const showLabel = (n) => n.refs >= 40 || (selection && nodeActive(n)) || hover?.id === n.id;

  const tip = (evt, id, title, lines) => {
    const box = evt.currentTarget.ownerSVGElement.getBoundingClientRect();
    setHover({ id, x: evt.clientX - box.left, y: evt.clientY - box.top, title, lines });
  };

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full select-none" role="img" aria-label="Graph of intertextual references between works" onClick={() => onSelect(null)}>
        {/* Author hulls */}
        {hulls.map((a) => {
          const active = !selection || selection.kind !== "author" || selection.id === a.id;
          const d = `M${a.hull.map((p) => p.join(",")).join("L")}Z`;
          return (
            <g key={a.id} className="cursor-pointer" opacity={active ? 1 : 0.45}
              onClick={(e) => { e.stopPropagation(); onSelect({ kind: "author", id: a.id, label: a.name, side: a.side }); }}
              onMouseMove={(e) => tip(e, a.id, a.name, [`${a.works.length} work${a.works.length > 1 ? "s" : ""} · ${a.refs} references`, `${a.cit} verbatim (cit.) · ${a.cf} allusions (cf.)`])}
              onMouseLeave={() => setHover(null)}>
              <path d={d} fill={a.color} stroke={a.color} strokeWidth="28" strokeLinejoin="round" opacity={selection?.kind === "author" && selection.id === a.id ? 0.22 : 0.11} />
              <text x={a.labelX} y={a.labelY - 8} textAnchor={a.side === "query" ? "start" : "end"} fill={a.color} style={{ fontSize: 15, fontWeight: 800 }}>{a.name}</text>
            </g>
          );
        })}

        {/* Edges */}
        {links.map((l) => {
          const active = isRelated(selection, l, "edge");
          const hovered = hover?.id === `${l.source.id}|${l.target.id}`;
          return (
            <path key={`${l.source.id}|${l.target.id}`} d={pathFor(l)} fill="none" stroke={edgeColor(l)}
              strokeWidth={edgeWidth(l.refs) + (hovered ? 2 : 0)} strokeLinecap="round" className="cursor-pointer transition-opacity"
              opacity={!selection ? 0.35 : active ? 0.85 : 0.06}
              onClick={(e) => { e.stopPropagation(); onSelect({ kind: "edge", source: l.source.id, target: l.target.id, label: `${l.source.work} → ${l.target.work}` }); }}
              onMouseMove={(e) => tip(e, `${l.source.id}|${l.target.id}`, `${l.source.work} → ${l.target.work}`, [`${l.refs} references`, `${l.cit} verbatim (cit.) · ${l.cf} allusions (cf.)`])}
              onMouseLeave={() => setHover(null)} />
          );
        })}

        {/* Document nodes */}
        {nodes.map((n) => {
          const active = nodeActive(n);
          const selected = selection?.kind === "work" && selection.id === n.id;
          return (
            <g key={n.id} className="cursor-pointer" opacity={active ? 1 : 0.3}
              onClick={(e) => { e.stopPropagation(); onSelect({ kind: "work", id: n.id, label: n.work, author: n.author, side: n.side }); }}
              onMouseMove={(e) => tip(e, n.id, n.work, [`${n.side === "query" ? "citing" : "source"} work · ${n.segments.toLocaleString()} segments`, `${n.refs} references: ${n.cit} cit. · ${n.cf} cf.`])}
              onMouseLeave={() => setHover(null)}>
              <circle cx={n.x} cy={n.y} r={n.r} fill={authorColor[n.author]} stroke={selected ? "#292433" : "#fff"} strokeWidth={selected ? 3 : 1.5} />
              {showLabel(n) && (
                <text x={n.side === "query" ? n.x - n.r - 5 : n.x + n.r + 5} y={n.y + 4} textAnchor={n.side === "query" ? "end" : "start"} className="fill-ink-2" style={{ fontSize: 11.5, fontFamily: "JetBrains Mono, monospace", fontWeight: 500, paintOrder: "stroke", stroke: "#fcfbf7", strokeWidth: 3 }}>{n.work}</text>
              )}
            </g>
          );
        })}
      </svg>

      {hover && (
        <div className="pointer-events-none absolute z-10 max-w-xs rounded-xl border border-line bg-surface px-3 py-2 text-[.8rem] shadow-card-lg"
          style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <div className="font-extrabold text-ink">{hover.title}</div>
          {hover.lines.map((l) => <div key={l} className="text-ink-2">{l}</div>)}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[.8rem] font-semibold text-muted">
        <span><span className="mr-1.5 inline-block h-2 w-6 rounded-full align-middle" style={{ background: CIT }} />mostly verbatim (cit.)</span>
        <span><span className="mr-1.5 inline-block h-2 w-6 rounded-full align-middle" style={{ background: CF }} />mostly allusions (cf.)</span>
        <span>line width = number of references · node size = references per work</span>
        <span className="ml-auto">click an author, work or edge to filter · click the background to reset</span>
      </div>
    </div>
  );
}
