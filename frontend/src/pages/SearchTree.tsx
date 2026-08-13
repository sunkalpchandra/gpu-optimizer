import { useMemo, useRef, useState, WheelEvent, MouseEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { api, fmtMs, TreeNode } from "../api";
import { useFetch } from "../components/hooks";
import {
  Empty,
  ErrorBox,
  Loading,
  ProvenanceBadge,
  SimBadge,
  StatusBadge,
} from "../components/ui";

const PROV_FILL: Record<string, string> = {
  rl: "#a78bfa",
  evolutionary: "#34d399",
  bo: "#22d3ee",
  random: "#94a3b8",
  grid: "#94a3b8",
  "baseline-naive": "#fbbf24",
  baseline: "#fbbf24",
  transfer: "#60a5fa",
  manual: "#94a3b8",
};

const STATUS_STROKE: Record<string, string> = {
  ok: "#10b981",
  compile_error: "#ef4444",
  runtime_error: "#ef4444",
  incorrect: "#f97316",
};

interface Placed extends TreeNode {
  x: number;
  y: number;
  r: number;
}

/** Layered layout: BFS depth from roots = row; siblings spread within row. */
function layout(nodes: TreeNode[]): { placed: Placed[]; width: number; height: number } {
  const byId = new Map(nodes.map((n) => [n.candidate_id, n]));
  const depth = new Map<string, number>();
  const depthOf = (n: TreeNode, seen: Set<string>): number => {
    const cached = depth.get(n.candidate_id);
    if (cached != null) return cached;
    if (!n.parent_id || !byId.has(n.parent_id) || seen.has(n.candidate_id)) {
      depth.set(n.candidate_id, 0);
      return 0;
    }
    seen.add(n.candidate_id);
    const d = depthOf(byId.get(n.parent_id)!, seen) + 1;
    depth.set(n.candidate_id, d);
    return d;
  };
  nodes.forEach((n) => depthOf(n, new Set()));

  const rows = new Map<number, TreeNode[]>();
  nodes.forEach((n) => {
    const d = depth.get(n.candidate_id) ?? 0;
    if (!rows.has(d)) rows.set(d, []);
    rows.get(d)!.push(n);
  });

  const bestOk = Math.min(
    ...nodes.filter((n) => n.latency_ms != null).map((n) => n.latency_ms as number),
  );
  const rowGap = 90;
  const maxRow = Math.max(1, ...Array.from(rows.values(), (r) => r.length));
  const width = Math.max(900, maxRow * 46);
  const placed: Placed[] = [];
  [...rows.entries()]
    .sort((a, b) => a[0] - b[0])
    .forEach(([d, row]) => {
      const gap = width / (row.length + 1);
      row.forEach((n, i) => {
        const improvement =
          n.latency_ms != null && isFinite(bestOk) ? bestOk / (n.latency_ms as number) : 0;
        placed.push({
          ...n,
          x: gap * (i + 1),
          y: 50 + d * rowGap,
          r: 7 + 7 * Math.min(Math.max(improvement, 0), 1),
        });
      });
    });
  const height = 100 + (Math.max(0, ...Array.from(rows.keys())) + 1) * rowGap;
  return { placed, width, height };
}

export default function SearchTree() {
  const { runId = "" } = useParams();
  const { data, error, loading } = useFetch(() => api.tree(runId), [runId]);
  const [selected, setSelected] = useState<Placed | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  const graph = useMemo(() => (data ? layout(data.nodes) : null), [data]);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!graph || !graph.placed.length)
    return <Empty>No candidates recorded for this run yet.</Empty>;

  const byId = new Map(graph.placed.map((n) => [n.candidate_id, n]));

  function onWheel(e: WheelEvent<SVGSVGElement>) {
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setView((v) => ({ ...v, scale: Math.min(4, Math.max(0.3, v.scale * factor)) }));
  }
  function onMouseDown(e: MouseEvent<SVGSVGElement>) {
    drag.current = { x: e.clientX, y: e.clientY };
  }
  function onMouseMove(e: MouseEvent<SVGSVGElement>) {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }));
  }

  return (
    <div>
      <div className="mb-4 flex items-center gap-3">
        <h1 className="text-lg font-bold text-slate-100">Search tree</h1>
        <Link to={`/runs/${runId}`} className="font-mono text-xs text-cyan-300 hover:underline">
          {runId}
        </Link>
        <span className="ml-auto text-xs text-slate-500">
          scroll to zoom · drag to pan · node size = closeness to best · click to inspect
        </span>
      </div>

      <div className="flex gap-4">
        <div className="card flex-1 overflow-hidden p-0">
          <svg
            className="h-[34rem] w-full cursor-grab active:cursor-grabbing"
            onWheel={onWheel}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={() => (drag.current = null)}
            onMouseLeave={() => (drag.current = null)}
          >
            <g transform={`translate(${view.x},${view.y}) scale(${view.scale})`}>
              {graph.placed.map((n) =>
                n.parent_id && byId.has(n.parent_id) ? (
                  <line
                    key={`e-${n.candidate_id}`}
                    x1={byId.get(n.parent_id)!.x}
                    y1={byId.get(n.parent_id)!.y}
                    x2={n.x}
                    y2={n.y}
                    stroke="#1e293b"
                    strokeWidth={1.5}
                  />
                ) : null,
              )}
              {graph.placed.map((n) => (
                <g
                  key={n.candidate_id}
                  transform={`translate(${n.x},${n.y})`}
                  className="cursor-pointer"
                  onClick={() => setSelected(n)}
                >
                  <circle
                    r={n.r}
                    fill={PROV_FILL[n.provenance] ?? "#94a3b8"}
                    fillOpacity={n.status === "ok" ? 0.9 : 0.25}
                    stroke={STATUS_STROKE[n.status] ?? "#64748b"}
                    strokeWidth={selected?.candidate_id === n.candidate_id ? 3 : 1.5}
                  />
                  {n.latency_ms != null && n.r > 12 && (
                    <text y={-n.r - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">
                      {fmtMs(n.latency_ms)}
                    </text>
                  )}
                </g>
              ))}
            </g>
          </svg>
        </div>

        <div className="w-80 shrink-0">
          {selected ? (
            <div className="card sticky top-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="font-mono text-xs text-slate-400">
                  {selected.candidate_id.slice(0, 12)}
                </span>
                <ProvenanceBadge value={selected.provenance} />
                <StatusBadge value={selected.status} />
              </div>
              <div className="mb-3 text-lg font-bold text-slate-100">
                {fmtMs(selected.latency_ms)}
                <SimBadge engine={selected.engine} />
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 border-t border-slate-800 pt-2">
                {Object.entries(selected.config).map(([k, v]) => (
                  <div key={k} className="contents">
                    <span className="text-xs text-slate-500">{k}</span>
                    <span className="text-right font-mono text-xs text-slate-200">{String(v)}</span>
                  </div>
                ))}
              </div>
              <Link className="btn mt-4 block text-center" to={`/kernels/${selected.candidate_id}`}>
                View kernel source
              </Link>
            </div>
          ) : (
            <Empty>Click a node to inspect its configuration.</Empty>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-500">
        legend:
        {Object.entries(PROV_FILL)
          .filter(([k]) => ["rl", "evolutionary", "bo", "random", "baseline-naive"].includes(k))
          .map(([k, c]) => (
            <span key={k} className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />
              {k}
            </span>
          ))}
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-red-500" /> failed
        </span>
      </div>
    </div>
  );
}
