import { MouseEvent, useMemo, useRef, useState, WheelEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { api, fmtMs, TreeNode } from "../api";
import { useFetch, useTitle } from "../components/hooks";
import {
  Empty,
  ErrorBox,
  Loading,
  PageHead,
  Panel,
  PROVENANCE_COLOR,
  ProvTag,
  SimTag,
  StatusDot,
} from "../components/ui";

const STATUS_STROKE: Record<string, string> = {
  ok: "var(--ok)",
  compile_error: "var(--err)",
  runtime_error: "var(--err)",
  incorrect: "var(--warn)",
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
  const rowGap = 84;
  const maxRow = Math.max(1, ...Array.from(rows.values(), (r) => r.length));
  const width = Math.max(900, maxRow * 42);
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
          y: 46 + d * rowGap,
          r: 6 + 7 * Math.min(Math.max(improvement, 0), 1),
        });
      });
    });
  const height = 92 + (Math.max(0, ...Array.from(rows.keys())) + 1) * rowGap;
  return { placed, width, height };
}

const LEGEND: Array<[string, string]> = [
  ["rl", PROVENANCE_COLOR.rl],
  ["evolutionary", PROVENANCE_COLOR.evolutionary],
  ["bo", PROVENANCE_COLOR.bo],
  ["random", PROVENANCE_COLOR.random],
  ["baseline", PROVENANCE_COLOR.baseline],
];

export default function SearchTree() {
  const { runId = "" } = useParams();
  useTitle(`${runId} tree`);
  const { data, error, loading } = useFetch(() => api.tree(runId), [runId]);
  const [selected, setSelected] = useState<Placed | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  const graph = useMemo(() => (data ? layout(data.nodes) : null), [data]);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!graph || !graph.placed.length) return <Empty>no candidates recorded for this run</Empty>;

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
    <div className="space-y-3">
      <PageHead
        crumbs={[
          <Link key="r" className="link" to="/">runs</Link>,
          <Link key="id" className="link mono" to={`/runs/${runId}`}>{runId}</Link>,
          "search tree",
        ]}
      />

      <div className="flex gap-3">
        <Panel
          title={`Lineage (${graph.placed.length} candidates)`}
          meta={
            <span className="flex items-center gap-3 text-[10.5px]" style={{ color: "var(--muted)" }}>
              {LEGEND.map(([k, c]) => (
                <span key={k} className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: c }} />
                  {k}
                </span>
              ))}
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full border" style={{ borderColor: "var(--err)" }} />
                failed
              </span>
            </span>
          }
        >
          <div className="relative">
            <svg
              className="h-[32rem] w-full cursor-grab active:cursor-grabbing"
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
                      stroke="var(--border-strong)"
                      strokeWidth={1.2}
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
                      fill={PROVENANCE_COLOR[n.provenance] ?? "var(--muted)"}
                      fillOpacity={n.status === "ok" ? 0.85 : 0.2}
                      stroke={STATUS_STROKE[n.status] ?? "var(--faint)"}
                      strokeWidth={selected?.candidate_id === n.candidate_id ? 2.5 : 1.2}
                    />
                    {n.latency_ms != null && n.r > 11 && (
                      <text y={-n.r - 4} textAnchor="middle" fontSize={9}
                            fill="var(--muted)" fontFamily="ui-monospace, Menlo, monospace">
                        {fmtMs(n.latency_ms)}
                      </text>
                    )}
                  </g>
                ))}
              </g>
            </svg>
            <span
              className="mono absolute bottom-2 right-3 text-[10.5px]"
              style={{ color: "var(--faint)" }}
            >
              wheel = zoom · drag = pan · size ∝ closeness to best
            </span>
          </div>
        </Panel>

        <div className="w-72 shrink-0">
          {selected ? (
            <Panel
              title="Candidate"
              meta={<span className="mono text-[10.5px]">{selected.candidate_id.slice(0, 12)}</span>}
            >
              <div className="p-3">
                <div className="mb-1 flex items-center gap-3">
                  <ProvTag value={selected.provenance} />
                  <StatusDot value={selected.status} />
                </div>
                <div className="mono mb-3 text-lg font-semibold">
                  {fmtMs(selected.latency_ms)}
                  <SimTag engine={selected.engine} />
                </div>
                <table className="w-full">
                  <tbody>
                    {Object.entries(selected.config).map(([k, v]) => (
                      <tr key={k}>
                        <td className="py-0.5 text-[11.5px]" style={{ color: "var(--muted)" }}>{k}</td>
                        <td className="mono py-0.5 text-right text-[11.5px]">{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <Link className="btn mt-3 block text-center" to={`/kernels/${selected.candidate_id}`}>
                  View kernel source
                </Link>
              </div>
            </Panel>
          ) : (
            <Panel title="Candidate">
              <Empty>select a node to inspect its configuration</Empty>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
