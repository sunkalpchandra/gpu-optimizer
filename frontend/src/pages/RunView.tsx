import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  fmtCount,
  fmtDuration,
  fmtMs,
  fmtPct,
  fmtSpeedup,
  Iteration,
  Run,
} from "../api";
import { usePoll, useTitle } from "../components/hooks";
import {
  Empty,
  ErrorBox,
  Kpi,
  Loading,
  PageHead,
  Panel,
  ProvTag,
  SimTag,
  StatusDot,
} from "../components/ui";

export default function RunView() {
  const { runId = "" } = useParams();
  useTitle(runId);
  const [run, setRun] = useState<Run | null>(null);
  const [iters, setIters] = useState<Iteration[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [logScale, setLogScale] = useState(false);
  const lastIter = useRef(0);
  const busy = useRef(false);       // one refresh in flight at a time
  const generation = useRef(0);     // invalidates stale responses on run change

  const refresh = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    const gen = generation.current;
    try {
      const r = await api.run(runId);
      const fresh = await api.iterations(runId, lastIter.current);
      if (gen !== generation.current) return; // run changed / unmounted
      setRun(r);
      if (fresh.length) {
        lastIter.current = Math.max(...fresh.map((i) => i.iteration));
        setIters((prev) => [...prev, ...fresh]);
      }
      setError(null); // a successful poll clears transient failures
    } catch (e) {
      if (gen === generation.current) setError((e as Error).message);
    } finally {
      busy.current = false;
    }
  }, [runId]);

  useEffect(() => {
    generation.current += 1;
    lastIter.current = 0;
    busy.current = false;
    setIters([]);
    setRun(null);
    setError(null);
    void refresh();
    return () => {
      generation.current += 1;
    };
  }, [refresh]);
  usePoll(refresh, 1500, run?.status === "running");

  const chartData = useMemo(
    () =>
      iters.map((it) => ({
        iteration: it.iteration,
        best: it.best_so_far_ms,
        actual: it.status === "ok" ? it.actual_ms : null,
      })),
    [iters],
  );

  if (error && !run) return <ErrorBox error={error} />;
  if (!run) return <Loading />;

  const speedup =
    run.best_latency_ms && run.baseline_torch_ms
      ? run.baseline_torch_ms / run.best_latency_ms
      : null;

  return (
    <div className="space-y-3">
      <PageHead
        crumbs={[<Link key="r" className="link" to="/">runs</Link>, <span key="id" className="mono">{run.run_id}</span>]}
        right={
          <>
            <StatusDot value={run.status} />
            <Link to={`/runs/${runId}/tree`} className="btn">
              Search tree
            </Link>
          </>
        }
      />
      {error && <ErrorBox error={error} />}

      <div
        className="mono flex flex-wrap items-center gap-x-5 gap-y-1 rounded-[4px] border px-3 py-1.5 text-[11.5px]"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}
      >
        <span>{run.task}</span>
        <span>{run.shape.join("×")}</span>
        <span>{run.algorithm}</span>
        <span>{run.gpu_name}</span>
        <span>
          engine {run.engine}
          <SimTag engine={run.engine} />
        </span>
        <span>elapsed {fmtDuration(run.started_at, run.finished_at ?? Date.now() / 1000)}</span>
      </div>
      {run.error && <ErrorBox error={run.error} />}

      <div className="kpis">
        <Kpi
          label="Best latency"
          value={
            <>
              {fmtMs(run.best_latency_ms)}
              <SimTag engine={run.engine} />
            </>
          }
        />
        <Kpi label="Speedup" value={fmtSpeedup(speedup)} sub="vs torch baseline" />
        <Kpi
          label="Baselines"
          value={fmtMs(run.baseline_torch_ms)}
          sub={`torch · naive ${fmtMs(run.baseline_naive_ms)}`}
        />
        <Kpi label="Evaluated" value={fmtCount(run.candidates_evaluated)} />
        <Kpi label="Compile success" value={fmtPct(run.compile_success_rate)} />
      </div>

      <Panel
        title="Convergence"
        meta={
          <button
            type="button"
            aria-pressed={logScale}
            onClick={() => setLogScale((s) => !s)}
            className="mono text-[11px]"
            style={{ color: "var(--accent)", cursor: "pointer" }}
          >
            y: {logScale ? "log" : "linear"}
          </button>
        }
      >
        <div className="h-64 p-2">
          {chartData.length ? (
            <ResponsiveContainer>
              <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                <XAxis
                  dataKey="iteration"
                  type="number"
                  domain={[1, "dataMax"]}
                  allowDecimals={false}
                  stroke="var(--faint)"
                  fontSize={10.5}
                  tickLine={false}
                />
                <YAxis
                  stroke="var(--faint)"
                  fontSize={10.5}
                  tickLine={false}
                  scale={logScale ? "log" : "auto"}
                  domain={logScale ? ["auto", "auto"] : [0, "auto"]}
                  tickFormatter={(v: number) => fmtMs(v)}
                  width={76}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--panel)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 4,
                    fontSize: 11.5,
                    fontFamily: "ui-monospace, Menlo, monospace",
                  }}
                  formatter={(v) => fmtMs(v as number)}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Scatter dataKey="actual" name="candidate" fill="var(--accent)" opacity={0.45} />
                <Line
                  dataKey="best"
                  name="best so far"
                  stroke="var(--ok)"
                  strokeWidth={1.6}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <Empty>waiting for iterations</Empty>
          )}
        </div>
      </Panel>

      <Panel title={`Iterations (${iters.length})`}>
        {iters.length ? (
          <div className="max-h-[30rem] overflow-auto">
            <table className="tbl">
              <thead className="thead-sticky sticky top-0" style={{ background: "var(--panel)" }}>
                <tr>
                  <th className="num">#</th>
                  <th>candidate</th>
                  <th>provenance</th>
                  <th>status</th>
                  <th className="num">predicted</th>
                  <th className="num">actual</th>
                  <th className="num">reward</th>
                  <th className="num">best</th>
                </tr>
              </thead>
              <tbody>
                {[...iters].reverse().map((it) => (
                  <tr key={it.id}>
                    <td className="num" style={{ color: "var(--faint)" }}>{it.iteration}</td>
                    <td>
                      <Link to={`/kernels/${it.candidate_id}`} className="link mono text-[11.5px]">
                        {it.candidate_id.slice(0, 10)}
                      </Link>
                    </td>
                    <td>{it.note ? <ProvTag value={it.note} /> : "—"}</td>
                    <td><StatusDot value={it.status} /></td>
                    <td className="num" style={{ color: "var(--muted)" }}>
                      {it.predicted_ms == null
                        ? "—"
                        : it.predicted_std == null
                          ? fmtMs(it.predicted_ms)
                          : `${fmtMs(it.predicted_ms)} ±${fmtMs(it.predicted_std)}`}
                    </td>
                    <td className="num">
                      {fmtMs(it.actual_ms)}
                      {it.actual_ms != null && <SimTag engine={run.engine} />}
                    </td>
                    <td className="num" style={{ color: (it.reward ?? 0) >= 0 ? "var(--ok)" : "var(--err)" }}>
                      {it.reward != null ? it.reward.toFixed(3) : "—"}
                    </td>
                    <td className="num" style={{ color: "var(--muted)" }}>{fmtMs(it.best_so_far_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>no iterations recorded</Empty>
        )}
      </Panel>
    </div>
  );
}
