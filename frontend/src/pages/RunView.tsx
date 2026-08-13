import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, fmtCount, fmtMs, fmtPct, fmtSpeedup, Iteration, Run } from "../api";
import { usePoll } from "../components/hooks";
import {
  Empty,
  ErrorBox,
  Loading,
  ProvenanceBadge,
  SectionTitle,
  SimBadge,
  Stat,
  StatusBadge,
} from "../components/ui";

export default function RunView() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [iters, setIters] = useState<Iteration[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [logScale, setLogScale] = useState(false);
  const lastIter = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const r = await api.run(runId);
      setRun(r);
      const fresh = await api.iterations(runId, lastIter.current);
      if (fresh.length) {
        lastIter.current = Math.max(...fresh.map((i) => i.iteration));
        setIters((prev) => [...prev, ...fresh]);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }, [runId]);

  useEffect(() => {
    lastIter.current = 0;
    setIters([]);
    setRun(null);
    void refresh();
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

  if (error) return <ErrorBox error={error} />;
  if (!run) return <Loading />;

  const speedup =
    run.best_latency_ms && run.baseline_torch_ms
      ? run.baseline_torch_ms / run.best_latency_ms
      : null;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg font-bold text-slate-100">{run.run_id}</h1>
        <StatusBadge value={run.status} />
        <span className="text-sm text-slate-400">
          {run.task} {run.shape.join("×")} · {run.algorithm} · {run.gpu_name}
        </span>
        <SimBadge engine={run.engine} />
        <Link to={`/runs/${runId}/tree`} className="btn ml-auto">
          Search tree →
        </Link>
      </div>
      {run.error && <ErrorBox error={run.error} />}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat
          label="Best latency"
          value={
            <>
              {fmtMs(run.best_latency_ms)}
              <SimBadge engine={run.engine} />
            </>
          }
          accent
        />
        <Stat label="Speedup vs torch" value={fmtSpeedup(speedup)} />
        <Stat
          label="Baselines"
          value={<span className="text-base">{fmtMs(run.baseline_torch_ms)}</span>}
          sub={`torch · naive ${fmtMs(run.baseline_naive_ms)}`}
        />
        <Stat label="Evaluated" value={fmtCount(run.candidates_evaluated)} />
        <Stat label="Compile success" value={fmtPct(run.compile_success_rate)} />
      </div>

      <SectionTitle>
        Convergence{" "}
        <button
          onClick={() => setLogScale((s) => !s)}
          className="ml-2 rounded border border-slate-700 px-1.5 py-0.5 text-[11px] font-normal normal-case text-slate-400 hover:text-slate-200"
        >
          {logScale ? "linear" : "log"} y
        </button>
      </SectionTitle>
      <div className="card h-72">
        {chartData.length ? (
          <ResponsiveContainer>
            <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="iteration" stroke="#64748b" fontSize={11} />
              <YAxis
                stroke="#64748b"
                fontSize={11}
                scale={logScale ? "log" : "auto"}
                domain={logScale ? ["auto", "auto"] : [0, "auto"]}
                tickFormatter={(v: number) => fmtMs(v)}
                width={80}
              />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
                formatter={(v) => fmtMs(v as number)}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Scatter dataKey="actual" name="candidate latency" fill="#38bdf8" opacity={0.55} />
              <Line
                dataKey="best"
                name="best so far"
                stroke="#34d399"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <Empty>Waiting for first iterations…</Empty>
        )}
      </div>

      <SectionTitle>Iterations</SectionTitle>
      {iters.length ? (
        <div className="card max-h-[32rem] overflow-auto p-0">
          <table className="w-full">
            <thead className="sticky top-0 border-b border-slate-800 bg-slate-950">
              <tr>
                <th className="th">#</th>
                <th className="th">candidate</th>
                <th className="th">source</th>
                <th className="th">status</th>
                <th className="th">predicted</th>
                <th className="th">actual</th>
                <th className="th">reward</th>
                <th className="th">best so far</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {[...iters].reverse().map((it) => (
                <tr key={it.id} className="hover:bg-slate-900/40">
                  <td className="td text-slate-500">{it.iteration}</td>
                  <td className="td">
                    <Link
                      to={`/kernels/${it.candidate_id}`}
                      className="font-mono text-xs text-cyan-300 hover:underline"
                    >
                      {it.candidate_id.slice(0, 10)}
                    </Link>
                  </td>
                  <td className="td">
                    <ProvenanceBadge value={it.note || "manual"} />
                  </td>
                  <td className="td">
                    <StatusBadge value={it.status} />
                  </td>
                  <td className="td text-slate-400">
                    {it.predicted_ms != null
                      ? `${fmtMs(it.predicted_ms)} ± ${fmtMs(it.predicted_std ?? 0)}`
                      : "—"}
                  </td>
                  <td className="td">
                    {fmtMs(it.actual_ms)}
                    {it.actual_ms != null && <SimBadge engine={run.engine} />}
                  </td>
                  <td className="td">
                    <span className={(it.reward ?? 0) >= 0 ? "text-emerald-300" : "text-red-300"}>
                      {it.reward?.toFixed(3) ?? "—"}
                    </span>
                  </td>
                  <td className="td text-slate-400">{fmtMs(it.best_so_far_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No iterations recorded yet.</Empty>
      )}
    </div>
  );
}
