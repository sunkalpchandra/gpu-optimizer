import { FormEvent, useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  api,
  apiMode,
  fmtCount,
  fmtDuration,
  fmtMs,
  fmtPct,
  fmtRelTime,
  fmtSpeedup,
  Run,
  Status,
  TaskInfo,
} from "../api";
import { useFetch, usePoll, useTitle } from "../components/hooks";
import {
  Empty,
  ErrorBox,
  Kpi,
  Loading,
  Panel,
  SimTag,
  StatusDot,
} from "../components/ui";

function LaunchForm({ tasks, algorithms }: { tasks: TaskInfo[]; algorithms: string[] }) {
  const nav = useNavigate();
  const [task, setTask] = useState(() => tasks[0]?.name ?? "");
  const [shape, setShape] = useState(() => (tasks[0]?.default_shapes[0] ?? []).join(", "));
  const [algorithm, setAlgorithm] = useState(() =>
    algorithms.includes("hybrid") ? "hybrid" : (algorithms[0] ?? ""),
  );
  const [budget, setBudget] = useState(120);
  const [engine, setEngine] = useState<"auto" | "cuda" | "simulated">("auto");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: mode } = useFetch(() => apiMode(), []);
  const readOnly = mode === "static";

  const taskInfo = tasks.find((t) => t.name === task);

  function pickTask(name: string) {
    setTask(name);
    const info = tasks.find((t) => t.name === name);
    if (info) setShape(info.default_shapes[0].join(", "));
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const parsed = shape
        .split(",")
        .map((s) => parseInt(s.trim(), 10))
        .filter((x) => !Number.isNaN(x));
      if (!parsed.length) throw new Error("invalid shape");
      const { run_id } = await api.optimize({
        task, shape: parsed, algorithm, max_evaluations: budget,
        batch_size: 8, seed: 0, engine, warmup: 10, iterations: 50,
      });
      nav(`/runs/${run_id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-wrap items-end gap-3 p-3">
      <label className="block">
        <div className="k-label mb-1">Task</div>
        <select className="input" value={task} onChange={(e) => pickTask(e.target.value)}>
          {tasks.map((t) => (
            <option key={t.name}>{t.name}</option>
          ))}
        </select>
      </label>
      <label className="block">
        <div className="k-label mb-1">
          Shape{taskInfo ? ` · ${fmtCount(taskInfo.space_size)} configs` : ""}
        </div>
        <input className="input mono w-56" value={shape} onChange={(e) => setShape(e.target.value)} />
      </label>
      <label className="block">
        <div className="k-label mb-1">Algorithm</div>
        <select className="input" value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
          {algorithms.map((a) => (
            <option key={a}>{a}</option>
          ))}
        </select>
      </label>
      <label className="block">
        <div className="k-label mb-1">Budget</div>
        <input type="number" min={1} max={5000} className="input mono w-20" value={budget}
               onChange={(e) => setBudget(parseInt(e.target.value || "1", 10))} />
      </label>
      <label className="block">
        <div className="k-label mb-1">Engine</div>
        <select className="input" value={engine} onChange={(e) => setEngine(e.target.value as typeof engine)}>
          <option value="auto">auto</option>
          <option value="cuda">cuda</option>
          <option value="simulated">simulated</option>
        </select>
      </label>
      <button type="submit" className="btn" disabled={busy || readOnly}>
        {busy ? "starting…" : "Start search"}
      </button>
      {readOnly && (
        <span className="text-[11.5px]" style={{ color: "var(--faint)" }}>
          disabled in snapshot mode
        </span>
      )}
      {error && <span className="text-[11.5px]" style={{ color: "var(--err)" }}>{error}</span>}
    </form>
  );
}

function RunsTable({ runs }: { runs: Run[] }) {
  if (!runs.length)
    return (
      <Empty>
        no runs recorded — launch one above or use{" "}
        <span className="mono ml-1">python optimize.py</span>
      </Empty>
    );
  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            <th>run</th>
            <th>task</th>
            <th>shape</th>
            <th>algorithm</th>
            <th>status</th>
            <th className="num">best</th>
            <th className="num">speedup</th>
            <th className="num">evals</th>
            <th className="num">duration</th>
            <th className="num">started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const speedup =
              r.best_latency_ms && r.baseline_torch_ms
                ? r.baseline_torch_ms / r.best_latency_ms
                : null;
            return (
              <tr key={r.run_id}>
                <td>
                  <Link className="link mono text-[11.5px]" to={`/runs/${r.run_id}`}>
                    {r.run_id}
                  </Link>
                </td>
                <td>{r.task}</td>
                <td className="mono text-[11.5px]">{r.shape.join("×")}</td>
                <td>{r.algorithm}</td>
                <td><StatusDot value={r.status} /></td>
                <td className="num">
                  {fmtMs(r.best_latency_ms)}
                  <SimTag engine={r.engine} />
                </td>
                <td className="num">{fmtSpeedup(speedup)}</td>
                <td className="num">{fmtCount(r.candidates_evaluated)}</td>
                <td className="num" style={{ color: "var(--muted)" }}>
                  {fmtDuration(r.started_at, r.finished_at)}
                </td>
                <td className="num" style={{ color: "var(--muted)" }}>
                  {fmtRelTime(r.started_at)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Overview() {
  useTitle("Overview");
  const [status, setStatus] = useState<Status | null>(null);
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { data: tasks } = useFetch(() => api.tasks(), []);
  const { data: algorithms } = useFetch(() => api.algorithms(), []);

  const refresh = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([api.status(), api.runs()]);
      setStatus(s);
      setRuns(r);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => void refresh(), [refresh]);
  const anyRunning = (runs ?? []).some((r) => r.status === "running");
  usePoll(refresh, 15000, !anyRunning); // steady refresh catches external runs
  usePoll(refresh, 3000, anyRunning);

  if (error && !status) return <ErrorBox error={error} />;
  if (!status) return <Loading />;

  const ov = status.overview;
  const env = status.environment;
  return (
    <div className="space-y-3">
      {error && <ErrorBox error={error} />}
      <div className="kpis">
        <Kpi label="Best speedup" value={fmtSpeedup(ov.best_speedup)} sub="vs PyTorch baseline" />
        <Kpi label="Kernels optimized" value={fmtCount(ov.kernels_optimized)} sub="task × shape" />
        <Kpi
          label="Benchmarks"
          value={fmtCount(ov.benchmarks_completed)}
          sub={`${fmtCount(ov.successful_benchmarks)} ok`}
        />
        <Kpi label="Compile success" value={fmtPct(ov.compile_success_rate)} />
        <Kpi
          label="Runs"
          value={fmtCount(ov.runs)}
          sub={status.active_runs.length ? `${status.active_runs.length} active` : "none active"}
        />
      </div>

      <div
        className="mono flex flex-wrap items-center gap-x-5 gap-y-1 rounded-[4px] border px-3 py-1.5 text-[11.5px]"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}
      >
        <span>python {env.python}</span>
        <span>torch {env.torch}</span>
        <span>
          cuda{" "}
          {env.cuda_available ? (env.cuda_version ?? "yes") : <span style={{ color: "var(--warn)" }}>unavailable</span>}
        </span>
        <span>
          triton{" "}
          {env.triton_available ? env.triton_version : <span style={{ color: "var(--warn)" }}>unavailable</span>}
        </span>
        {env.gpu_name && <span>{env.gpu_name}</span>}
      </div>

      <Panel title="Launch search">
        {tasks?.length && algorithms?.length ? (
          <LaunchForm tasks={tasks} algorithms={algorithms} />
        ) : (
          <Empty>loading task registry</Empty>
        )}
      </Panel>

      <Panel title={`Runs (${runs?.length ?? 0})`}>
        {runs ? <RunsTable runs={runs} /> : <Loading />}
      </Panel>
    </div>
  );
}
