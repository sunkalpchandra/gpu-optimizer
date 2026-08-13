import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, fmtCount, fmtMs, fmtPct, fmtSpeedup, Run, TaskInfo } from "../api";
import { useFetch, usePoll } from "../components/hooks";
import {
  Empty,
  ErrorBox,
  Loading,
  SectionTitle,
  SimBadge,
  Stat,
  StatusBadge,
} from "../components/ui";

function NewRunForm({ tasks, algorithms }: { tasks: TaskInfo[]; algorithms: string[] }) {
  const nav = useNavigate();
  const [task, setTask] = useState("matmul");
  const [shape, setShape] = useState("1024, 1024, 1024");
  const [algorithm, setAlgorithm] = useState("hybrid");
  const [budget, setBudget] = useState(120);
  const [engine, setEngine] = useState<"auto" | "cuda" | "simulated">("auto");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const taskInfo = tasks.find((t) => t.name === task);
  useEffect(() => {
    if (taskInfo) setShape(taskInfo.default_shapes[0].join(", "));
  }, [task]); // eslint-disable-line react-hooks/exhaustive-deps

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
        task,
        shape: parsed,
        algorithm,
        max_evaluations: budget,
        batch_size: 8,
        seed: 0,
        engine,
        warmup: 10,
        iterations: 50,
      });
      nav(`/runs/${run_id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="card">
      <div className="mb-3 text-sm font-bold text-slate-200">New optimization</div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Task</span>
          <select className="input w-full" value={task} onChange={(e) => setTask(e.target.value)}>
            {tasks.map((t) => (
              <option key={t.name}>{t.name}</option>
            ))}
          </select>
        </label>
        <label className="col-span-2 block">
          <span className="mb-1 block text-xs text-slate-500">
            Shape {taskInfo && <span className="text-slate-600">(space: {fmtCount(taskInfo.space_size)} configs)</span>}
          </span>
          <input className="input w-full" value={shape} onChange={(e) => setShape(e.target.value)} />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Algorithm</span>
          <select
            className="input w-full"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
          >
            {algorithms.map((a) => (
              <option key={a}>{a}</option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Budget</span>
          <input
            type="number"
            min={1}
            max={5000}
            className="input w-full"
            value={budget}
            onChange={(e) => setBudget(parseInt(e.target.value || "1", 10))}
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Engine</span>
          <select
            className="input w-full"
            value={engine}
            onChange={(e) => setEngine(e.target.value as typeof engine)}
          >
            <option value="auto">auto</option>
            <option value="cuda">cuda</option>
            <option value="simulated">simulated</option>
          </select>
        </label>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button className="btn" disabled={busy}>
          {busy ? "Starting…" : "Start search"}
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </form>
  );
}

function RunsTable({ runs }: { runs: Run[] }) {
  if (!runs.length)
    return <Empty>No runs yet — start one above, or run `python optimize.py`.</Empty>;
  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full">
        <thead className="border-b border-slate-800">
          <tr>
            <th className="th">run</th>
            <th className="th">task</th>
            <th className="th">shape</th>
            <th className="th">algorithm</th>
            <th className="th">status</th>
            <th className="th">best</th>
            <th className="th">speedup</th>
            <th className="th">evals</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {runs.map((r) => {
            const speedup =
              r.best_latency_ms && r.baseline_torch_ms
                ? r.baseline_torch_ms / r.best_latency_ms
                : null;
            return (
              <tr key={r.run_id} className="hover:bg-slate-900/40">
                <td className="td">
                  <Link className="font-mono text-xs text-cyan-300 hover:underline" to={`/runs/${r.run_id}`}>
                    {r.run_id}
                  </Link>
                </td>
                <td className="td">{r.task}</td>
                <td className="td font-mono text-xs">{r.shape.join("×")}</td>
                <td className="td">{r.algorithm}</td>
                <td className="td">
                  <StatusBadge value={r.status} />
                </td>
                <td className="td">
                  {fmtMs(r.best_latency_ms)}
                  <SimBadge engine={r.engine} />
                </td>
                <td className="td">{fmtSpeedup(speedup)}</td>
                <td className="td">{fmtCount(r.candidates_evaluated)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Overview() {
  const { data: status, error, loading } = useFetch(() => api.status(), []);
  const { data: tasks } = useFetch(() => api.tasks(), []);
  const { data: algorithms } = useFetch(() => api.algorithms(), []);
  const [runs, setRuns] = useState<Run[] | null>(null);

  const refreshRuns = () => api.runs().then(setRuns).catch(() => {});
  useEffect(() => void refreshRuns(), []);
  usePoll(refreshRuns, 3000, (runs ?? []).some((r) => r.status === "running"));

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!status) return null;

  const ov = status.overview;
  const env = status.environment;
  return (
    <div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat
          label="Best speedup"
          value={fmtSpeedup(ov.best_speedup)}
          accent
          sub="vs PyTorch baseline"
        />
        <Stat label="Kernels optimized" value={fmtCount(ov.kernels_optimized)} sub="task × shape pairs" />
        <Stat label="Benchmarks" value={fmtCount(ov.benchmarks_completed)} sub={`${fmtCount(ov.successful_benchmarks)} successful`} />
        <Stat label="Compile success" value={fmtPct(ov.compile_success_rate)} />
        <Stat label="Search runs" value={fmtCount(ov.runs)} sub={status.active_runs.length ? `${status.active_runs.length} live` : undefined} />
      </div>

      <SectionTitle>Environment</SectionTitle>
      <div className="card grid grid-cols-2 gap-x-8 gap-y-1 text-sm lg:grid-cols-4">
        <div>
          <span className="text-slate-500">Python</span> {env.python}
        </div>
        <div>
          <span className="text-slate-500">PyTorch</span> {env.torch}
        </div>
        <div>
          <span className="text-slate-500">CUDA</span>{" "}
          {env.cuda_available ? env.cuda_version ?? "yes" : <span className="text-amber-300">not available</span>}
        </div>
        <div>
          <span className="text-slate-500">Triton</span>{" "}
          {env.triton_available ? env.triton_version : <span className="text-amber-300">not available</span>}
        </div>
        {env.gpu_name && (
          <div className="col-span-2">
            <span className="text-slate-500">GPU</span> {env.gpu_name}
          </div>
        )}
        {env.notes.map((n, i) => (
          <div key={i} className="col-span-full text-xs text-amber-300/80">
            note: {n}
          </div>
        ))}
      </div>

      <SectionTitle>New optimization</SectionTitle>
      {tasks && algorithms ? <NewRunForm tasks={tasks} algorithms={algorithms} /> : <Loading />}

      <SectionTitle>Recent runs</SectionTitle>
      {runs ? <RunsTable runs={runs} /> : <Loading />}
    </div>
  );
}
