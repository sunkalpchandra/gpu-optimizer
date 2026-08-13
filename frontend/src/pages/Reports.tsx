import { api, RankMetrics, Report } from "../api";
import { useFetch } from "../components/hooks";
import { Empty, ErrorBox, Loading, SectionTitle } from "../components/ui";

function RankTable({ rows }: { rows: Record<string, RankMetrics> }) {
  const entries = Object.entries(rows);
  if (!entries.length) return <Empty>No data.</Empty>;
  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full">
        <thead className="border-b border-slate-800">
          <tr>
            <th className="th">case</th>
            <th className="th">n</th>
            <th className="th">spearman ρ</th>
            <th className="th">top-1 regret</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {entries.map(([k, m]) => (
            <tr key={k}>
              <td className="td font-mono text-xs">{k}</td>
              <td className="td">{m.n}</td>
              <td className="td">
                <span
                  className={
                    (m.spearman ?? 0) > 0.6
                      ? "text-emerald-300"
                      : (m.spearman ?? 0) > 0.3
                        ? "text-amber-300"
                        : "text-red-300"
                  }
                >
                  {m.spearman?.toFixed(3) ?? "—"}
                </span>
              </td>
              <td className="td">{m.top1_regret_pct != null ? `${m.top1_regret_pct}%` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReportView({ report }: { report: Report }) {
  return (
    <div className="mb-10">
      <div className="mb-2 flex items-center gap-3">
        <h1 className="text-lg font-bold text-slate-100">{report.name}</h1>
        <span className="text-xs text-slate-500">
          {report.gpu} · seed {report.seed} · engine {report.engine_label}
        </span>
      </div>
      {report.engine_label === "simulated" && (
        <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-200">
          Simulated results (no CUDA GPU) — development data, not hardware measurements.
        </div>
      )}

      <SectionTitle>Shape interpolation (unseen mid-range sizes)</SectionTitle>
      <RankTable rows={report.interpolation} />
      <SectionTitle>Shape extrapolation (small → large)</SectionTitle>
      <RankTable rows={report.extrapolation} />
      <SectionTitle>Workload transfer (unseen workloads)</SectionTitle>
      <RankTable rows={report.workload_transfer} />

      <SectionTitle>Hardware transfer</SectionTitle>
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead className="border-b border-slate-800">
            <tr>
              <th className="th">case</th>
              <th className="th">transferred</th>
              <th className="th">native best</th>
              <th className="th">penalty</th>
              <th className="th">top-k needed</th>
              <th className="th">spearman ρ</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {Object.entries(report.hardware_transfer).map(([k, m]) => (
              <tr key={k}>
                <td className="td font-mono text-xs">{k}</td>
                <td className="td">
                  {typeof m.transferred_ms === "number" ? `${m.transferred_ms} ms` : m.transferred_ms}
                </td>
                <td className="td">{m.native_ms} ms</td>
                <td className="td">
                  {typeof m.penalty_pct === "number" ? `${m.penalty_pct}%` : m.penalty_pct}
                </td>
                <td className="td">{m.top_k_needed ?? "—"}</td>
                <td className="td">{m.spearman?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SectionTitle>Search efficiency (equal budget)</SectionTitle>
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead className="border-b border-slate-800">
            <tr>
              <th className="th">algorithm</th>
              <th className="th">best latency</th>
              <th className="th">speedup vs torch</th>
              <th className="th">evals to best</th>
              <th className="th">compile rate</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {Object.entries(report.search_efficiency).map(([algo, m]) => (
              <tr key={algo}>
                <td className="td font-semibold">{algo}</td>
                <td className="td">{m.best_ms} ms</td>
                <td className="td">{m.speedup_vs_torch}×</td>
                <td className="td">{m.evals_to_best ?? "—"}</td>
                <td className="td">{(m.compile_rate * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Reports() {
  const { data, error, loading } = useFetch(() => api.reports(), []);
  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!data?.length)
    return (
      <Empty>
        No generalization reports yet — run `python scripts/generalization.py`.
      </Empty>
    );
  return (
    <div>
      {data.map((r) => (
        <ReportView key={r.name} report={r} />
      ))}
    </div>
  );
}
