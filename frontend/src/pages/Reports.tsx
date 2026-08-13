import { api, RankMetrics, Report } from "../api";
import { useFetch, useTitle } from "../components/hooks";
import { Empty, ErrorBox, Loading, Notice, PageHead, Panel } from "../components/ui";

function rhoColor(rho: number | null | undefined): string {
  if (rho == null) return "var(--faint)";
  if (rho > 0.6) return "var(--ok)";
  if (rho > 0.3) return "var(--warn)";
  return "var(--err)";
}

function RankTable({ rows }: { rows: Record<string, RankMetrics> }) {
  const entries = Object.entries(rows);
  if (!entries.length) return <Empty>no data</Empty>;
  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>case</th>
          <th className="num">n</th>
          <th className="num">spearman ρ</th>
          <th className="num">top-1 regret</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([k, m]) => (
          <tr key={k}>
            <td className="mono text-[11.5px]">{k}</td>
            <td className="num">{m.n}</td>
            <td className="num" style={{ color: rhoColor(m.spearman) }}>
              {m.spearman?.toFixed(3) ?? "—"}
            </td>
            <td className="num">{m.top1_regret_pct != null ? `${m.top1_regret_pct}%` : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReportView({ report }: { report: Report }) {
  return (
    <div className="space-y-3">
      <PageHead
        crumbs={["reports", <span key="n" className="mono">{report.name}</span>]}
        right={
          <span className="mono text-[11.5px]" style={{ color: "var(--muted)" }}>
            {report.gpu} · seed {report.seed} · engine {report.engine_label}
          </span>
        }
      />
      {report.engine_label === "simulated" && (
        <Notice tone="warn">
          Simulated results (no CUDA GPU) — these validate search behavior and transfer, not
          hardware performance.
        </Notice>
      )}

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Shape interpolation">
          <RankTable rows={report.interpolation} />
        </Panel>
        <Panel title="Shape extrapolation">
          <RankTable rows={report.extrapolation} />
        </Panel>
      </div>
      <Panel title="Workload transfer">
        <RankTable rows={report.workload_transfer} />
      </Panel>

      <Panel title="Hardware transfer (top-k protocol)">
        <table className="tbl">
          <thead>
            <tr>
              <th>case</th>
              <th className="num">transferred</th>
              <th className="num">native best</th>
              <th className="num">penalty</th>
              <th className="num">k needed</th>
              <th className="num">spearman ρ</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.hardware_transfer).map(([k, m]) => (
              <tr key={k}>
                <td className="mono text-[11.5px]">{k}</td>
                <td className="num">
                  {typeof m.transferred_ms === "number" ? `${m.transferred_ms} ms` : m.transferred_ms}
                </td>
                <td className="num">{m.native_ms} ms</td>
                <td className="num">
                  {typeof m.penalty_pct === "number" ? `${m.penalty_pct}%` : m.penalty_pct}
                </td>
                <td className="num">{m.top_k_needed ?? "—"}</td>
                <td className="num" style={{ color: rhoColor(m.spearman) }}>
                  {m.spearman?.toFixed(3) ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel title="Search efficiency · equal budget">
        <table className="tbl">
          <thead>
            <tr>
              <th>algorithm</th>
              <th className="num">best latency</th>
              <th className="num">speedup vs torch</th>
              <th className="num">evals to best</th>
              <th className="num">compile rate</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.search_efficiency).map(([algo, m]) => (
              <tr key={algo}>
                <td className="font-semibold">{algo}</td>
                <td className="num">{m.best_ms} ms</td>
                <td className="num">{m.speedup_vs_torch}×</td>
                <td className="num">{m.evals_to_best ?? "—"}</td>
                <td className="num">{(m.compile_rate * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

export default function Reports() {
  useTitle("Reports");
  const { data, error, loading } = useFetch(() => api.reports(), []);
  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!data?.length)
    return <Empty>no generalization reports — run `python scripts/generalization.py`</Empty>;
  return (
    <div className="space-y-8">
      {data.map((r) => (
        <ReportView key={r.name} report={r} />
      ))}
    </div>
  );
}
