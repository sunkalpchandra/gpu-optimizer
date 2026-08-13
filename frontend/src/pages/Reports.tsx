import { api, fmtMs, fmtSpeedup, RankMetrics, Report } from "../api";
import { useFetch, useTitle } from "../components/hooks";
import { Empty, ErrorBox, Loading, Notice, PageHead, Panel } from "../components/ui";

function rhoColor(rho: number | null | undefined): string {
  if (rho == null) return "var(--faint)";
  if (rho > 0.6) return "var(--ok)";
  if (rho > 0.3) return "var(--warn)";
  return "var(--err)";
}

const pct = (v: number | string | null | undefined) =>
  typeof v === "number" ? `${v.toFixed(1)}%` : (v ?? "—");

function RankTable({ rows }: { rows: Record<string, RankMetrics> }) {
  const entries = Object.entries(rows);
  if (!entries.length) return <Empty>no data</Empty>;
  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>case</th>
          <th className="num">n</th>
          <th className="num">spearman rho</th>
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
            <td className="num">{pct(m.top1_regret_pct)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReportView({ report }: { report: Report }) {
  return (
    <div className="space-y-3">
      <div
        className="mono flex flex-wrap items-center gap-x-5 gap-y-1 rounded-[4px] border px-3 py-1.5 text-[11.5px]"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}
      >
        <span style={{ color: "var(--text)" }}>{report.name}</span>
        <span>{report.gpu}</span>
        <span>seed {report.seed}</span>
        <span>engine {report.engine_label}</span>
      </div>
      {report.engine_label === "simulated" && (
        <Notice tone="warn">
          Simulated results — these validate search behavior and transfer, not hardware
          performance.
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

      <Panel title="Hardware transfer">
        <table className="tbl">
          <thead>
            <tr>
              <th>case</th>
              <th className="num">transferred</th>
              <th className="num">native best</th>
              <th className="num">penalty</th>
              <th className="num">k needed</th>
              <th className="num">spearman rho</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.hardware_transfer).map(([k, m]) => (
              <tr key={k}>
                <td className="mono text-[11.5px]">{k}</td>
                <td className="num">
                  {typeof m.transferred_ms === "number" ? fmtMs(m.transferred_ms) : m.transferred_ms}
                </td>
                <td className="num">{fmtMs(m.native_ms)}</td>
                <td className="num">{pct(m.penalty_pct)}</td>
                <td className="num">{m.top_k_needed ?? "—"}</td>
                <td className="num" style={{ color: rhoColor(m.spearman) }}>
                  {m.spearman?.toFixed(3) ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel title="Search efficiency">
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
                <td className="num">{fmtMs(m.best_ms)}</td>
                <td className="num">{fmtSpeedup(m.speedup_vs_torch)}</td>
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
  const head = <PageHead crumbs={["reports"]} />;
  if (loading) return <div className="space-y-3">{head}<Loading /></div>;
  if (error) return <div className="space-y-3">{head}<ErrorBox error={error} /></div>;
  return (
    <div className="space-y-3">
      {head}
      {data?.length ? (
        data.map((r) => <ReportView key={r.name} report={r} />)
      ) : (
        <Panel title="Generalization">
          <Empty>
            no reports — run <span className="mono ml-1">python scripts/generalization.py</span>
          </Empty>
        </Panel>
      )}
    </div>
  );
}
