import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, GpuSpec } from "../api";
import { useFetch, useTitle } from "../components/hooks";
import { ErrorBox, Loading, PageHead, Panel } from "../components/ui";

function SpecTable({ spec }: { spec: GpuSpec }) {
  const rows: Array<[string, string]> = [
    ["Compute capability", `sm_${spec.compute_capability[0]}${spec.compute_capability[1]}`],
    ["Streaming multiprocessors", String(spec.sm_count)],
    ["Memory", `${spec.memory_gb.toFixed(0)} GB`],
    ["Memory bandwidth", `${spec.memory_bandwidth_gbs.toFixed(0)} GB/s`],
    ["L2 cache", `${spec.l2_cache_mb.toFixed(0)} MB`],
    ["FP32 peak", `${spec.fp32_tflops.toFixed(1)} TFLOPS`],
    ["FP16/BF16 peak", `${spec.fp16_tflops.toFixed(1)} TFLOPS`],
    ["Shared memory / SM", `${spec.shared_mem_per_sm_kb} KB`],
    ["Registers / SM", spec.registers_per_sm.toLocaleString()],
    ["Warp size", String(spec.warp_size)],
    ["Precisions", `fp32${spec.supports_tf32 ? " · tf32" : ""} · fp16${spec.supports_bf16 ? " · bf16" : ""}`],
  ];
  return (
    <table className="tbl">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={{ color: "var(--muted)" }}>{k}</td>
            <td className="num">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function GpuMetrics() {
  useTitle("Hardware");
  const { data, error, loading } = useFetch(() => api.gpu(), []);
  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;

  const catalog = Object.entries(data.catalog).map(([key, s]) => ({
    key,
    fp16: s.fp16_tflops,
    fp32: s.fp32_tflops,
    bandwidth_tbs: s.memory_bandwidth_gbs / 1000,
    ...s,
  }));

  return (
    <div className="space-y-3">
      <PageHead crumbs={["hardware"]} />

      <div className="flex flex-wrap gap-3">
        <div className="min-w-72 flex-1">
          <Panel
            title="Current target"
            meta={
              data.is_simulated ? (
                <span className="mono font-semibold" style={{ color: "var(--warn)" }}>
                  SIMULATED
                </span>
              ) : undefined
            }
          >
            <div className="px-3 pt-2 text-[13px] font-semibold">{data.current.name}</div>
            <SpecTable spec={data.current} />
            {!data.cuda_available && (
              <p className="px-3 pb-2 pt-1 text-[11.5px]" style={{ color: "var(--warn)" }}>
                No CUDA device present — this catalog spec drives the simulated engine; its
                results are labeled accordingly.
              </p>
            )}
          </Panel>
        </div>

        <div className="min-w-96 flex-[1.4]">
          <Panel title="Catalog throughput">
            <div className="h-72 p-2">
              <ResponsiveContainer>
                <BarChart data={catalog} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="key" stroke="var(--faint)" fontSize={10.5} tickLine={false} />
                  <YAxis stroke="var(--faint)" fontSize={10.5} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "var(--panel)",
                      border: "1px solid var(--border-strong)",
                      borderRadius: 4,
                      fontSize: 11.5,
                    }}
                    cursor={{ fill: "rgba(110,168,254,0.05)" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="fp16" name="FP16 TFLOPS" fill="#3fb950" />
                  <Bar dataKey="fp32" name="FP32 TFLOPS" fill="#6ea8fe" />
                  <Bar dataKey="bandwidth_tbs" name="Bandwidth TB/s" fill="#a371f7" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>
      </div>

      <Panel title={`GPU catalog (${catalog.length})`}>
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>gpu</th>
                <th>arch</th>
                <th className="num">SMs</th>
                <th className="num">memory</th>
                <th className="num">bandwidth</th>
                <th className="num">fp32</th>
                <th className="num">fp16</th>
                <th className="num">smem/SM</th>
              </tr>
            </thead>
            <tbody>
              {catalog.map((g) => (
                <tr key={g.key}>
                  <td>{g.key}</td>
                  <td className="mono text-[11.5px]">
                    sm_{g.compute_capability[0]}{g.compute_capability[1]}
                  </td>
                  <td className="num">{g.sm_count}</td>
                  <td className="num">{g.memory_gb.toFixed(0)} GB</td>
                  <td className="num">{g.memory_bandwidth_gbs.toFixed(0)} GB/s</td>
                  <td className="num">{g.fp32_tflops.toFixed(1)}</td>
                  <td className="num">{g.fp16_tflops.toFixed(1)}</td>
                  <td className="num">{g.shared_mem_per_sm_kb} KB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
