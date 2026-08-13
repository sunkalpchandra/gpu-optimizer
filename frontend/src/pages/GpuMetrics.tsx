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
import { useFetch } from "../components/hooks";
import { ErrorBox, Loading, SectionTitle } from "../components/ui";

function SpecRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-slate-800/60 py-1.5 text-sm last:border-0">
      <span className="text-slate-500">{label}</span>
      <span className="font-mono text-slate-200">{value}</span>
    </div>
  );
}

function CurrentGpu({ spec, simulated }: { spec: GpuSpec; simulated: boolean }) {
  return (
    <div className="card">
      <div className="mb-2 flex items-center gap-3">
        <div className="text-base font-bold text-slate-100">{spec.name}</div>
        {simulated && (
          <span className="rounded border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-xs font-bold text-amber-300">
            SIMULATED
          </span>
        )}
      </div>
      <div className="grid gap-x-10 lg:grid-cols-2">
        <div>
          <SpecRow
            label="Compute capability"
            value={`sm_${spec.compute_capability[0]}${spec.compute_capability[1]}`}
          />
          <SpecRow label="SM count" value={String(spec.sm_count)} />
          <SpecRow label="Memory" value={`${spec.memory_gb.toFixed(0)} GB`} />
          <SpecRow label="Memory bandwidth" value={`${spec.memory_bandwidth_gbs.toFixed(0)} GB/s`} />
          <SpecRow label="L2 cache" value={`${spec.l2_cache_mb.toFixed(0)} MB`} />
        </div>
        <div>
          <SpecRow label="FP32 peak" value={`${spec.fp32_tflops.toFixed(1)} TFLOPS`} />
          <SpecRow label="FP16/BF16 peak" value={`${spec.fp16_tflops.toFixed(1)} TFLOPS`} />
          <SpecRow label="Shared mem / SM" value={`${spec.shared_mem_per_sm_kb} KB`} />
          <SpecRow label="Registers / SM" value={spec.registers_per_sm.toLocaleString()} />
          <SpecRow
            label="Precisions"
            value={`fp32${spec.supports_tf32 ? " tf32" : ""} fp16${spec.supports_bf16 ? " bf16" : ""}`}
          />
        </div>
      </div>
    </div>
  );
}

export default function GpuMetrics() {
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
    <div>
      <SectionTitle>Current target</SectionTitle>
      <CurrentGpu spec={data.current} simulated={data.is_simulated} />
      {!data.cuda_available && (
        <p className="mt-2 text-xs text-amber-300/80">
          No CUDA device present — this catalog spec drives the simulated engine; all its results
          are labeled accordingly.
        </p>
      )}

      <SectionTitle>Throughput comparison (catalog)</SectionTitle>
      <div className="card h-80">
        <ResponsiveContainer>
          <BarChart data={catalog} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="key" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="fp16" name="FP16 TFLOPS" fill="#34d399" />
            <Bar dataKey="fp32" name="FP32 TFLOPS" fill="#22d3ee" />
            <Bar dataKey="bandwidth_tbs" name="Bandwidth (TB/s)" fill="#a78bfa" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <SectionTitle>GPU catalog</SectionTitle>
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead className="border-b border-slate-800">
            <tr>
              <th className="th">GPU</th>
              <th className="th">arch</th>
              <th className="th">SMs</th>
              <th className="th">mem</th>
              <th className="th">bandwidth</th>
              <th className="th">fp32</th>
              <th className="th">fp16</th>
              <th className="th">smem/SM</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {catalog.map((g) => (
              <tr key={g.key} className="hover:bg-slate-900/40">
                <td className="td">{g.key}</td>
                <td className="td font-mono text-xs">
                  sm_{g.compute_capability[0]}
                  {g.compute_capability[1]}
                </td>
                <td className="td">{g.sm_count}</td>
                <td className="td">{g.memory_gb.toFixed(0)} GB</td>
                <td className="td">{g.memory_bandwidth_gbs.toFixed(0)} GB/s</td>
                <td className="td">{g.fp32_tflops.toFixed(1)} TF</td>
                <td className="td">{g.fp16_tflops.toFixed(1)} TF</td>
                <td className="td">{g.shared_mem_per_sm_kb} KB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
