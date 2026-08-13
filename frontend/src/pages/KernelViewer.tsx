import { useParams } from "react-router-dom";
import { api, fmtMs } from "../api";
import CodeBlock from "../components/CodeBlock";
import { useFetch } from "../components/hooks";
import { ErrorBox, Loading, SimBadge, StatusBadge } from "../components/ui";

export default function KernelViewer() {
  const { candidateId = "" } = useParams();
  const { data, error, loading } = useFetch(() => api.source(candidateId), [candidateId]);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-bold text-slate-100">
          {data.task} <span className="text-slate-500">{data.shape.join("×")}</span>
        </h1>
        <StatusBadge value={data.status} />
        <span className="text-sm text-slate-400">
          {fmtMs(data.latency_ms)}
          <SimBadge engine={data.engine} />
        </span>
        <span className="ml-auto font-mono text-xs text-slate-600">{data.candidate_id}</span>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        {Object.entries(data.config).map(([k, v]) => (
          <span
            key={k}
            className="rounded-full border border-slate-700 bg-slate-900 px-2.5 py-1 font-mono text-xs"
          >
            <span className="text-slate-500">{k}=</span>
            <span className="text-emerald-300">{String(v)}</span>
          </span>
        ))}
      </div>

      <CodeBlock code={data.source} />
    </div>
  );
}
