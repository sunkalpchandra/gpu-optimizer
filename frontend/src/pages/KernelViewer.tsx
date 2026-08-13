import { Link, useParams } from "react-router-dom";
import { api, fmtMs } from "../api";
import CodeBlock from "../components/CodeBlock";
import { useFetch, useTitle } from "../components/hooks";
import { ErrorBox, Loading, PageHead, Panel, SimTag, StatusDot } from "../components/ui";

export default function KernelViewer() {
  const { candidateId = "" } = useParams();
  useTitle(`kernel ${candidateId.slice(0, 10)}`);
  const { data, error, loading } = useFetch(() => api.source(candidateId), [candidateId]);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;

  return (
    <div className="space-y-3">
      <PageHead
        crumbs={[
          <Link key="r" className="link" to="/">runs</Link>,
          "kernel",
          <span key="id" className="mono">{data.candidate_id.slice(0, 12)}</span>,
        ]}
        right={<StatusDot value={data.status} />}
      />

      <div
        className="mono flex flex-wrap items-center gap-x-5 gap-y-1 rounded-[4px] border px-3 py-1.5 text-[11.5px]"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}
      >
        <span>{data.task}</span>
        <span>{data.shape.join("×")}</span>
        <span style={{ color: "var(--text)" }}>
          {fmtMs(data.latency_ms)}
          <SimTag engine={data.engine} />
        </span>
        <span className="flex flex-wrap gap-x-3">
          {Object.entries(data.config).map(([k, v]) => (
            <span key={k}>
              <span style={{ color: "var(--faint)" }}>{k}=</span>
              <span style={{ color: "var(--text)" }}>{String(v)}</span>
            </span>
          ))}
        </span>
      </div>

      <Panel title={`${data.task} kernel · Triton`} meta={<span className="mono">{data.candidate_id}</span>}>
        <CodeBlock code={data.source} />
      </Panel>
    </div>
  );
}
