import { ReactNode } from "react";

// ------------------------------------------------------------------ badges

const PROVENANCE_COLORS: Record<string, string> = {
  rl: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  evolutionary: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  bo: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
  random: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  grid: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  "baseline-naive": "bg-amber-500/15 text-amber-300 border-amber-500/30",
  baseline: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  transfer: "bg-blue-500/15 text-blue-300 border-blue-500/30",
  manual: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

export function ProvenanceBadge({ value }: { value: string }) {
  const cls = PROVENANCE_COLORS[value] ?? PROVENANCE_COLORS.manual;
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[11px] font-medium ${cls}`}>
      {value}
    </span>
  );
}

const STATUS_COLORS: Record<string, string> = {
  ok: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  finished: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  running: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30 animate-pulse",
  compile_error: "bg-red-500/15 text-red-300 border-red-500/30",
  runtime_error: "bg-red-500/15 text-red-300 border-red-500/30",
  incorrect: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  error: "bg-red-500/15 text-red-300 border-red-500/30",
};

export function StatusBadge({ value }: { value: string }) {
  const cls = STATUS_COLORS[value] ?? STATUS_COLORS.ok;
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[11px] font-medium ${cls}`}>
      {value.replace("_", " ")}
    </span>
  );
}

/** Amber marker attached to any number produced by the simulated engine. */
export function SimBadge({ engine }: { engine: string | undefined }) {
  if (engine !== "simulated") return null;
  return (
    <span
      title="Simulated engine estimate — not a GPU measurement"
      className="ml-1 inline-block rounded border border-amber-500/40 bg-amber-500/15
        px-1 py-px align-middle text-[10px] font-semibold text-amber-300"
    >
      sim
    </span>
  );
}

// ------------------------------------------------------------------- stats

export function Stat({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="card">
      <div className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${accent ? "text-emerald-400" : "text-slate-100"}`}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

// ------------------------------------------------------------ empty/loading

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="card flex min-h-32 items-center justify-center text-sm text-slate-500">
      {children}
    </div>
  );
}

export function Loading() {
  return (
    <div className="space-y-3">
      {[...Array(3)].map((_, i) => (
        <div key={i} className="card h-20 animate-pulse bg-slate-900" />
      ))}
    </div>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="card border-red-500/30 text-sm text-red-300">
      Failed to load: {error}
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return <h2 className="mb-2 mt-6 text-sm font-bold uppercase tracking-wider text-slate-400">{children}</h2>;
}
