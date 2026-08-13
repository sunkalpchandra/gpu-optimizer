import { ReactNode } from "react";

// ------------------------------------------------------------------ panels

export function Panel({
  title,
  meta,
  children,
  pad = false,
}: {
  title?: ReactNode;
  meta?: ReactNode;
  children: ReactNode;
  pad?: boolean;
}) {
  return (
    <section className="panel">
      {title != null && (
        <header className="panel-head">
          <span>{title}</span>
          {meta != null && <span className="ml-auto font-normal normal-case tracking-normal">{meta}</span>}
        </header>
      )}
      <div className={pad ? "p-3" : undefined}>{children}</div>
    </section>
  );
}

// ---------------------------------------------------------------- statuses

const STATUS_COLOR: Record<string, string> = {
  ok: "var(--ok)",
  finished: "var(--ok)",
  running: "var(--accent)",
  compile_error: "var(--err)",
  runtime_error: "var(--err)",
  error: "var(--err)",
  incorrect: "var(--warn)",
};

/** Dot + label, the way real consoles mark state — color is the signal. */
export function StatusDot({ value }: { value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${value === "running" ? "animate-pulse" : ""}`}
        style={{ background: STATUS_COLOR[value] ?? "var(--faint)" }}
      />
      <span className="text-[12px]" style={{ color: "var(--muted)" }}>
        {value.replace("_", " ")}
      </span>
    </span>
  );
}

export const PROVENANCE_COLOR: Record<string, string> = {
  rl: "#a371f7",
  evolutionary: "#3fb950",
  bo: "#39c5cf",
  random: "#8b949e",
  grid: "#8b949e",
  "baseline-naive": "#d29922",
  baseline: "#d29922",
  transfer: "#6ea8fe",
  manual: "#8b949e",
};

/** Compact mono tag; tinted text, no background pill. */
export function ProvTag({ value }: { value: string }) {
  return (
    <span
      className="mono text-[11px]"
      style={{ color: PROVENANCE_COLOR[value] ?? "var(--muted)" }}
    >
      {value}
    </span>
  );
}

/** Marks a number produced by the simulated engine. */
export function SimTag({ engine }: { engine: string | undefined }) {
  if (engine !== "simulated") return null;
  return (
    <span
      title="Simulated engine estimate — not a GPU measurement"
      className="mono ml-1 align-middle text-[10px] font-semibold"
      style={{ color: "var(--warn)" }}
    >
      sim
    </span>
  );
}

// -------------------------------------------------------------------- KPIs

export function Kpi({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="kpi">
      <div className="k-label">{label}</div>
      <div className="k-value">{value}</div>
      {sub != null && <div className="k-sub">{sub}</div>}
    </div>
  );
}

// ------------------------------------------------------------- page header

export function PageHead({
  crumbs,
  right,
}: {
  crumbs: ReactNode[];
  right?: ReactNode;
}) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <nav className="flex min-w-0 items-center gap-1.5 text-[13px]" style={{ color: "var(--muted)" }}>
        {crumbs.map((c, i) => (
          <span key={i} className="flex min-w-0 items-center gap-1.5">
            {i > 0 && <span style={{ color: "var(--faint)" }}>/</span>}
            <span className={i === crumbs.length - 1 ? "truncate font-semibold text-[color:var(--text)]" : ""}>
              {c}
            </span>
          </span>
        ))}
      </nav>
      {right != null && <div className="ml-auto flex items-center gap-2">{right}</div>}
    </div>
  );
}

// ------------------------------------------------------------ empty/loading

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-24 items-center justify-center p-4 text-[12.5px]" style={{ color: "var(--faint)" }}>
      {children}
    </div>
  );
}

export function Loading() {
  return (
    <div className="space-y-2">
      {[...Array(3)].map((_, i) => (
        <div key={i} className="panel h-16 animate-pulse" />
      ))}
    </div>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="panel p-3 text-[12.5px]" style={{ borderColor: "#5c2a2e", color: "var(--err)" }}>
      request failed: {error}
    </div>
  );
}

// -------------------------------------------------------------------- misc

export function Notice({
  tone,
  children,
}: {
  tone: "warn" | "info";
  children: ReactNode;
}) {
  const color = tone === "warn" ? "var(--warn)" : "var(--accent)";
  return (
    <div
      className="mb-3 flex items-start gap-2 rounded-[4px] border px-3 py-2 text-[12.5px]"
      style={{
        borderColor: `color-mix(in srgb, ${color} 35%, transparent)`,
        background: `color-mix(in srgb, ${color} 7%, transparent)`,
        color: "var(--text)",
      }}
    >
      <span className="mt-[3px] inline-block h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: color }} />
      <div>{children}</div>
    </div>
  );
}
