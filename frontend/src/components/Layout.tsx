import { NavLink, Outlet } from "react-router-dom";
import { api, apiMode } from "../api";
import { useFetch } from "./hooks";
import { Notice } from "./ui";

function Icon({ d }: { d: string }) {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

const NAV = [
  // gauge
  { to: "/", label: "Overview", d: "M8 13a5 5 0 1 1 5-5M8 13a5 5 0 0 1-5-5m5 5V8m0 0 3-3" },
  // chip
  { to: "/gpu", label: "Hardware", d: "M5 5h6v6H5zM5 2v2m3-2v2m3-2v2M5 12v2m3-2v2m3-2v2M2 5h2m-2 3h2m-2 3h2m8-6h2m-2 3h2m-2 3h2" },
  // document
  { to: "/reports", label: "Reports", d: "M4 2h6l2 2v10H4zM6 7h4M6 9.5h4M6 12h2.5" },
];

export default function Layout() {
  const { data: status } = useFetch(() => api.status(), []);
  const { data: mode } = useFetch(() => apiMode(), []);

  const env = status?.environment;
  const engineLabel = env ? (env.cuda_available ? env.gpu_name ?? "cuda" : "simulated engine") : "…";

  return (
    <div className="flex min-h-screen flex-col">
      {/* status topbar */}
      <header
        className="flex h-9 items-center gap-4 border-b px-3"
        style={{ borderColor: "var(--border)", background: "var(--panel)" }}
      >
        <span className="text-[12px] font-bold tracking-[0.12em]" style={{ color: "var(--text)" }}>
          GPU-OPTIMIZER
        </span>
        <span className="mono text-[11px]" style={{ color: "var(--faint)" }}>
          v0.1.0
        </span>
        <span className="mono ml-auto flex items-center gap-1.5 text-[11px]" style={{ color: "var(--muted)" }}>
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ background: env?.cuda_available ? "var(--ok)" : "var(--warn)" }}
          />
          {engineLabel}
        </span>
        <span className="mono text-[11px] uppercase" style={{ color: mode === "static" ? "var(--accent)" : "var(--muted)" }}>
          {mode === "static" ? "snapshot" : mode === "live" ? "live" : ""}
        </span>
        <a
          className="link mono text-[11px]"
          href="https://github.com/sunkalpchandra/gpu-optimizer"
          target="_blank"
          rel="noreferrer"
        >
          github
        </a>
      </header>

      <div className="flex flex-1">
        <aside
          className="w-40 shrink-0 border-r py-2"
          style={{ borderColor: "var(--border)" }}
        >
          <nav className="space-y-0.5 px-2">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.to === "/"}
                className={({ isActive }) =>
                  "flex items-center gap-2 rounded-[3px] px-2 py-1.5 text-[12.5px] " +
                  (isActive ? "font-semibold" : "")
                }
                style={({ isActive }) => ({
                  color: isActive ? "var(--text)" : "var(--muted)",
                  background: isActive ? "color-mix(in srgb, var(--accent) 10%, transparent)" : undefined,
                })}
              >
                <Icon d={n.d} /> {n.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main className="min-w-0 flex-1 p-4">
          {mode === "static" && (
            <Notice tone="info">
              Read-only snapshot of recorded runs (GitHub Pages, no backend). Clone{" "}
              <a className="link" href="https://github.com/sunkalpchandra/gpu-optimizer" target="_blank" rel="noreferrer">
                the repository
              </a>{" "}
              and start <span className="mono">uvicorn server.api.main:app</span> to run live searches.
            </Notice>
          )}
          {status?.simulated_data_present && (
            <Notice tone="warn">
              Simulated-engine data present. Values marked <span className="mono font-semibold" style={{ color: "var(--warn)" }}>sim</span>{" "}
              are deterministic model estimates for development — not GPU measurements.
            </Notice>
          )}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
