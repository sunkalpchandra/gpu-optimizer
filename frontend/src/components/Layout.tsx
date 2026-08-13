import { NavLink, Outlet } from "react-router-dom";
import { api, apiMode } from "../api";
import { useFetch } from "./hooks";

const NAV = [
  { to: "/", label: "Overview", icon: "◈" },
  { to: "/gpu", label: "GPU metrics", icon: "▦" },
  { to: "/reports", label: "Reports", icon: "≣" },
];

export default function Layout() {
  const { data: status } = useFetch(() => api.status(), []);
  const { data: mode } = useFetch(() => apiMode(), []);

  return (
    <div className="flex min-h-screen">
      <aside className="fixed inset-y-0 left-0 flex w-52 flex-col border-r border-slate-800 bg-slate-950/80 p-4">
        <div className="mb-6">
          <div className="text-lg font-black tracking-tight text-slate-100">
            gpu<span className="text-emerald-400">-optimizer</span>
          </div>
          <div className="text-[11px] text-slate-500">autonomous kernel search</div>
        </div>
        <nav className="space-y-1">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium ${
                  isActive
                    ? "bg-emerald-500/10 text-emerald-300"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`
              }
            >
              <span className="text-xs">{n.icon}</span> {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-4 border-t border-slate-800 pt-3 text-[11px] leading-5 text-slate-500">
          Runs, search trees and kernel sources are reachable from Overview.
        </div>
        <div className="mt-auto pt-4 text-[11px] text-slate-600">
          <a
            className="hover:text-slate-400"
            href="https://github.com/sunkalpchandra/gpu-optimizer"
            target="_blank"
            rel="noreferrer"
          >
            github.com/sunkalpchandra/gpu-optimizer
          </a>
        </div>
      </aside>
      <main className="ml-52 flex-1 p-6">
        {mode === "static" && (
          <div className="mb-4 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-4 py-2.5 text-sm text-cyan-200">
            Static demo snapshot (no backend) — recorded data from simulated runs. Clone{" "}
            <a
              className="underline"
              href="https://github.com/sunkalpchandra/gpu-optimizer"
              target="_blank"
              rel="noreferrer"
            >
              the repo
            </a>{" "}
            and run <code className="font-mono text-xs">uvicorn server.api.main:app</code> for
            live searches.
          </div>
        )}
        {status?.simulated_data_present && (
          <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-200">
            ⚠ Simulated engine data present — numbers labeled{" "}
            <span className="rounded border border-amber-500/40 bg-amber-500/15 px-1 text-[10px] font-semibold">
              sim
            </span>{" "}
            are model estimates, not GPU measurements.
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}
