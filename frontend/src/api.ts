// Typed client for the gpu-optimizer FastAPI backend.

export interface Environment {
  python: string;
  torch: string;
  cuda_available: boolean;
  cuda_version: string | null;
  gpu_name: string | null;
  triton_available: boolean;
  triton_version: string | null;
  notes: string[];
}

export interface Overview {
  best_speedup: number | null;
  kernels_optimized: number;
  benchmarks_completed: number;
  successful_benchmarks: number;
  compile_success_rate: number | null;
  runs: number;
  results_by_engine: Record<string, number>;
}

export interface Status {
  environment: Environment;
  overview: Overview;
  simulated_data_present: boolean;
  active_runs: string[];
}

export interface TaskInfo {
  name: string;
  default_shapes: number[][];
  space_size: number;
  supported_dtypes: string[];
}

export interface GpuSpec {
  name: string;
  compute_capability: [number, number];
  sm_count: number;
  memory_gb: number;
  memory_bandwidth_gbs: number;
  fp32_tflops: number;
  fp16_tflops: number;
  shared_mem_per_sm_kb: number;
  registers_per_sm: number;
  warp_size: number;
  l2_cache_mb: number;
  supports_bf16: boolean;
  supports_tf32: boolean;
  is_simulated: boolean;
}

export interface GpuInfo {
  current: GpuSpec;
  is_simulated: boolean;
  catalog: Record<string, GpuSpec>;
  cuda_available: boolean;
}

export type Engine = "cuda" | "simulated";
export type RunStatus = "running" | "finished" | "error";

export interface Run {
  run_id: string;
  task: string;
  shape: number[];
  algorithm: string;
  engine: Engine;
  gpu_name: string;
  status: RunStatus;
  best_candidate_id: string | null;
  best_latency_ms: number | null;
  baseline_torch_ms: number | null;
  baseline_naive_ms: number | null;
  candidates_evaluated: number;
  compile_success_rate: number | null;
  started_at: number;
  finished_at: number | null;
  error?: string;
}

export interface Iteration {
  id: number;
  run_id: string;
  iteration: number;
  candidate_id: string;
  predicted_ms: number | null;
  predicted_std: number | null;
  actual_ms: number | null;
  reward: number | null;
  best_so_far_ms: number | null;
  status: string;
  note: string;
  created_at: number;
}

export type CandidateStatus = "ok" | "compile_error" | "runtime_error" | "incorrect";

export interface TreeNode {
  candidate_id: string;
  parent_id: string | null;
  config: Record<string, string | number>;
  status: CandidateStatus;
  provenance: string;
  latency_ms: number | null;
  engine: Engine;
}

export interface Tree {
  run_id: string;
  nodes: TreeNode[];
}

export interface KernelSource {
  candidate_id: string;
  task: string;
  shape: number[];
  config: Record<string, string | number>;
  engine: Engine;
  status: CandidateStatus;
  latency_ms: number | null;
  source: string;
}

export interface RankMetrics {
  n: number;
  spearman: number | null;
  top1_regret_pct: number | null;
}

export interface HardwareTransferMetrics {
  transferred_ms: number | string;
  native_ms: number;
  penalty_pct: number | string;
  top_k_needed: number | null;
  spearman: number | null;
}

export interface EfficiencyMetrics {
  best_ms: number;
  speedup_vs_torch: number;
  evals_to_best: number | null;
  compile_rate: number;
}

export interface Report {
  name: string;
  engine_label: string;
  gpu: string;
  seed: number;
  interpolation: Record<string, RankMetrics>;
  extrapolation: Record<string, RankMetrics>;
  workload_transfer: Record<string, RankMetrics>;
  hardware_transfer: Record<string, HardwareTransferMetrics>;
  search_efficiency: Record<string, EfficiencyMetrics>;
}

export interface OptimizeRequest {
  task: string;
  shape: number[];
  algorithm: string;
  max_evaluations: number;
  batch_size: number;
  seed: number;
  engine: "auto" | "cuda" | "simulated";
  warmup: number;
  iterations: number;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  status: () => get<Status>("/api/status"),
  tasks: () => get<TaskInfo[]>("/api/tasks"),
  algorithms: () => get<string[]>("/api/algorithms"),
  gpu: () => get<GpuInfo>("/api/gpu"),
  runs: () => get<Run[]>("/api/runs"),
  run: (id: string) => get<Run>(`/api/runs/${id}`),
  iterations: (id: string, after = 0) =>
    get<Iteration[]>(`/api/runs/${id}/iterations?after=${after}`),
  tree: (id: string) => get<Tree>(`/api/runs/${id}/tree`),
  source: (candidateId: string) =>
    get<KernelSource>(`/api/candidates/${candidateId}/source`),
  reports: () => get<Report[]>("/api/reports"),
  optimize: async (req: OptimizeRequest): Promise<{ run_id: string }> => {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`optimize failed (HTTP ${res.status}): ${body}`);
    }
    return res.json();
  },
};

// ---------------------------------------------------------------- formatting

export function fmtMs(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  if (v < 0.01) return `${(v * 1000).toFixed(2)} µs`;
  if (v < 1) return `${v.toFixed(4)} ms`;
  if (v < 100) return `${v.toFixed(3)} ms`;
  return `${v.toFixed(1)} ms`;
}

export function fmtCount(v: number | null | undefined): string {
  if (v == null) return "—";
  return v.toLocaleString("en-US");
}

export function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export function fmtSpeedup(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  return `${v.toFixed(2)}×`;
}
