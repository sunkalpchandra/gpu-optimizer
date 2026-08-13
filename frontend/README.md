# gpu-optimizer dashboard

React + TypeScript + Tailwind (v4) + Recharts frontend for the optimizer.

## Development

```bash
# terminal 1 — backend (repo root)
uvicorn server.api.main:app --port 8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api → :8000)
```

## Production build

```bash
npm run build        # type-checks then emits frontend/dist
```

The FastAPI server serves `frontend/dist` automatically when it exists, so
after building, the whole demo is just `uvicorn server.api.main:app`.

## Pages

- **Overview** — stat tiles, environment, new-optimization launcher, runs table
- **Run view** (`/#/runs/:id`) — live iteration table + convergence chart
- **Search tree** (`/#/runs/:id/tree`) — clickable candidate-lineage SVG
- **Kernel viewer** (`/#/kernels/:candidateId`) — annotated Triton source
- **GPU metrics** (`/#/gpu`) — current target + catalog comparison
- **Reports** (`/#/reports`) — generalization study tables

Simulated-engine numbers always carry an amber `sim` badge; a global banner
appears whenever simulated data is present.
