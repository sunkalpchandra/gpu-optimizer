"""Stage 8 backend: API endpoints over a temp DB, including a live run."""

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import server.api.main as api
    from benchmarks.db import BenchmarkDB

    test_db = BenchmarkDB(tmp_path / "api.sqlite")
    monkeypatch.setattr(api, "_db", test_db)
    return TestClient(api.app)


def test_status_tasks_algorithms_gpu(client):
    s = client.get("/api/status").json()
    assert "environment" in s and "overview" in s
    assert isinstance(s["environment"]["cuda_available"], bool)

    t = client.get("/api/tasks").json()
    names = {x["name"] for x in t}
    assert {"matmul", "vecadd", "softmax", "attention"} <= names
    assert all(x["space_size"] > 0 for x in t)

    a = client.get("/api/algorithms").json()
    assert "hybrid" in a and "random" in a

    g = client.get("/api/gpu").json()
    assert g["current"]["sm_count"] > 0
    assert "catalog" in g


def test_missing_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/candidates/nope/source").status_code == 404


def test_optimize_run_lifecycle(client):
    req = {"task": "vecadd", "shape": [1 << 20], "algorithm": "random",
           "max_evaluations": 12, "batch_size": 4, "engine": "simulated",
           "warmup": 1, "iterations": 5}
    r = client.post("/api/optimize", json=req)
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    # poll until finished (background thread; simulated engine is fast)
    for _ in range(200):
        detail = client.get(f"/api/runs/{run_id}")
        if detail.status_code == 200 and detail.json()["status"] == "finished":
            break
        time.sleep(0.1)
    else:
        pytest.fail("run did not finish in time")

    detail = detail.json()
    assert detail["engine"] == "simulated"          # honesty label
    assert detail["candidates_evaluated"] == 12
    assert detail["best_latency_ms"] > 0

    iters = client.get(f"/api/runs/{run_id}/iterations").json()
    assert len(iters) == 12
    # incremental polling
    tail = client.get(f"/api/runs/{run_id}/iterations", params={"after": 8}).json()
    assert len(tail) == 4

    tree = client.get(f"/api/runs/{run_id}/tree").json()
    assert len(tree["nodes"]) >= 12
    ok_nodes = [n for n in tree["nodes"] if n["status"] == "ok"]
    assert ok_nodes and all(n["engine"] == "simulated" for n in tree["nodes"])

    src = client.get(f"/api/candidates/{ok_nodes[-1]['candidate_id']}/source").json()
    assert "@triton.jit" in src["source"]
    assert "vecadd_kernel" in src["source"]

    runs = client.get("/api/runs").json()
    assert any(x["run_id"] == run_id for x in runs)


def test_optimize_validation(client):
    assert client.post("/api/optimize", json={"task": "nope"}).status_code == 400
    assert client.post("/api/optimize",
                       json={"task": "matmul", "algorithm": "nope"}).status_code == 400
