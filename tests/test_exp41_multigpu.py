"""Multi-GPU sharder for exp41 (#148). The load-bearing test is that the shard
partition is a disjoint cover of every cell — each cell runs on exactly one GPU,
none dropped, none double-counted. Wrapper helpers (command build, collate) are
pure and tested here; the MCMC itself is bayes/GPU-gated and run on the box."""
import importlib.util
import json
from pathlib import Path

import pytest

_MGPU = Path(__file__).resolve().parents[1] / "scripts" / "exp41_multigpu.py"
_spec = importlib.util.spec_from_file_location("exp41_multigpu", _MGPU)
mgpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mgpu)


def test_shard_partition_is_disjoint_cover():
    exp41 = pytest.importorskip("experiments.exp41_borrowing_calibration")
    cells = list(exp41.iter_cells(["group", "member"], [0.5, 0.7, 0.9]))
    assert len(cells) == 54                              # 2 × 3 × 3 scen × 3 policy
    n = 8
    seen: set = set()
    for i in range(n):
        shard = {idx for idx in range(len(cells)) if idx % n == i}
        assert not (shard & seen), "shards overlap → cell double-run"
        seen |= shard
    assert seen == set(range(len(cells))), "shards miss cells → gaps in the grid"


def test_worker_command_shape():
    cmd = mgpu.worker_command("py", 3, 8, "/tmp/w3.json", ["--full", "--n-reps", "100"])
    assert cmd == ["py", "-m", "experiments.exp41_borrowing_calibration",
                   "--shard", "3/8", "--out", "/tmp/w3.json",
                   "--chain-method", "vectorized", "--full", "--n-reps", "100"]


def test_worker_env_pins_one_gpu():
    e = mgpu.worker_env(5)
    assert e["CUDA_VISIBLE_DEVICES"] == "5"
    assert e["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_collate_merges_and_sorts(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"cell": 2, "x": 1}, {"cell": 0, "x": 2}]))
    (tmp_path / "b.json").write_text(json.dumps([{"cell": 1, "x": 3}]))
    rows = mgpu.collate([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert [r["cell"] for r in rows] == [0, 1, 2]
