"""Unit tests: program graph IR, parameter space, actions, encodings."""

import random

import numpy as np
import pytest

from compiler.ir import NODE_FEATURE_DIM, ProgramGraph
from compiler.transformations.actions import (
    ACTION_CATALOG,
    NUM_ACTIONS,
    STOP_ACTION_INDEX,
    apply_action,
    valid_action_mask,
)
from compiler.transformations.space import (
    CANDIDATE_FEATURE_DIM,
    Candidate,
    ParamSpace,
    ParamSpec,
    encode_config,
)


def small_space() -> ParamSpace:
    return ParamSpace(
        [
            ParamSpec("BLOCK_M", choices=(16, 32, 64)),
            ParamSpec("num_warps", choices=(2, 4, 8)),
            ParamSpec("dtype", choices=("float32", "float16")),
        ],
        name="test",
    )


class TestProgramGraph:
    def test_build_and_features(self):
        g = ProgramGraph("t")
        a = g.input((64, 32))
        b = g.input((32, 16))
        c = g.add("matmul", (64, 16), inputs=(a, b), reduction_axes=(1,))
        g.add("output", (64, 16), inputs=(c,))
        assert len(g) == 4
        feats = g.node_features()
        assert feats.shape == (4, NODE_FEATURE_DIM)
        assert np.isfinite(feats).all()
        adj = g.adjacency()
        assert adj[a.node_id, c.node_id] == 1.0
        assert adj[c.node_id, 3] == 1.0
        assert adj.sum() == 3  # a->c, b->c, c->output

    def test_topological_enforcement(self):
        g = ProgramGraph()
        with pytest.raises(ValueError):
            g.add("relu", (4,), inputs=(5,))

    def test_unknown_op_rejected(self):
        g = ProgramGraph()
        with pytest.raises(ValueError):
            g.add("convolve", (4,))


class TestParamSpace:
    def test_sample_validate_roundtrip(self):
        space = small_space()
        rng = random.Random(0)
        for _ in range(20):
            cfg = space.sample(rng)
            space.validate(cfg)  # no raise

    def test_validate_rejects(self):
        space = small_space()
        with pytest.raises(ValueError):
            space.validate({"BLOCK_M": 48, "num_warps": 4, "dtype": "float32"})
        with pytest.raises(ValueError):
            space.validate({"BLOCK_M": 16, "num_warps": 4})

    def test_grid_size(self):
        space = small_space()
        assert space.size() == 3 * 3 * 2
        assert len(space.grid()) == 18

    def test_neighbors_are_one_step(self):
        space = small_space()
        cfg = {"BLOCK_M": 32, "num_warps": 2, "dtype": "float32"}
        nbrs = space.neighbors(cfg)
        # BLOCK_M has 2 neighbors, num_warps 1 (at low edge), dtype 1
        assert len(nbrs) == 4
        for n in nbrs:
            diff = [k for k in cfg if n[k] != cfg[k]]
            assert len(diff) == 1


class TestEncoding:
    def test_encode_shape_and_determinism(self):
        cfg = {"BLOCK_M": 64, "num_warps": 4, "dtype": "float16"}
        v = encode_config(cfg)
        assert v.shape == (CANDIDATE_FEATURE_DIM,)
        assert np.allclose(v, encode_config(dict(cfg)))

    def test_candidate_id_stable_and_config_sensitive(self):
        c1 = Candidate("matmul", (64, 64, 64), {"BLOCK_M": 32})
        c2 = Candidate("matmul", (64, 64, 64), {"BLOCK_M": 32})
        c3 = Candidate("matmul", (64, 64, 64), {"BLOCK_M": 64})
        assert c1.candidate_id == c2.candidate_id
        assert c1.candidate_id != c3.candidate_id


class TestActions:
    def test_mask_and_apply(self):
        space = small_space()
        cfg = {"BLOCK_M": 16, "num_warps": 8, "dtype": "float32"}
        mask = valid_action_mask(space, cfg)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask[STOP_ACTION_INDEX]
        for i, act in enumerate(ACTION_CATALOG):
            if not mask[i]:
                continue
            new = apply_action(space, cfg, i)
            if act.is_stop:
                assert new is None
            else:
                space.validate(new)
                assert new != cfg

    def test_boundary_masked(self):
        space = small_space()
        cfg = {"BLOCK_M": 64, "num_warps": 2, "dtype": "float32"}
        mask = valid_action_mask(space, cfg)
        for i, act in enumerate(ACTION_CATALOG):
            if act.param == "BLOCK_M" and act.direction > 0:
                assert not mask[i]  # already at max
            if act.param == "num_warps" and act.direction < 0:
                assert not mask[i]  # already at min
            if act.param == "BLOCK_N":
                assert not mask[i]  # absent from space
