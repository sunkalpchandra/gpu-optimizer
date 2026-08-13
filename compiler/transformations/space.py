"""Structured kernel-transformation space.

Candidates are *structured configurations* (tiling, warps, pipeline stages,
precision, strategy flags) over a per-task :class:`ParamSpace` — the policy
never emits free-form source code.  The same structures double as:

- the mutation space for the evolutionary optimizer,
- the discrete action space for the RL policy (see ``actions.py``),
- the feature encoding consumed by the learned performance model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ConfigValue = int | float | str
Config = dict[str, ConfigValue]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable kernel parameter.

    ``choices`` is an *ordered* tuple: adjacent entries are considered
    neighbors, which defines local mutations and the RL inc/dec actions.
    Continuous parameters use ``low``/``high`` instead of ``choices`` and are
    handled by the policy's Gaussian head.
    """

    name: str
    choices: tuple[ConfigValue, ...] | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if (self.choices is None) == (self.low is None or self.high is None):
            raise ValueError(f"{self.name}: specify either choices or (low, high)")

    @property
    def is_continuous(self) -> bool:
        return self.choices is None

    def sample(self, rng: random.Random) -> ConfigValue:
        if self.is_continuous:
            return rng.uniform(self.low, self.high)  # type: ignore[arg-type]
        return rng.choice(self.choices)  # type: ignore[arg-type]

    def index_of(self, value: ConfigValue) -> int:
        if self.is_continuous:
            raise TypeError(f"{self.name} is continuous")
        return self.choices.index(value)  # type: ignore[union-attr]


class ParamSpace:
    """Ordered collection of :class:`ParamSpec` plus validity constraints."""

    def __init__(self, specs: Sequence[ParamSpec], name: str = "space") -> None:
        self.name = name
        self.specs: dict[str, ParamSpec] = {s.name: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError("duplicate parameter names")

    def __iter__(self) -> Iterator[ParamSpec]:
        return iter(self.specs.values())

    def __getitem__(self, name: str) -> ParamSpec:
        return self.specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    @property
    def discrete_specs(self) -> list[ParamSpec]:
        return [s for s in self if not s.is_continuous]

    def size(self) -> int:
        """Number of discrete configurations (continuous params excluded)."""
        n = 1
        for s in self.discrete_specs:
            n *= len(s.choices)  # type: ignore[arg-type]
        return n

    def sample(self, rng: random.Random) -> Config:
        return {s.name: s.sample(rng) for s in self}

    def grid(self, limit: int | None = None) -> list[Config]:
        """Full cartesian product over discrete params (continuous at midpoint)."""
        names = [s.name for s in self.discrete_specs]
        pools = [s.choices for s in self.discrete_specs]  # type: ignore[misc]
        out: list[Config] = []
        for combo in itertools.product(*pools):
            cfg: Config = dict(zip(names, combo))
            for s in self:
                if s.is_continuous:
                    cfg[s.name] = (s.low + s.high) / 2.0  # type: ignore[operator]
            out.append(cfg)
            if limit is not None and len(out) >= limit:
                break
        return out

    def validate(self, config: Config) -> None:
        for s in self:
            if s.name not in config:
                raise ValueError(f"missing param {s.name}")
            v = config[s.name]
            if s.is_continuous:
                if not (s.low <= float(v) <= s.high):  # type: ignore[operator]
                    raise ValueError(f"{s.name}={v} outside [{s.low}, {s.high}]")
            elif v not in s.choices:  # type: ignore[operator]
                raise ValueError(f"{s.name}={v!r} not in {s.choices}")

    def neighbors(self, config: Config) -> list[Config]:
        """All configs one discrete step away (the local-mutation neighborhood)."""
        out = []
        for s in self.discrete_specs:
            idx = s.index_of(config[s.name])
            for d in (-1, +1):
                j = idx + d
                if 0 <= j < len(s.choices):  # type: ignore[arg-type]
                    cfg = dict(config)
                    cfg[s.name] = s.choices[j]  # type: ignore[index]
                    out.append(cfg)
        return out


# --------------------------------------------------------------------------
# Global candidate encoding (shared across tasks)
# --------------------------------------------------------------------------
# Fixed slot layout so one surrogate/policy can embed candidates from any
# task.  Numeric params are encoded as log2(value)/16, strategy/dtype params
# one-hot.  Absent params get value 0 with presence-mask 0.

GLOBAL_NUMERIC_PARAMS: tuple[str, ...] = (
    "BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_SIZE", "BLOCK_D",
    "GROUP_M", "num_warps", "num_stages",
)
GLOBAL_DTYPE_CHOICES: tuple[str, ...] = ("float32", "float16", "bfloat16", "tf32")
GLOBAL_STRATEGY_CHOICES: tuple[str, ...] = ("loop", "atomic", "two_pass", "fused", "unfused")

CANDIDATE_FEATURE_DIM = (
    2 * len(GLOBAL_NUMERIC_PARAMS) + len(GLOBAL_DTYPE_CHOICES) + len(GLOBAL_STRATEGY_CHOICES)
)


def encode_config(config: Mapping[str, ConfigValue]) -> np.ndarray:
    """Encode a config dict into the fixed global feature layout."""
    vals = np.zeros(len(GLOBAL_NUMERIC_PARAMS), dtype=np.float32)
    mask = np.zeros(len(GLOBAL_NUMERIC_PARAMS), dtype=np.float32)
    for i, p in enumerate(GLOBAL_NUMERIC_PARAMS):
        if p in config:
            vals[i] = float(np.log2(max(float(config[p]), 1.0))) / 16.0
            mask[i] = 1.0
    dt = np.zeros(len(GLOBAL_DTYPE_CHOICES), dtype=np.float32)
    dtype = str(config.get("dtype", "float32"))
    if dtype in GLOBAL_DTYPE_CHOICES:
        dt[GLOBAL_DTYPE_CHOICES.index(dtype)] = 1.0
    strat = np.zeros(len(GLOBAL_STRATEGY_CHOICES), dtype=np.float32)
    strategy = str(config.get("strategy", ""))
    if strategy in GLOBAL_STRATEGY_CHOICES:
        strat[GLOBAL_STRATEGY_CHOICES.index(strategy)] = 1.0
    return np.concatenate([vals, mask, dt, strat])


@dataclass
class Candidate:
    """A concrete kernel configuration for one task instance."""

    task: str                       # registry key, e.g. "matmul"
    shape: tuple[int, ...]          # problem shape, e.g. (M, N, K)
    config: Config = field(default_factory=dict)
    parent_id: str | None = None    # search-tree lineage
    provenance: str = "manual"      # random | grid | evolutionary | bo | rl | manual

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(
            {"task": self.task, "shape": list(self.shape), "config": self.config},
            sort_keys=True, default=str,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    @property
    def dtype(self) -> str:
        return str(self.config.get("dtype", "float32"))

    def encode(self) -> np.ndarray:
        return encode_config(self.config)

    def mutated(self, config: Config, provenance: str) -> "Candidate":
        return Candidate(task=self.task, shape=self.shape, config=config,
                         parent_id=self.candidate_id, provenance=provenance)

    def describe(self) -> str:
        cfg = ", ".join(f"{k}={v}" for k, v in sorted(self.config.items()))
        return f"{self.task}{list(self.shape)} [{cfg}]"
