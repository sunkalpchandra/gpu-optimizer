"""Discrete transformation actions for the RL policy.

The policy edits a kernel configuration through a fixed global catalog of
structured actions ("increase BLOCK_M", "decrease num_warps", "switch
precision", ... , STOP).  A fixed catalog keeps the policy output head a
constant size across tasks; per-state validity masks disable actions that
don't apply (parameter absent from the task's space, or already at the edge
of its choice list).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compiler.transformations.space import (
    GLOBAL_NUMERIC_PARAMS,
    Config,
    ParamSpace,
)

# Params the RL policy may edit.  dtype/strategy are ordered categorical
# choices in each task's ParamSpace, so inc/dec traverses them uniformly.
TUNABLE_PARAMS: tuple[str, ...] = GLOBAL_NUMERIC_PARAMS + ("dtype", "strategy")


@dataclass(frozen=True)
class Action:
    param: str          # parameter to edit, or "STOP"
    direction: int      # +1 / -1 step through the ordered choices; 0 for STOP

    @property
    def is_stop(self) -> bool:
        return self.param == "STOP"

    def describe(self) -> str:
        if self.is_stop:
            return "STOP"
        arrow = "+" if self.direction > 0 else "-"
        return f"{self.param}{arrow}"


ACTION_CATALOG: tuple[Action, ...] = tuple(
    Action(p, d) for p in TUNABLE_PARAMS for d in (+1, -1)
) + (Action("STOP", 0),)

NUM_ACTIONS = len(ACTION_CATALOG)
STOP_ACTION_INDEX = NUM_ACTIONS - 1


def valid_action_mask(space: ParamSpace, config: Config) -> np.ndarray:
    """Boolean mask over :data:`ACTION_CATALOG` for the given state."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for i, a in enumerate(ACTION_CATALOG):
        if a.is_stop:
            mask[i] = True
            continue
        if a.param not in space:
            continue
        spec = space[a.param]
        if spec.is_continuous:
            continue  # continuous params are handled by the Gaussian head
        idx = spec.index_of(config[a.param])
        j = idx + a.direction
        mask[i] = 0 <= j < len(spec.choices)  # type: ignore[arg-type]
    return mask


def apply_action(space: ParamSpace, config: Config, action_index: int) -> Config | None:
    """Apply a catalog action; returns the new config, or ``None`` for STOP.

    Raises ``ValueError`` if the action is invalid in this state (callers
    should sample under :func:`valid_action_mask`).
    """
    action = ACTION_CATALOG[action_index]
    if action.is_stop:
        return None
    if action.param not in space:
        raise ValueError(f"{action.describe()}: param not in space {space.name!r}")
    spec = space[action.param]
    idx = spec.index_of(config[action.param])
    j = idx + action.direction
    if not (0 <= j < len(spec.choices)):  # type: ignore[arg-type]
        raise ValueError(f"{action.describe()}: at boundary")
    new = dict(config)
    new[action.param] = spec.choices[j]  # type: ignore[index]
    return new
