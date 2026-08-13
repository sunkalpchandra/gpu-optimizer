"""Graph intermediate representation of tensor programs.

Every optimization task is represented as a small dataflow graph of tensor
operations.  The graph is the *program representation* consumed by the learned
encoders (see ``optimizer/policy/encoders.py``): each node exposes a fixed-size
numeric feature vector, and edges express dataflow dependencies.

The IR is deliberately compact — enough structure for representation learning
(op identity, shapes, dtypes, reduction axes, dependencies), without trying to
be a full compiler IR.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

# Fixed operation vocabulary.  Order matters: it defines the one-hot layout of
# node features, so append new ops at the end rather than reordering.
OP_VOCAB: tuple[str, ...] = (
    "input",
    "output",
    "matmul",
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "exp",
    "sqrt",
    "rsqrt",
    "tanh",
    "scale",
    "reduce_sum",
    "reduce_max",
    "reduce_mean",
    "broadcast",
    "where",
)

DTYPE_VOCAB: tuple[str, ...] = ("float32", "float16", "bfloat16")

_MAX_RANK = 4  # feature vectors reserve slots for up to 4D shapes


@dataclass
class OpNode:
    """A single tensor operation in the program graph."""

    node_id: int
    op: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    inputs: tuple[int, ...] = ()
    # Axes (of this node's *input*) reduced away, e.g. (1,) for a row reduction.
    reduction_axes: tuple[int, ...] = ()
    # Free-form attributes (broadcasting info, scale constants, ...).
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in OP_VOCAB:
            raise ValueError(f"unknown op {self.op!r}; extend OP_VOCAB")
        if self.dtype not in DTYPE_VOCAB:
            raise ValueError(f"unknown dtype {self.dtype!r}")
        if len(self.shape) > _MAX_RANK:
            raise ValueError(f"rank {len(self.shape)} > {_MAX_RANK} unsupported")

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def feature_vector(self) -> np.ndarray:
        """Encode this node as a fixed-size float vector.

        Layout: [op one-hot | dtype one-hot | rank/4 | log2(numel)/40 |
                 padded log2(dims)/20 | is_reduction | reduction axis mask]
        """
        op_oh = np.zeros(len(OP_VOCAB), dtype=np.float32)
        op_oh[OP_VOCAB.index(self.op)] = 1.0
        dt_oh = np.zeros(len(DTYPE_VOCAB), dtype=np.float32)
        dt_oh[DTYPE_VOCAB.index(self.dtype)] = 1.0

        rank = np.array([len(self.shape) / _MAX_RANK], dtype=np.float32)
        log_numel = np.array([np.log2(max(self.numel, 1)) / 40.0], dtype=np.float32)
        dims = np.zeros(_MAX_RANK, dtype=np.float32)
        for i, d in enumerate(self.shape):
            dims[i] = np.log2(max(d, 1)) / 20.0

        is_red = np.array([1.0 if self.reduction_axes else 0.0], dtype=np.float32)
        red_mask = np.zeros(_MAX_RANK, dtype=np.float32)
        for a in self.reduction_axes:
            if 0 <= a < _MAX_RANK:
                red_mask[a] = 1.0

        return np.concatenate([op_oh, dt_oh, rank, log_numel, dims, is_red, red_mask])


NODE_FEATURE_DIM = len(OP_VOCAB) + len(DTYPE_VOCAB) + 1 + 1 + _MAX_RANK + 1 + _MAX_RANK


class ProgramGraph:
    """A dataflow graph of :class:`OpNode` with builder helpers.

    Nodes are stored in insertion order, which is required to be topological
    (an op may only reference previously added nodes).
    """

    def __init__(self, name: str = "program") -> None:
        self.name = name
        self.nodes: list[OpNode] = []

    # ------------------------------------------------------------- builders
    def add(
        self,
        op: str,
        shape: Sequence[int],
        *,
        inputs: Sequence[OpNode | int] = (),
        dtype: str = "float32",
        reduction_axes: Sequence[int] = (),
        **attrs,
    ) -> OpNode:
        ids = tuple(n.node_id if isinstance(n, OpNode) else int(n) for n in inputs)
        for i in ids:
            if not 0 <= i < len(self.nodes):
                raise ValueError(f"input id {i} not yet defined (topological order required)")
        node = OpNode(
            node_id=len(self.nodes),
            op=op,
            shape=tuple(int(s) for s in shape),
            dtype=dtype,
            inputs=ids,
            reduction_axes=tuple(reduction_axes),
            attrs=dict(attrs),
        )
        self.nodes.append(node)
        return node

    def input(self, shape: Sequence[int], dtype: str = "float32", **attrs) -> OpNode:
        return self.add("input", shape, dtype=dtype, **attrs)

    # ------------------------------------------------------------ accessors
    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[OpNode]:
        return iter(self.nodes)

    def edges(self) -> list[tuple[int, int]]:
        """Directed (producer, consumer) pairs."""
        return [(src, node.node_id) for node in self.nodes for src in node.inputs]

    # ------------------------------------------------------------ encodings
    def node_features(self) -> np.ndarray:
        """(num_nodes, NODE_FEATURE_DIM) feature matrix."""
        if not self.nodes:
            return np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)
        return np.stack([n.feature_vector() for n in self.nodes])

    def adjacency(self) -> np.ndarray:
        """Dense adjacency matrix A[i, j] = 1 iff edge i→j (producer→consumer)."""
        n = len(self.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        for src, dst in self.edges():
            adj[src, dst] = 1.0
        return adj

    def summary(self) -> str:
        lines = [f"ProgramGraph({self.name!r}, {len(self.nodes)} nodes)"]
        for n in self.nodes:
            deps = f" <- {list(n.inputs)}" if n.inputs else ""
            red = f" reduce{list(n.reduction_axes)}" if n.reduction_axes else ""
            lines.append(f"  [{n.node_id:2d}] {n.op:<12} {n.dtype:<9} {list(n.shape)}{red}{deps}")
        return "\n".join(lines)
