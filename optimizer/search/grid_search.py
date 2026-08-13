"""Grid search baseline: exhaustive sweep, optionally shuffled and capped.

Practical only for small spaces; the loop stops when the grid is exhausted.
"""

from __future__ import annotations

from compiler.transformations.space import Candidate
from optimizer.search.base import SearchContext, Searcher


class GridSearcher(Searcher):
    name = "grid"

    def __init__(self, ctx: SearchContext, shuffle: bool = True,
                 limit: int | None = None) -> None:
        super().__init__(ctx)
        grid = ctx.space.grid(limit=None)
        if shuffle:
            ctx.rng.shuffle(grid)
        if limit is not None:
            grid = grid[:limit]
        self._queue = grid

    def propose(self, n: int) -> list[Candidate]:
        batch, self._queue = self._queue[:n], self._queue[n:]
        return [self.ctx.candidate(cfg, "grid") for cfg in batch]


from optimizer.search.factory import register_searcher


@register_searcher("grid")
def _build(ctx: SearchContext, shuffle: bool = True, limit: int | None = None,
           **_ignored) -> GridSearcher:
    return GridSearcher(ctx, shuffle=shuffle, limit=limit)
