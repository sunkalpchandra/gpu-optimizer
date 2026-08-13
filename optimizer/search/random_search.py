"""Random search baseline: uniform samples from the parameter space."""

from __future__ import annotations

from compiler.transformations.space import Candidate
from optimizer.search.base import SearchContext, Searcher


class RandomSearcher(Searcher):
    name = "random"

    def __init__(self, ctx: SearchContext, max_proposals: int | None = None) -> None:
        super().__init__(ctx)
        self.max_proposals = max_proposals
        self._proposed = 0

    def propose(self, n: int) -> list[Candidate]:
        if self.max_proposals is not None:
            n = min(n, self.max_proposals - self._proposed)
        out = [self.ctx.candidate(self.ctx.space.sample(self.ctx.rng), "random")
               for _ in range(max(n, 0))]
        self._proposed += len(out)
        return out


from optimizer.search.factory import register_searcher


@register_searcher("random")
def _build(ctx: SearchContext, max_proposals: int | None = None, **_ignored) -> RandomSearcher:
    return RandomSearcher(ctx, max_proposals=max_proposals)
