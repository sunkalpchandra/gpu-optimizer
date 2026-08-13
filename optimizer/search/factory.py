"""Searcher registry/factory: algorithm name → constructed searcher.

Later stages (evolutionary, bayesian, rl, hybrid) register here; the
experiment runner and CLI only ever reference algorithm names.
"""

from __future__ import annotations

from collections.abc import Callable

from optimizer.search.base import SearchContext, Searcher

_BUILDERS: dict[str, Callable[..., Searcher]] = {}


def register_searcher(name: str):
    def deco(builder: Callable[..., Searcher]):
        _BUILDERS[name] = builder
        return builder
    return deco


def available_searchers() -> list[str]:
    _import_all()
    return sorted(_BUILDERS)


def make_searcher(name: str, ctx: SearchContext, **params) -> Searcher:
    _import_all()
    if name not in _BUILDERS:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](ctx, **params)


def _import_all() -> None:
    """Import all strategy modules so their registrations run."""
    import importlib

    for mod in ("optimizer.search.random_search", "optimizer.search.grid_search",
                "optimizer.evolutionary.ga", "optimizer.search.bayesian",
                "optimizer.rl.searcher", "optimizer.search.hybrid"):
        try:
            importlib.import_module(mod)
        except ImportError:  # stages not yet built / optional deps missing
            pass
