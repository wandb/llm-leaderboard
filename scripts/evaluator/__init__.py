"""Lazy evaluator module exports.

Importing every evaluator at package import time pulls optional heavyweight
dependencies such as transformers, COMET, Docker clients, and benchmark-specific
packages even when the corresponding benchmark is disabled. Keep the historical
`from evaluator import foo` surface, but import each submodule only on first use.
"""

from importlib import import_module


__all__ = [
    "jaster",
    "jbbq",
    "mtbench",
    "jaster_translation",
    "toxicity",
    "jtruthfulqa",
    "aggregate",
    "bfcl",
    "swe_bench",
    "hallulens",
    "arc_agi",
    "hle",
    "m_ifeval",
    "agentic_math",
    "swebench_pro",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
