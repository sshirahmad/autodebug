from .pipeline import run_pipeline

__all__ = ["run_pipeline", "graph", "build_graph"]


def __getattr__(name):
    # Lazy: importing the interactive graph builds the Manager agent (model +
    # registry), which the eval/run_pipeline path doesn't need. Defer it to first
    # access of `graph`/`build_graph` (langgraph dev, CLI --stream, tests).
    if name in ("graph", "build_graph"):
        from . import interactive

        return getattr(interactive, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
