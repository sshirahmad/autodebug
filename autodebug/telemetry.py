"""Phoenix Arize OpenTelemetry tracing for AutoDebug.

Instruments all LangChain/LangGraph calls automatically — model responses,
tool calls, chain execution — with no changes needed in agent code.

Activation (opt-in via env vars):
    AUTODEBUG_PHOENIX_ENABLED=true
    AUTODEBUG_PHOENIX_ENDPOINT=http://localhost:6006   # default

Run Phoenix locally before starting AutoDebug:
    python -m phoenix.server.main serve
    # UI at http://localhost:6006
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # must run before reading any env vars below

_instrumented = False


def setup_tracing(project_name: str = "autodebug") -> None:
    """Register Phoenix OTel tracing. Safe to call multiple times."""
    global _instrumented
    if _instrumented:
        return

    # Read env vars here (not at module level) so load_dotenv() has already run
    enabled  = os.getenv("AUTODEBUG_PHOENIX_ENABLED", "false").lower() == "true"
    endpoint = os.getenv("AUTODEBUG_PHOENIX_ENDPOINT", "http://localhost:6006")

    if not enabled:
        return

    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        raise ImportError(
            "Phoenix tracing dependencies are not installed. Run:\n"
            "  pip install 'autodebug[tracing]'"
        )

    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        verbose=False,
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    _instrumented = True
    print(f"[autodebug] Phoenix tracing enabled → {endpoint}")
