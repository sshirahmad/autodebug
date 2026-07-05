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

    # We deviate from the Phoenix docs' `auto_instrument=True` recipe because
    # arize-phoenix (the server, installed in this env so `phoenix serve` works)
    # hard-depends on openinference-instrumentation-openai. With both LangChain
    # and OpenAI instrumentors active, the OpenAI tracer uses OTel context
    # propagation while the LangChain tracer uses callbacks — they can't see
    # each other, so every LLM call gets an extra orphan ChatCompletion root
    # span. Activating LangChain explicitly avoids the duplicate.
    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # HITL resume (Command(resume=…)) makes newer LangChain fire an `on_resume`
    # callback that openinference's tracer (<=0.1.66) doesn't implement, logging a
    # noisy but harmless AttributeError on every resume. Add a no-op so resumes stay
    # quiet. Best-effort — never let this break tracing setup.
    try:
        from openinference.instrumentation.langchain._tracer import OpenInferenceTracer

        if not hasattr(OpenInferenceTracer, "on_resume"):
            OpenInferenceTracer.on_resume = lambda self, *a, **k: None
    except Exception:  # noqa: BLE001
        pass

    _instrumented = True
    print(f"[autodebug] Phoenix tracing enabled → {endpoint}")
