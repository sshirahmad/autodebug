"""Shared infrastructure: budget tracking, middleware, and model factory.

Each agent is built with LangChain's `create_agent` — see autodebug/agents/{repro,bisect,...}.py.
The agent's tool-execution loop is handled by create_agent; budgets are enforced
via the `budget_middleware` (a pair of before_model/after_model hooks).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.agents.middleware import AgentState, after_model, before_model
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime

load_dotenv()

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_PROVIDER = "anthropic"


class BudgetExceeded(Exception):
    """Raised by budget_middleware when any limit is hit."""


@dataclass
class Budget:
    time_seconds: int | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    tokens_used: int = field(default=0, init=False)
    cost_used: float = field(default=0.0, init=False)
    _deadline: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._deadline = (time.monotonic() + self.time_seconds) if self.time_seconds else None

    def add_tokens(self, input_tokens: int, output_tokens: int = 0) -> float:
        self.tokens_used += input_tokens + output_tokens
        cost = (
            input_tokens * self.cost_per_1k_input / 1000
            + output_tokens * self.cost_per_1k_output / 1000
        )
        self.cost_used += cost
        return cost

    def check(self) -> None:
        if self._deadline and time.monotonic() > self._deadline:
            raise BudgetExceeded(
                f"Time budget exceeded ({self.tokens_used} tokens, ${self.cost_used:.4f})"
            )
        if self.tokens is not None and self.tokens_used > self.tokens:
            raise BudgetExceeded(
                f"Token budget exceeded ({self.tokens_used} > {self.tokens})"
            )
        if self.cost_usd is not None and self.cost_used > self.cost_usd:
            raise BudgetExceeded(
                f"Cost budget exceeded (${self.cost_used:.4f} > ${self.cost_usd})"
            )

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        return cls(
            time_seconds=cfg.time_budget_seconds,
            tokens=cfg.token_budget,
            cost_usd=cfg.cost_budget_usd,
            cost_per_1k_input=cfg.cost_per_1k_input_tokens or 0.0,
            cost_per_1k_output=cfg.cost_per_1k_output_tokens or 0.0,
        )


def budget_middleware(budget: Budget) -> list:
    """Two-piece middleware: check budget before each model call, track tokens after."""

    @before_model
    def _check_budget(state: AgentState, runtime: Runtime) -> None:
        budget.check()
        return None

    @after_model
    def _track_tokens(state: AgentState, runtime: Runtime) -> None:
        msg = state["messages"][-1]
        meta = getattr(msg, "usage_metadata", None) or {}
        budget.add_tokens(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        return None

    return [_check_budget, _track_tokens]


def build_model(
    model_id: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
):
    """Construct a chat model from env vars / explicit overrides.

    OpenRouter example:
        AUTODEBUG_MODEL=openrouter/owl-alpha
        AUTODEBUG_MODEL_PROVIDER=openai
        OPENAI_API_BASE=https://openrouter.ai/api/v1
    """
    mid = model_id or os.getenv("AUTODEBUG_MODEL", _DEFAULT_MODEL)
    prv = provider or os.getenv("AUTODEBUG_MODEL_PROVIDER", _DEFAULT_PROVIDER)
    kwargs: dict = {}
    resolved_base = base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_BASE_URL")
    if resolved_base and prv == "openai":
        kwargs["base_url"] = resolved_base
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(mid, model_provider=prv, **kwargs)
