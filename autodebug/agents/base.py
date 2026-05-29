"""Base class for all AutoDebug agents.

Model selection via environment variables:
    AUTODEBUG_MODEL          — model ID (default: claude-sonnet-4-6)
    AUTODEBUG_MODEL_PROVIDER — provider hint passed to init_chat_model
                               (anthropic | openai | ollama | ...)
                               Leave unset to let LangChain auto-detect.

OpenRouter example:
    AUTODEBUG_MODEL=anthropic/claude-3.5-sonnet
    AUTODEBUG_MODEL_PROVIDER=openai
    OPENAI_API_KEY=sk-or-v1-...
    OPENAI_API_BASE=https://openrouter.ai/api/v1
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # load .env from cwd or any parent directory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

_DEFAULT_MODEL    = "claude-sonnet-4-6"
_DEFAULT_PROVIDER = "anthropic"


class BudgetExceeded(Exception):
    """Raised when an agent exhausts its time, token, or cost budget."""


class Budget:
    """Tracks elapsed time, token spend, and USD cost; raises BudgetExceeded when any limit is hit."""

    def __init__(
        self,
        time_seconds: int | None,
        tokens: int | None,
        cost_usd: float | None = None,
        cost_per_1k_input_tokens: float | None = None,
        cost_per_1k_output_tokens: float | None = None,
    ) -> None:
        self._deadline = time.monotonic() + time_seconds if time_seconds else None
        self._token_limit = tokens
        self._cost_limit = cost_usd
        self._tokens_used = 0
        self._cost_used = 0.0
        self._input_rate  = (cost_per_1k_input_tokens  or 0.0) / 1000
        self._output_rate = (cost_per_1k_output_tokens or 0.0) / 1000

    def add_tokens(self, input_tokens: int, output_tokens: int = 0) -> float:
        """Track tokens and cost. Returns the USD cost for this call."""
        self._tokens_used += input_tokens + output_tokens
        cost = input_tokens * self._input_rate + output_tokens * self._output_rate
        self._cost_used += cost
        return cost

    @property
    def cost_used(self) -> float:
        return self._cost_used

    def check(self) -> None:
        if self._deadline and time.monotonic() > self._deadline:
            raise BudgetExceeded(
                f"Time budget exceeded ({self._tokens_used} tokens, ${self._cost_used:.4f} spent)"
            )
        if self._token_limit and self._tokens_used > self._token_limit:
            raise BudgetExceeded(
                f"Token budget exceeded ({self._tokens_used} > {self._token_limit})"
            )
        if self._cost_limit and self._cost_used > self._cost_limit:
            raise BudgetExceeded(
                f"Cost budget exceeded (${self._cost_used:.4f} > ${self._cost_limit})"
            )


def _build_model(
    model_id: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
):
    mid = model_id or os.getenv("AUTODEBUG_MODEL", _DEFAULT_MODEL)
    prv = provider or os.getenv("AUTODEBUG_MODEL_PROVIDER", _DEFAULT_PROVIDER)
    kwargs: dict = {}
    # base_url: explicit arg wins over env var (for OpenRouter / Ollama)
    resolved_base = base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_BASE_URL")
    if resolved_base and prv == "openai":
        kwargs["base_url"] = resolved_base
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(mid, model_provider=prv, **kwargs)


class BaseAgent(ABC):
    def __init__(
        self,
        model_id: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        base_url: str | None = None,
        time_budget_seconds: int | None = None,
        token_budget: int | None = None,
        cost_budget_usd: float | None = None,
        cost_per_1k_input_tokens: float | None = None,
        cost_per_1k_output_tokens: float | None = None,
        max_retries: int = 0,
    ):
        self.llm = _build_model(model_id, provider, temperature, base_url)
        self._time_budget_seconds = time_budget_seconds
        self._token_budget = token_budget
        self._cost_budget_usd = cost_budget_usd
        self._cost_per_1k_input = cost_per_1k_input_tokens
        self._cost_per_1k_output = cost_per_1k_output_tokens
        self._max_retries = max_retries

    def _new_budget(self) -> Budget:
        return Budget(
            self._time_budget_seconds,
            self._token_budget,
            self._cost_budget_usd,
            self._cost_per_1k_input,
            self._cost_per_1k_output,
        )

    # ------------------------------------------------------------------
    # Unified chat helper — hides LangChain message types from subclasses
    # ------------------------------------------------------------------

    def _chat(
        self,
        messages: list[BaseMessage],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> AIMessage:
        model = self.llm.bind_tools(tools) if tools else self.llm
        all_messages: list[BaseMessage] = (
            [SystemMessage(content=system)] + messages if system else messages
        )
        response = model.invoke(all_messages)
        assert isinstance(response, AIMessage)
        return response

    # ------------------------------------------------------------------
    # Convenience constructors for message types
    # ------------------------------------------------------------------

    @staticmethod
    def human(content: str) -> HumanMessage:
        return HumanMessage(content=content)

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> ToolMessage:
        return ToolMessage(content=content, tool_call_id=tool_call_id)

    @staticmethod
    def invoke_tool(tool_map: dict, tc: dict) -> ToolMessage:
        name = tc["name"]
        if name not in tool_map:
            available = ", ".join(sorted(tool_map))
            content = f"Error: unknown tool '{name}'. Available tools: {available}"
        else:
            content = str(tool_map[name].invoke(tc["args"]))
        return ToolMessage(content=content, tool_call_id=tc["id"])

    @staticmethod
    def _usage(response: AIMessage) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) from response metadata."""
        meta = response.usage_metadata or {}
        return meta.get("input_tokens", 0), meta.get("output_tokens", 0)

    @staticmethod
    def count_tokens(response: AIMessage) -> int:
        meta = response.usage_metadata or {}
        return meta.get("input_tokens", 0) + meta.get("output_tokens", 0)

    @abstractmethod
    def run(self, state) -> object:
        """Execute this agent and return updated state."""
        ...
