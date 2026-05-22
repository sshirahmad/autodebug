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
from abc import ABC, abstractmethod

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

_DEFAULT_MODEL    = "claude-sonnet-4-6"
_DEFAULT_PROVIDER = "anthropic"


def _build_model(model_id: str | None = None, provider: str | None = None):
    mid = model_id or os.getenv("AUTODEBUG_MODEL", _DEFAULT_MODEL)
    prv = provider or os.getenv("AUTODEBUG_MODEL_PROVIDER", _DEFAULT_PROVIDER)
    kwargs: dict = {}
    # OpenRouter / custom base URL support
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_BASE_URL")
    if base_url and prv == "openai":
        kwargs["base_url"] = base_url
    return init_chat_model(mid, model_provider=prv, **kwargs)


class BaseAgent(ABC):
    def __init__(self, model_id: str | None = None, provider: str | None = None):
        self.llm = _build_model(model_id, provider)

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
    def count_tokens(response: AIMessage) -> int:
        meta = response.usage_metadata or {}
        return meta.get("input_tokens", 0) + meta.get("output_tokens", 0)

    @abstractmethod
    def run(self, state) -> object:
        """Execute this agent and return updated state."""
        ...
