"""Pydantic models for all AutoDebug configuration."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SandboxConfig(BaseModel):
    image: str = "autodebug-sandbox:latest"
    timeout_seconds: int = 300
    memory_limit: str = "512m"


class TracingConfig(BaseModel):
    enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006"
    project_name: str = "autodebug"


class AgentConfig(BaseModel):
    system_prompt: str
    tools: list[str]
    time_budget_seconds: int | None = None
    token_budget: int | None = None
    max_retries: int = 0
    model: str | None = None      # overrides AUTODEBUG_MODEL if set
    provider: str | None = None   # overrides AUTODEBUG_MODEL_PROVIDER if set
    extra: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
