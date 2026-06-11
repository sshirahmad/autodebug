"""Shared pipeline state passed between all LangGraph nodes."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    INIT = "init"
    REPRO = "repro"
    BISECT = "bisect"
    ROOT_CAUSE = "root_cause"
    FIX = "fix"
    DONE = "done"
    FAILED = "failed"


class ReproResult(BaseModel):
    repro_script: str
    error_output: str
    confirmed: bool


class BisectResult(BaseModel):
    culprit_commit: str
    commit_message: str
    commit_diff: str
    steps_taken: int


class RootCauseResult(BaseModel):
    summary: str
    relevant_lines: list[str]
    hypothesis: str


class FixResult(BaseModel):
    patch: str
    attempts: int
    test_output: str
    pr_url: Optional[str] = None


class DebugState(BaseModel):
    """Single state object threaded through the entire LangGraph pipeline."""

    # --- Input (production-real only; benchmark/test metadata lives in the eval
    # harness, never here — the agents must run as they would in production) ---
    repo_url: str
    bug_report: str
    known_good_commit: Optional[str] = None  # optional real hint: a version where it worked
    github_issue_url: Optional[str] = None
    ref: Optional[str] = None                # commit/branch to check out (defaults to HEAD)

    # --- Runtime ---
    stage: PipelineStage = PipelineStage.INIT
    manager_phase: Optional[str] = None  # last Manager FSM phase (manager mode only)
    repo_volume: Optional[str] = None  # Docker volume name holding the cloned repo
    error: Optional[str] = None

    # --- Agent outputs (filled in as pipeline progresses) ---
    repro: Optional[ReproResult] = None
    bisect: Optional[BisectResult] = None
    root_cause: Optional[RootCauseResult] = None
    fix: Optional[FixResult] = None

    # --- Metadata ---
    messages: list[dict] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    class Config:
        use_enum_values = True
