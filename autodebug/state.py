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

    # --- Input ---
    repo_url: str
    bug_report: str
    known_good_commit: Optional[str] = None
    github_issue_url: Optional[str] = None

    # --- Runtime ---
    stage: PipelineStage = PipelineStage.INIT
    repo_local_path: Optional[str] = None
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

    class Config:
        use_enum_values = True
