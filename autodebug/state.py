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
    pre_fix_commit: Optional[str] = None     # fixed_commit_id~1; clone is checked out here
    fixed_commit_id: Optional[str] = None    # the commit where the fix landed (for test-file sync)
    test_file: Optional[str] = None          # path to the relevant test file
    test_command: Optional[str] = None       # command to run targeted regression tests
    test_patch: Optional[str] = None         # explicit test-only diff to apply at clone

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
