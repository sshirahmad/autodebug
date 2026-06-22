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
    evidence: str = ""   # runtime evidence the agent observed (postmortem/inspect_at)
    fix_plan: str = ""   # the concrete change the fixer should EXECUTE (not re-derive)
    # Ranked alternative hypotheses to fall back to if the fix for the primary
    # one fails — so a loop-back EXPLORES a different cause instead of re-deriving
    # the same (failed) one. See DebugState.hypothesis_attempts for what's been tried.
    alternatives: list[str] = Field(default_factory=list)


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
    # Production-real environment (NOT test metadata): the deployment's pinned
    # dependencies and setup step, used at clone time so the agent runs against the
    # versions the code actually expects.
    requirements: Optional[str] = None
    setup_command: Optional[str] = None

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

    # The hypothesis→fix attempt "tree": one node per fix attempt, recording the
    # hypothesis tried, a digest of the patch, and the outcome ("pass"/"fail").
    # Lets a loop-back avoid re-trying a hypothesis+patch that already failed and
    # steer root_cause toward an untried alternative.
    hypothesis_attempts: list[dict] = Field(default_factory=list)

    # --- Metadata ---
    messages: list[dict] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    class Config:
        use_enum_values = True
