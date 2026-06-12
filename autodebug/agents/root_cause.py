"""Root Cause Agent — explains WHY the bug happens, backed by observed evidence.

It returns a structured `RootCauseReport` (via create_agent's response_format)
rather than calling a submit tool. The report REQUIRES an `evidence` field, and
the driver rejects any report produced without actually running an inspection
tool (run_repro_with_traceback / inspect_at) — forcing the agent to observe the
failure instead of speculating.
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from autodebug.agents.base import (
    Budget, BudgetExceeded, attempt_trajectory, build_model, budget_middleware,
    maybe_optimize_prompt, model_retry_middleware, planning_middleware,
    retry_feedback, summarization_middleware, tool_call_limit_middleware,
)
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage, RootCauseReport, RootCauseResult

_INSPECTION_TOOLS = {"run_repro_with_traceback", "inspect_at"}


def _used_inspection_tool(messages: list) -> bool:
    """Did the agent actually observe the failure at runtime (not just read code)?"""
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("name") in _INSPECTION_TOOLS:
                    return True
    return False


def run_root_cause(state: DebugState, *, registry) -> DebugState:
    """Drive the root_cause agent until it returns an evidence-backed report."""
    assert state.repo_volume
    assert state.bisect
    assert state.repro

    cfg = registry.get_config("root_cause")
    system_prompt = registry.system_prompt("root_cause")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    # Refine rather than regenerate: a prior hypothesis means the fix based on it
    # failed, so the manager looped back. Surface it so the agent corrects what was
    # wrong instead of re-deriving (and possibly re-deriving the same mistake).
    refine_note = ""
    if state.root_cause is not None:
        refine_note = (
            "\nNOTE: a previous root-cause hypothesis was produced but the fix based "
            "on it FAILED verification, so it was wrong or incomplete. Previous "
            f"hypothesis:\n{state.root_cause.hypothesis}\n{state.root_cause.summary[:800]}\n"
            "Identify what that analysis got wrong and produce a corrected hypothesis "
            "(in particular, distinguish the real defect from any environment/import "
            "artifact in the sandbox).\n"
        )

    crashed = False
    run_error: str | None = None
    with Sandbox(volume=state.repo_volume) as sandbox:
        initial_run = sandbox.run_script(state.repro.repro_script)
        initial_text = (
            f"Bug report (the symptom your hypothesis MUST explain):\n"
            f"{state.bug_report}\n\n"
            f"Culprit commit: {state.bisect.commit_message}\n"
            f"SHA: {state.bisect.culprit_commit}\n\n"
            f"Commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
            f"Reproduction script output:\n```\n{initial_run.output[-3000:]}\n```\n"
            f"{refine_note}\n"
            "Identify the root cause of the bug described above. You MUST observe "
            "the failure at runtime — call run_repro_with_traceback or inspect_at — "
            "before answering, and base your `evidence` on what you actually saw."
        )

        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools(
                "root_cause",
                sandbox=sandbox,
                parent_sha=f"{state.bisect.culprit_commit}^",
                repro_script=state.repro.repro_script,
            )
            saver = InMemorySaver()
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                response_format=RootCauseReport,
                checkpointer=saver,
                middleware=(
                    budget_middleware(budget)
                    + model_retry_middleware()
                    + planning_middleware()
                    + summarization_middleware(cfg.model, cfg.provider)
                    + tool_call_limit_middleware(cfg.tool_call_limits)
                ),
            )
            invoke_config = {
                "recursion_limit": sys.maxsize,
                "configurable": {"thread_id": f"root_cause-{attempt}"},
            }

            crashed = False
            final: dict | None = None
            try:
                final = agent.invoke(
                    {"messages": [HumanMessage(content=initial_text)]},
                    config=invoke_config,
                )
            except BudgetExceeded:
                crashed = True
            except Exception as exc:  # noqa: BLE001 — degrade, don't abort the pipeline
                crashed = True
                run_error = f"{type(exc).__name__}: {str(exc)[:300]}"

            state.total_tokens += budget.tokens_used
            state.total_cost += budget.cost_used
            state.total_llm_calls += budget.calls

            report = (final or {}).get("structured_response")
            messages = (final or {}).get("messages") or attempt_trajectory(agent, invoke_config)

            if isinstance(report, RootCauseReport):
                if not _used_inspection_tool(messages):
                    # Speculation, not observation — reject and force a real probe.
                    run_error = "submitted a hypothesis without observing the failure"
                    system_prompt += (
                        "\n\nYOU DID NOT OBSERVE THE FAILURE. Before answering you MUST "
                        "call run_repro_with_traceback or inspect_at and base `evidence` "
                        "on its actual output. A hypothesis without a real observation "
                        "is rejected."
                    )
                else:
                    state.root_cause = RootCauseResult(
                        summary=report.summary,
                        relevant_lines=report.relevant_lines,
                        hypothesis=report.hypothesis,
                        evidence=report.evidence,
                    )
                    state.stage = PipelineStage.FIX
                    return state

            if attempt >= cfg.max_retries:
                break

            # Retrying (crash or rejected report): optimize the prompt from history.
            system_prompt = maybe_optimize_prompt(
                system_prompt,
                attempt_trajectory(agent, invoke_config),
                retry_feedback("determining the root cause with observed evidence"),
                model_id=cfg.model, provider=cfg.provider,
            )

    state.stage = PipelineStage.FAILED
    state.error = (
        f"RootCauseAgent: {run_error}" if run_error else
        "RootCauseAgent: budget exceeded with no report" if crashed else
        "RootCauseAgent: could not determine root cause"
    )
    return state
