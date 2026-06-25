# AutoDebug

An end-to-end agentic debugger: given a repository and a bug report, it
**reproduces**, **bisects**, **root-causes**, and **fixes** the bug
automatically — then verifies the fix against a reproduction it built itself.

The agents see only what a human triager would: the repo and a natural-language
bug report. They are never given the benchmark's test — that is held out and
used only for independent scoring (see [Evaluation](#evaluation)).

---

## How it works

A **Manager** agent orchestrates four specialist sub-agents through a finite
state machine, calling each as a tool and deciding the next move from the signal
it returns:

```
                 ┌─────────────────────────────────────────────┐
                 │                  Manager (FSM)               │
                 │   init → reproduced → bisected → analyzed    │
                 │                    ↘  revising  ↙            │
                 └─────────────────────────────────────────────┘
   run_repro_agent     run_bisect_agent   run_root_cause_agent   run_fix_agent
        │                    │                    │                    │
   reproduce the        find the commit      explain WHY it       write & verify
   bug as a script      that introduced it   broke (observed      the patch against
   (the success         (git pickaxe /       at runtime)          the reproduction
   oracle)              bisect)
```

| Stage | Agent | Output |
|-------|-------|--------|
| **Reproduce** | `repro` | a minimal Python script that fails on the bug and passes once fixed — the oracle for the rest of the run |
| **Bisect** | `bisect` | the SHA of the commit that introduced the regression |
| **Root cause** | `root_cause` | a runtime-observed hypothesis + a concrete fix plan |
| **Fix** | `fix` | a patch that makes the reproduction (and targeted tests) pass |

If a fix fails verification the Manager enters **revising** and loops back to the
weakest link (repro, root cause, or another patch) until it succeeds or exhausts
its budget.

Setting `AUTODEBUG_MANAGER=0` falls back to a classic linear
`repro → bisect → root_cause → fix` pipeline with no Manager.

### Design notes

- **Isolated sandbox.** Every stage runs in a Docker container attached to a
  shared per-run volume; the host never touches the cloned files, so symlinks
  and on-disk state stay intact. One long-lived container per stage — filesystem
  state persists across tool calls within a stage; a hard per-command timeout
  keeps any single call from hanging the run.
- **Tool registry.** Any `make_<name>_tool` factory exported from
  [autodebug/tools/](autodebug/tools/) is auto-discovered; each agent's
  `config/agents/<name>.json` lists which tools it gets.
- **Prompt + budget config** live in [config/](config/) — system prompts in
  `config/prompts/*.yaml`, per-agent model/budget/tool settings in
  `config/agents/*.json` — so behavior is tunable without code changes.

---

## Setup

**Requirements:** Python ≥ 3.11, Docker, and an LLM API key.

```bash
# 1. Environment + dependencies
conda create -n autodebug python=3.11 && conda activate autodebug
pip install -e ".[dev,eval,tracing]"

# 2. Build the sandbox image the agents run code in
docker build -t autodebug-sandbox:latest ./docker/sandbox

# 3. Configure credentials and models
cp .env.example .env   # then edit — see Configuration below
```

---

## Usage

```bash
autodebug debug https://github.com/psf/black \
  --bug "black crashes in AWS Lambda: ProcessPoolExecutor fails when /dev/shm is unavailable" \
  --good <known-good-commit-or-date>     # optional, helps bisect
```

Pass `--stream` to watch progress live instead of waiting in silence — and to be
prompted for guidance if the run gets stuck (human-in-the-loop, see below). Add
`--tokens` to also stream the sub-agents' LLM output token-by-token:

```bash
autodebug debug https://github.com/psf/black --bug "..." --stream
autodebug debug https://github.com/psf/black --bug "..." --tokens   # live token stream
```

Or from Python:

```python
from autodebug.graph import run_pipeline

state = run_pipeline(
    repo_url="https://github.com/psf/black",
    bug_report="black crashes in AWS Lambda ...",
    known_good_commit=None,   # optional
)
print(state.stage, state.fix and state.fix.patch)
```

---

## Interactive serving (Agent Chat UI + human-in-the-loop)

The blocking CLI runs a bug to completion in silence. For an interactive workflow —
live progress, a chat UI, and the ability to step in when the agent gets stuck —
AutoDebug also ships as a **compiled LangGraph graph** ([autodebug/graph/interactive.py](autodebug/graph/interactive.py))
with a `messages` channel and a human-in-the-loop (HITL) review step.

```bash
# Install the serving extra (the LangGraph dev server)
pip install -e ".[serve]"

# Serve the graph on http://localhost:2024 (also prints a LangGraph Studio URL)
langgraph dev
```

`langgraph dev` reads [langgraph.json](langgraph.json), serves the `autodebug` graph
over the standard LangGraph protocol, and provides persistence so threads, streaming,
and interrupts work. Two ways to use it — **no custom backend needed**:

- **LangGraph Studio** — open the `smith.langchain.com/studio?baseUrl=…` URL printed
  by `langgraph dev`. It has a **config panel**, so set `repo_url` (and optional
  `ref`/`known_good`/`requirements`/…) there and type the bug report as the message.
  This is the quickest way to drive a run and watch its state.
- **Agent Chat UI** — point [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui)
  (a separate Next.js frontend) at `http://localhost:2024` with assistant id
  `autodebug`. (It has no config panel, so the run target must come from Studio or be
  parsed from the first message.)

**Streaming granularity.** The chat shows **milestone** messages (one per stage + the
failure/summary). For a **live token stream** of the sub-agents, set
`AUTODEBUG_STREAM_TOKENS=1` (or the CLI `--tokens` flag). Tokens are forwarded from
every sub-agent's model through the graph stream as LangGraph `custom` events
(`{"token": …}`) / inline CLI output. It's gated because the default and the eval
harness want clean, non-streaming behavior; the toggle uses a contextvar (not a
global), so a server can stream per-run without a race.

**Human-in-the-loop.** When a run exhausts its retries or is about to fail, the
graph's `human_review` node `interrupt()`s with a summary of what was tried and
why it failed, instead of silently returning a failure. Clients stream the run,
detect the `__interrupt__`, and resume with `Command(resume=<feedback>)` (Studio /
Agent Chat UI render this automatically; the CLI `--stream` prompts at the terminal).
The feedback is threaded into the next attempt's prompt, so the retry is **steered**,
not a blind replay. Each sub-agent keeps its own checkpointer threads (so
already-paid-for repro/bisect/root-cause are served from cache); only the top graph
interrupts.

> The non-interactive `run_pipeline` and the eval harness are unchanged — the graph
> is an additive surface that reuses the same stage runners.

---

## Configuration

All configuration is via environment variables (loaded from `.env`). The most
common ones — see [.env.example](.env.example) for the full list:

| Variable | Purpose | Default |
|----------|---------|---------|
| `AUTODEBUG_MODEL` / `AUTODEBUG_MODEL_PROVIDER` | LLM and provider (any LangChain `init_chat_model` target — Anthropic, OpenAI, OpenRouter, Ollama, …) | `claude-sonnet-4-6` / `anthropic` |
| `AUTODEBUG_FIX_MODEL` | Optional stronger model just for fix generation | — |
| `AUTODEBUG_MANAGER` | `1` = Manager FSM, `0` = linear pipeline | `1` |
| `AUTODEBUG_PROMPT_OPTIM` | Rewrite a failing agent's prompt from its trajectory on retry | `1` |
| `AUTODEBUG_MEMORY_ENABLED` | Memory **recall** via LangMem (memories are always written; this gates reads) | `0` |
| `SANDBOX_IMAGE` / `SANDBOX_TIMEOUT_SECONDS` / `SANDBOX_MEM_LIMIT` | Docker sandbox knobs | `autodebug-sandbox:latest` / `300` / `512m` |
| `GITHUB_TOKEN` | For reading issues / opening PRs | — |

Per-agent knobs (time/cost budget, model, allowed tools, tool-call limits) live
in `config/agents/*.json`; their system prompts are in `config/prompts/*.yaml`.

---

## Evaluation

The eval harness runs AutoDebug against a dataset of real bugs and scores each
stage independently — the agents never see the test.

```bash
# Run the whole dataset (BugsInPy-derived, 501 instances across 17 projects)
python eval/run_eval.py

# First N instances, or specific ones
python eval/run_eval.py eval/datasets/buginspy.json 10
python eval/run_eval.py --ids black-1,pandas-23
```

Metrics are printed and saved to `eval/results/run_<timestamp>.json`:
`repro_rate`, `bisect_accuracy`, `fix_rate`, plus token/cost/wall-time averages.

### Fix scoring distinguishes a bad fix from a broken harness

A fix is scored only after the harness proves the instance is **scoreable** with
a *gold baseline*: the official test must **pass at the fixed commit** and
**fail at the buggy commit**. If it doesn't — a missing test-only dependency, a
broken import, a wrong test command — the instance is flagged `harness_invalid`
and **excluded** from `fix_rate` instead of silently counting as a fix failure.
The metrics report `fix_scoreable` and `harness_invalid` counts separately, so a
0% fix rate always means the agent failed, not the environment.

To make instances scoreable, the sandbox installs the project's **test/dev**
dependencies (e.g. `.[test]`, `.[dev]`, dev-requirement files) on top of its
runtime deps, so the official test module can actually import.

Useful toggles: `AUTODEBUG_EVAL_BASELINE=0` disables gold-baseline validation
(falls back to output heuristics); `AUTODEBUG_PIP_INSTALL=0` skips dependency
installation.

---

## Tracing

AutoDebug auto-instruments every LangChain/LangGraph call (model responses, tool
calls, chain execution) with Phoenix / OpenTelemetry.

```bash
# 1. Start Phoenix (UI at http://localhost:6006)
python -m phoenix.server.main serve

# 2. Enable tracing for the run
export AUTODEBUG_PHOENIX_ENABLED=true

# 3. Inspect the latest pipeline runs from the CLI
python fetch_traces.py
```

---

## Testing

```bash
conda run -n autodebug python -m pytest tests/ -q
```

---

## Project layout

```
autodebug/
  agents/      manager + repro/bisect/root_cause/fix sub-agents
  tools/       auto-discovered make_<name>_tool factories
  sandbox/     Docker volume + long-lived container runner
  graph/       pipeline.py (imperative run_pipeline) + interactive.py (compiled graph + HITL)
  fsm.py       Manager finite state machine
  registry.py  config loading + tool building
  state.py     DebugState and per-stage result models
langgraph.json   serves the graph for `langgraph dev` / Studio / Agent Chat UI
config/
  agents/      per-agent model/budget/tool config (JSON)
  prompts/     per-agent system prompts (YAML)
eval/
  run_eval.py  evaluation harness + independent fix scoring
  datasets/    buginspy.json
.skills/       agent-loadable debugging skills (investigate, bisect-tricks, …)
docker/sandbox Dockerfile for the execution image
fetch_traces.py  pretty-print the latest Phoenix traces
```

---

## Contributing

Contributions are very welcome — agents, tools, sandbox improvements, eval
coverage, docs, and bug fixes alike. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for
setup, the test workflow, and code style. Good first issues include tidying lint
findings and adding eval coverage. Please open an issue to discuss anything
non-trivial before sending a large PR.

## License

[MIT](LICENSE) © 2026 Shayan Shirahmad

