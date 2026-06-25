# Contributing to AutoDebug

Thanks for your interest in contributing! AutoDebug is an end-to-end agentic
debugger (reproduce → bisect → root-cause → fix), and there's plenty to improve —
from agent prompts and tools to the sandbox, the eval harness, and docs.

This guide gets you set up and explains how we work.

## Getting started

**Requirements:** Python ≥ 3.11, Docker (running), and an LLM API key.

```bash
# 1. Fork + clone, then create the env
conda create -n autodebug python=3.11 && conda activate autodebug
pip install -e ".[dev,eval,tracing]"

# 2. Build the sandbox image the agents run code in
docker build -t autodebug-sandbox:latest ./docker/sandbox

# 3. Configure credentials
cp .env.example .env   # then add your ANTHROPIC_API_KEY (or other provider)
```

See the [README](README.md) for the full architecture and configuration reference.

## Running the tests

```bash
pytest tests/ -q
```

Most tests mock Docker and the LLM, so they run without an API key or a live
sandbox. Please add or update tests for any behavior you change — every PR should
keep the suite green.

## Development workflow

1. **Open an issue first** for anything non-trivial, so we can agree on the approach
   before you invest time. (Small fixes/typos can go straight to a PR.)
2. **Branch** off `main`: `git checkout -b feature/short-description`.
3. Make your change, **with tests**.
4. Run the suite and the linter locally (see below).
5. **Open a PR** against `main`, fill in the template, and link the issue.

We squash-merge; keep PRs focused and reasonably small. A reviewer will take a look.

## Code style

- **Lint:** `ruff check .` — and `ruff check . --fix` for the auto-fixable ones.
- **Format:** `ruff format .`
- Match the surrounding code: comment density, naming, and idiom. Comments should
  explain *why*, not restate the code.

> **Good first issue:** the repo currently has a backlog of small lint findings
> (unused imports, ambiguous names, etc.). Tidying these up — file by file — is a
> great, low-risk way to make your first contribution. CI reports them but doesn't
> block on them yet.

## Where to start

- **`autodebug/agents/`** — the repro/bisect/root_cause/fix/manager agents.
- **`autodebug/tools/`** — auto-discovered `make_<name>_tool` factories an agent can use.
- **`config/prompts/*.yaml`** + **`config/agents/*.json`** — prompts and per-agent
  budgets/tools, tunable without code changes.
- **`autodebug/sandbox/runner.py`** — the Docker sandbox + per-bug environment build.
- **`eval/run_eval.py`** — the evaluation harness and independent fix scoring.

## Reporting bugs / requesting features

Use the issue templates. For bugs, please include repro steps, what you expected,
what happened, and environment details (OS, Python, Docker version).

## Security

Please **do not** open public issues for security vulnerabilities. Instead, report
them privately to the maintainer (see the repository's security policy / contact).

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
