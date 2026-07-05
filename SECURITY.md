# Security Policy

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, report them privately by email to **shayan.shirahmadi@gmail.com**.

Include as much as you can so we can reproduce and assess quickly:

- a description of the issue and its impact,
- steps to reproduce (proof-of-concept if possible),
- affected version / commit,
- any suggested fix or mitigation.

## What to expect

- We aim to **acknowledge** your report within a few days.
- We'll work with you to understand and validate the issue, and keep you updated on
  progress toward a fix.
- Please give us a reasonable window to release a fix before any public disclosure.

We appreciate responsible disclosure and will credit reporters who wish to be
acknowledged.

## Scope note

AutoDebug executes code in a Docker sandbox and drives LLM agents that run shell
commands and apply patches. Run it against repositories and bug reports you trust,
in an isolated environment — issues stemming from intentionally adversarial inputs
to the sandbox are in scope, but routine "the agent ran a command" behavior is by
design.
