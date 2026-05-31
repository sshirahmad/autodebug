---
name: systematic-debugging
description: Four-phase debugging process for the fix and root_cause agents. Iron law — no fixes without root cause first. Phases are read the culprit diff, trace the failure through the call stack, verify the hypothesis by running the repro, then apply the minimal fix. Includes a pattern table mapping common error types to their root causes.
---

# Systematic Debugging

## Iron Law

**NO FIXES WITHOUT ROOT CAUSE FIRST.**

Symptom fixes create whack-a-mole. Find the root cause, then fix it.

---

## AutoDebug Context

In the AutoDebug pipeline you already have:
- The **culprit commit** — the bisect result that first introduced the bug
- The **repro script** — reproduces the failure deterministically
- The **traceback** — exact error output

Start from these. Don't re-investigate things you already know.

---

## Phase 1: Understand the Culprit Diff

Before touching any other file:

1. Read the culprit commit diff in full. What exactly changed?
2. Read the files that changed. What did they do before? What do they do now?
3. Use `read_file_at_parent(path=...)` to see pre-change versions.

**Questions to answer:**
- What invariant does this change break?
- What callers, subclasses, or dependents assumed the old behavior?
- Is the break a direct effect (called code crashes) or indirect (side effect missing)?

---

## Phase 2: Trace the Failure

Map the traceback to the diff:

```
Traceback (innermost last):
  File "test_collection.py", line 89, in test_verify_no_version  ← symptom
  File "collection.py", line 312, in verify_collection           ← where it crashes
TypeError: missing required argument 'version'                   ← what broke
```

Now connect it to the diff:
- The diff removed the `version=None` default from line 312 → **that's the root cause**
- The test at line 89 calls with no version → **that's what triggers it**

The root cause is always **the diff-introduced change that started the failure chain**.

---

## Phase 3: Verify Before Fixing

Run the repro to confirm the traceback matches your theory. Root_cause agent: use `run_repro_with_traceback`. Fix agent: use `run_repro`.

If the error doesn't match your theory: go back to Phase 1. Don't guess.

**3-strike rule:** If 3 hypotheses all fail, the issue is likely structural — not a simple code read. Escalate with what you've found.

---

## Phase 4: Fix

Fix the root cause, not the symptom:

1. **Minimal change** — fewest lines, fewest files. Resist the urge to refactor.
2. **One change at a time** — don't bundle multiple fixes.
3. **Verify** — run the repro script AND the targeted test command.
4. **If fix touches >3 files** — pause. Is this really the right layer to fix?

---

## Red Flags

Stop and re-analyze if:
- "Quick fix for now" — there is no now, fix it right.
- You're guessing what the root cause is instead of reading the diff.
- Each fix reveals a new problem elsewhere — you're fixing the wrong layer.
- You've tried 3+ fixes and none worked — wrong mental model, start over.

---

## Pattern Reference

| Error | Common Root Cause |
|-------|------------------|
| `TypeError: missing argument` | Default removed, parameter made required |
| `AttributeError: no attribute 'X'` | Rename, move, or deletion of attribute |
| `KeyError` in registry/dict | Import side effect removed, registration missing |
| `TypeError: 'X' is not iterable` | Return type changed |
| `ImportError` / `ModuleNotFoundError` | Module reorganization in the diff |
| Test passes but wrong value | Logic branch changed, edge case not covered |
| Works in isolation, fails in suite | State pollution, global modified in culprit |

---

## Output Checklist

Before submitting root cause:
- [ ] I read the culprit diff completely
- [ ] I can name the exact file and line where the contract broke
- [ ] I can trace the path from the diff change to the traceback error
- [ ] I verified by running the repro that the error matches my theory (root_cause agent: use `run_repro_with_traceback`; fix agent: use `run_repro`)
- [ ] My hypothesis is one specific, falsifiable sentence
