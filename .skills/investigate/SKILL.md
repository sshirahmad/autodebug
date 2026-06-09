---
name: investigate
description: Root cause analysis for the AutoDebug root_cause agent. You arrive with a culprit commit diff, a repro script, and a traceback — start from the diff, trace the call chain to the error site, and produce a single precise causal hypothesis naming the file and line where the contract broke.
---

# Root Cause Analysis — AutoDebug Edition

## What You Have

You arrive with:
- **Bug report** — symptom description
- **Culprit commit** — the SHA that introduced the bug (from bisect), its message, and its diff
- **Repro script** — a Python script that reproduces the failure
- **Traceback** — the actual error output from running the repro at the culprit

Your job: explain precisely **what changed** and **why it broke**.

---

## The Process

### Step 1 — Read the culprit diff first

The diff tells you exactly what changed. Before reading any other file, read and understand every line of it.

Ask:
- What was added, removed, or changed?
- What invariant does this change break?
- What callers, subclasses, or dependents relied on the old behavior?

### Step 2 — Read the traceback

Map the traceback to the diff. The line that throws is a symptom — trace backwards to find where the diff-introduced change reaches it.

```
Error at: lib/foo/bar.py:147 in process()
    ← called by lib/foo/baz.py:89 in handle()
    ← triggered by the culprit removing the default value at lib/foo/api.py:23
```

That last line is the root cause.

### Step 3 — Read the relevant files

Read the file(s) changed in the culprit diff and the files that the traceback touches. Understand:
- What the code did before the change (use `read_file_at_parent` to read at the parent commit)
- What it does now (use `read_file` for the current version)
- The exact place the contract was broken

### Step 4 — Form a precise hypothesis

Write one sentence in this form:

> "The culprit removed/changed `X` in `file:line`, which broke the assumption in `file:line` that `Y`, causing `Z` when `W`."

Concrete example:
> "The culprit changed `verify_collection` in `galaxy/collection.py:312` to require an explicit `version` parameter, breaking callers in `test_collection.py:89` that relied on the default `None` value, causing `TypeError: verify_collection() missing 1 required argument` when running `test_verify_collections_no_version`."

That sentence is your root cause. It should be:
- Specific (names files and lines)
- Causal (explains the chain from diff → failure)
- Falsifiable (you can verify it by reading the code)

---

## Tools to Use

### Read the file at parent commit
```
read_file_at_parent(path="lib/foo/bar.py")
```
This reads the file as it was BEFORE the culprit commit. Compare with the current version to understand what changed.

### Run the repro with traceback
```
run_repro_with_traceback()
```
Runs the repro script and captures the full traceback. Use this to confirm your hypothesis — does the error match your theory?

### Search memory for prior bugs
```
search_memory(query="similar bug in galaxy collection")
```
Check if this bug pattern has been seen before.

---

## Common Root Cause Patterns

### Removed default parameter
The culprit made a required parameter that used to be optional (or removed a default value). Callers that relied on the default break.

Look for: `TypeError: func() missing N required positional argument`

### Changed return type / structure
The culprit changed what a function returns (dict → list, added a required key, etc.). Callers that unpacked or accessed specific fields break.

Look for: `KeyError`, `AttributeError`, `IndexError`, `TypeError: 'X' object is not iterable`

### Renamed or moved attribute
The culprit renamed a class attribute or moved a function. Code that referenced the old name breaks.

Look for: `AttributeError: 'X' object has no attribute 'Y'`

### Behavior change under edge case
The culprit changed logic that works for the common case but breaks for an edge case (empty list, None, missing key, specific version string).

Look for: the test that exercises an edge case failing, while similar tests pass.

### Import side effect removed
The culprit removed an import that had a side effect (registering a handler, setting a global, etc.). Code that relied on the side effect silently fails.

Look for: subtle behavior changes, missing registrations, `KeyError` in a registry lookup.

---

## Output Format

When you call `submit_root_cause`, provide:

**summary**: 1-2 sentences. What broke and why.
> "The `verify_collection` function now requires an explicit `version` argument. Callers passing no version hit a TypeError."

**hypothesis**: The causal chain.
> "Commit abc123 removed the `version=None` default from `verify_collection()` (collection.py:312). The test `test_verify_collections_no_version` calls it without a version argument (test_collection.py:89), triggering `TypeError: missing required argument`."

**relevant_lines**: File + line references where the break occurs.
```
["lib/ansible/galaxy/collection.py:312", "test/units/galaxy/test_collection.py:89"]
```

---

## Rules

1. **Read the diff before anything else.** The diff is the ground truth for what changed.
2. **Trace the call chain.** The error site in the traceback is a symptom; the culprit diff is the cause. Connect them.
3. **Use `read_file_at_parent`** to see what the code looked like before — don't guess.
4. **Be specific.** "Something changed in galaxy" is not a root cause. "Line 312 removed the default value" is.
5. **One root cause.** If you find multiple issues, report the one that directly causes the repro failure.
