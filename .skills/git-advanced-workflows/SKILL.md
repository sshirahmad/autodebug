---
name: git-advanced-workflows
description: How to use git bisect and explore commit history inside the AutoDebug sandbox. All git commands must go through subprocess.run() because run_script executes Python, not bash. Covers automated bisect with git bisect run, manual binary search, unshallowing shallow clones, and handling merge commits with empty diffs.
---

# Git Advanced Workflows — AutoDebug Edition

## CRITICAL: Sandbox Execution Model

`run_script` **runs a Python script**, not a bash script. Every git operation must go through `subprocess.run()`.

**WRONG — will cause SyntaxError:**
```
run_script("git bisect start")
run_script("git log --oneline abc..HEAD")
```

**CORRECT — always write Python:**
```python
import subprocess, sys

r = subprocess.run(
    ['git', 'bisect', 'start'],
    cwd='/workspace/repo',
    capture_output=True, text=True
)
print(r.stdout, r.stderr)
```

---

## Git Bisect — Finding the First Bad Commit

Your primary job. Given a known-good commit and HEAD (known-bad), find the FIRST commit that broke the behavior.

### Automated bisect with a Python test script

This is the fastest approach. Write one script that exits 0 = good, 1 = bad.

**Step 1 — write the test script to a file:**
```python
import subprocess

test_script = '''
import subprocess, sys
r = subprocess.run(
    ['python', '-c', 'import ansible; ...'],   # or whatever repro logic
    cwd='/workspace/repo',
    capture_output=True, text=True, timeout=60
)
sys.exit(0 if r.returncode == 0 else 1)
'''

with open('/tmp/bisect_test.py', 'w') as f:
    f.write(test_script)
```

**Step 2 — run automated bisect:**
```python
import subprocess

# Start bisect with range
r = subprocess.run(
    ['git', 'bisect', 'start', 'HEAD', 'KNOWN_GOOD_SHA'],
    cwd='/workspace/repo', capture_output=True, text=True
)
print(r.stdout)

# Run automated bisect — git checks out commits and runs the script
r = subprocess.run(
    ['git', 'bisect', 'run', 'python', '/tmp/bisect_test.py'],
    cwd='/workspace/repo', capture_output=True, text=True, timeout=600
)
print(r.stdout[-3000:])  # Last 3000 chars contains the result

# Clean up
subprocess.run(['git', 'bisect', 'reset'], cwd='/workspace/repo')
```

The output will contain a line like:
```
abc1234def is the first bad commit
```

That SHA is your culprit. Call `submit_culprit` with it.

### Manual bisect (when automated bisect won't work)

Use when the test requires complex setup between checkout steps.

```python
import subprocess

cwd = '/workspace/repo'

def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip(), r.returncode

# Get the commit range
log_out, _ = git('log', '--oneline', 'KNOWN_GOOD..HEAD')
commits = [line.split()[0] for line in log_out.splitlines()]
print(f"Range: {len(commits)} commits to search")

# Binary search manually
lo, hi = 0, len(commits) - 1
while lo < hi:
    mid = (lo + hi) // 2
    sha = commits[mid]
    git('checkout', sha)
    
    # Run repro
    r = subprocess.run(['python', '/tmp/repro.py'], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        hi = mid      # bad: bug present, search earlier
    else:
        lo = mid + 1  # good: bug absent, search later

git('checkout', '-')   # restore HEAD
first_bad = commits[lo]
print(f"First bad commit: {first_bad}")
```

### Unshallowing before bisect

Clones are often shallow. Bisect requires full history.

```python
import subprocess

# Check if shallow
r = subprocess.run(
    ['git', 'rev-parse', '--is-shallow-repository'],
    cwd='/workspace/repo', capture_output=True, text=True
)
if r.stdout.strip() == 'true':
    subprocess.run(
        ['git', 'fetch', '--unshallow'],
        cwd='/workspace/repo', capture_output=True, text=True, timeout=120
    )
```

### Merge commits — empty diffs

When a commit is a merge, `git diff sha^ sha` returns empty because merges don't introduce changes themselves. Use `git show` instead:

```python
r = subprocess.run(
    ['git', 'show', '--stat', sha],
    cwd='/workspace/repo', capture_output=True, text=True
)
# Or to see the full merge diff:
r = subprocess.run(
    ['git', 'show', '--format=fuller', sha],
    cwd='/workspace/repo', capture_output=True, text=True
)
```

For merge commits, the real change is in the commit that was merged in — look at the parents:

```python
r = subprocess.run(
    ['git', 'log', '--oneline', f'{sha}^1..{sha}^2'],
    cwd='/workspace/repo', capture_output=True, text=True
)
print(r.stdout)  # commits between the two parents
```

---

## Exploring Commit History

```python
import subprocess

cwd = '/workspace/repo'

# Log between two commits
r = subprocess.run(
    ['git', 'log', '--oneline', 'GOOD_SHA..HEAD'],
    cwd=cwd, capture_output=True, text=True
)
print(r.stdout)

# Files changed by a commit
r = subprocess.run(
    ['git', 'diff', '--name-only', f'{sha}^', sha],
    cwd=cwd, capture_output=True, text=True
)
print(r.stdout)

# Full diff of a commit
r = subprocess.run(
    ['git', 'show', '--format=', sha],
    cwd=cwd, capture_output=True, text=True
)
print(r.stdout[:5000])

# Commits touching a specific file
r = subprocess.run(
    ['git', 'log', '--oneline', 'GOOD_SHA..HEAD', '--', 'path/to/file.py'],
    cwd=cwd, capture_output=True, text=True
)
print(r.stdout)

# When was a function last changed
r = subprocess.run(
    ['git', 'log', '-p', '--pickaxe-regex', '-S', 'def my_function',
     'GOOD_SHA..HEAD', '--', '*.py'],
    cwd=cwd, capture_output=True, text=True
)
print(r.stdout[:5000])
```

---

## Checkout and Restore

```python
import subprocess

cwd = '/workspace/repo'

# Checkout a specific commit to test it
subprocess.run(['git', 'checkout', sha], cwd=cwd)

# Restore HEAD
subprocess.run(['git', 'checkout', '-'], cwd=cwd)
# or by name:
subprocess.run(['git', 'checkout', 'HEAD'], cwd=cwd)

# Read a file at a specific commit without checking out
r = subprocess.run(
    ['git', 'show', f'{sha}:path/to/file.py'],
    cwd=cwd, capture_output=True, text=True
)
print(r.stdout)
```

---

## Key Rules for AutoDebug

1. **Always use `subprocess.run()` — never bare shell strings.**
2. **Unshallow before bisect** — check `--is-shallow-repository` first.
3. **The culprit is the FIRST bad commit** between `known_good_commit` and HEAD.
4. **Automated `git bisect run`** is faster than manual looping — prefer it.
5. **On merge commits**, look at `git show --stat` and the parents, not `git diff sha^ sha`.
6. **Capture both stdout AND stderr** — build failures often go to stderr.
7. **Set a timeout** on subprocess calls — a hanging test will stall the bisect.
