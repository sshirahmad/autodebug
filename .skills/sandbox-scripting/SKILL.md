---
name: sandbox-scripting
description: How to write Python scripts for the AutoDebug sandbox. The run_script tool executes a Python string, not bash — shell commands must use subprocess.run(). Covers subprocess patterns for git, pytest, grep, file I/O, exit codes for git bisect run, and common mistakes to avoid.
---

# Sandbox Scripting — AutoDebug Edition

## The Execution Model

`run_script(code: str)` takes a **Python script as a string** and runs it inside the sandbox container. The container has the target repository checked out at `/workspace/repo`.

- Input: Python source code (string)
- Output: combined stdout/stderr, exit code
- Working directory: the repo root (`/workspace/repo`)
- The script can import anything installed in the repo's environment

**This is NOT a bash executor. Shell commands must go through `subprocess.run()`.**

---

## Basic Pattern

```python
import subprocess, sys

r = subprocess.run(
    ['git', 'log', '--oneline', '-10'],
    cwd='/workspace/repo',
    capture_output=True,
    text=True,
    timeout=30,
)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr, file=sys.stderr)
```

---

## Running Tests

```python
import subprocess, sys

# Run a specific test
r = subprocess.run(
    ['python', '-m', 'pytest', 'test/units/galaxy/test_collection.py::test_verify_no_version',
     '-x', '--tb=short', '-q'],
    cwd='/workspace/repo',
    capture_output=True, text=True, timeout=120,
)
print(r.stdout[-3000:])
print(r.stderr[-1000:])
sys.exit(r.returncode)
```

```python
# Run the repro script inline
import subprocess, sys, textwrap

repro = textwrap.dedent('''
    import ansible.galaxy.collection as col
    col.verify_collection('some.namespace', 'module', ...)
''')

with open('/tmp/repro.py', 'w') as f:
    f.write(repro)

r = subprocess.run(
    ['python', '/tmp/repro.py'],
    cwd='/workspace/repo',
    capture_output=True, text=True, timeout=60,
)
print("exit code:", r.returncode)
print(r.stdout)
print(r.stderr)
sys.exit(r.returncode)
```

---

## Reading Files

```python
# Read a file from the repo
with open('/workspace/repo/lib/ansible/galaxy/collection.py') as f:
    content = f.read()
print(content[5000:8000])  # specific section
```

Or use the `read_file` tool — it's faster and stays within budget.

---

## Grep / Search

```python
import subprocess

r = subprocess.run(
    ['grep', '-n', '-r', 'def verify_collection', '/workspace/repo/lib/'],
    capture_output=True, text=True,
)
print(r.stdout)
```

---

## Git Operations

```python
import subprocess

cwd = '/workspace/repo'

def git(*args, check=False):
    r = subprocess.run(['git'] + list(args), cwd=cwd,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args} failed:\n{r.stderr}")
    return r.stdout.strip(), r.returncode

# Common patterns:
sha, _ = git('rev-parse', 'HEAD')
log, _ = git('log', '--oneline', 'abc123..HEAD')
diff, _ = git('diff', 'abc123^', 'abc123')
files, _ = git('diff', '--name-only', 'abc123^', 'abc123')
```

---

## Handling Long Output

Scripts can produce a lot of output. Print only what you need:

```python
# Tail of output
output = r.stdout
print(output[-3000:])  # last 3000 chars

# First N lines
lines = output.splitlines()
print('\n'.join(lines[:50]))

# Filter relevant lines
for line in output.splitlines():
    if 'error' in line.lower() or 'fail' in line.lower():
        print(line)
```

---

## Exit Codes for Bisect

When writing a bisect test script, exit codes matter:

```python
import subprocess, sys

r = subprocess.run(
    ['python', '-m', 'pytest', 'test/units/galaxy/test_collection.py::test_verify_no_version'],
    cwd='/workspace/repo',
    capture_output=True, text=True, timeout=120,
)

# For git bisect run:
#   0   = good (bug not present)
#   1   = bad  (bug present)
# 125   = skip (can't test this commit — e.g., doesn't compile)

if r.returncode == 0:
    sys.exit(0)   # good
elif 'SyntaxError' in r.stderr or 'ImportError' in r.stderr:
    sys.exit(125) # skip — untestable commit
else:
    sys.exit(1)   # bad
```

---

## Installing Dependencies

If a commit is missing a dependency:

```python
import subprocess, sys

r = subprocess.run(
    ['pip', 'install', '-q', 'some-package==1.2.3'],
    capture_output=True, text=True, timeout=60,
)
print(r.stdout, r.stderr)
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `run_script("git log --oneline")` | Use `subprocess.run(['git', 'log', '--oneline'], cwd=cwd, ...)` |
| No `cwd` argument | Always pass `cwd='/workspace/repo'` for repo operations |
| No `timeout` | Always set a timeout — a hanging process stalls the agent |
| Not checking `r.returncode` | Check it; zero = success, non-zero = failure |
| Printing all output | Truncate with `[-3000:]` or filter lines |
| Assuming stdout has the error | Errors usually go to stderr — print both |
