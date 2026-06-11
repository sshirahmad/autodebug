"""Helpers for splitting and applying unified diff patches.

The benchmark's `ground_truth_patch` typically bundles both the code fix and the
new tests that demonstrate the bug. The pipeline only applies the *test* portion
at clone time — the code fix is what the agents are supposed to produce.
"""

from __future__ import annotations

import re

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)


def is_test_path(path: str) -> bool:
    """Heuristic: does this path belong to test code rather than source?"""
    parts = path.split("/")
    if any(p in ("test", "tests", "testing") for p in parts):
        return True
    filename = parts[-1] if parts else ""
    return filename.startswith("test_") or filename.endswith("_test.py")


# Backwards-compatible private alias.
_is_test_path = is_test_path


def split_patch(unified_diff: str) -> tuple[str, str]:
    """Split a unified diff into (code_patch, test_patch) by file path.

    Each `diff --git a/<path> b/<path>` block goes wholesale into the test
    patch if the path looks like a test file, otherwise into the code patch.
    """
    if not unified_diff:
        return "", ""

    # Find every "diff --git" header position so we can slice into per-file chunks.
    headers = list(_DIFF_HEADER_RE.finditer(unified_diff))
    if not headers:
        return unified_diff, ""

    code_chunks: list[str] = []
    test_chunks: list[str] = []
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(unified_diff)
        chunk = unified_diff[start:end]
        path = match.group(1)
        (test_chunks if _is_test_path(path) else code_chunks).append(chunk)

    return "".join(code_chunks), "".join(test_chunks)
