"""Fetch a GitHub issue/PR so a run can be driven from just an issue URL.

The graph's `github_issue_url` field routes through here: given an issue or pull
URL, we fetch its title + body via the GitHub REST API and fold it into the bug
report. Best-effort — any failure (bad URL, network, 404, rate limit) returns None
so a run never breaks on issue fetching. Uses ``GITHUB_TOKEN`` when set (higher
rate limits + private repos), but works unauthenticated for public issues.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

# github.com/<owner>/<repo>/issues/<n>  (PRs live at /pull/<n> but the issues API
# serves them too, so accept both).
_ISSUE_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/(?:issues|pull)/(\d+)")


def issue_api_url(url: str) -> str | None:
    """Map a browser issue/PR URL to its GitHub REST API endpoint, or None."""
    m = _ISSUE_RE.search(url or "")
    if not m:
        return None
    owner, repo, num = m.groups()
    return f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"


def repo_from_issue_url(url: str) -> str | None:
    """The parent repo URL of an issue/PR link (so a pasted issue URL yields the
    repo to clone too), or None if `url` isn't an issue/PR link."""
    m = _ISSUE_RE.search(url or "")
    if not m:
        return None
    owner, repo, _ = m.groups()
    return f"https://github.com/{owner}/{repo}"


def fetch_issue(url: str, *, timeout: float = 10.0, max_chars: int = 8000) -> str | None:
    """Return ``"<title>\\n\\n<body>"`` for a GitHub issue/PR URL, or None.

    Best-effort and never raises: a malformed URL, network error, 404, or rate
    limit all return None. Body is truncated to `max_chars` so a huge issue can't
    blow up the prompt.
    """
    api = issue_api_url(url)
    if not api:
        return None
    req = urllib.request.Request(
        api,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "autodebug"},
    )
    token = os.getenv("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — issue fetch is best-effort, never fatal
        return None
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    text = (f"{title}\n\n{body}" if title else body).strip()
    return text[:max_chars] or None
