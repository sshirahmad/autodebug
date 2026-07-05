"""Tests for autodebug/github.py — issue/PR URL parsing + best-effort fetch."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug import github  # noqa: E402


class TestUrlParsing:
    def test_issue_api_url_for_issue(self):
        assert github.issue_api_url("https://github.com/psf/black/issues/1234") == \
            "https://api.github.com/repos/psf/black/issues/1234"

    def test_issue_api_url_for_pull(self):
        # PRs are served by the issues API too.
        assert github.issue_api_url("https://github.com/psf/black/pull/56") == \
            "https://api.github.com/repos/psf/black/issues/56"

    def test_issue_api_url_none_for_plain_repo(self):
        assert github.issue_api_url("https://github.com/psf/black") is None
        assert github.issue_api_url("not a url") is None

    def test_repo_from_issue_url(self):
        assert github.repo_from_issue_url("https://github.com/psf/black/issues/9") == \
            "https://github.com/psf/black"
        assert github.repo_from_issue_url("https://github.com/psf/black") is None


class TestFetchIssue:
    def test_fetch_returns_title_and_body(self, monkeypatch):
        payload = json.dumps({"title": "Crash on empty input", "body": "Steps: ..."}).encode()

        class _Resp:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(github.urllib.request, "urlopen", lambda req, timeout=0: _Resp())
        out = github.fetch_issue("https://github.com/psf/black/issues/1")
        assert out == "Crash on empty input\n\nSteps: ..."

    def test_fetch_none_on_network_error(self, monkeypatch):
        def boom(*a, **k): raise OSError("no network")
        monkeypatch.setattr(github.urllib.request, "urlopen", boom)
        assert github.fetch_issue("https://github.com/psf/black/issues/1") is None

    def test_fetch_none_for_non_issue_url(self):
        # Never even hits the network for a non-issue URL.
        assert github.fetch_issue("https://github.com/psf/black") is None

    def test_body_truncated_to_max_chars(self, monkeypatch):
        payload = json.dumps({"title": "T", "body": "x" * 50000}).encode()

        class _Resp:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(github.urllib.request, "urlopen", lambda req, timeout=0: _Resp())
        out = github.fetch_issue("https://github.com/psf/black/issues/1", max_chars=100)
        assert len(out) == 100
