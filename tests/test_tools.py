"""Tests that need no paid calls. The arXiv tests use the network and are skipped when it is unavailable."""

import urllib.error
from pathlib import Path

import pytest

from rtaste import tools


def test_repo_tools_stay_inside_the_repo(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    assert "src/" in tools.repo_ls(tmp_path, "")
    assert "print('hi')" in tools.repo_read(tmp_path, "src/main.py")
    assert tools.repo_read(tmp_path, "../../etc/passwd").startswith("No such file")
    assert tools.repo_ls(tmp_path, "/etc").startswith("No such directory")


def test_arxiv_search_respects_the_launch_day():
    try:
        early = tools.arxiv_search('"KV cache eviction" AND reasoning', launch_day="2020-01-01")
        late = tools.arxiv_search('"KV cache eviction" AND reasoning', launch_day="2026-10-02")
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("arXiv is not reachable")
    assert early == "No papers found."
    assert "KV" in late


def test_arxiv_read_refuses_papers_after_the_launch_day():
    try:
        refused = tools.arxiv_read("2609.03430", launch_day="2026-08-01")
        allowed = tools.arxiv_read("2609.03430", launch_day="2026-10-02")
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("arXiv is not reachable")
    assert "after the launch day" in refused
    assert "Characters 0 to" in allowed
