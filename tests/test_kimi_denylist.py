"""Kimi Code's deny list: disallowedTools in the project's .kimi-code/agents/agent.md."""
import os

import pytest

from tokeneconomizer.plugin import denylist
from tokeneconomizer.plugin.store import Store


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path / "te"))
    root = tmp_path / "repo"
    root.mkdir()
    return str(root)


def _store(tmp_path):
    return Store(str(tmp_path / "te"))


def test_apply_creates_the_agent_file_and_restore_removes_it(repo, tmp_path):
    store = _store(tmp_path)
    assert denylist.apply(repo, ["WebSearch", "FetchURL"], store, "kimi") == ["FetchURL",
                                                                               "WebSearch"]
    path = denylist.settings_path(repo, "kimi")
    text = open(path, encoding="utf-8").read()
    assert "name: agent" in text and "override: true" in text
    assert "- WebSearch" in text and "${base_prompt}" in text
    assert denylist.apply(repo, ["WebSearch"], store, "kimi") == []   # idempotent
    assert denylist.restore(repo, store, host="kimi") == ["FetchURL", "WebSearch"]
    assert not os.path.exists(path)                                   # te-created, now empty


def test_apply_merges_into_a_persons_main_agent_file(repo, tmp_path):
    store = _store(tmp_path)
    path = denylist.settings_path(repo, "kimi")
    os.makedirs(os.path.dirname(path))
    open(path, "w", encoding="utf-8").write(
        "---\nname: agent\noverride: true\ndescription: my own\ndisallowedTools:\n"
        "  - Bash\n---\n\nMy own prompt body.\n")
    added = denylist.apply(repo, ["WebSearch"], store, "kimi")
    assert added == ["WebSearch"]
    text = open(path, encoding="utf-8").read()
    assert "  - Bash" in text and "  - WebSearch" in text              # theirs kept
    assert "My own prompt body." in text                              # body untouched
    assert os.path.exists(path)
    assert denylist.restore(repo, store, host="kimi") == ["WebSearch"]
    text = open(path, encoding="utf-8").read()
    assert "  - Bash" in text and "WebSearch" not in text


def test_apply_refuses_what_it_cannot_own(repo, tmp_path):
    store = _store(tmp_path)
    path = denylist.settings_path(repo, "kimi")
    os.makedirs(os.path.dirname(path))
    open(path, "w", encoding="utf-8").write(
        "---\nname: agent\noverride: true\ndisallowedTools: [Bash]\n---\n\nBody.\n")
    with pytest.raises(RuntimeError, match="not a block list"):
        denylist.apply(repo, ["WebSearch"], store, "kimi")
    open(path, "w", encoding="utf-8").write(
        "---\nname: reviewer\noverride: true\n---\n\nBody.\n")
    with pytest.raises(RuntimeError, match="default main agent"):
        denylist.apply(repo, ["WebSearch"], store, "kimi")
    open(path, "w", encoding="utf-8").write("no frontmatter at all\n")
    with pytest.raises(RuntimeError, match="no --- frontmatter"):
        denylist.apply(repo, ["WebSearch"], store, "kimi")


def test_bili_tools_are_never_proposed():
    from tokeneconomizer.plugin import denylist
    assert denylist.never_propose("mcp__bili__acp")
    assert denylist.never_propose("mcp__bili__acp_cache")
    assert denylist.never_propose("bili")
    assert not denylist.never_propose("mcp__github__get_issue")
    assert not denylist.never_propose("Bash")
    assert not denylist.never_propose("WebSearch")
