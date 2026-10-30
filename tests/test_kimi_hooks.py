"""Kimi Code host adapter and hook installer (see plugin/kimihooks.py)."""
import json
import os

from tokeneconomizer.plugin import hosts, kimihooks


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path / "te"))
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "kimi"))
    return dict(os.environ)


def _write_kimi_session(tmp_path, sid="session_abc"):
    """A session dir whose wire path the adapter should resolve from the session id."""
    sess = tmp_path / "kimi" / "sessions" / "wd_x" / sid
    (sess / "agents" / "main").mkdir(parents=True)
    wire = sess / "agents" / "main" / "wire.jsonl"
    wire.write_text('{"type":"agent.message.appended"}\n')
    return str(wire)


def test_kimi_transcript_resolves_from_the_session_id(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    wire = _write_kimi_session(tmp_path)
    host = hosts.Kimi()
    p = {"session_id": "session_abc", "cwd": "/w/repo"}
    assert host.transcript(p) == wire
    assert host.transcript({"session_id": "", "cwd": "/w"}) == ""
    assert hosts.detect({"transcript_path": wire}).name == "kimi"


def test_kimi_prompt_context_and_start_context_flags():
    host = hosts.Kimi()
    assert host.prompt_context is True
    assert host.keeps_start_context is False       # SessionStart is observation-only


def test_kimi_hooks_install_is_idempotent_and_preserves_config(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(hosts.__file__))))
    monkeypatch.setenv("TE_PLUGIN_ROOT", os.path.join(repo, "plugin"))
    cfg = tmp_path / "kimi" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('default_model = "m"\n\n[providers."P"]\nbase_url = "u"\n')
    path, changed, command = kimihooks.install()
    assert changed and command
    text = cfg.read_text()
    assert 'default_model = "m"' in text and '[providers."P"]' in text
    assert text.count("[[hooks]]") == len(kimihooks.EVENTS)
    assert "UserPromptSubmit" in text and command in text
    assert "--host kimi" in open(command, encoding="utf-8").read()  # baked into the shim
    # every registered event carries the hook command and a timeout
    for event, timeout in kimihooks.EVENTS:
        assert f'event = "{event}"' in text and f"timeout = {timeout}" in text
    again, changed2, _ = kimihooks.install()
    assert not changed2 and cfg.read_text() == text     # second install: byte-identical
    path2, removed = kimihooks.uninstall()
    assert removed
    assert cfg.read_text() == 'default_model = "m"\n\n[providers."P"]\nbase_url = "u"\n'
    _p, removed2 = kimihooks.uninstall()
    assert not removed2                                  # nothing left to strip


def test_kimi_hooks_block_roundtrips_without_trailing_content(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(hosts.__file__))))
    monkeypatch.setenv("TE_PLUGIN_ROOT", os.path.join(repo, "plugin"))
    cfg = tmp_path / "kimi" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("")
    kimihooks.install()
    assert kimihooks.uninstall()[1]
    assert cfg.read_text() == ""


def test_kimi_refresh_catalog_reads_the_newest_wire_snapshot(tmp_path, monkeypatch):
    from tokeneconomizer.plugin import toolcheck
    _env(tmp_path, monkeypatch)
    _write_kimi_session(tmp_path, sid="session_s1")
    wire = tmp_path / "kimi" / "sessions" / "wd_x" / "session_s1" / "agents" / "main" / \
        "wire.jsonl"
    wire.write_text(json.dumps({"type": "llm.tools_snapshot", "agentId": "main", "tools": [
        {"name": "Bash", "description": "sh"},
        {"name": "WebSearch", "description": "web"}]}) + "\n")
    path, tools = toolcheck.refresh_catalog(str(tmp_path / "te"), host="kimi")
    assert tools == ["Bash", "WebSearch"]
    assert json.load(open(path))["tools"] == ["Bash", "WebSearch"]


def test_runner_for_routes_kimi_to_its_own_cli(monkeypatch):
    from tokeneconomizer.plugin import sidecall
    seen = []
    monkeypatch.setattr(sidecall, "kimi_call",
                        lambda prompt, model="": seen.append(prompt) or ("OK", None))
    runner = sidecall.runner_for("kimi", type("C", (), {"paid_model": ""})())
    assert runner("hi") == ("OK", None)
    assert seen == ["hi"]
