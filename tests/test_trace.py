"""Trace seam: load_trajectory / tool_usage / parse_catalog over synthetic transcripts."""
import argparse
import os

from tokeneconomizer import trace
from fixtures import write_claude_transcript, write_codex_transcript


def test_load_trajectory_claude(tmp_path):
    p = str(tmp_path / "t.jsonl")
    write_claude_transcript(p, [
        ("bash", {"command": "python -m pytest tests/test_x.py"}, "2 passed"),
        ("edit", {"file_path": "pkg/mod.py"}, "ok")])
    traj = trace.load_trajectory(p)
    names = [name for name, _a, _r, _ok in traj]
    assert "bash" in names and "edit" in names


def test_a_claude_request_blob_repeating_the_conversation_adds_no_call(tmp_path):
    """Claude Code 2.1.300 writes `api-request-blob` entries that repeat a request's messages,
    earlier tool calls and results included. Read as calls, each one counted twice."""
    import json
    use = {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pytest -q"}}
    res = {"type": "tool_result", "tool_use_id": "toolu_1",
           "content": "/bin/bash: line 1: pytest: command not found", "is_error": True}
    lines = [{"type": "assistant", "cwd": "/w/repo", "message": {"role": "assistant",
                                                                "content": [use]}},
             {"type": "user", "message": {"role": "user", "content": [res]}},
             {"type": "api-request-blob", "message": {"role": "assistant", "content": [use]}},
             {"type": "api-request-blob", "message": {"role": "user", "content": [res]}},
             {"type": "assistant", "message": {"role": "assistant", "content": [use]}}]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    calls = trace.load_trajectory(str(p))
    assert [(c[0], c[1]["command"], c[3]) for c in calls] == [("Bash", "pytest -q", False)]
    assert trace.call_cwds(str(p)) == ["/w/repo"]
    assert trace.tool_usage(str(p)) == {"Bash": 1}
    from tokeneconomizer.microskills import distiller
    assert len(distiller.load_calls(str(p))) == 1


def test_tool_usage_counts(tmp_path):
    p = str(tmp_path / "t.jsonl")
    write_claude_transcript(p, [
        ("bash", {"command": "ls"}, ""),
        ("bash", {"command": "pwd"}, ""),
        ("edit", {"file_path": "a.py"}, "")])
    usage = trace.tool_usage(p)
    assert usage.get("bash", 0) >= 2


def test_parse_catalog_codex_default():
    cat = trace.parse_catalog("/nonexistent", "codex")
    assert "shell" in cat and "apply_patch" in cat


def test_parse_catalog_missing_returns_empty(tmp_path):
    # Claude path with no transcript file -> empty catalog, no crash.
    assert trace.parse_catalog(str(tmp_path), "claude") == {}


# ---- Copilot tool-catalog sources: one pipeline across CLI builds -------------------
#
# The CLI has moved the loaded tool catalog twice, and each move made the then-current
# reader return {} -- which untooling cannot tell apart from "no tools loaded", so it
# degraded to a silent no-op for a whole sweep (measured: CLI 1.0.82 dropped 8 of 17 tools
# on 30/31 tasks; 1.0.83, same harness and repo, dropped nothing on 31/31). These tests
# pin all three source formats and the auto-selection between them.
import json as _json

from tokeneconomizer import trace as _trace
from tokeneconomizer.trace import catalog as _catalog

_SCHEMA_LOG = (
    '{"body":"{\\"tools\\":[{\\"name\\":\\"bash\\",\\"title\\":null,'
    '\\"description\\":\\"Run a shell command\\"},'
    '{\\"name\\":\\"view\\",\\"description\\":\\"Read a file\\"}]}"}'
)
_FLAT_LOG = '{\n  "tools": [\n    "bash:66921ccc",\n    "view:0963ceac",\n    "edit:9e640c64"\n  ]\n}'


def _checkpoint_transcript(*names):
    return _json.dumps({
        "type": "session.usage_checkpoint",
        "data": {"promptCacheBreakState": [
            {"models": {"m": {"tools": [{"name": n} for n in names]}}}]},
    })


def test_flat_catalog_parses_name_fingerprint_form():
    """CLI >= 1.0.82 publishes `"tools": ["bash:66921ccc", ...]` and no schemas."""
    cat = _catalog.parse_tool_catalog_flat(_FLAT_LOG)
    assert set(cat) == {"bash", "view", "edit"}
    assert all(v == "" for v in cat.values())      # names only, no descriptions
    assert _catalog.parse_tool_catalog(_FLAT_LOG) == {}   # schema reader stays schema-only


def test_schema_catalog_still_wins_and_keeps_descriptions(tmp_path):
    (tmp_path / "copilot_debug.log").write_text(_SCHEMA_LOG + "\n" + _FLAT_LOG)
    cat, src = _trace.parse_catalog_with_source(str(tmp_path), "copilot")
    assert src == "schema_debug_log"               # richest: it has descriptions
    assert cat["bash"].startswith("Run a shell command")


def test_flat_used_when_schemas_and_checkpoint_are_gone(tmp_path):
    """The 1.0.83 shape: no schemas, no usage_checkpoint -- must NOT parse empty."""
    (tmp_path / "copilot_debug.log").write_text(_FLAT_LOG)
    (tmp_path / "transcript.jsonl").write_text('{"type":"assistant.message","data":{}}')
    cat, src = _trace.parse_catalog_with_source(str(tmp_path), "copilot")
    assert src == "flat_debug_log"
    assert set(cat) == {"bash", "view", "edit"}


def test_usage_checkpoint_source_is_still_read(tmp_path):
    """The 1.0.82 shape must keep working -- supporting the new build is not a swap."""
    (tmp_path / "transcript.jsonl").write_text(_checkpoint_transcript("bash", "view"))
    cat, src = _trace.parse_catalog_with_source(str(tmp_path), "copilot")
    assert src == "usage_checkpoint"
    assert set(cat) == {"bash", "view"}


def test_no_source_reports_none_not_empty_catalog(tmp_path):
    """No source must be distinguishable from an empty catalog: {} is not 'no tools'."""
    (tmp_path / "copilot_debug.log").write_text("nothing useful here")
    cat, src = _trace.parse_catalog_with_source(str(tmp_path), "copilot")
    assert cat == {}
    assert src is None                              # defect signal, not an empty catalog


# ---- Web-Bench pass@1 vs pass@N ------------------------------------------------------
#
# Web-Bench grades each task with up to `retry` attempts, so a task has TWO defensible
# resolve metrics, and they differ materially (measured on prisma: 1-4 tasks per arm).
# Web-Bench's own headline is pass@1. The runner breaks out of the attempt loop on the
# first pass, which makes pass@1 recoverable from older rows that predate the explicit
# fields -- these pin that derivation, since getting it wrong reports pass@N under the
# pass@1 column.
def _p1p2(row):
    """The aggregator's per-row pass@1/pass@N resolution, mirrored for test."""
    p1, p2 = row.get("resolved_p1", ""), row.get("resolved_p2", "")
    att = row.get("webbench_attempts", "")
    if p1 == "" and p2 == "":
        if att != "":
            att = int(att or 0)
            if att == 0:
                return None, None
            p2 = row["resolved"] is True
            p1 = p2 and att == 1
        else:
            p1 = p2 = row["resolved"]
    return p1, p2


def test_explicit_attempt_fields_win():
    assert _p1p2({"resolved": True, "resolved_p1": False,
                  "resolved_p2": True, "webbench_attempts": 2}) == (False, True)


def test_legacy_row_second_attempt_pass_is_not_pass_at_1():
    """resolved=True after 2 attempts means attempt 1 FAILED."""
    assert _p1p2({"resolved": True, "webbench_attempts": 2}) == (False, True)


def test_legacy_row_first_attempt_pass():
    assert _p1p2({"resolved": True, "webbench_attempts": 1}) == (True, True)


def test_legacy_row_failed_every_attempt():
    assert _p1p2({"resolved": False, "webbench_attempts": 2}) == (False, False)


def test_unreached_task_counts_in_neither_metric():
    """Never attempted is not a miss -- it must not enter numerator or denominator."""
    assert _p1p2({"resolved": False, "webbench_attempts": 0}) == (None, None)


def test_single_attempt_domain_collapses_to_resolved():
    """SWE-bench rows carry no attempt fields; pass@1 and pass@N are the same number."""
    assert _p1p2({"resolved": True}) == (True, True)
    assert _p1p2({"resolved": False}) == (False, False)


def _jsonl(path, records):
    import json
    with open(path, "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def test_a_claude_session_counts_the_prompts_the_person_gave(tmp_path):
    """Typed prompts and slash commands; not tool results, a local command's output, an
    interrupt marker, a compaction summary, meta text, or a subagent's prompts."""
    def user(content, **flags):
        return dict({"type": "user", "message": {"role": "user", "content": content}}, **flags)
    p = _jsonl(tmp_path / "c.jsonl", [
        user("fix the test"),
        user([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]),
        user([{"type": "text", "text": "and the docs"}, {"type": "image", "source": {}}]),
        user("<command-name>/tokeneconomizer:status</command-name>"),
        user("<local-command-stdout>done</local-command-stdout>"),
        user("[Request interrupted by user]"),
        user("Caveat: the messages below were generated by the user", isMeta=True),
        user("This session is being continued...", isCompactSummary=True),
        user("subagent task", isSidechain=True),
        {"type": "assistant", "message": {"role": "assistant", "content": "done"}}])
    assert trace.claude_prompt_count(p) == 3


def test_a_codex_session_counts_one_prompt_per_turn(tmp_path):
    turn = [{"type": "turn_context", "payload": {"turn_id": "t"}},
            {"type": "event_msg", "payload": {"type": "task_started"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user"}}]
    assert trace.codex_prompt_count(_jsonl(tmp_path / "r.jsonl", turn * 3)) == 3
    # a rollout without task_started events counts its turn contexts
    older = [r for r in turn if r["type"] != "event_msg"]
    assert trace.codex_prompt_count(_jsonl(tmp_path / "o.jsonl", older * 2)) == 2


def test_a_copilot_session_counts_its_user_messages(tmp_path):
    p = _jsonl(tmp_path / "events.jsonl", [
        {"type": "session.start", "data": {}}, {"type": "user.message", "data": {}},
        {"type": "tool.execution_start", "data": {}}, {"type": "user.message", "data": {}}])
    assert trace.copilot_prompt_count(p) == 2



def test_a_pi_session_counts_the_prompts_on_its_active_branch(tmp_path):
    """Pi keeps a conversation as a tree; a prompt on a branch the person left is not counted."""
    def msg(eid, parent, role):
        return {"type": "message", "id": eid, "parentId": parent, "message": {"role": role}}
    p = _jsonl(tmp_path / "pi.jsonl", [
        {"type": "session", "id": "s"}, msg("a", None, "user"), msg("b", "a", "assistant"),
        msg("x", "b", "user"), msg("y", "x", "assistant"),        # abandoned
        msg("c", "b", "user"), msg("d", "c", "toolResult"), msg("e", "d", "assistant")])
    assert trace.pi_prompt_count(p) == 2


# ---- Kimi Code wire.jsonl --------------------------------------------------------------------

def _kimi_session(tmp_path, events, cwd="/w/repo"):
    """A synthetic Kimi session: <tmp>/sessions/wd_x/session_s1/{state.json, agents/main/wire.jsonl}."""
    import json as _json
    sess = tmp_path / "sessions" / "wd_x" / "session_s1"
    (sess / "agents" / "main").mkdir(parents=True)
    (sess / "state.json").write_text(_json.dumps(
        {"id": "session_s1", "cwd": cwd, "createdAt": 1000000, "updatedAt": 2000000}))
    wire = sess / "agents" / "main" / "wire.jsonl"
    wire.write_text("\n".join(_json.dumps(e) for e in events))
    return str(wire)


def _kimi_wire(tmp_path):
    """The events of one Kimi session: two prompts, a failed Bash call, a subagent call."""
    def appended(role, **kw):
        return {"type": "agent.message.appended",
                "message": {"message": dict({"role": role}, **kw), "meta": {"source": "x"}}}
    return _kimi_session(tmp_path, [
        {"type": "metadata", "protocol_version": "1.5"},
        {"type": "turn.prompt", "agentId": "main",
         "input": [{"type": "text", "text": "run the tests"}], "turnId": 0},
        appended("user", content=[{"type": "text", "text": "run the tests"}],
                 id="msg_1", toolCalls=[]),
        appended("assistant",
                 content=[{"type": "text", "text": "running"}],
                 toolCalls=[{"type": "function", "id": "call_1", "name": "Bash",
                             "arguments": '{"command": "pytest -q"}'}]),
        appended("tool", content=[{"type": "text", "text": "2 failed\n"
                                   "Command failed with exit code: 1."}],
                 toolCallId="call_1"),
        {"type": "turn.prompt", "agentId": "main",
         "input": [{"type": "text", "text": "fix it"}], "turnId": 1},
        {"type": "context.append_message", "agentId": "main",
         "message": {"role": "user", "content": [{"type": "text", "text": "<system-reminder>"}],
                     "toolCalls": [], "origin": {"kind": "injection"}}},
        appended("assistant", content=[], toolCalls=[
            {"type": "function", "id": "call_2", "name": "Agent",
             "arguments": '{"prompt": "dig"}'}]),
        appended("tool", content=[{"type": "text", "text": "done"}],
                 toolCallId="call_2"),
        {"type": "usage.record", "agentId": "main", "model": "kimi-code/kimi-for-coding",
         "usage": {"inputOther": 100, "output": 10, "inputCacheRead": 50,
                   "inputCacheCreation": 5}, "usageScope": "turn"},
        {"type": "usage.record", "agentId": "main", "model": "kimi-code/kimi-for-coding",
         "usage": {"inputOther": 200, "output": 20, "inputCacheRead": 60,
                   "inputCacheCreation": 0}, "usageScope": "turn"},
    ])


def test_kimi_wire_reads_as_calls_with_the_session_cwd(tmp_path):
    p = _kimi_wire(tmp_path)
    traj = trace.load_trajectory(p)
    assert [(c[0], c[1].get("command") or c[1].get("prompt"), c[3]) for c in traj] == [
        ("Bash", "pytest -q", False), ("Agent", "dig", True)]
    assert trace.call_cwds(p) == ["/w/repo", "/w/repo"]
    assert trace.tool_usage(p) == {"Bash": 1, "Agent": 1}


def test_kimi_prompts_are_turn_prompts_injections_ignored(tmp_path):
    assert trace.kimi_prompt_count(_kimi_wire(tmp_path)) == 2


def test_kimi_usage_sums_every_step_record(tmp_path):
    u = trace.kimi_session_usage(_kimi_wire(tmp_path))
    assert u["input_tokens"] == 300
    assert u["cache_read_input_tokens"] == 110
    assert u["cache_creation_input_tokens"] == 5
    assert u["output_tokens"] == 30
    assert u["requests"] == 2
    assert u["model"] == "kimi-code/kimi-for-coding"


def test_kimi_tools_snapshot_is_the_catalog(tmp_path):
    snap = {"type": "llm.tools_snapshot", "agentId": "main", "tools": [
        {"name": "Bash", "description": "Run a shell command."},
        {"name": "Agent", "description": "Launch a subagent."}]}
    later = {"type": "llm.tools_snapshot", "agentId": "main", "tools": [
        {"name": "Bash", "description": "Run a shell command."}]}
    p = _kimi_session(tmp_path, [snap, later])
    cat = trace.parse_tool_catalog_kimi(open(p, encoding="utf-8").read())
    assert cat == {"Bash": "Run a shell command."}   # the last snapshot wins
    run = tmp_path / "run"          # the fold's run dir: transcript.jsonl -> the wire log
    run.mkdir()
    (run / "transcript.jsonl").write_text(open(p, encoding="utf-8").read())
    assert trace.parse_catalog_with_source(str(run), "kimi")[1] == "kimi_tools_snapshot"


def test_kimi_import_records_sessions_and_skips_known_ones(tmp_path, monkeypatch):
    from tokeneconomizer.plugin import cli
    wire = _kimi_wire(tmp_path)
    home = tmp_path / "kimi-home"
    home.mkdir()
    import shutil
    shutil.move(str(tmp_path / "sessions"), str(home / "sessions"))
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path / "te-state"))
    monkeypatch.setenv("TE_FOLD_SYNC", "1")
    args = argparse.Namespace(home=str(home), no_fold=True)
    assert cli.cmd_kimi_import(args) == 0
    from tokeneconomizer.plugin.store import Store
    from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
    store = Store(repo_state_dir(repo_identity("/w/repo"), "kimi"))
    rec = store.read(os.path.join("sessions", "session_s1.json"))
    assert rec["host"] == "kimi" and rec["transcript"].endswith("wire.jsonl") \
        and rec["ended"] == 2000.0
    assert cli.cmd_kimi_import(args) == 0   # second run skips the recorded session
