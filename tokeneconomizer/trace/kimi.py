"""Kimi Code's session log: tool calls, token usage, prompts, subagents, tool catalog.

Kimi Code (>= 2.0) persists a session as `wire.jsonl` under
`<KIMI_CODE_HOME>/sessions/<workspace>/<session>/agents/<agent>/`: a stream of typed events.
The conversation's calls ride `agent.message.appended` records -- assistant messages carry
`toolCalls` (`{type: "function", id, name, arguments}` with `arguments` a JSON string), tool
messages answer with `toolCallId`. What the person typed is `turn.prompt` (user-origin only;
date reminders and permission notices are separate `context.append_message` injections, and
steering is `turn.steer`, so counting `turn.prompt` does not need the exclusions Claude's
transcript needs). Per-step token usage is `usage.record`
(`{inputOther, output, inputCacheRead, inputCacheCreation}`); the loaded tool list is
`llm.tools_snapshot` (full `{name, description}` schemas). The session's cwd lives in the
sibling `state.json`, not on wire events. Subagents each get their own `agents/<id>/wire.jsonl`
under the same session directory.

Verified against Kimi Code 2.4.1 wire.jsonl; stdlib only.
"""
import json
import os
import re

from .jsonl import lines as _lines

# usage.record -> the four breakdown fields pricing.effective_tokens reads (Claude's names).
_USAGE_KEYS = (("input_tokens", "inputOther"),
               ("cache_read_input_tokens", "inputCacheRead"),
               ("cache_creation_input_tokens", "inputCacheCreation"),
               ("output_tokens", "output"))
# Sniffed by trace.calls before the Claude/Copilot shapes: only kimi writes this event type.
_MARKER = '"agent.message.appended"'
# A failed Bash result ends with this line (verified on Kimi Code 2.4.1 wire.jsonl).
_FAILED_RE = re.compile(r"Command failed with exit code:?\s*-?\d")


def is_kimi_wire(lines):
    """True when the lines are a Kimi Code wire.jsonl (sniffed, like the other agents)."""
    return any(_MARKER in line for line in lines)


def _session_dir(wire_path):
    """The session directory holding state.json: <session>/agents/<agent>/wire.jsonl -> <session>."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(wire_path))))


def kimi_session_cwd(wire_path):
    """The session's working directory from its state.json; None when unavailable."""
    try:
        with open(os.path.join(_session_dir(wire_path), "state.json"), encoding="utf-8") as fh:
            cwd = json.load(fh).get("cwd")
        return cwd if isinstance(cwd, str) else None
    except (OSError, ValueError):
        return None


def kimi_subagent_files(wire_path):
    """Sibling agents' wire files (`agents/<id>/wire.jsonl`) of one agent's log."""
    if not wire_path:
        return []
    agents_dir = os.path.dirname(os.path.dirname(os.path.abspath(wire_path)))
    out = []
    try:
        for name in sorted(os.listdir(agents_dir)):
            p = os.path.join(agents_dir, name, "wire.jsonl")
            if name != "main" and os.path.isfile(p):
                out.append(p)
    except OSError:
        pass
    return out


def kimi_calls(wire_path):
    """[(name, args, result_text, success, cwd)] for a Kimi wire.jsonl: each assistant
    `toolCalls` entry paired with its tool message by id; `arguments` is a JSON string and
    stays a dict through json.loads ({} when it does not parse). cwd is the session's,
    from state.json -- wire events do not carry one."""
    out, order, seen, results = [], [], {}, {}
    for line in _lines(wire_path):
        if _MARKER not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") != "agent.message.appended":
            continue
        msg = (ev.get("message") or {}).get("message") or {}
        if msg.get("role") == "assistant":
            for tc in msg.get("toolCalls") or []:
                if not isinstance(tc, dict):
                    continue
                tid = tc.get("id")
                if not tid or tid in seen:
                    continue
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                except ValueError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                seen[tid] = (tc.get("name"), args)
                order.append(tid)
        elif msg.get("role") == "tool":
            tid = msg.get("toolCallId")
            text = " ".join(c.get("text", "") for c in msg.get("content") or []
                            if isinstance(c, dict))
            results[tid] = (text, not _FAILED_RE.search(text[-300:]))
    cwd = kimi_session_cwd(wire_path)
    for tid in order:
        name, args = seen[tid]
        res_text, ok = results.get(tid, ("", True))
        out.append((name, args, res_text, ok, cwd))
    return out


def kimi_prompt_count(wire_path):
    """How many prompts the person gave the session: `turn.prompt` records (the origin is the
    user by construction; injections never take this event type)."""
    n = 0
    for line in _lines(wire_path):
        if '"turn.prompt"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        n += isinstance(ev, dict) and ev.get("type") == "turn.prompt"
    return n


def kimi_session_usage(wire_path, include_subagents=True):
    """Token breakdown of a Kimi session: every `usage.record` is one step's request, so all
    are summed (no per-block over-counting to undo, unlike Claude's transcript). Returns the
    four breakdown fields `pricing.effective_tokens` reads, plus `model` (the most frequent),
    `requests` (usage records seen) and `files` read."""
    files = [wire_path] + (kimi_subagent_files(wire_path) if include_subagents else [])
    totals = {k: 0 for k, _ in _USAGE_KEYS}
    models, requests = {}, 0
    for path in files:
        for line in _lines(path):
            if '"usage.record"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") != "usage.record":
                continue
            usage = ev.get("usage")
            if not isinstance(usage, dict):
                continue
            requests += 1
            for out_key, in_key in _USAGE_KEYS:
                try:
                    totals[out_key] += int(usage.get(in_key) or 0)
                except (TypeError, ValueError):
                    pass
            model = ev.get("model")
            if model:
                models[model] = models.get(model, 0) + 1
    totals["requests"] = requests
    totals["model"] = max(models, key=models.get) if models else None
    totals["files"] = len([f for f in files if f and os.path.exists(f)])
    return totals


def kimi_tool_usage(wire_path, include_subagents=True):
    """{tool: calls} over a session and, by default, its subagent wires."""
    counts = {}
    files = [wire_path] + (kimi_subagent_files(wire_path) if include_subagents else [])
    for path in files:
        for name, _args, _res, _ok, _cwd in kimi_calls(path):
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts
