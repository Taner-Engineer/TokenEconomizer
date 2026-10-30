"""Installing TokenEconomizer's hooks into Kimi Code's `[[hooks]]` config.

Kimi Code has no plugin marketplace entry to carry hook rules, so `te kimi-hooks install`
writes them directly into `$KIMI_CODE_HOME/config.toml` as one managed block between marker
comments. The block is idempotent: installing again replaces the block, uninstalling strips
it, and everything outside the markers is left byte-identical. The hook command is the same
stable launcher `ensure_launcher` maintains (`<state root>/bin/te-hook`), so an update to the
plugin's checkout keeps working without re-installing.

Only `te`'s six events are registered; `SubagentStart`/`SubagentStop` fire for sub-agents too,
but the controller keys its state by the main session's id and the wire log already carries
the sub-agents' calls, so registering them would double-count.
"""
import json
import os
import shlex

from .hosts import kimi_home
from .paths import ensure_launcher, launcher_path

BEGIN = "# >>> TokenEconomizer (kimi)"
END = "# <<< TokenEconomizer (kimi)"
# (event, timeout seconds) -- the same budget the Claude hook manifest gives each event.
EVENTS = (("SessionStart", 15), ("UserPromptSubmit", 20), ("PostToolUse", 60),
          ("PostToolUseFailure", 60), ("Stop", 10), ("SessionEnd", 10))


def config_path(env=None):
    env = os.environ if env is None else env
    return os.path.join(kimi_home(env), "config.toml")


def _strip_block(text):
    """The config text without a managed block; (text, had_block)."""
    lines = text.splitlines()
    out, i, had = [], 0, False
    while i < len(lines):
        if lines[i].strip() == BEGIN:
            had = True
            while i < len(lines) and lines[i].strip() != END:
                i += 1
            i += 1                       # drop the END line too
            while i < len(lines) and not lines[i].strip():
                i += 1                   # and the blank line after it
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), had


def _block(command):
    rows = [BEGIN, "# Managed by `te kimi-hooks` -- edit with that command, not by hand."]
    for event, timeout in EVENTS:
        rows += ["", "[[hooks]]", f'event = "{event}"',
                 f"command = {json.dumps(command)}", f"timeout = {timeout}"]
    rows.append(END)
    return "\n".join(rows)


def _hook_command_path(env=None):
    env = os.environ if env is None else env
    return os.path.join(os.path.dirname(launcher_path(env)), "te-hook")


def _ensure_hook_launcher(env=None):
    """`<state root>/bin/te-hook` pointing at this plugin's scripts/te-hook -- like
    ensure_launcher's te launcher, but carrying `--host kimi` so plain config lines need
    no per-host substitution (kimi's config substitutes nothing)."""
    env = os.environ if env is None else env
    plugin = env.get("TE_PLUGIN_ROOT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "plugin")
    target = os.path.join(plugin, "scripts", "te-hook")
    if not os.access(target, os.X_OK):
        return None
    path = _hook_command_path(env)
    body = ("#!/bin/sh\n# Written by the TokenEconomizer plugin at install: its te-hook, "
            "fixed to the kimi host.\nexec %s --host kimi \"$@\"\n"
            % shlex.quote(target))
    try:
        with open(path, encoding="utf-8") as fh:
            if fh.read() == body:
                return path
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
    os.chmod(tmp, 0o700)
    os.replace(tmp, path)
    return path


def _hook_command_path(env=None):
    env = os.environ if env is None else env
    return os.path.join(os.path.dirname(launcher_path(env)), "te-hook")


def install(env=None):
    """Write (or refresh) the managed hooks block. Returns (config path, changed, command)."""
    env = os.environ if env is None else env
    command = _ensure_hook_launcher(env)
    if not command:
        return config_path(env), False, None
    path = config_path(env)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        text = ""
    stripped, _ = _strip_block(text)
    block = _block(command)
    new = (stripped.rstrip() + "\n\n" + block + "\n") if stripped.strip() else block + "\n"
    if new != text:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
        os.replace(tmp, path)
        return path, True, command
    return path, False, command


def uninstall(env=None):
    """Remove the managed block. Returns (config path, changed)."""
    env = os.environ if env is None else env
    path = config_path(env)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return path, False
    stripped, had = _strip_block(text)
    if not had:
        return path, False
    new = stripped.rstrip("\n")
    new = (new + "\n") if new else ""
    if new != text:
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
        os.replace(tmp, path)
    return path, True
