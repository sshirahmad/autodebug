"""Harness builders for runtime inspection tools.

These produce self-contained Python programs that are run via `sandbox.run_script`
(which transports them safely as base64). They give agents *causal* feedback —
the actual failure state, not just a pass/fail — so debugging isn't blind.

- `postmortem_harness`: run a script; on any uncaught exception, print the full
  traceback PLUS the local variables at every stack frame.
- `inspect_harness`: a non-interactive tracepoint — run a driver and, each time a
  given file:line is reached, evaluate expressions in that frame and record them
  with the frame's locals.

The target script/driver is embedded with `repr()` so arbitrary source (quotes,
newlines) survives as a valid string literal.
"""

from __future__ import annotations


def postmortem_harness(script: str) -> str:
    """A program that runs `script` and post-mortems any uncaught exception.

    Output is ordered most-relevant-first: the exception, then frames
    innermost-first (crash site first), dunder noise filtered — so reading or
    head-truncating it keeps the useful part.
    """
    return (
        "import sys, linecache\n"
        f"_SCRIPT = {script!r}\n"
        "try:\n"
        "    exec(compile(_SCRIPT, '<repro>', 'exec'), {'__name__': '__main__'})\n"
        "except SystemExit:\n"
        "    raise\n"
        "except BaseException as _e:\n"
        "    _tb = _e.__traceback__\n"
        "    _frames = []\n"
        "    while _tb is not None:\n"
        "        _frames.append((_tb.tb_frame, _tb.tb_lineno)); _tb = _tb.tb_next\n"
        "    out = ['EXCEPTION: %s: %s' % (type(_e).__name__, _e), '',\n"
        "           '=== FRAME STATE (innermost first) ==='] \n"
        "    for fr, lineno in reversed(_frames[-9:]):\n"
        "        if '_SCRIPT' in fr.f_locals: continue  # skip this harness's own frame\n"
        "        co = fr.f_code\n"
        "        out.append('%s:%d in %s' % (co.co_filename, lineno, co.co_name))\n"
        "        line = linecache.getline(co.co_filename, lineno).strip()\n"
        "        if line: out.append('    > ' + line)\n"
        "        for _k, _v in list(fr.f_locals.items())[:15]:\n"
        "            if _k.startswith('__') and _k.endswith('__'): continue\n"
        "            try: _r = repr(_v)\n"
        "            except Exception: _r = '<unrepr>'\n"
        "            out.append('      %s = %s' % (_k, _r[:200]))\n"
        "    print(chr(10).join(out))\n"
        "    sys.exit(1)\n"
    )


def inspect_harness(
    target_file: str,
    target_line: int,
    expressions: list[str],
    driver: str,
    max_hits: int = 5,
) -> str:
    """A program that probes `target_file:target_line` while running `driver`.

    Each hit records the evaluated `expressions` plus the frame locals; stops
    after `max_hits`. Emits a JSON list of hits (empty if the line is never hit).
    """
    return (
        "import sys, json, linecache\n"
        f"TARGET_FILE = {target_file!r}\n"
        f"TARGET_LINE = {int(target_line)}\n"
        f"EXPRS = {list(expressions)!r}\n"
        f"MAX_HITS = {int(max_hits)}\n"
        "HITS = []\n"
        "def _local(frame, event, arg):\n"
        "    if event == 'line' and frame.f_lineno == TARGET_LINE:\n"
        "        row = {'locals': {k: repr(v)[:120] for k, v in list(frame.f_locals.items())[:20]\n"
        "                          if not (k.startswith('__') and k.endswith('__'))}}\n"
        "        for ex in EXPRS:\n"
        "            try: row[ex] = repr(eval(ex, frame.f_globals, frame.f_locals))[:200]\n"
        "            except Exception as e: row[ex] = '<err: %s>' % e\n"
        "        HITS.append(row)\n"
        "        if len(HITS) >= MAX_HITS: sys.settrace(None)\n"
        "    return _local\n"
        "def _global(frame, event, arg):\n"
        "    if event == 'call' and frame.f_code.co_filename.endswith(TARGET_FILE):\n"
        "        return _local\n"
        "    return None\n"
        f"_DRIVER = {driver!r}\n"
        "sys.settrace(_global)\n"
        "try:\n"
        "    exec(compile(_DRIVER, '<driver>', 'exec'), {'__name__': '__main__'})\n"
        "except SystemExit:\n"
        "    pass\n"
        "except BaseException:\n"
        "    import traceback; traceback.print_exc()\n"
        "finally:\n"
        "    sys.settrace(None)\n"
        "print(json.dumps(HITS, indent=2))\n"
    )
