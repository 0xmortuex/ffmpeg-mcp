"""Run ffmpeg safely: argument lists (no shell), progress, timeout, tree kill.

`run_steps` executes one or more ffmpeg invocations in order. For each it:

* starts the process with stdin closed and stdout/stderr piped,
* parses `-progress pipe:1` key=value blocks on stdout for the output time,
* keeps only the tail of stderr (that is all an error message needs),
* polls for a timeout or a cancel request and kills the whole process tree.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

from .errors import MediaError


@dataclass
class Step:
    argv: list[str]
    label: str = ""


def quote_arg(arg: str) -> str:
    """Quote one argument for pasting into a Windows shell (cmd or PowerShell)."""
    if sys.platform != "win32":
        import shlex

        return shlex.quote(arg)
    quoted = subprocess.list2cmdline([arg])
    safe = all(c.isalnum() or c in "_@%+=:,./\\-" for c in arg)
    if quoted == arg and not safe:
        # list2cmdline only quotes spaces; also quote ; & | [ ] ( ) ' so shells keep it whole.
        return '"' + arg + '"'
    return quoted


def quote_command(argv: list[str]) -> str:
    return " ".join(quote_arg(a) for a in argv)


def kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the process and everything it started."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _popen(argv: list[str]) -> subprocess.Popen[str]:
    common: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    try:
        if sys.platform == "win32":
            # No console window when launched from a GUI client; own process group.
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            return subprocess.Popen(argv, creationflags=flags, **common)
        return subprocess.Popen(argv, start_new_session=True, **common)
    except OSError as e:
        raise MediaError(f"could not start {argv[0]!r}: {e}") from e


class _Reader(threading.Thread):
    """Drain a pipe on a thread so ffmpeg never blocks on a full buffer."""

    def __init__(self, stream: IO[str], handle: Callable[[str], None]) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._handle = handle
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for line in self._stream:
                self._handle(line)
        except BaseException as e:  # reported by the caller, not swallowed
            self.error = e


def run_steps(
    steps: list[Step],
    *,
    timeout_s: float,
    cancel: threading.Event | None = None,
    on_progress: Callable[[float], None] | None = None,
    total_s: float | None = None,
    on_spawn: Callable[[int], None] | None = None,
) -> None:
    """Run each step in order. Raises MediaError on failure, timeout or cancel.

    on_progress receives a 0..1 fraction of the whole job (all steps), at most
    about twice per second. total_s is the expected output duration per step.
    """
    deadline = time.monotonic() + timeout_s
    n = len(steps)
    for i, step in enumerate(steps):
        _run_one(step, i, n, deadline, timeout_s, cancel, on_progress, total_s, on_spawn)
    if on_progress:
        on_progress(1.0)


def _run_one(
    step: Step,
    index: int,
    count: int,
    deadline: float,
    timeout_s: float,
    cancel: threading.Event | None,
    on_progress: Callable[[float], None] | None,
    total_s: float | None,
    on_spawn: Callable[[int], None] | None,
) -> None:
    proc = _popen(step.argv)
    assert proc.stdout is not None and proc.stderr is not None
    if on_spawn:
        on_spawn(proc.pid)
    err_tail: collections.deque[str] = collections.deque(maxlen=40)
    last_report = 0.0

    def handle_progress(line: str) -> None:
        nonlocal last_report
        key, _, value = line.strip().partition("=")
        if key not in ("out_time_us", "out_time_ms") or not value.isdigit():
            return
        now = time.monotonic()
        if on_progress is None or not total_s or now - last_report < 0.5:
            return
        last_report = now
        frac = min(int(value) / 1e6 / total_s, 1.0)
        on_progress((index + frac) / count)

    out_reader = _Reader(proc.stdout, handle_progress)
    err_reader = _Reader(proc.stderr, err_tail.append)
    out_reader.start()
    err_reader.start()
    try:
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if out_reader.error is not None:
                kill_tree(proc)
                raise MediaError(f"progress reporting failed: {out_reader.error}")
            if cancel is not None and cancel.is_set():
                kill_tree(proc)
                raise MediaError("cancelled")
            if time.monotonic() > deadline:
                kill_tree(proc)
                raise MediaError(
                    f"ffmpeg did not finish within {timeout_s:.0f}s and was stopped "
                    "(raise timeout_s, or use a faster option such as a smaller clip)"
                )
    except BaseException:
        kill_tree(proc)
        raise
    finally:
        out_reader.join(timeout=5)
        err_reader.join(timeout=5)
        proc.stdout.close()
        proc.stderr.close()
    if out_reader.error is not None:
        raise MediaError(f"progress reporting failed: {out_reader.error}")
    if proc.returncode != 0:
        tail = [ln.rstrip() for ln in err_tail if ln.strip()][-8:]
        where = f" ({step.label})" if step.label else ""
        detail = "\n".join("  " + ln for ln in tail) or "  (ffmpeg printed no error text)"
        raise MediaError(f"ffmpeg failed{where} with exit code {proc.returncode}:\n{detail}")
