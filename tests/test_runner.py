"""The process runner against real processes: progress, timeout, cancel, tree kill, errors."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mediakit_mcp.errors import MediaError
from mediakit_mcp.plans import ffmpeg_argv
from mediakit_mcp.runner import Step, quote_arg, quote_command, run_steps


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True
        ).stdout
        return f'"{pid}"' in out
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_dead(pid: int, seconds: float = 10.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def realtime_job(out: Path, seconds: int) -> Step:
    """ffmpeg encoding a lavfi source in real time, so it takes `seconds` of wall clock."""
    return Step(
        ffmpeg_argv(
            "-re", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
            "-c:v", "libx264", str(out),
        )  # fmt: skip
    )


def test_progress_is_parsed_and_monotonic(tmp_path: Path) -> None:
    seen: list[float] = []
    run_steps(
        [realtime_job(tmp_path / "o.mp4", 3)],
        timeout_s=60,
        on_progress=seen.append,
        total_s=3.0,
    )
    assert seen[-1] == 1.0
    assert any(0.0 < f < 1.0 for f in seen), seen
    assert seen == sorted(seen)
    assert (tmp_path / "o.mp4").stat().st_size > 0


def test_progress_spans_multiple_steps(tmp_path: Path) -> None:
    seen: list[float] = []
    run_steps(
        [realtime_job(tmp_path / "a.mp4", 2), realtime_job(tmp_path / "b.mp4", 2)],
        timeout_s=60,
        on_progress=seen.append,
        total_s=2.0,
    )
    assert any(f > 0.5 for f in seen[:-1]), "second step should report beyond 50%"
    assert seen[-1] == 1.0


def test_timeout_kills_ffmpeg(tmp_path: Path) -> None:
    pids: list[int] = []
    t0 = time.monotonic()
    with pytest.raises(MediaError, match="did not finish within 1s"):
        run_steps([realtime_job(tmp_path / "o.mp4", 60)], timeout_s=1, on_spawn=pids.append)
    assert time.monotonic() - t0 < 15
    assert wait_dead(pids[0])


def test_cancel_kills_ffmpeg(tmp_path: Path) -> None:
    pids: list[int] = []
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(MediaError, match="cancelled"):
        run_steps(
            [realtime_job(tmp_path / "o.mp4", 60)], timeout_s=120, cancel=cancel, on_spawn=pids.append
        )
    assert time.monotonic() - t0 < 15
    assert wait_dead(pids[0])


def test_kill_takes_down_the_whole_process_tree(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "open(sys.argv[1], 'w').write(str(c.pid))\n"
        "time.sleep(120)\n"
    )
    parent: list[int] = []
    with pytest.raises(MediaError, match="did not finish"):
        run_steps(
            [Step([sys.executable, "-c", script, str(pidfile)])],
            timeout_s=3,
            on_spawn=parent.append,
        )
    grandchild = int(pidfile.read_text())
    assert wait_dead(parent[0])
    assert wait_dead(grandchild), "the grandchild survived the kill"


def test_failure_reports_the_tail_of_stderr_not_the_log(tmp_path: Path) -> None:
    step = Step(ffmpeg_argv("-i", str(tmp_path / "nope.mp4"), str(tmp_path / "o.mp4")))
    with pytest.raises(MediaError) as err:
        run_steps([step], timeout_s=30)
    msg = str(err.value)
    assert "ffmpeg failed with exit code" in msg and "nope.mp4" in msg
    assert len(msg.splitlines()) <= 10
    assert "Stream #" not in msg  # no banner/log spam


def test_failure_in_a_later_step_is_labelled(tmp_path: Path) -> None:
    ok = realtime_job(tmp_path / "a.mp4", 1)
    bad = Step(ffmpeg_argv("-i", str(tmp_path / "nope.mp4"), str(tmp_path / "b.mp4")), "pass 2")
    with pytest.raises(MediaError, match=r"\(pass 2\)"):
        run_steps([ok, bad], timeout_s=60)


def test_missing_executable_is_a_clear_error() -> None:
    with pytest.raises(MediaError, match="could not start"):
        run_steps([Step(["definitely-not-a-real-binary-xyz"])], timeout_s=5)


def test_a_failing_progress_callback_is_not_swallowed(tmp_path: Path) -> None:
    def boom(_: float) -> None:
        raise RuntimeError("client went away")

    with pytest.raises(MediaError, match="progress reporting failed: client went away"):
        run_steps([realtime_job(tmp_path / "o.mp4", 30)], timeout_s=30, on_progress=boom, total_s=30)


# ------------------------------------------------------------- quoting


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell quoting")
def test_windows_quoting() -> None:
    assert quote_arg(r"C:\Users\me\clip.mp4") == r"C:\Users\me\clip.mp4"
    assert quote_arg(r"C:\My Videos\clip.mp4") == r'"C:\My Videos\clip.mp4"'
    graph = "fps=12,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    assert quote_arg(graph) == f'"{graph}"'  # ';' must not split the command in PowerShell
    assert quote_arg("") == '""'
    assert quote_command(["ffmpeg", "-i", r"C:\a b\c.mp4"]) == r'ffmpeg -i "C:\a b\c.mp4"'


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell quoting")
def test_posix_quoting() -> None:
    assert quote_arg("a b") == "'a b'"
    assert quote_arg("plain.mp4") == "plain.mp4"


@pytest.mark.skipif(sys.platform != "win32", reason="needs PowerShell")
def test_quoted_windows_command_round_trips_through_powershell(tmp_path: Path) -> None:
    weird = tmp_path / "it has spaces; and [brackets].txt"
    script = "import sys; print(repr(sys.argv[1:]))"
    cmd = quote_command([sys.executable, "-c", script, str(weird), "a;b"])
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"& {cmd}"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == repr([str(weird), "a;b"])


def test_one_enormous_stderr_line_does_not_flood_the_error() -> None:
    script = "import sys; sys.stderr.write('x' * 2_000_000); sys.exit(1)"
    with pytest.raises(MediaError) as err:
        run_steps([Step([sys.executable, "-c", script])], timeout_s=30)
    assert len(str(err.value)) < 1000
