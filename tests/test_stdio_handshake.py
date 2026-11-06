"""MCP stdio test: launch the real installed server and drive it like a client would.

Covers the tool list, real jobs with progress notifications, dry runs, the --dir
allow-list, that instructive errors survive the trip to the agent, and cancelling
a running job (ffmpeg must die and leave no partial file behind).
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

from conftest import ffprobe_json  # noqa: E402

EXPECTED_TOOLS = {
    "media_info",
    "convert",
    "compress",
    "trim",
    "extract_audio",
    "make_gif",
    "extract_frames",
    "concat",
    "transform",
}


@asynccontextmanager
async def session(
    *args: str, env: dict[str, str] | None = None, cwd: Path | None = None
) -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mediakit_mcp", *args], env=env, cwd=cwd
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        yield s


def text(res: object) -> str:
    return res.content[0].text  # type: ignore[attr-defined,no-any-return]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_handshake_and_tool_list() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mediakit_mcp"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        init = await s.initialize()
        assert init.server_info.name == "mediakit"
        tools = (await s.list_tools()).tools
        assert {t.name for t in tools} == EXPECTED_TOOLS
        for t in tools:
            # Every tool is documented for the agent, and the safety switches are in the schema.
            assert t.description and len(t.description) > 80, t.name
            if t.name != "media_info":
                assert {"dry_run", "overwrite"} <= set(t.input_schema["properties"]), t.name


@pytest.mark.anyio
async def test_installed_console_script_starts() -> None:
    scripts = Path(sysconfig.get_path("scripts"))
    script = scripts / ("mediakit-mcp.exe" if sys.platform == "win32" else "mediakit-mcp")
    if not script.exists():
        pytest.skip("console script not installed in this environment")
    out = subprocess.run([str(script), "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "mediakit-mcp 0.1.0" in out.stdout


@pytest.mark.anyio
async def test_real_jobs_over_mcp_with_progress(work: Path) -> None:
    updates: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        updates.append((progress, total, message))

    async with session() as s:
        info = await s.call_tool("media_info", {"input": str(work / "video.mp4")})
        assert not info.is_error and "h264" in text(info) and "6.00s" in text(info)

        res = await s.call_tool(
            "compress",
            {"input": str(work / "video.mp4"), "target_mb": 1},
            progress_callback=on_progress,
        )
        assert not res.is_error, text(res)
        assert "target 1 MB: met" in text(res)
        out = work / "video.compressed.mp4"
        assert out.stat().st_size <= 1024 * 1024
        assert ffprobe_json(out)["streams"][0]["codec_name"] == "h264"

        res = await s.call_tool("make_gif", {"input": str(work / "video.mp4"), "duration": 1, "fps": 8})
        assert not res.is_error and (work / "video.gif").exists()
        res = await s.call_tool("extract_frames", {"input": str(work / "video.mp4"), "times": [1, "0:02"]})
        assert not res.is_error and "Wrote 2 files" in text(res)
        res = await s.call_tool("extract_audio", {"input": str(work / "video.mp4"), "format": "wav"})
        assert not res.is_error and (work / "video.audio.wav").exists()
        res = await s.call_tool("transform", {"input": str(work / "video.mp4"), "width": 160, "speed": 2})
        assert not res.is_error and (work / "video.transformed.mp4").exists()
        res = await s.call_tool("convert", {"input": str(work / "tone.wav"), "format": "flac"})
        assert not res.is_error and (work / "tone.flac").exists()
        res = await s.call_tool("trim", {"input": str(work / "video.mp4"), "start": 1, "end": 2, "mode": "fast"})
        assert not res.is_error, text(res)
        a, b = work / "video.trimmed.mp4", work / "b.mp4"
        b.write_bytes(a.read_bytes())
        res = await s.call_tool("concat", {"inputs": [str(a), str(b)]})
        assert not res.is_error and (work / "video.trimmed.concat.mp4").exists(), text(res)

    assert updates, "no progress notifications reached the client"
    assert all(total == 100.0 for _, total, _ in updates)
    assert updates == sorted(updates)


@pytest.mark.anyio
async def test_dry_run_returns_the_command_and_writes_nothing(work: Path) -> None:
    before = sorted(p.name for p in work.iterdir())
    async with session() as s:
        for tool, args in [
            ("compress", {"input": str(work / "video.mp4"), "preset": "discord-10mb"}),
            ("convert", {"input": str(work / "video.mp4"), "format": "webm"}),
            ("trim", {"input": str(work / "video.mp4"), "start": 1, "duration": 2}),
            ("make_gif", {"input": str(work / "video.mp4")}),
            ("extract_frames", {"input": str(work / "video.mp4"), "every_s": 2}),
            ("extract_audio", {"input": str(work / "video.mp4")}),
            ("transform", {"input": str(work / "video.mp4"), "crop": "100:100"}),
            ("concat", {"inputs": [str(work / "video.mp4"), str(work / "video.mp4")]}),
        ]:
            res = await s.call_tool(tool, {**args, "dry_run": True})
            assert not res.is_error, (tool, text(res))
            assert text(res).startswith("DRY RUN") and "ffmpeg" in text(res), tool
            assert "-progress" in text(res)
    assert sorted(p.name for p in work.iterdir()) == before


@pytest.mark.anyio
async def test_instructive_errors_reach_the_agent(work: Path) -> None:
    async with session() as s:
        res = await s.call_tool("media_info", {"input": str(work / "nope.mp4")})
        assert res.is_error and "input file not found" in text(res)
        res = await s.call_tool("media_info", {"input": str(work / "corrupt.mp4")})
        assert res.is_error and "may be corrupt" in text(res)
        res = await s.call_tool("extract_audio", {"input": str(work / "silent.mp4")})
        assert res.is_error and "no audio stream" in text(res)
        await s.call_tool("trim", {"input": str(work / "video.mp4"), "start": 0, "duration": 1})
        res = await s.call_tool("trim", {"input": str(work / "video.mp4"), "start": 0, "duration": 1})
        assert res.is_error and "already exists" in text(res) and "overwrite=true" in text(res)
        res = await s.call_tool("compress", {"input": str(work / "video.mp4"), "target_mb": 0.001})
        assert res.is_error and "too small" in text(res)
        res = await s.call_tool("compress", {"input": str(work / "video.mp4"), "preset": "nope"})
        assert res.is_error and "unknown preset" in text(res)
        res = await s.call_tool("trim", {"input": str(work / "video.mp4"), "start": "soon"})
        assert res.is_error and "not a time" in text(res)


@pytest.mark.anyio
async def test_missing_ffmpeg_is_reported_with_install_hints(work: Path, tmp_path: Path) -> None:
    env = {"FFMPEG_PATH": str(tmp_path / "nope.exe"), "FFPROBE_PATH": str(tmp_path / "nope.exe")}
    async with session(env=env) as s:
        assert {t.name for t in (await s.list_tools()).tools} == EXPECTED_TOOLS  # still starts
        res = await s.call_tool("media_info", {"input": str(work / "video.mp4")})
        assert res.is_error
        assert "FFPROBE_PATH is set to" in text(res) and "winget install Gyan.FFmpeg" in text(res)


@pytest.mark.anyio
async def test_dir_flag_restricts_the_server(work: Path) -> None:
    allowed = work / "allowed"
    allowed.mkdir()
    (allowed / "ok.mp4").write_bytes((work / "silent.mp4").read_bytes())
    async with session("--dir", str(allowed)) as s:
        res = await s.call_tool("media_info", {"input": str(allowed / "ok.mp4")})
        assert not res.is_error
        res = await s.call_tool("media_info", {"input": str(work / "video.mp4")})
        assert res.is_error and "outside the allowed directories" in text(res)
        res = await s.call_tool(
            "trim", {"input": str(allowed / "ok.mp4"), "start": 0, "duration": 1,
                     "output": str(work / "escape.mp4")}
        )  # fmt: skip
        assert res.is_error and "outside the allowed directories" in text(res)
        assert not (work / "escape.mp4").exists()


def ffmpeg_pids_mentioning(fragment: str) -> list[int]:
    if sys.platform == "win32":
        cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='ffmpeg.exe'\" | "
            f"Where-Object {{ $_.CommandLine -like '*{fragment}*' }} | "
            "ForEach-Object { $_.ProcessId }"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    else:
        out = subprocess.run(["pgrep", "-f", fragment], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


@pytest.mark.anyio
async def test_cancelling_a_call_kills_ffmpeg_and_cleans_up(work: Path) -> None:
    # 0.25x speed of a 6 s clip = 24 s of VP9 output: long enough to cancel mid-way.
    marker = work.name
    async with session() as s:
        with anyio.move_on_after(2.5):
            await s.call_tool(
                "transform", {"input": str(work / "video.mp4"), "speed": 0.25, "width": 640,
                              "output": str(work / "slow.webm")},
                read_timeout_seconds=None,
            )  # fmt: skip
        # Leaving the scope cancelled the request (the client sends notifications/cancelled).
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and ffmpeg_pids_mentioning(marker):
            await anyio.sleep(0.3)
        assert ffmpeg_pids_mentioning(marker) == [], "ffmpeg survived the cancel"
        await anyio.sleep(0.5)
        assert not list(work.glob("*.partial.*")), "partial output was left behind"
        assert not (work / "slow.webm").exists()
        # The server is still healthy afterwards.
        res = await s.call_tool("media_info", {"input": str(work / "video.mp4")})
        assert not res.is_error
