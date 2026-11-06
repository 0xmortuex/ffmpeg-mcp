"""mediakit-mcp: everyday media jobs for AI agents, by intent rather than by flag."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

import anyio
import anyio.from_thread
import anyio.lowlevel
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer

from . import __version__, ops
from .errors import MediaError
from .paths import allowed_dirs, resolve_input, set_allowed_dirs
from .plans import Plan, execute, format_dry_run
from .probe import describe, probe

mcp = MCPServer(
    "mediakit",
    instructions=(
        "Everyday audio/video jobs powered by ffmpeg. Start with media_info to see what a file "
        "is. Then pick the tool for the intent: compress (to a size like 10 MB or a preset such "
        "as discord-10mb / whatsapp), convert (mp4, webm, mkv, mp3, wav, gif...), trim, "
        "extract_audio, make_gif, extract_frames (thumbnails), concat (join same-format clips) "
        "and transform (resize, crop, speed). Every tool accepts dry_run=true and then returns "
        "the exact ffmpeg command line without running it. Outputs are written next to the "
        "input (clip.compressed.mp4) and existing files are never overwritten unless "
        "overwrite=true. Paths may be absolute or relative to the server's working directory."
    ),
)

Timeout = 900


async def _run(
    planner: Callable[[], Plan], dry_run: bool, timeout_s: int, ctx: Context[Any, Any]
) -> str:
    if timeout_s < 1:
        raise MediaError("timeout_s must be at least 1")
    plan = await anyio.to_thread.run_sync(planner)
    if dry_run:
        return format_dry_run(plan)

    cancel = threading.Event()
    name = plan.tool

    # Progress is reported from the runner's reader thread, so hand it the loop explicitly.
    token = anyio.lowlevel.current_token()

    def progress(fraction: float) -> None:
        anyio.from_thread.run(
            ctx.report_progress, fraction * 100, 100.0, f"{name}: encoding", token=token
        )

    def work() -> str:
        return execute(plan, timeout_s=timeout_s, cancel=cancel, on_progress=progress)

    try:
        # abandon_on_cancel: if the client cancels, return at once; the worker sees the
        # event, kills ffmpeg's process tree and removes the partial output.
        return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        cancel.set()
        raise


@mcp.tool()
async def media_info(input: str) -> str:
    """Summarize a media file with ffprobe: container, duration, size, bitrate, and every
    stream (codec, resolution, fps, sample rate, channels). Use it first to learn what you have
    before compressing or converting, and to check results afterwards.

    input: path to an audio/video/image file.
    """

    def job() -> str:
        return describe(probe(resolve_input(input)))

    return await anyio.to_thread.run_sync(job)


@mcp.tool()
async def convert(
    input: str,
    ctx: Context[Any, Any],
    format: str | None = None,
    output: str | None = None,
    quality: str = "medium",
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Convert a file to another format with sensible codecs.

    Video: mp4/mov/mkv (H.264 + AAC), webm (VP9 + Opus). Audio: mp3, wav, flac, ogg, m4a, opus,
    aac. gif: the first 5 seconds (use make_gif for control). Choose the target with format
    (e.g. "webm") or by the extension of output; the default output sits next to the input.

    quality: low | medium | high (trades size for fidelity). For a size limit use compress.
    overwrite: replace an existing output (default: refuse). dry_run: return the ffmpeg command
    instead of running it. timeout_s: stop ffmpeg after this many seconds.
    """
    return await _run(
        lambda: ops.plan_convert(input, format, output, quality, overwrite), dry_run, timeout_s, ctx
    )


@mcp.tool()
async def compress(
    input: str,
    ctx: Context[Any, Any],
    target_mb: float | None = None,
    preset: str | None = None,
    quality: str = "medium",
    max_height: int | None = None,
    output: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Make a video smaller. Three ways to say how:

    - target_mb: aim for a file size in MB (MiB). Uses a two-pass encode with bitrate maths
      (video bitrate = target size / duration - audio, with a 5% safety margin) and lowers
      the resolution automatically when the bitrate would be too low to look decent. The
      result reports whether the target was met; errors if the target is impossibly small.
    - preset: discord-10mb, discord-50mb, whatsapp (16 MB, 720p max), email-25mb (aims at
      18 MB because attachments grow ~37% when mailed), or web (quality-based, 1080p max).
    - neither: a quality-based (CRF) re-encode controlled by quality low|medium|high.

    max_height: never exceed this height in pixels. output ending in .webm uses VP9/Opus,
    otherwise H.264/AAC (default output: name.compressed.mp4 next to the input).
    overwrite: replace an existing output. dry_run: return the ffmpeg commands (both passes)
    without running them. timeout_s: stop after this many seconds.
    """
    return await _run(
        lambda: ops.plan_compress(
            input, target_mb, preset, quality, max_height, output, overwrite
        ),
        dry_run,
        timeout_s,
        ctx,
    )


@mcp.tool()
async def trim(
    input: str,
    ctx: Context[Any, Any],
    start: str | float | None = None,
    end: str | float | None = None,
    duration: str | float | None = None,
    mode: str = "accurate",
    output: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Cut a section out of a video or audio file.

    start, end, duration: seconds (12.5) or [HH:]MM:SS[.ms] ("1:02:03.5"). Give start and
    end, or start and duration; with only start the cut runs to the end of the file.
    mode: "accurate" (default) re-encodes so the cut is frame-exact; "fast" copies the streams
    without re-encoding (instant, lossless) but the cut snaps to the keyframe before start,
    so the clip may begin a little early. Default output: name.trimmed.ext.
    """
    return await _run(
        lambda: ops.plan_trim(input, start, end, duration, mode, output, overwrite),
        dry_run,
        timeout_s,
        ctx,
    )


@mcp.tool()
async def extract_audio(
    input: str,
    ctx: Context[Any, Any],
    format: str = "mp3",
    bitrate: str | None = None,
    quality: str = "medium",
    output: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Pull the audio track out of a video (or re-encode an audio file).

    format: mp3 (default), wav, flac, ogg, m4a, opus, aac, or "copy" to keep the original
    audio stream untouched (fast, lossless; picks the matching extension). bitrate: for lossy
    formats, e.g. "192k" (overrides quality). Fails clearly if the input has no audio.
    Default output: name.audio.mp3 next to the input.
    """
    return await _run(
        lambda: ops.plan_extract_audio(input, format, bitrate, quality, output, overwrite),
        dry_run,
        timeout_s,
        ctx,
    )


@mcp.tool()
async def make_gif(
    input: str,
    ctx: Context[Any, Any],
    start: str | float | None = None,
    duration: str | float | None = None,
    fps: int = 12,
    width: int = 480,
    loop: bool = True,
    output: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Make a high-quality animated GIF from a video (two-step palette: palettegen + paletteuse).

    start: where to begin (default 0). duration: seconds to capture (default 5, max 30; it is
    shortened if the clip ends sooner). fps 1-30 (default 12). width in pixels (default 480,
    height keeps the aspect ratio; never upscaled). loop: repeat forever (default true).
    GIFs grow fast: for long clips use convert to mp4/webm instead.
    """
    return await _run(
        lambda: ops.plan_gif(input, output, start, duration, fps, width, loop, overwrite),
        dry_run,
        timeout_s,
        ctx,
    )


@mcp.tool()
async def extract_frames(
    input: str,
    ctx: Context[Any, Any],
    times: list[str | float] | None = None,
    every_s: float | None = None,
    format: str = "png",
    width: int | None = None,
    max_frames: int = 100,
    output_dir: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Save still images (thumbnails) from a video.

    Either times (a list like [1, 12.5, "0:01:30"]: one image per time) or every_s (one image
    every N seconds, starting at 0). With neither, one frame from the middle. format: png
    (default) or jpg. width: resize to this width (aspect kept). max_frames guards against
    producing thousands of files. Images go to output_dir (default: a "name.frames" folder next
    to the input) as name.frame-12.5s.png or name.frame-0001.png.
    """
    return await _run(
        lambda: ops.plan_frames(
            input, times, every_s, format, width, max_frames, output_dir, overwrite
        ),
        dry_run,
        timeout_s,
        ctx,
    )


@mcp.tool()
async def concat(
    inputs: list[str],
    ctx: Context[Any, Any],
    output: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Join clips end to end without re-encoding (fast and lossless).

    The clips must share codec, resolution, pixel format, frame rate and audio parameters (for
    example several recordings from one camera or pieces made by trim). If they differ, the
    error says exactly which property differs; convert/transform them to match first.
    Default output: first.concat.ext next to the first input.
    """
    return await _run(lambda: ops.plan_concat(inputs, output, overwrite), dry_run, timeout_s, ctx)


@mcp.tool()
async def transform(
    input: str,
    ctx: Context[Any, Any],
    width: int | None = None,
    height: int | None = None,
    crop: str | None = None,
    speed: float | None = None,
    output: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_s: int = Timeout,
) -> str:
    """Resize, crop and/or change the speed of a video (re-encodes with H.264/AAC, or VP9 for .webm).

    width / height: target size in pixels; give one and the other keeps the aspect ratio.
    crop: "W:H" (centered) or "W:H:X:Y" in pixels, applied before resizing. speed: 0.25-4
    (2 = twice as fast, 0.5 = slow motion); audio is retimed to match. Default output:
    name.transformed.ext.
    """
    return await _run(
        lambda: ops.plan_transform(input, width, height, crop, speed, output, overwrite),
        dry_run,
        timeout_s,
        ctx,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mediakit-mcp", description="MCP server for everyday ffmpeg jobs."
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="DIR",
        help="only allow input/output files inside this directory (repeatable; also "
        "MEDIAKIT_ALLOWED_DIRS, separated by the OS path separator). Default: unrestricted.",
    )
    parser.add_argument("--version", action="version", version=f"mediakit-mcp {__version__}")
    args = parser.parse_args(argv)
    dirs: list[str] = list(args.dir)
    env = os.environ.get("MEDIAKIT_ALLOWED_DIRS")
    if env:
        dirs.extend(d for d in env.split(os.pathsep) if d)
    try:
        set_allowed_dirs(dirs)
    except MediaError as e:
        sys.exit(f"mediakit-mcp: {e}")
    if allowed_dirs():
        print(
            "mediakit-mcp: restricted to " + ", ".join(str(d) for d in allowed_dirs()),
            file=sys.stderr,
        )
    mcp.run()


if __name__ == "__main__":
    main()
