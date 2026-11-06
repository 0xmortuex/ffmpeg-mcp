"""Real media for every test: generated once per session with ffmpeg's lavfi sources."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mediakit_mcp.binaries import find_tool
from mediakit_mcp.paths import set_allowed_dirs


def ffmpeg(*args: str) -> None:
    proc = subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr


def ffprobe_json(path: Path, *, count_frames: bool = False) -> dict:  # type: ignore[type-arg]
    cmd = [find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
           "-show_streams"]
    if count_frames:
        cmd.append("-count_frames")
    proc = subprocess.run([*cmd, str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)  # type: ignore[no-any-return]


def stream(data: dict, kind: str) -> dict:  # type: ignore[type-arg]
    return next(s for s in data["streams"] if s["codec_type"] == kind)


@pytest.fixture(autouse=True)
def _unrestricted() -> None:
    set_allowed_dirs([])


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of generated source files (treat as read-only; copy to modify)."""
    d = tmp_path_factory.mktemp("media")
    if shutil.which("ffmpeg") is None:
        find_tool("ffmpeg")
    # 6 s, 640x360@30, noisy so it is several MB and compression is meaningful; with audio.
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
        "-vf", "noise=alls=25:allf=t+u", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(d / "video.mp4"),
    )  # fmt: skip
    # Same content without audio.
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "silent.mp4"),
    )  # fmt: skip
    # Odd dimensions (H.264 needs even): 321x241.
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
        "-c:v", "libx264", "-vf", "crop=319:239", "-pix_fmt", "yuv444p", str(d / "odd.mkv"),
    )  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=880:duration=3", str(d / "tone.wav"))
    (d / "corrupt.mp4").write_bytes(b"this is definitely not an mp4 file" * 50)
    good = (d / "video.mp4").read_bytes()
    (d / "truncated.mp4").write_bytes(good[:3000])
    assert (d / "video.mp4").stat().st_size > 2 * 1024 * 1024, "fixture should be several MB"
    return d


@pytest.fixture
def work(media: Path, tmp_path: Path) -> Path:
    """A fresh per-test copy of the generated media."""
    for f in media.iterdir():
        shutil.copy(f, tmp_path / f.name)
    return tmp_path
