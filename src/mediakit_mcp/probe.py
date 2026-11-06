"""ffprobe wrapper: turn the JSON into a small typed summary."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .binaries import find_tool
from .errors import MediaError


@dataclass
class Stream:
    index: int
    kind: str
    codec: str
    profile: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    pix_fmt: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    bit_rate: int | None = None
    attached_pic: bool = False


@dataclass
class MediaInfo:
    path: Path
    format_name: str
    duration: float | None
    size: int
    bit_rate: int | None
    streams: list[Stream] = field(default_factory=list)

    @property
    def video(self) -> Stream | None:
        return next((s for s in self.streams if s.kind == "video" and not s.attached_pic), None)

    @property
    def audio(self) -> Stream | None:
        return next((s for s in self.streams if s.kind == "audio"), None)

    def require_duration(self) -> float:
        if self.duration is None or self.duration <= 0:
            raise MediaError(
                f"cannot determine the duration of {self.path.name!r} "
                "(streaming or broken container?); convert it to mp4 first"
            )
        return self.duration


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fps(v: Any) -> float | None:
    try:
        f = Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        return None
    return float(f) if f > 0 else None


def tail_lines(text: str, n: int = 8) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join("  " + ln for ln in lines[-n:])


def probe(path: Path, timeout_s: float = 60.0) -> MediaInfo:
    ffprobe = find_tool("ffprobe")
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise MediaError(f"ffprobe timed out after {timeout_s:.0f}s reading {path.name!r}") from None
    if proc.returncode != 0:
        raise MediaError(
            f"ffprobe could not read {path.name!r}: the file may be corrupt, truncated, or not "
            f"media at all.\n{tail_lines(proc.stderr)}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise MediaError(f"ffprobe returned unreadable output for {path.name!r}") from None
    fmt = data.get("format") or {}
    streams: list[Stream] = []
    for s in data.get("streams") or []:
        streams.append(
            Stream(
                index=int(s.get("index", len(streams))),
                kind=str(s.get("codec_type", "unknown")),
                codec=str(s.get("codec_name", "unknown")),
                profile=s.get("profile"),
                width=_int(s.get("width")),
                height=_int(s.get("height")),
                fps=_fps(s.get("avg_frame_rate")),
                pix_fmt=s.get("pix_fmt"),
                sample_rate=_int(s.get("sample_rate")),
                channels=_int(s.get("channels")),
                channel_layout=s.get("channel_layout"),
                bit_rate=_int(s.get("bit_rate")),
                attached_pic=bool((s.get("disposition") or {}).get("attached_pic")),
            )
        )
    if not any(s.kind in ("video", "audio") for s in streams):
        raise MediaError(f"{path.name!r} contains no audio or video streams")
    return MediaInfo(
        path=path,
        format_name=str(fmt.get("format_name", "unknown")),
        duration=_float(fmt.get("duration")),
        size=_int(fmt.get("size")) or path.stat().st_size,
        bit_rate=_int(fmt.get("bit_rate")),
        streams=streams,
    )


def fmt_size(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    n /= 1024
    for unit in ("KB", "MB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def fmt_duration(s: float | None) -> str:
    if s is None:
        return "unknown"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{int(h)}:{int(m):02d}:{sec:05.2f}"
    if m:
        return f"{int(m)}:{sec:05.2f}"
    return f"{sec:.2f}s"


def fmt_rate(bps: float | None) -> str:
    if not bps:
        return "unknown"
    return f"{bps / 1e6:.2f} Mb/s" if bps >= 1e6 else f"{bps / 1e3:.0f} kb/s"


def describe(info: MediaInfo) -> str:
    lines = [
        str(info.path),
        f"  container: {info.format_name} | duration: {fmt_duration(info.duration)} | "
        f"size: {fmt_size(info.size)} | bitrate: {fmt_rate(info.bit_rate)}",
    ]
    for s in info.streams:
        if s.kind == "video":
            extra = " (cover art)" if s.attached_pic else ""
            fps = f" {s.fps:.3g} fps" if s.fps else ""
            prof = f" ({s.profile})" if s.profile else ""
            rate = f" {fmt_rate(s.bit_rate)}" if s.bit_rate else ""
            lines.append(
                f"  video #{s.index}: {s.codec}{prof} {s.width}x{s.height}{fps} "
                f"{s.pix_fmt or ''}{rate}{extra}"
            )
        elif s.kind == "audio":
            rate = f" {fmt_rate(s.bit_rate)}" if s.bit_rate else ""
            lines.append(
                f"  audio #{s.index}: {s.codec} {s.sample_rate} Hz "
                f"{s.channel_layout or s.channels}{rate}"
            )
        else:
            lines.append(f"  {s.kind} #{s.index}: {s.codec}")
    if info.audio is None:
        lines.append("  (no audio stream)")
    if info.video is None:
        lines.append("  (no video stream)")
    return "\n".join(lines)
