"""Bitrate maths for hitting a target file size.

size = (video_bitrate + audio_bitrate) * duration / 8, so for a target we solve
for the video bitrate after reserving audio and a safety margin for container
overhead and rate-control error. Pure functions: no ffmpeg needed to test them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import MediaError

# Aim a little under the target: muxing overhead plus two-pass inaccuracy.
SAFETY = 0.95
# libvpx-vp9 rate control wanders more than x264 (measured up to 99.7% of target at 0.95).
SAFETY_VP9 = 0.90
MIB = 1024 * 1024
AUDIO_STEPS = (128_000, 96_000, 64_000, 32_000)
MIN_VIDEO_BPS = 60_000
# Below this many bits per pixel per frame, H.264 looks like mush: shrink instead.
BITS_PER_PIXEL_FLOOR = 0.045
HEIGHT_LADDER = (2160, 1440, 1080, 720, 480, 360, 240, 144)


@dataclass(frozen=True)
class BitratePlan:
    video_bps: int
    audio_bps: int
    height: int
    scaled: bool


def plan_bitrates(
    target_mb: float,
    duration_s: float,
    *,
    has_audio: bool,
    width: int,
    height: int,
    fps: float | None,
    max_height: int | None = None,
    safety: float = SAFETY,
) -> BitratePlan:
    if target_mb <= 0:
        raise MediaError("target_mb must be positive")
    if duration_s <= 0:
        raise MediaError("cannot plan a size target without a known duration")
    total_bps = target_mb * MIB * 8 * safety / duration_s

    audio_bps = 0
    if has_audio:
        audio_bps = AUDIO_STEPS[-1]
        for step in AUDIO_STEPS:
            if step <= 0.25 * total_bps:
                audio_bps = step
                break
    video_bps = int(total_bps - audio_bps)
    if video_bps < MIN_VIDEO_BPS:
        need = (MIN_VIDEO_BPS + audio_bps) * duration_s / 8 / MIB / safety
        raise MediaError(
            f"{target_mb:g} MB is too small for {duration_s:.1f}s of video "
            f"(it would need about {need:.2f} MB even at minimum quality). "
            "Trim the clip first or raise the target."
        )

    start_h = min(height, max_height) if max_height else height
    candidates = [start_h] + [h for h in HEIGHT_LADDER if h < start_h]
    chosen = candidates[-1]
    rate = fps or 30.0
    for h in candidates:
        w = width * h / height
        if video_bps / (w * h * rate) >= BITS_PER_PIXEL_FLOOR:
            chosen = h
            break
    return BitratePlan(video_bps, audio_bps, chosen, scaled=chosen != height)
