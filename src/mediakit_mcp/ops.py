"""Planners: turn an intent ("compress to 10 MB") into exact ffmpeg command lines.

Every planner validates its arguments, probes the input (read-only) and returns a
Plan. Nothing is written until the Plan is executed, which is what makes dry_run
cheap and honest: the dry run prints these same argument lists.
"""

from __future__ import annotations

import math
import re
import tempfile
import uuid
from pathlib import Path

from .binaries import require_encoders
from .errors import MediaError
from .paths import derived, resolve_dir, resolve_input, resolve_output
from .plans import Plan, commit_file, ffmpeg_argv, simple_plan
from .probe import MediaInfo, probe
from .runner import Step
from .sizing import SAFETY, SAFETY_VP9, plan_bitrates

VIDEO_EXTS = ("mp4", "mov", "mkv", "webm")
AUDIO_EXTS = ("mp3", "wav", "flac", "ogg", "m4a", "opus", "aac")
ALL_EXTS = (*VIDEO_EXTS, *AUDIO_EXTS, "gif")
QUALITIES = ("low", "medium", "high")

H264_CRF = {"low": "28", "medium": "23", "high": "19"}
VP9_CRF = {"low": "38", "medium": "33", "high": "28"}
AAC_BITRATE = {"low": "96k", "medium": "128k", "high": "192k"}
MP3_Q = {"low": "6", "medium": "4", "high": "2"}
VORBIS_Q = {"low": "3", "medium": "5", "high": "7"}
OPUS_BITRATE = {"low": "64k", "medium": "96k", "high": "160k"}

# Presets: target size (MB, MiB-based) and an optional height cap.
# email-25mb aims at 18 MB because attachments are base64-encoded (about +37%).
PRESETS: dict[str, tuple[float | None, int | None, str]] = {
    "discord-10mb": (10.0, None, "Discord free-tier upload limit"),
    "discord-50mb": (50.0, None, "Discord boosted upload limit"),
    "whatsapp": (16.0, 720, "WhatsApp 16 MB video limit, capped at 720p"),
    "email-25mb": (18.0, None, "25 MB mail limit, minus ~37% base64 overhead"),
    "web": (None, 1080, "quality-based H.264 for web pages, capped at 1080p, no size target"),
}

COPY_EXT = {
    "aac": "m4a",
    "alac": "m4a",
    "mp3": "mp3",
    "opus": "opus",
    "vorbis": "ogg",
    "flac": "flac",
    "pcm_s16le": "wav",
}

GIF_MAX_SECONDS = 30.0


# ---------------------------------------------------------------- helpers


def parse_time(value: str | float | int, name: str) -> float:
    """Seconds as a number, or 'SS', 'MM:SS', 'HH:MM:SS(.ms)'."""
    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
        else:
            text = value.strip().lower().removesuffix("s")
            parts = text.split(":")
            if not 1 <= len(parts) <= 3:
                raise ValueError
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60 + float(part)
    except ValueError:
        raise MediaError(
            f"{name}={value!r} is not a time. Use seconds (12.5) or [HH:]MM:SS[.ms] (1:02:03.5)."
        ) from None
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        raise MediaError(f"{name}={value!r} must be zero or positive")
    return seconds


def _t(seconds: float) -> str:
    return f"{seconds:.3f}"


def _quality(q: str) -> str:
    if q not in QUALITIES:
        raise MediaError(f"quality must be one of {', '.join(QUALITIES)} (got {q!r})")
    return q


def _need(info: MediaInfo, *, video: bool = False, audio: bool = False, why: str) -> None:
    if video and info.video is None:
        raise MediaError(f"{info.path.name!r} has no video stream, so it cannot be used to {why}")
    if audio and info.audio is None:
        raise MediaError(f"{info.path.name!r} has no audio stream, so it cannot be used to {why}")


def _ext_of(output: str | None, fmt: str | None) -> str | None:
    from_out = Path(output).suffix.lower().lstrip(".") if output else None
    from_fmt = fmt.lower().lstrip(".") if fmt else None
    if from_out and from_fmt and from_out != from_fmt:
        raise MediaError(
            f"format={fmt!r} contradicts the output extension .{from_out}; "
            "pass only one of them"
        )
    return from_out or from_fmt


def _check_ext(ext: str | None, allowed: tuple[str, ...], what: str) -> str:
    if not ext:
        raise MediaError(f"no target format: pass format=... or an output with an extension ({what})")
    if ext not in allowed:
        raise MediaError(f"unsupported format {ext!r}; supported: {', '.join(allowed)}")
    return ext


def _even_filter(info: MediaInfo) -> list[str]:
    v = info.video
    if v and v.width and v.height and (v.width % 2 or v.height % 2):
        return ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    return []


def _vf(filters: list[str]) -> list[str]:
    return ["-vf", ",".join(filters)] if filters else []


def video_codec_args(ext: str, quality: str) -> tuple[list[str], tuple[str, ...]]:
    """Codec arguments (video + audio) for a video container, and encoders it needs."""
    if ext == "webm":
        args = [
            "-c:v", "libvpx-vp9", "-crf", VP9_CRF[quality], "-b:v", "0",
            "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-pix_fmt", "yuv420p",
            "-c:a", "libopus", "-b:a", OPUS_BITRATE[quality],
        ]  # fmt: skip
        return args, ("libvpx-vp9", "libopus")
    args = [
        "-c:v", "libx264", "-preset", "medium", "-crf", H264_CRF[quality], "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AAC_BITRATE[quality],
    ]  # fmt: skip
    if ext in ("mp4", "mov"):
        args += ["-movflags", "+faststart"]
    return args, ("libx264", "aac")


def audio_codec_args(
    ext: str, quality: str, bitrate: str | None = None
) -> tuple[list[str], tuple[str, ...]]:
    if bitrate and ext in ("wav", "flac"):
        raise MediaError(f"bitrate does not apply to lossless {ext}")
    if ext == "wav":
        return ["-c:a", "pcm_s16le"], ()
    if ext == "flac":
        return ["-c:a", "flac"], ("flac",)
    if ext == "mp3":
        return (
            ["-c:a", "libmp3lame", *(["-b:a", bitrate] if bitrate else ["-q:a", MP3_Q[quality]])],
            ("libmp3lame",),
        )
    if ext == "ogg":
        return (
            ["-c:a", "libvorbis", *(["-b:a", bitrate] if bitrate else ["-q:a", VORBIS_Q[quality]])],
            ("libvorbis",),
        )
    if ext == "opus":
        return ["-c:a", "libopus", "-b:a", bitrate or OPUS_BITRATE[quality]], ("libopus",)
    if ext in ("m4a", "aac"):
        return ["-c:a", "aac", "-b:a", bitrate or AAC_BITRATE[quality]], ("aac",)
    raise MediaError(f"unsupported audio format {ext!r}; supported: {', '.join(AUDIO_EXTS)}")


def _default_video_ext(src: Path) -> str:
    ext = src.suffix.lower().lstrip(".")
    return ext if ext in VIDEO_EXTS else "mp4"


# ---------------------------------------------------------------- convert


def plan_convert(
    input: str, format: str | None, output: str | None, quality: str, overwrite: bool
) -> Plan:
    src = resolve_input(input)
    quality = _quality(quality)
    ext = _check_ext(_ext_of(output, format), ALL_EXTS, "for example mp4, webm, mp3, gif")
    info = probe(src)
    if ext == "gif":
        plan = plan_gif(input, output, 0.0, 5.0, 12, 480, True, overwrite)
        plan.tool = "convert"
        plan.notes.append(
            "gif output keeps only the first 5 seconds at 12 fps / 480 px; "
            "use make_gif to choose the range, fps and width"
        )
        return plan

    is_video = ext in VIDEO_EXTS
    if is_video:
        _need(info, video=True, why=f"make a .{ext} video (use extract_audio for sound only)")
        codec, encoders = video_codec_args(ext, quality)
    else:
        _need(info, audio=True, why=f"make a .{ext} audio file")
        codec, encoders = audio_codec_args(ext, quality)
    if src.suffix.lower() == f".{ext}":
        default = derived(src, "converted", ext)
    else:
        default = src.with_suffix(f".{ext}")
    final = resolve_output(output, default, overwrite=overwrite, inputs=[src])
    require_encoders(*encoders)

    def build(part: Path) -> list[Step]:
        if is_video:
            argv = ffmpeg_argv(
                "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                *_vf(_even_filter(info)), *codec, str(part),
            )  # fmt: skip
        else:
            argv = ffmpeg_argv("-i", str(src), "-map", "0:a:0", "-vn", *codec, str(part))
        return [Step(argv)]

    return simple_plan(
        "convert", src, final, build, overwrite=overwrite, duration_s=info.duration
    )


# ---------------------------------------------------------------- compress


def plan_compress(
    input: str,
    target_mb: float | None,
    preset: str | None,
    quality: str,
    max_height: int | None,
    output: str | None,
    overwrite: bool,
) -> Plan:
    src = resolve_input(input)
    quality = _quality(quality)
    if preset is not None and target_mb is not None:
        raise MediaError("pass either preset or target_mb, not both")
    notes: list[str] = []
    if preset is not None:
        if preset not in PRESETS:
            raise MediaError(f"unknown preset {preset!r}; choose one of: {', '.join(PRESETS)}")
        target_mb, preset_height, why = PRESETS[preset]
        max_height = max_height or preset_height
        notes.append(f"preset {preset}: {why}")
    if max_height is not None and max_height < 16:
        raise MediaError("max_height must be at least 16 pixels")
    info = probe(src)
    _need(info, video=True, why="compress (use convert or extract_audio for audio files)")
    video = info.video
    assert video is not None and video.width and video.height

    out_ext = Path(output).suffix.lower().lstrip(".") if output else None
    if out_ext is None:
        out_ext = "mp4" if preset is not None else _default_video_ext(src)
    if out_ext not in VIDEO_EXTS:
        raise MediaError(f"compress writes {', '.join(VIDEO_EXTS)} files, not .{out_ext}")
    final = resolve_output(output, derived(src, "compressed", out_ext), overwrite=overwrite, inputs=[src])
    codec_args, encoders = video_codec_args(out_ext, quality)
    require_encoders(*encoders)
    is_vp9 = out_ext == "webm"
    duration = info.require_duration() if target_mb is not None else info.duration

    if target_mb is None:
        filters = _even_filter(info)
        if max_height and video.height > max_height:
            filters = [f"scale=-2:{max_height}"]
            notes.append(f"scaled down to {max_height}p")

        def build_crf(part: Path) -> list[Step]:
            return [
                Step(
                    ffmpeg_argv(
                        "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                        *_vf(filters), *codec_args, str(part),
                    )  # fmt: skip
                )
            ]

        notes.append(f"quality-based encode (quality={quality}); no size target")
        return simple_plan(
            "compress", src, final, build_crf, overwrite=overwrite, duration_s=duration, notes=notes
        )

    bp = plan_bitrates(
        target_mb,
        duration or 0,
        has_audio=info.audio is not None,
        width=video.width,
        height=video.height,
        fps=video.fps,
        max_height=max_height,
        safety=SAFETY_VP9 if is_vp9 else SAFETY,
    )
    filters = [f"scale=-2:{bp.height}"] if bp.height != video.height else _even_filter(info)
    if bp.height != video.height:
        notes.append(f"scaled to {bp.height}p so the bitrate is enough for decent quality")
    if info.size <= target_mb * 1024 * 1024:
        notes.append("the input was already under the target; it was re-encoded anyway")
    notes.append(
        f"two-pass encode: video {bp.video_bps // 1000} kb/s"
        + (f", audio {bp.audio_bps // 1000} kb/s" if bp.audio_bps else ", no audio")
    )
    passlog = str(Path(tempfile.gettempdir()) / f"mediakit-{uuid.uuid4().hex[:10]}")
    venc = (
        ["-c:v", "libvpx-vp9", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4"]
        if is_vp9
        else ["-c:v", "libx264", "-preset", "medium"]
    )
    rate = ["-b:v", str(bp.video_bps), "-pix_fmt", "yuv420p"]
    if not is_vp9:
        # Cap the peaks: plain two-pass ABR overshoots by ~5-8% on noisy short clips.
        rate += ["-maxrate", str(bp.video_bps), "-bufsize", str(bp.video_bps * 2)]
    pass_args = ["-passlogfile", passlog]
    aenc = ["-c:a", "libopus" if is_vp9 else "aac", "-b:a", str(bp.audio_bps)]
    tail = ["-movflags", "+faststart"] if out_ext in ("mp4", "mov") else []

    def build_two_pass(part: Path) -> list[Step]:
        first = ffmpeg_argv(
            "-i", str(src), "-map", "0:v:0", *_vf(filters), *venc, *rate, "-pass", "1",
            *pass_args, "-an", "-f", "null", "-",
        )  # fmt: skip
        audio = ["-map", "0:a:0?", *aenc] if bp.audio_bps else ["-an"]
        second = ffmpeg_argv(
            "-i", str(src), "-map", "0:v:0", *_vf(filters), *venc, *rate, "-pass", "2",
            *pass_args, *audio, *tail, str(part),
        )  # fmt: skip
        return [Step(first, "analysis pass"), Step(second, "encode pass")]

    plan = simple_plan(
        "compress", src, final, build_two_pass, overwrite=overwrite, duration_s=duration, notes=notes
    )
    plan.temp_globs.append(passlog + "*")
    plan.target_mb = target_mb
    return plan


# ---------------------------------------------------------------- trim


def plan_trim(
    input: str,
    start: str | float | None,
    end: str | float | None,
    duration: str | float | None,
    mode: str,
    output: str | None,
    overwrite: bool,
) -> Plan:
    src = resolve_input(input)
    if mode not in ("accurate", "fast"):
        raise MediaError("mode must be 'accurate' (re-encode, frame-exact) or 'fast' (stream copy)")
    if end is not None and duration is not None:
        raise MediaError("pass end or duration, not both")
    info = probe(src)
    total = info.require_duration()
    s = parse_time(start, "start") if start is not None else 0.0
    if s >= total:
        raise MediaError(f"start={s:g}s is at or past the end of the media ({total:.2f}s)")
    if end is not None:
        e = parse_time(end, "end")
        if e <= s:
            raise MediaError(f"end={e:g}s must be after start={s:g}s")
        if e > total + 0.05:
            raise MediaError(f"end={e:g}s is past the end of the media ({total:.2f}s)")
        length = min(e, total) - s
    elif duration is not None:
        length = parse_time(duration, "duration")
        if length <= 0:
            raise MediaError("duration must be greater than zero")
        if s + length > total + 0.05:
            raise MediaError(
                f"start + duration = {s + length:g}s is past the end of the media ({total:.2f}s)"
            )
    else:
        length = total - s
    if start is None and end is None and duration is None:
        raise MediaError("nothing to trim: pass start, end and/or duration")

    ext = (Path(output).suffix.lower() if output else src.suffix.lower()).lstrip(".")
    final = resolve_output(output, derived(src, "trimmed"), overwrite=overwrite, inputs=[src])
    notes: list[str] = []
    seek = ["-ss", _t(s), "-i", str(src), "-t", _t(length)]
    if mode == "fast":
        tail = ["-map", "0:v?", "-map", "0:a?", "-c", "copy", "-avoid_negative_ts", "make_zero"]
        notes.append(
            "fast mode copies streams without re-encoding: the cut snaps to the nearest "
            "keyframe before start, so the result can begin up to a few seconds early"
        )
    else:
        if ext in VIDEO_EXTS:
            _need(info, video=True, why="be trimmed into a video file")
            codec, encoders = video_codec_args(ext, "high")
            tail = ["-map", "0:v:0", "-map", "0:a:0?", *_vf(_even_filter(info)), *codec]
        elif ext in AUDIO_EXTS:
            _need(info, audio=True, why="be trimmed into an audio file")
            codec, encoders = audio_codec_args(ext, "high")
            tail = ["-map", "0:a:0", "-vn", *codec]
        else:
            raise MediaError(
                f"accurate mode cannot write .{ext}; choose an output with one of "
                f"{', '.join((*VIDEO_EXTS, *AUDIO_EXTS))} or use mode='fast'"
            )
        require_encoders(*encoders)

    def build(part: Path) -> list[Step]:
        return [Step(ffmpeg_argv(*seek, *tail, str(part)))]

    return simple_plan("trim", src, final, build, overwrite=overwrite, duration_s=length, notes=notes)


# ---------------------------------------------------------------- extract_audio


def plan_extract_audio(
    input: str, format: str, bitrate: str | None, quality: str, output: str | None, overwrite: bool
) -> Plan:
    src = resolve_input(input)
    quality = _quality(quality)
    info = probe(src)
    audio = info.audio
    if audio is None:
        raise MediaError(f"{src.name!r} has no audio stream, so there is nothing to extract")
    if bitrate is not None and not re.fullmatch(r"\d{2,3}k", bitrate):
        raise MediaError("bitrate looks like '128k' or '320k'")
    fmt = (_ext_of(output, format) or "mp3")
    encoders: tuple[str, ...]
    if fmt == "copy":
        ext = COPY_EXT.get(audio.codec)
        if ext is None:
            raise MediaError(
                f"cannot copy {audio.codec!r} audio into a standalone file; "
                f"pick a format: {', '.join(AUDIO_EXTS)}"
            )
        codec, encoders = ["-c:a", "copy"], ()
    else:
        ext = _check_ext(fmt, AUDIO_EXTS, "audio format")
        codec, encoders = audio_codec_args(ext, quality, bitrate)
    final = resolve_output(output, derived(src, "audio", ext), overwrite=overwrite, inputs=[src])
    require_encoders(*encoders)

    def build(part: Path) -> list[Step]:
        return [Step(ffmpeg_argv("-i", str(src), "-map", "0:a:0", "-vn", *codec, str(part)))]

    return simple_plan(
        "extract_audio", src, final, build, overwrite=overwrite, duration_s=info.duration
    )


# ---------------------------------------------------------------- make_gif


def plan_gif(
    input: str,
    output: str | None,
    start: str | float | None,
    duration: str | float | None,
    fps: int,
    width: int,
    loop: bool,
    overwrite: bool,
) -> Plan:
    src = resolve_input(input)
    if not 1 <= fps <= 30:
        raise MediaError("fps must be between 1 and 30")
    if not 16 <= width <= 1920:
        raise MediaError("width must be between 16 and 1920 pixels")
    info = probe(src)
    _need(info, video=True, why="make a GIF")
    video = info.video
    assert video is not None and video.width
    total = info.require_duration()
    s = parse_time(start, "start") if start is not None else 0.0
    if s >= total:
        raise MediaError(f"start={s:g}s is at or past the end of the media ({total:.2f}s)")
    want = parse_time(duration, "duration") if duration is not None else 5.0
    if want <= 0:
        raise MediaError("duration must be greater than zero")
    if want > GIF_MAX_SECONDS:
        raise MediaError(
            f"duration={want:g}s is too long for a GIF (limit {GIF_MAX_SECONDS:g}s: GIFs get "
            "huge). Pick a shorter range, or convert to mp4/webm instead."
        )
    length = min(want, total - s)
    notes = []
    if length < want - 0.01:
        notes.append(f"the clip ends at {total:.2f}s, so the GIF is {length:.2f}s long")
    default = src.with_suffix(".gif") if src.suffix.lower() != ".gif" else derived(src, "new", "gif")
    final = resolve_output(output, default, overwrite=overwrite, inputs=[src])

    scale = f",scale={width}:-1:flags=lanczos" if video.width > width else ""
    graph = (
        f"fps={fps}{scale},split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )

    def build(part: Path) -> list[Step]:
        return [
            Step(
                ffmpeg_argv(
                    "-ss", _t(s), "-t", _t(length), "-i", str(src), "-map", "0:v:0", "-an",
                    "-vf", graph, "-loop", "0" if loop else "-1", str(part),
                )  # fmt: skip
            )
        ]

    return simple_plan(
        "make_gif", src, final, build, overwrite=overwrite, duration_s=length, notes=notes
    )


# ---------------------------------------------------------------- extract_frames


def plan_frames(
    input: str,
    times: list[str | float] | None,
    every_s: float | None,
    format: str,
    width: int | None,
    max_frames: int,
    output_dir: str | None,
    overwrite: bool,
) -> Plan:
    src = resolve_input(input)
    fmt = format.lower().lstrip(".")
    if fmt not in ("png", "jpg"):
        raise MediaError("format must be png or jpg")
    if times is not None and every_s is not None:
        raise MediaError("pass times or every_s, not both")
    if every_s is not None and every_s <= 0:
        raise MediaError("every_s must be greater than zero")
    if width is not None and not 16 <= width <= 7680:
        raise MediaError("width must be between 16 and 7680 pixels")
    if max_frames < 1:
        raise MediaError("max_frames must be at least 1")
    info = probe(src)
    _need(info, video=True, why="extract frames from")
    total = info.require_duration()
    out_dir = resolve_dir(output_dir, src.with_name(f"{src.stem}.frames"))
    part_dir = out_dir / ".mediakit-partial"
    scale = [f"scale={width}:-2"] if width else []
    qual = ["-q:v", "2"] if fmt == "jpg" else []
    steps: list[Step] = []
    notes: list[str] = []
    pairs: list[tuple[Path, Path]] = []

    if every_s is None:
        wanted = [parse_time(t, "times") for t in times] if times else [total / 2]
        if not times:
            notes.append(f"no times given: took one frame from the middle ({total / 2:.2f}s)")
        if len(wanted) > max_frames:
            raise MediaError(f"{len(wanted)} frames requested, max_frames is {max_frames}")
        for i, t in enumerate(wanted):
            if t >= total:
                raise MediaError(f"time {t:g}s is past the end of the media ({total:.2f}s)")
            part = part_dir / f"f{i:04d}.{fmt}"
            final = out_dir / f"{src.stem}.frame-{t:g}s.{fmt}"
            pairs.append((part, final))
            steps.append(
                Step(
                    ffmpeg_argv(
                        "-ss", _t(t), "-i", str(src), "-map", "0:v:0", "-frames:v", "1",
                        *_vf(scale), *qual, str(part),
                    ),  # fmt: skip
                    f"frame at {t:g}s",
                )
            )
        for _, final in pairs:
            _refuse_existing(final, overwrite)
        duration_s: float | None = None
    else:
        bound = math.ceil(total / every_s) + 1
        if bound > max_frames:
            raise MediaError(
                f"every_s={every_s:g} on {total:.1f}s of video makes up to {bound} frames "
                f"(max_frames is {max_frames}). Raise every_s or max_frames."
            )
        for i in range(1, bound + 1):
            _refuse_existing(out_dir / f"{src.stem}.frame-{i:04d}.{fmt}", overwrite)
        steps.append(
            Step(
                ffmpeg_argv(
                    "-i", str(src), "-map", "0:v:0", "-vf", ",".join([f"fps=1/{every_s:g}", *scale]),
                    *qual, str(part_dir / f"%04d.{fmt}"),
                )  # fmt: skip
            )
        )
        duration_s = total

    def prepare() -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        part_dir.mkdir(exist_ok=True)

    def finalize() -> list[Path]:
        moves = pairs
        if every_s is not None:
            made = sorted(part_dir.glob(f"*.{fmt}"))
            moves = [(p, out_dir / f"{src.stem}.frame-{i:04d}.{fmt}") for i, p in enumerate(made, 1)]
        if not moves:
            raise MediaError("ffmpeg produced no frames")
        return [commit_file(p, f, overwrite) for p, f in moves]

    return Plan(
        tool="extract_frames",
        input=src,
        steps=steps,
        outputs=[f for _, f in pairs] or [out_dir / f"{src.stem}.frame-0001.{fmt}"],
        duration_s=duration_s,
        notes=notes,
        temp_paths=[part_dir],
        prepare=prepare,
        finalize=finalize,
    )


def _refuse_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MediaError(
            f"'{path}' already exists. Pass overwrite=true to replace existing frames, "
            "or choose another output_dir."
        )


# ---------------------------------------------------------------- concat


def _signature(info: MediaInfo) -> dict[str, object]:
    v, a = info.video, info.audio
    return {
        "video codec": v.codec if v else None,
        "resolution": f"{v.width}x{v.height}" if v else None,
        "pixel format": v.pix_fmt if v else None,
        "frame rate": round(v.fps, 2) if v and v.fps else None,
        "audio codec": a.codec if a else None,
        "sample rate": a.sample_rate if a else None,
        "channels": a.channels if a else None,
    }


def plan_concat(inputs: list[str], output: str | None, overwrite: bool) -> Plan:
    if len(inputs) < 2:
        raise MediaError("concat needs at least two input files")
    srcs = [resolve_input(p) for p in inputs]
    infos = [probe(p) for p in srcs]
    base = _signature(infos[0])
    for p, info in zip(srcs[1:], infos[1:], strict=True):
        sig = _signature(info)
        diffs = [f"{k}: {base[k]} vs {sig[k]}" for k in base if base[k] != sig[k]]
        if diffs:
            raise MediaError(
                f"{p.name!r} does not match {srcs[0].name!r} ({'; '.join(diffs)}). Joining "
                "without re-encoding needs identical codecs and parameters: convert them to "
                "the same format first (convert / transform), then concat."
            )
    total = sum(i.require_duration() for i in infos)
    final = resolve_output(
        output, derived(srcs[0], "concat"), overwrite=overwrite, inputs=srcs
    )
    listing = Path(tempfile.gettempdir()) / f"mediakit-concat-{uuid.uuid4().hex[:10]}.txt"

    def esc(p: Path) -> str:
        return p.as_posix().replace("'", "'\\''")

    text = "".join(f"file '{esc(p)}'\n" for p in srcs)

    def prepare() -> None:
        listing.write_text(text, encoding="utf-8")

    def build(part: Path) -> list[Step]:
        return [
            Step(
                ffmpeg_argv(
                    "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v?",
                    "-map", "0:a?", "-c", "copy", str(part),
                )  # fmt: skip
            )
        ]

    plan = simple_plan(
        "concat",
        srcs[0],
        final,
        build,
        overwrite=overwrite,
        duration_s=total,
        notes=[f"joined {len(srcs)} files without re-encoding (list file contents: {text.strip()!r})"],
    )
    plan.prepare = prepare
    plan.temp_paths.append(listing)
    plan.input_bytes_extra = sum(p.stat().st_size for p in srcs[1:])
    return plan


# ---------------------------------------------------------------- transform


def _atempo_chain(speed: float) -> str:
    parts: list[str] = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.6g}")
    return ",".join(parts)


def plan_transform(
    input: str,
    width: int | None,
    height: int | None,
    crop: str | None,
    speed: float | None,
    output: str | None,
    overwrite: bool,
) -> Plan:
    src = resolve_input(input)
    if width is None and height is None and crop is None and speed is None:
        raise MediaError("nothing to do: pass width/height (resize), crop and/or speed")
    for name, val in (("width", width), ("height", height)):
        if val is not None and val < 16:
            raise MediaError(f"{name} must be at least 16 pixels")
    if speed is not None and not 0.25 <= speed <= 4.0:
        raise MediaError("speed must be between 0.25 (4x slower) and 4 (4x faster)")
    info = probe(src)
    _need(info, video=True, why="be resized, cropped or retimed (video only)")
    video = info.video
    assert video is not None and video.width and video.height

    filters: list[str] = []
    notes: list[str] = []
    out_w, out_h = video.width, video.height
    if crop is not None:
        m = re.fullmatch(r"(\d+):(\d+)(?::(\d+):(\d+))?", crop.strip())
        if not m:
            raise MediaError("crop looks like 'W:H' (centered) or 'W:H:X:Y' in pixels, e.g. 320:240:10:20")
        cw, ch = int(m[1]), int(m[2])
        cx = int(m[3]) if m[3] else (video.width - cw) // 2
        cy = int(m[4]) if m[4] else (video.height - ch) // 2
        if cw < 16 or ch < 16 or cx + cw > video.width or cy + ch > video.height:
            raise MediaError(
                f"crop {cw}x{ch} at ({cx},{cy}) does not fit inside the "
                f"{video.width}x{video.height} frame"
            )
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        out_w, out_h = cw, ch
    if width is not None or height is not None:
        filters.append(f"scale={width or -2}:{height or -2}")
        if width and height:
            out_w, out_h = width, height
        elif width:
            out_w, out_h = width, round(out_h * width / out_w / 2) * 2
        else:
            assert height is not None
            out_w, out_h = round(out_w * height / out_h / 2) * 2, height
    if (out_w % 2 or out_h % 2) and (width and height or crop):
        raise MediaError(
            f"the result would be {out_w}x{out_h}; video encoders need even width and height"
        )
    length = info.duration
    audio_filter: list[str] = []
    if speed is not None and speed != 1.0:
        filters.append(f"setpts=PTS/{speed:.6g}")
        if info.audio:
            audio_filter = ["-af", _atempo_chain(speed)]
        if length:
            length /= speed
        notes.append(f"speed x{speed:g}" + ("" if info.audio else " (no audio to retime)"))

    ext = (Path(output).suffix.lower().lstrip(".") if output else _default_video_ext(src))
    if ext not in VIDEO_EXTS:
        raise MediaError(f"transform writes {', '.join(VIDEO_EXTS)} files, not .{ext}")
    final = resolve_output(output, derived(src, "transformed", ext), overwrite=overwrite, inputs=[src])
    codec, encoders = video_codec_args(ext, "high")
    require_encoders(*encoders)

    def build(part: Path) -> list[Step]:
        return [
            Step(
                ffmpeg_argv(
                    "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", *_vf(filters),
                    *audio_filter, *codec, str(part),
                )  # fmt: skip
            )
        ]

    return simple_plan(
        "transform", src, final, build, overwrite=overwrite, duration_s=length, notes=notes
    )
