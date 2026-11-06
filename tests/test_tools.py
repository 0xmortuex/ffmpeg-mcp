"""Run every tool for real against generated media and verify the output with ffprobe."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ffprobe_json, stream
from mediakit_mcp import ops
from mediakit_mcp.errors import MediaError
from mediakit_mcp.plans import Plan, execute, format_dry_run
from mediakit_mcp.probe import describe, probe


def go(plan: Plan) -> str:
    return execute(plan, timeout_s=300)


def dur(path: Path) -> float:
    return float(ffprobe_json(path)["format"]["duration"])


# ------------------------------------------------------------- media_info


def test_media_info_summarizes_streams(work: Path) -> None:
    text = describe(probe(work / "video.mp4"))
    assert "h264" in text and "640x360" in text and "30 fps" in text
    assert "aac" in text and "48000 Hz" in text
    assert "duration: 6.00s" in text


def test_media_info_flags_missing_audio(work: Path) -> None:
    assert "(no audio stream)" in describe(probe(work / "silent.mp4"))


# ------------------------------------------------------------- convert


def test_convert_to_webm_uses_vp9_and_opus(work: Path) -> None:
    out = work / "video.webm"
    report = go(ops.plan_convert(str(work / "video.mp4"), "webm", None, "medium", False))
    assert out.exists() and "Wrote" in report
    data = ffprobe_json(out)
    assert stream(data, "video")["codec_name"] == "vp9"
    assert stream(data, "audio")["codec_name"] == "opus"
    assert dur(out) == pytest.approx(6.0, abs=0.3)


def test_convert_to_mp3_and_wav(work: Path) -> None:
    go(ops.plan_convert(str(work / "video.mp4"), "mp3", None, "medium", False))
    go(ops.plan_convert(str(work / "tone.wav"), None, str(work / "tone.flac"), "medium", False))
    assert stream(ffprobe_json(work / "video.mp3"), "audio")["codec_name"] == "mp3"
    assert stream(ffprobe_json(work / "tone.flac"), "audio")["codec_name"] == "flac"
    assert dur(work / "video.mp3") == pytest.approx(6.0, abs=0.3)


def test_convert_odd_dimensions_to_mp4_gets_even_size(work: Path) -> None:
    go(ops.plan_convert(str(work / "odd.mkv"), "mp4", None, "medium", False))
    v = stream(ffprobe_json(work / "odd.mp4"), "video")
    assert v["width"] % 2 == 0 and v["height"] % 2 == 0 and v["codec_name"] == "h264"


def test_convert_same_extension_gets_a_derived_name(work: Path) -> None:
    go(ops.plan_convert(str(work / "silent.mp4"), "mp4", None, "low", False))
    assert (work / "silent.converted.mp4").exists()


def test_convert_to_gif_notes_the_limits(work: Path) -> None:
    report = go(ops.plan_convert(str(work / "silent.mp4"), "gif", None, "medium", False))
    assert "first 5 seconds" in report
    assert stream(ffprobe_json(work / "silent.gif"), "video")["codec_name"] == "gif"


def test_convert_video_from_audio_only_input_fails_clearly(work: Path) -> None:
    with pytest.raises(MediaError, match="no video stream"):
        ops.plan_convert(str(work / "tone.wav"), "mp4", None, "medium", False)


def test_convert_unknown_format_lists_supported(work: Path) -> None:
    with pytest.raises(MediaError, match="unsupported format 'xyz'.*mp4"):
        ops.plan_convert(str(work / "video.mp4"), "xyz", None, "medium", False)


def test_convert_format_vs_output_extension_conflict(work: Path) -> None:
    with pytest.raises(MediaError, match="contradicts"):
        ops.plan_convert(str(work / "video.mp4"), "webm", str(work / "x.mp4"), "medium", False)


# ------------------------------------------------------------- compress


def test_compress_hits_target_size_with_two_passes(work: Path) -> None:
    src = work / "video.mp4"
    target_mb = 1.0
    plan = ops.plan_compress(str(src), target_mb, None, "medium", None, None, False)
    assert len(plan.steps) == 2
    report = go(plan)
    out = work / "video.compressed.mp4"
    size = out.stat().st_size
    assert size <= target_mb * 1024 * 1024, report
    assert size > 0.6 * target_mb * 1024 * 1024, "should use most of the budget, not undershoot wildly"
    assert "target 1 MB: met" in report
    data = ffprobe_json(out)
    assert stream(data, "video")["codec_name"] == "h264"
    assert stream(data, "audio")["codec_name"] == "aac"
    assert dur(out) == pytest.approx(6.0, abs=0.3)
    assert list(Path(__import__("tempfile").gettempdir()).glob("mediakit-*-0.log*")) == []


def test_compress_tiny_target_downscales_instead_of_mush(work: Path) -> None:
    plan = ops.plan_compress(str(work / "video.mp4"), 0.3, None, "medium", None, None, False)
    go(plan)
    out = work / "video.compressed.mp4"
    v = stream(ffprobe_json(out), "video")
    assert v["height"] < 360
    assert out.stat().st_size <= 0.3 * 1024 * 1024


def test_compress_impossible_target_is_an_error(work: Path) -> None:
    with pytest.raises(MediaError, match="too small for 6.0s"):
        ops.plan_compress(str(work / "video.mp4"), 0.01, None, "medium", None, None, False)


def test_compress_silent_video_has_no_audio_stream(work: Path) -> None:
    plan = ops.plan_compress(str(work / "silent.mp4"), 0.2, None, "medium", None, None, False)
    go(plan)
    data = ffprobe_json(work / "silent.compressed.mp4")
    assert [s["codec_type"] for s in data["streams"]] == ["video"]


def test_compress_preset_discord_and_webm_output(work: Path) -> None:
    plan = ops.plan_compress(
        str(work / "video.mp4"), None, "discord-10mb", "medium", None, None, False
    )
    assert plan.target_mb == 10
    go(plan)
    assert (work / "video.compressed.mp4").stat().st_size <= 10 * 1024 * 1024
    plan = ops.plan_compress(
        str(work / "video.mp4"), 1.0, None, "medium", None, str(work / "v.webm"), False
    )
    go(plan)
    assert stream(ffprobe_json(work / "v.webm"), "video")["codec_name"] == "vp9"
    assert (work / "v.webm").stat().st_size <= 1024 * 1024


def test_compress_quality_mode_and_max_height(work: Path) -> None:
    plan = ops.plan_compress(str(work / "video.mp4"), None, "web", "medium", 180, None, False)
    assert len(plan.steps) == 1
    go(plan)
    out = work / "video.compressed.mp4"
    assert stream(ffprobe_json(out), "video")["height"] == 180
    assert out.stat().st_size < (work / "video.mp4").stat().st_size


def test_compress_rejects_preset_and_target_together(work: Path) -> None:
    with pytest.raises(MediaError, match="not both"):
        ops.plan_compress(str(work / "video.mp4"), 5, "whatsapp", "medium", None, None, False)
    with pytest.raises(MediaError, match="unknown preset"):
        ops.plan_compress(str(work / "video.mp4"), None, "tiktok", "medium", None, None, False)


def test_compress_audio_only_input_is_refused(work: Path) -> None:
    with pytest.raises(MediaError, match="no video stream"):
        ops.plan_compress(str(work / "tone.wav"), 1, None, "medium", None, None, False)


# ------------------------------------------------------------- trim


def test_trim_accurate_is_frame_exact(work: Path) -> None:
    go(ops.plan_trim(str(work / "video.mp4"), 1.5, None, 2.0, "accurate", None, False))
    out = work / "video.trimmed.mp4"
    assert dur(out) == pytest.approx(2.0, abs=0.1)
    assert stream(ffprobe_json(out), "video")["codec_name"] == "h264"


def test_trim_with_timecodes_and_end(work: Path) -> None:
    go(ops.plan_trim(str(work / "video.mp4"), "0:00:01", "0:03.5", None, "accurate", None, False))
    assert dur(work / "video.trimmed.mp4") == pytest.approx(2.5, abs=0.1)


def test_trim_fast_copies_streams(work: Path) -> None:
    report = go(ops.plan_trim(str(work / "video.mp4"), 2, 4, None, "fast", None, False))
    out = work / "video.trimmed.mp4"
    assert "keyframe" in report
    # Stream copy keeps the original encoder settings (same profile and size class).
    assert stream(ffprobe_json(out), "video")["codec_name"] == "h264"
    assert 1.5 <= dur(out) <= 6.0  # snaps to a keyframe; never shorter than asked, roughly


def test_trim_audio_file(work: Path) -> None:
    go(ops.plan_trim(str(work / "tone.wav"), 1, 2.5, None, "accurate", None, False))
    assert dur(work / "tone.trimmed.wav") == pytest.approx(1.5, abs=0.05)


def test_trim_bad_ranges(work: Path) -> None:
    src = str(work / "video.mp4")
    with pytest.raises(MediaError, match="past the end"):
        ops.plan_trim(src, 10, None, None, "fast", None, False)
    with pytest.raises(MediaError, match="must be after start"):
        ops.plan_trim(src, 3, 2, None, "fast", None, False)
    with pytest.raises(MediaError, match="not both"):
        ops.plan_trim(src, 1, 2, 1, "fast", None, False)
    with pytest.raises(MediaError, match="not a time"):
        ops.plan_trim(src, "soon", None, None, "fast", None, False)
    with pytest.raises(MediaError, match="nothing to trim"):
        ops.plan_trim(src, None, None, None, "fast", None, False)
    with pytest.raises(MediaError, match="mode must be"):
        ops.plan_trim(src, 1, 2, None, "sloppy", None, False)


# ------------------------------------------------------------- extract_audio


def test_extract_audio_mp3_and_copy(work: Path) -> None:
    go(ops.plan_extract_audio(str(work / "video.mp4"), "mp3", "128k", "medium", None, False))
    out = work / "video.audio.mp3"
    data = ffprobe_json(out)
    assert stream(data, "audio")["codec_name"] == "mp3"
    assert "video" not in [s["codec_type"] for s in data["streams"]]
    assert dur(out) == pytest.approx(6.0, abs=0.3)

    go(ops.plan_extract_audio(str(work / "video.mp4"), "copy", None, "medium", None, False))
    assert stream(ffprobe_json(work / "video.audio.m4a"), "audio")["codec_name"] == "aac"


def test_extract_audio_without_audio_stream(work: Path) -> None:
    with pytest.raises(MediaError, match="no audio stream"):
        ops.plan_extract_audio(str(work / "silent.mp4"), "mp3", None, "medium", None, False)


def test_extract_audio_bad_bitrate(work: Path) -> None:
    with pytest.raises(MediaError, match="bitrate looks like"):
        ops.plan_extract_audio(str(work / "video.mp4"), "mp3", "fast", "medium", None, False)
    with pytest.raises(MediaError, match="lossless"):
        ops.plan_extract_audio(str(work / "video.mp4"), "wav", "128k", "medium", None, False)


# ------------------------------------------------------------- make_gif


def test_make_gif_frame_count_and_width(work: Path) -> None:
    go(ops.plan_gif(str(work / "video.mp4"), None, 1, 2, 10, 320, True, False))
    out = work / "video.gif"
    data = ffprobe_json(out, count_frames=True)
    v = stream(data, "video")
    assert v["codec_name"] == "gif"
    assert v["width"] == 320
    assert int(v["nb_read_frames"]) == pytest.approx(20, abs=2)  # 2 s at 10 fps


def test_make_gif_never_upscales_and_caps_duration(work: Path) -> None:
    report = go(ops.plan_gif(str(work / "silent.mp4"), None, 3, 10, 5, 1000, False, False))
    data = ffprobe_json(work / "silent.gif", count_frames=True)
    assert stream(data, "video")["width"] == 640
    assert int(stream(data, "video")["nb_read_frames"]) == pytest.approx(5, abs=2)  # 1 s left
    assert "ends at 4.00s" in report


def test_make_gif_limits(work: Path) -> None:
    src = str(work / "video.mp4")
    with pytest.raises(MediaError, match="too long for a GIF"):
        ops.plan_gif(src, None, 0, 45, 10, 320, True, False)
    with pytest.raises(MediaError, match="fps must be"):
        ops.plan_gif(src, None, 0, 2, 60, 320, True, False)
    with pytest.raises(MediaError, match="width must be"):
        ops.plan_gif(src, None, 0, 2, 10, 4000, True, False)
    with pytest.raises(MediaError, match="no video stream"):
        ops.plan_gif(str(work / "tone.wav"), None, 0, 2, 10, 320, True, False)


# ------------------------------------------------------------- extract_frames


def test_extract_frames_at_times(work: Path) -> None:
    report = go(ops.plan_frames(str(work / "video.mp4"), [1, "0:03.5"], None, "png", 320, 100, None, False))
    d = work / "video.frames"
    files = sorted(p.name for p in d.iterdir())
    assert files == ["video.frame-1s.png", "video.frame-3.5s.png"], files
    v = stream(ffprobe_json(d / files[0]), "video")
    assert v["width"] == 320 and v["codec_name"] == "png"
    assert "Wrote 2 files" in report
    assert not (d / ".mediakit-partial").exists()


def test_extract_frames_every_n_seconds_jpg(work: Path) -> None:
    go(ops.plan_frames(str(work / "video.mp4"), None, 2.0, "jpg", None, 100, None, False))
    files = sorted(p.name for p in (work / "video.frames").iterdir())
    assert 3 <= len(files) <= 4, files  # 6 s / 2 s -> frames at 0, 2, 4 (and maybe 6)
    assert files[0] == "video.frame-0001.jpg"
    assert stream(ffprobe_json(work / "video.frames" / files[0]), "video")["codec_name"] == "mjpeg"


def test_extract_frames_default_is_one_middle_frame(work: Path) -> None:
    go(ops.plan_frames(str(work / "silent.mp4"), None, None, "png", None, 100, None, False))
    assert [p.name for p in (work / "silent.frames").iterdir()] == ["silent.frame-2s.png"]


def test_extract_frames_errors(work: Path) -> None:
    src = str(work / "video.mp4")
    with pytest.raises(MediaError, match="past the end"):
        ops.plan_frames(src, [99], None, "png", None, 100, None, False)
    with pytest.raises(MediaError, match="max_frames"):
        ops.plan_frames(src, None, 0.01, "png", None, 100, None, False)
    with pytest.raises(MediaError, match="not both"):
        ops.plan_frames(src, [1], 2.0, "png", None, 100, None, False)
    with pytest.raises(MediaError, match="png or jpg"):
        ops.plan_frames(src, [1], None, "bmp", None, 100, None, False)


def test_extract_frames_refuses_existing_unless_overwrite(work: Path) -> None:
    src = str(work / "video.mp4")
    go(ops.plan_frames(src, [1], None, "png", None, 100, None, False))
    with pytest.raises(MediaError, match="already exists"):
        ops.plan_frames(src, [1], None, "png", None, 100, None, False)
    go(ops.plan_frames(src, [1], None, "png", None, 100, None, True))


# ------------------------------------------------------------- concat


def test_concat_joins_same_codec_clips(work: Path) -> None:
    src = str(work / "video.mp4")
    go(ops.plan_trim(src, 0, None, 2, "accurate", str(work / "a.mp4"), False))
    go(ops.plan_trim(src, 2, None, 3, "accurate", str(work / "b.mp4"), False))
    report = go(ops.plan_concat([str(work / "a.mp4"), str(work / "b.mp4")], None, False))
    out = work / "a.concat.mp4"
    assert dur(out) == pytest.approx(5.0, abs=0.3), report
    data = ffprobe_json(out)
    assert stream(data, "video")["codec_name"] == "h264"
    assert stream(data, "audio")["codec_name"] == "aac"


def test_concat_rejects_mismatched_clips_naming_the_difference(work: Path) -> None:
    with pytest.raises(MediaError, match="audio codec: aac vs None"):
        ops.plan_concat([str(work / "video.mp4"), str(work / "silent.mp4")], None, False)
    with pytest.raises(MediaError, match="at least two"):
        ops.plan_concat([str(work / "video.mp4")], None, False)


def test_concat_handles_quotes_in_file_names(work: Path) -> None:
    import shutil

    shutil.copy(work / "silent.mp4", work / "it's here.mp4")
    go(ops.plan_concat([str(work / "silent.mp4"), str(work / "it's here.mp4")], None, False))
    assert dur(work / "silent.concat.mp4") == pytest.approx(8.0, abs=0.3)


# ------------------------------------------------------------- transform


def test_transform_resize_keeps_aspect(work: Path) -> None:
    go(ops.plan_transform(str(work / "video.mp4"), 320, None, None, None, None, False))
    v = stream(ffprobe_json(work / "video.transformed.mp4"), "video")
    assert (v["width"], v["height"]) == (320, 180)


def test_transform_crop_and_speed(work: Path) -> None:
    go(ops.plan_transform(str(work / "video.mp4"), None, None, "320:240", 2.0, None, False))
    out = work / "video.transformed.mp4"
    data = ffprobe_json(out)
    v = stream(data, "video")
    assert (v["width"], v["height"]) == (320, 240)
    assert dur(out) == pytest.approx(3.0, abs=0.3)  # twice as fast
    assert stream(data, "audio")["codec_name"] == "aac"


def test_transform_slow_motion_extreme_speed_chains_atempo(work: Path) -> None:
    go(ops.plan_transform(str(work / "silent.mp4"), None, None, None, 0.25, None, False))
    assert dur(work / "silent.transformed.mp4") == pytest.approx(16.0, abs=0.5)
    plan = ops.plan_transform(str(work / "video.mp4"), None, None, None, 4.0, None, False)
    assert "atempo=2.0,atempo=2" in " ".join(plan.steps[0].argv)


def test_transform_errors(work: Path) -> None:
    src = str(work / "video.mp4")
    with pytest.raises(MediaError, match="nothing to do"):
        ops.plan_transform(src, None, None, None, None, None, False)
    with pytest.raises(MediaError, match="does not fit"):
        ops.plan_transform(src, None, None, "900:900", None, None, False)
    with pytest.raises(MediaError, match="crop looks like"):
        ops.plan_transform(src, None, None, "big", None, None, False)
    with pytest.raises(MediaError, match="speed must be"):
        ops.plan_transform(src, None, None, None, 9, None, False)
    with pytest.raises(MediaError, match="even width"):
        ops.plan_transform(src, 321, 181, None, None, None, False)


# ------------------------------------------------------------- dry run


def test_dry_run_prints_exact_commands_and_writes_nothing(work: Path) -> None:
    before = sorted(p.name for p in work.iterdir())
    plan = ops.plan_compress(str(work / "video.mp4"), 2.0, None, "medium", None, None, False)
    text = format_dry_run(plan)
    assert text.startswith("DRY RUN")
    assert "Step 1/2 (analysis pass)" in text and "Step 2/2 (encode pass)" in text
    assert "-pass 1" in text and "-pass 2" in text and "libx264" in text
    assert "video.compressed.mp4" in text
    assert sorted(p.name for p in work.iterdir()) == before


def test_dry_run_command_actually_runs(work: Path) -> None:
    """The printed arguments are the ones that execute: run the plan's argv by hand."""
    import subprocess

    plan = ops.plan_trim(str(work / "video.mp4"), 1, None, 1, "accurate", None, False)
    argv = plan.steps[0].argv
    assert subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL).returncode == 0
    assert plan.temp_paths[0].exists()  # wrote to the partial name
    assert not (work / "video.trimmed.mp4").exists()
