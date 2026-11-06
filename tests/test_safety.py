"""Overwrite rules, the directory allow-list, broken inputs, missing ffmpeg, sizing maths."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ffprobe_json
from mediakit_mcp import ops
from mediakit_mcp.binaries import find_tool
from mediakit_mcp.errors import MediaError
from mediakit_mcp.paths import set_allowed_dirs
from mediakit_mcp.plans import execute, simple_plan
from mediakit_mcp.probe import describe, probe
from mediakit_mcp.runner import Step
from mediakit_mcp.sizing import plan_bitrates

# ------------------------------------------------------------- never overwrite


def test_existing_output_is_refused_then_replaced_with_overwrite(work: Path) -> None:
    src = str(work / "video.mp4")
    execute(ops.plan_trim(src, 0, None, 1, "accurate", None, False), timeout_s=120)
    out = work / "video.trimmed.mp4"
    first = out.read_bytes()
    with pytest.raises(MediaError, match="already exists.*overwrite=true"):
        ops.plan_trim(src, 0, None, 2, "accurate", None, False)
    assert out.read_bytes() == first, "refusal must not touch the existing file"
    execute(ops.plan_trim(src, 0, None, 2, "accurate", None, True), timeout_s=120)
    assert out.read_bytes() != first
    assert float(ffprobe_json(out)["format"]["duration"]) == pytest.approx(2.0, abs=0.1)


def test_failed_overwrite_leaves_the_original_and_no_partial(work: Path) -> None:
    existing = work / "precious.mp4"
    existing.write_bytes(b"precious")
    src = work / "video.mp4"

    def build(part: Path) -> list[Step]:
        from mediakit_mcp.plans import ffmpeg_argv

        # Fails after being told to write to the partial file.
        return [Step(ffmpeg_argv("-i", str(work / "missing.mp4"), str(part)))]

    plan = simple_plan("trim", src, existing, build, overwrite=True, duration_s=1)
    with pytest.raises(MediaError, match="ffmpeg failed"):
        execute(plan, timeout_s=30)
    assert existing.read_bytes() == b"precious"
    assert not list(work.glob("*.partial.*"))


def test_successful_run_leaves_no_partial_or_pass_logs(work: Path) -> None:
    execute(ops.plan_compress(str(work / "video.mp4"), 1, None, "medium", None, None, False), timeout_s=120)
    assert not list(work.glob("*.partial.*"))
    assert not list(work.glob("*.log*")) and not list(work.glob("*.mbtree"))


def test_output_equal_to_input_is_refused(work: Path) -> None:
    src = str(work / "video.mp4")
    with pytest.raises(MediaError, match="same file as the input"):
        ops.plan_trim(src, 0, None, 1, "accurate", src, True)
    with pytest.raises(MediaError, match="same file as the input"):
        ops.plan_compress(src, 1, None, "medium", None, src, True)


def test_output_directory_must_exist(work: Path) -> None:
    with pytest.raises(MediaError, match="output directory .* does not exist"):
        ops.plan_trim(str(work / "video.mp4"), 0, None, 1, "fast", str(work / "nope" / "x.mp4"), False)


# ------------------------------------------------------------- broken inputs


def test_missing_file(work: Path) -> None:
    with pytest.raises(MediaError, match="input file not found"):
        probe_via_tool(work / "nope.mp4")


def test_directory_as_input(work: Path) -> None:
    with pytest.raises(MediaError, match="not a file"):
        probe_via_tool(work)


@pytest.mark.parametrize("name", ["corrupt.mp4", "truncated.mp4"])
def test_corrupt_and_truncated_files_give_a_readable_error(work: Path, name: str) -> None:
    with pytest.raises(MediaError, match=rf"ffprobe could not read '{name}'"):
        probe_via_tool(work / name)
    with pytest.raises(MediaError, match="could not read"):
        ops.plan_compress(str(work / name), 1, None, "medium", None, None, False)
    with pytest.raises(MediaError, match="could not read"):
        ops.plan_trim(str(work / name), 0, None, 1, "fast", None, False)


def test_empty_path(work: Path) -> None:
    with pytest.raises(MediaError, match="empty path"):
        probe_via_tool(Path(" "))


def probe_via_tool(path: Path) -> str:
    from mediakit_mcp.paths import resolve_input

    return describe(probe(resolve_input(str(path))))


# ------------------------------------------------------------- allow-list


def test_allowed_dirs_restrict_inputs_and_outputs(work: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    inside = work / "inside"
    inside.mkdir()
    (inside / "ok.mp4").write_bytes((work / "silent.mp4").read_bytes())
    set_allowed_dirs([inside])
    out = str(inside / "ok.trimmed.mp4")
    execute(ops.plan_trim(str(inside / "ok.mp4"), 0, None, 1, "fast", out, False), timeout_s=60)
    with pytest.raises(MediaError, match="input .* outside the allowed directories"):
        ops.plan_trim(str(work / "video.mp4"), 0, None, 1, "fast", None, False)
    with pytest.raises(MediaError, match="output .* outside the allowed directories"):
        ops.plan_trim(str(inside / "ok.mp4"), 0, None, 1, "fast", str(work / "escape.mp4"), False)
    # Dot-dot traversal resolves first, so it cannot sneak out.
    with pytest.raises(MediaError, match="outside the allowed directories"):
        ops.plan_trim(str(inside / ".." / "video.mp4"), 0, None, 1, "fast", None, False)
    with pytest.raises(MediaError, match="outside the allowed directories"):
        ops.plan_frames(str(inside / "ok.mp4"), [1], None, "png", None, 5, str(work / "frames"), False)
    with pytest.raises(MediaError, match="outside the allowed directories"):
        ops.plan_concat([str(inside / "ok.mp4"), str(work / "silent.mp4")], None, False)


def test_allowed_dir_must_exist(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="does not exist"):
        set_allowed_dirs([tmp_path / "nope"])


# ------------------------------------------------------------- finding ffmpeg


def test_missing_ffmpeg_gives_install_instructions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MediaError) as err:
        find_tool("ffmpeg")
    msg = str(err.value)
    assert "winget install Gyan.FFmpeg" in msg and "brew install ffmpeg" in msg
    assert "apt install ffmpeg" in msg and "FFMPEG_PATH" in msg


def test_bad_env_override_is_reported_not_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FFPROBE_PATH", str(tmp_path / "nope.exe"))
    with pytest.raises(MediaError, match="FFPROBE_PATH is set to .* no ffprobe executable"):
        find_tool("ffprobe")


def test_env_override_wins_and_accepts_a_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    real = Path(find_tool("ffmpeg"))
    monkeypatch.setenv("FFMPEG_PATH", str(real.parent))
    assert Path(find_tool("ffmpeg")).samefile(real)
    monkeypatch.setenv("FFMPEG_PATH", str(real))
    assert Path(find_tool("ffmpeg")).samefile(real)


def test_ffprobe_is_found_next_to_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = Path(find_tool("ffmpeg"))
    monkeypatch.setenv("FFMPEG_PATH", str(real))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("FFPROBE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Path(find_tool("ffprobe")).parent.samefile(real.resolve().parent)


# ------------------------------------------------------------- bitrate maths


def test_bitrate_maths_matches_the_formula() -> None:
    # 10 MB over 100 s with audio: 10*1048576*8*0.95/100 = 796917 b/s total.
    bp = plan_bitrates(10, 100, has_audio=True, width=1280, height=720, fps=30)
    assert bp.audio_bps == 128_000
    assert bp.video_bps == pytest.approx(796_917 - 128_000, abs=2)
    total_bytes = (bp.video_bps + bp.audio_bps) * 100 / 8
    assert total_bytes <= 10 * 1024 * 1024


def test_no_audio_means_all_budget_to_video() -> None:
    bp = plan_bitrates(10, 100, has_audio=False, width=1280, height=720, fps=30)
    assert bp.audio_bps == 0 and bp.video_bps == int(10 * 1048576 * 8 * 0.95 / 100)


def test_low_budget_steps_audio_down() -> None:
    bp = plan_bitrates(2, 100, has_audio=True, width=640, height=360, fps=30)  # ~159 kb/s total
    assert bp.audio_bps == 32_000


def test_resolution_ladder_steps_down_until_bits_per_pixel_is_sane() -> None:
    rich = plan_bitrates(100, 60, has_audio=True, width=1920, height=1080, fps=30)
    assert rich.height == 1080 and not rich.scaled
    poor = plan_bitrates(5, 300, has_audio=True, width=1920, height=1080, fps=30)
    assert poor.height < 1080 and poor.scaled
    capped = plan_bitrates(100, 60, has_audio=True, width=1920, height=1080, fps=30, max_height=720)
    assert capped.height == 720


def test_hopeless_target_explains_the_minimum() -> None:
    with pytest.raises(MediaError, match=r"too small for 3600.0s.*about \d+\.\d+ MB"):
        plan_bitrates(5, 3600, has_audio=True, width=1920, height=1080, fps=30)
    with pytest.raises(MediaError, match="must be positive"):
        plan_bitrates(0, 10, has_audio=True, width=640, height=360, fps=30)
