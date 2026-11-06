"""A Plan is "what we would run": built (and shown for dry_run) before anything executes."""

from __future__ import annotations

import glob
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .binaries import find_tool
from .errors import MediaError
from .probe import fmt_duration, fmt_size, probe
from .runner import Step, quote_command, run_steps

BASE_FLAGS = ["-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-y"]


def ffmpeg_argv(*parts: str) -> list[str]:
    """ffmpeg + common flags + the given arguments."""
    return [find_tool("ffmpeg"), *BASE_FLAGS, *parts]


def partial_of(final: Path) -> Path:
    """clip.mp4 -> clip.partial.mp4: ffmpeg writes here, then we rename on success."""
    return final.with_name(f"{final.stem}.partial{final.suffix}")


def commit_file(partial: Path, final: Path, overwrite: bool) -> Path:
    if not overwrite and final.exists():
        raise MediaError(
            f"'{final}' appeared while ffmpeg was running; not overwriting it "
            f"(the new file is at '{partial}')"
        )
    try:
        os.replace(partial, final)
    except OSError as e:
        raise MediaError(f"could not move the result into place at '{final}': {e}") from e
    return final


@dataclass
class Plan:
    tool: str
    input: Path
    steps: list[Step]
    outputs: list[Path]
    duration_s: float | None = None  # expected output duration, for progress
    notes: list[str] = field(default_factory=list)
    temp_paths: list[Path] = field(default_factory=list)  # removed afterwards if present
    temp_globs: list[str] = field(default_factory=list)  # e.g. two-pass log files
    prepare: Callable[[], None] | None = None
    finalize: Callable[[], list[Path]] | None = None
    target_mb: float | None = None
    input_bytes_extra: int = 0  # concat: the other inputs' sizes


def simple_plan(
    tool: str,
    src: Path,
    final: Path,
    build: Callable[[Path], list[Step]],
    *,
    overwrite: bool,
    duration_s: float | None,
    notes: list[str] | None = None,
) -> Plan:
    """The common case: one output file, written to a partial name then renamed."""
    part = partial_of(final)
    return Plan(
        tool=tool,
        input=src,
        steps=build(part),
        outputs=[final],
        duration_s=duration_s,
        notes=notes or [],
        temp_paths=[part],
        finalize=lambda: [commit_file(part, final, overwrite)],
    )


def format_dry_run(plan: Plan) -> str:
    lines = ["DRY RUN: nothing was executed. This is what would run:"]
    n = len(plan.steps)
    for i, step in enumerate(plan.steps, 1):
        label = f" ({step.label})" if step.label else ""
        if n > 1:
            lines.append(f"Step {i}/{n}{label}:")
        lines.append("  " + quote_command(step.argv))
    for out in plan.outputs[:10]:
        lines.append(f"Result would be written to: {out}")
    if len(plan.outputs) > 10:
        lines.append(f"  ... and {len(plan.outputs) - 10} more files")
    lines.append(
        "(ffmpeg writes to a temporary '.partial' name first and renames it on success, "
        "so a failed run never leaves a half-written output or damages an existing one.)"
    )
    lines.extend(f"Note: {n}" for n in plan.notes)
    return "\n".join(lines)


def _cleanup(plan: Plan) -> None:
    for p in plan.temp_paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    for pattern in plan.temp_globs:
        for f in glob.glob(pattern):
            try:
                os.unlink(f)
            except OSError:
                pass


def execute(
    plan: Plan,
    *,
    timeout_s: float,
    cancel: threading.Event | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_spawn: Callable[[int], None] | None = None,
) -> str:
    """Run the plan and return a concise report."""
    before = plan.input.stat().st_size + plan.input_bytes_extra
    t0 = time.monotonic()
    try:
        if plan.prepare:
            plan.prepare()
        run_steps(
            plan.steps,
            timeout_s=timeout_s,
            cancel=cancel,
            on_progress=on_progress,
            total_s=plan.duration_s,
            on_spawn=on_spawn,
        )
        assert plan.finalize is not None
        finals = plan.finalize()
    finally:
        _cleanup(plan)
    return report(plan, finals, before, time.monotonic() - t0)


def report(plan: Plan, finals: list[Path], before: int, took: float) -> str:
    if len(finals) == 1:
        out = finals[0]
        size = out.stat().st_size
        pct = f" ({size / before * 100:.0f}% of the original)" if before else ""
        lines = [f"Wrote {out}", f"  size: {fmt_size(before)} -> {fmt_size(size)}{pct}"]
        try:
            info = probe(out)
        except MediaError as e:
            lines.append(f"  (the file was written but could not be re-read for verification: {e})")
        else:
            bits = [f"duration: {fmt_duration(info.duration)}"]
            if info.video:
                bits.append(f"video: {info.video.codec} {info.video.width}x{info.video.height}")
            if info.audio:
                bits.append(f"audio: {info.audio.codec}")
            lines.append("  " + " | ".join(bits))
        if plan.target_mb is not None:
            limit = plan.target_mb * 1024 * 1024
            if size <= limit:
                lines.append(f"  target {plan.target_mb:g} MB: met")
            else:
                lines.append(
                    f"  target {plan.target_mb:g} MB: MISSED by {fmt_size(size - limit)}. "
                    "Re-run with a smaller target_mb (for example 10% lower)."
                )
    else:
        total = sum(f.stat().st_size for f in finals)
        lines = [
            f"Wrote {len(finals)} files to {finals[0].parent}",
            f"  first: {finals[0].name} | last: {finals[-1].name} | total {fmt_size(total)}",
        ]
    lines.extend(f"  note: {n}" for n in plan.notes)
    lines.append(f"  took {took:.1f}s")
    return "\n".join(lines)
