"""Path resolution, the optional directory allow-list, and output naming."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from .errors import MediaError

_allowed: list[Path] = []


def set_allowed_dirs(dirs: Iterable[str | Path]) -> None:
    """Restrict every input and output to these directories (empty = unrestricted)."""
    resolved: list[Path] = []
    for d in dirs:
        p = Path(os.path.expanduser(str(d))).resolve()
        if not p.is_dir():
            raise MediaError(f"allowed directory '{d}' does not exist or is not a directory")
        resolved.append(p)
    _allowed[:] = resolved


def allowed_dirs() -> list[Path]:
    return list(_allowed)


def _resolve(raw: str) -> Path:
    if not raw or not raw.strip():
        raise MediaError("empty path")
    return Path(os.path.expanduser(raw.strip())).resolve()


def check_allowed(path: Path, what: str) -> None:
    if not _allowed:
        return
    if not any(path == d or path.is_relative_to(d) for d in _allowed):
        listing = ", ".join(str(d) for d in _allowed)
        raise MediaError(
            f"{what} '{path}' is outside the allowed directories ({listing}). "
            "The server was started with --dir to restrict file access."
        )


def resolve_input(raw: str) -> Path:
    path = _resolve(raw)
    check_allowed(path, "input")
    if not path.exists():
        raise MediaError(f"input file not found: '{path}'")
    if not path.is_file():
        raise MediaError(f"input is not a file: '{path}'")
    return path


def resolve_dir(raw: str | None, default: Path) -> Path:
    """An output directory (for extract_frames). It is created on demand."""
    path = _resolve(raw) if raw else default
    check_allowed(path, "output directory")
    if path.exists() and not path.is_dir():
        raise MediaError(f"output directory '{path}' exists and is not a directory")
    return path


def resolve_output(
    output: str | None, default: Path, *, overwrite: bool, inputs: Iterable[Path] = ()
) -> Path:
    """Pick the output path and refuse to clobber anything unless overwrite=True."""
    path = _resolve(output) if output else default
    check_allowed(path, "output")
    for src in inputs:
        if path == src:
            raise MediaError(
                f"output '{path}' is the same file as the input; choose a different output"
            )
    if path.is_dir():
        raise MediaError(f"output '{path}' is a directory; give a file name")
    if not path.parent.is_dir():
        raise MediaError(f"output directory '{path.parent}' does not exist")
    if path.exists() and not overwrite:
        raise MediaError(
            f"output '{path}' already exists. Pass overwrite=true to replace it, "
            "or choose another output name."
        )
    return path


def derived(src: Path, tag: str, ext: str | None = None) -> Path:
    """clip.mp4 + 'compressed' -> clip.compressed.mp4 (next to the input)."""
    suffix = ext if ext is not None else src.suffix
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    return src.with_name(f"{src.stem}.{tag}{suffix}")
