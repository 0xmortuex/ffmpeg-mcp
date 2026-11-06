"""Locate ffmpeg and ffprobe, or fail with instructions that fix it."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys

from .errors import MediaError

INSTALL_HINT = (
    "Install ffmpeg and make sure it is on PATH: "
    "Windows: `winget install Gyan.FFmpeg` | macOS: `brew install ffmpeg` | "
    "Debian/Ubuntu: `sudo apt install ffmpeg`. "
    "Or point to the binaries with the FFMPEG_PATH / FFPROBE_PATH environment variables "
    "(a full path to the executable). Restart the MCP client afterwards."
)


def _from_env(name: str, env_var: str) -> str | None:
    value = os.environ.get(env_var)
    if not value:
        return None
    if os.path.isdir(value):
        exe = name + (".exe" if sys.platform == "win32" else "")
        candidate = os.path.join(value, exe)
    else:
        candidate = value
    if not os.path.isfile(candidate):
        raise MediaError(
            f"{env_var} is set to {value!r}, but no {name} executable exists there. "
            f"Fix or unset {env_var}. {INSTALL_HINT}"
        )
    return candidate


def find_tool(name: str) -> str:
    """Return the path of `ffmpeg` or `ffprobe`.

    Order: FFMPEG_PATH / FFPROBE_PATH, then PATH, then (ffprobe only) the
    folder that holds the ffmpeg found by the first two steps.
    """
    env_var = f"{name.upper()}_PATH"
    found = _from_env(name, env_var) or shutil.which(name)
    if found is None and name == "ffprobe":
        try:
            ffmpeg = _from_env("ffmpeg", "FFMPEG_PATH") or shutil.which("ffmpeg")
        except MediaError:
            ffmpeg = None
        if ffmpeg:
            sibling = os.path.join(
                os.path.dirname(os.path.realpath(ffmpeg)),
                "ffprobe" + (".exe" if sys.platform == "win32" else ""),
            )
            if os.path.isfile(sibling):
                found = sibling
    if found is None:
        raise MediaError(f"`{name}` was not found on PATH and {env_var} is not set. {INSTALL_HINT}")
    return found


@functools.lru_cache(maxsize=8)
def _encoders_of(ffmpeg: str) -> frozenset[str]:
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        # " V....D libx264   libx264 H.264 ..." : flags first, then the name
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


def require_encoders(*encoders: str) -> None:
    """Raise an actionable error if this ffmpeg build lacks an encoder."""
    have = _encoders_of(find_tool("ffmpeg"))
    missing = [e for e in encoders if e not in have]
    if missing:
        raise MediaError(
            f"this ffmpeg build lacks the encoder(s): {', '.join(missing)}. "
            "Install a full ffmpeg build (Windows: `winget install Gyan.FFmpeg`; "
            "macOS: `brew install ffmpeg`; Linux: your distro's ffmpeg package)."
        )
