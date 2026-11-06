# Changelog

## 0.1.0 - 2026-11-06

First release.

- Nine tools: `media_info`, `convert`, `compress`, `trim`, `extract_audio`, `make_gif`,
  `extract_frames`, `concat`, `transform`.
- `compress` solves for a target size (two-pass encode, audio reserved, 5% safety margin, VBV peak
  cap, automatic downscale when the bitrate is too low) and has presets: discord-10mb,
  discord-50mb, whatsapp, email-25mb, web.
- Every tool takes `dry_run=true` (prints the exact, shell-quoted ffmpeg command line) and
  `overwrite` (existing outputs are never replaced by default).
- Outputs are written to a `.partial` name and renamed on success; a failed run never damages an
  existing file.
- Runs ffmpeg without a shell, parses `-progress pipe:1` into MCP progress notifications, enforces
  a timeout, and kills the whole process tree on timeout or client cancel.
- `--dir` / `MEDIAKIT_ALLOWED_DIRS` restrict which directories the server may read and write.
- ffmpeg/ffprobe found through `FFMPEG_PATH` / `FFPROBE_PATH` or PATH, with install hints
  (winget, brew, apt) when missing.
