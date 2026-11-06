# mediakit-mcp

**Everyday media jobs for your AI agent: say what you want, not which ffmpeg flags to use.**

<!-- mcp-name: io.github.0xmortuex/ffmpeg-mcp -->

mediakit-mcp is an [MCP](https://modelcontextprotocol.io) server that wraps ffmpeg in nine
intent-level tools: compress a video to fit Discord's 10 MB limit, turn a clip into a GIF, pull the
audio out, cut a section, grab thumbnails, join clips. The agent never has to remember
`-c:v libx264 -pass 1 -passlogfile ...`; you get the result and a short report, not a screen of
ffmpeg log.

Every tool has a **dry run** that returns the exact ffmpeg command it would execute, and nothing
is ever overwritten unless you say so.

> The PyPI package is `mediakit-mcp` (the name `ffmpeg-mcp` was already taken by another project).
> The GitHub repository is `ffmpeg-mcp`.

## What it looks like

Real output from the test suite:

```
> compress video.mp4 target_mb=1
Wrote C:\clips\video.compressed.mp4
  size: 21.23 MB -> 0.93 MB (4% of the original)
  duration: 6.02s | video: h264 640x360 | audio: aac
  target 1 MB: met
  note: two-pass encode: video 1200 kb/s, audio 128 kb/s
  took 1.1s

> compress video.mp4 preset=discord-10mb dry_run=true
DRY RUN: nothing was executed. This is what would run:
Step 1/2 (analysis pass):
  C:\...\ffmpeg.exe -hide_banner -nostdin -loglevel error -nostats -progress pipe:1 -y -i C:\clips\video.mp4 -map 0:v:0 -c:v libx264 -preset medium -b:v 1200000 -pix_fmt yuv420p -maxrate 1200000 -bufsize 2400000 -pass 1 -passlogfile ... -an -f null -
Step 2/2 (encode pass):
  ...
```

## Install

You need ffmpeg and ffprobe on PATH (or pointed to by `FFMPEG_PATH` / `FFPROBE_PATH`):

| OS | Command |
|----|---------|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |

Use a full build: the server needs libx264, libvpx-vp9, libopus and libmp3lame, and tells you
which one is missing if it isn't there.

**Claude Code**

```bash
claude mcp add mediakit -- uvx mediakit-mcp
```

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "mediakit": {
      "command": "uvx",
      "args": ["mediakit-mcp"]
    }
  }
}
```

Restrict it to the folders you work in (recommended):

```json
{
  "mcpServers": {
    "mediakit": {
      "command": "uvx",
      "args": ["mediakit-mcp", "--dir", "C:\\Users\\you\\Videos", "--dir", "C:\\Users\\you\\Downloads"]
    }
  }
}
```

Or with pip: `pip install mediakit-mcp`, then run `mediakit-mcp`.

## Try asking

- *"What's in `C:\Videos\demo.mp4`?"*
- *"Compress demo.mp4 so I can send it on Discord."* (`preset=discord-10mb`)
- *"Get this under 25 MB for email."*
- *"Make a GIF of the first 4 seconds of demo.mp4, 480 px wide."*
- *"Cut from 1:20 to 1:45 and keep the quality."* / *"...fast, I don't need it frame-exact."*
- *"Extract the audio as an mp3 at 192k."*
- *"Give me a thumbnail every 30 seconds."*
- *"Join part1.mp4 and part2.mp4."*
- *"Make it 720p wide and play it at 1.5x."*
- *"Show me the ffmpeg command you'd use to convert this to webm, but don't run it."*

## Tools

| Tool | What it does |
|------|--------------|
| `media_info` | ffprobe summary: container, duration, size, bitrate, every stream (codec, resolution, fps, sample rate, channels). |
| `convert` | To mp4, mov, mkv (H.264+AAC), webm (VP9+Opus), mp3, wav, flac, ogg, m4a, opus, aac, or a short gif. `quality` low/medium/high. |
| `compress` | `target_mb=N` (two-pass, bitrate maths), `preset` (discord-10mb, discord-50mb, whatsapp, email-25mb, web), or a quality-based re-encode. `max_height` caps the resolution. |
| `trim` | `start` / `end` / `duration` (seconds or `HH:MM:SS`). `mode="accurate"` re-encodes; `mode="fast"` stream-copies. |
| `extract_audio` | mp3, wav, flac, ogg, m4a, opus, aac, or `copy` the original stream. Clear error if there is no audio. |
| `make_gif` | Palette-based GIF (palettegen + paletteuse). `fps` 1-30, `width`, `start`, `duration` (max 30 s), `loop`. |
| `extract_frames` | PNG/JPG stills at `times=[...]` or every `every_s` seconds, with a `max_frames` guard. |
| `concat` | Joins same-codec clips without re-encoding; says exactly which property differs if they don't match. |
| `transform` | Resize (`width`/`height`), `crop` (`W:H[:X:Y]`), `speed` (0.25-4, audio retimed). |

Every tool except `media_info` also takes `output`, `overwrite` (default false), `dry_run` and
`timeout_s`. Outputs go next to the input with a derived name: `clip.mp4` becomes
`clip.compressed.mp4`, `clip.trimmed.mp4`, `clip.gif`, `clip.audio.mp3`, `clip.frames/`.

### How `compress` hits a size

For `target_mb=T` and duration `D`: the total bitrate budget is `T * 1 MiB * 8 * 0.95 / D` (the 5%
is a margin for container overhead and rate-control error; VP9 uses 10%). Audio gets 128 kb/s, or
less if that would eat more than a quarter of the budget, and the rest goes to video. The video is
encoded in two passes with a peak cap, and if the bitrate works out below about 0.045 bits per pixel
per frame the resolution steps down a ladder (1080, 720, 480, 360, ...) rather than producing mush.
A target that is impossible even at minimum quality is an error that says how big the file would
need to be. The result always says whether the target was met. In the test suite, targets from
0.5 to 10 MB landed at 91-97% of the target.

`email-25mb` aims at 18 MB because attachments grow by about 37% when mailed (base64).

## Safety

- **No shell.** ffmpeg is started with an argument list, so file names with spaces, quotes or
  `; & |` cannot inject anything. Dry-run output is quoted for pasting into cmd or PowerShell.
- **Never overwrites by default.** An existing output is an error unless `overwrite=true`.
  ffmpeg writes to `name.partial.ext` and the file is renamed only on success, so a failed or
  cancelled run leaves no half-written file and does not damage an existing one.
- **Directory allow-list.** Start with `--dir DIR` (repeatable) or `MEDIAKIT_ALLOWED_DIRS`
  (separated by `;` on Windows, `:` elsewhere) and every input and output path must resolve inside
  one of them. `..` and symlinks are resolved first. Without it, the server can read and write any
  file the user can, which is what a local ffmpeg tool normally does.
- **Bounded runs.** Each job has a timeout (default 900 s). On timeout, or when the client cancels
  the call, ffmpeg's whole process tree is killed and temporary files are removed.
- **Progress, not log spam.** ffmpeg runs with `-loglevel error`; progress arrives as MCP progress
  notifications, and a failure returns only the last few relevant stderr lines.
- It does not touch the network: it only runs `ffmpeg` and `ffprobe`. Input names can't be turned
  into options because every path is resolved to an absolute path first.

## Limits

- `concat` joins without re-encoding, so the clips must match. Use `convert`/`transform` first if
  they don't.
- `trim mode="fast"` can only cut on keyframes, so the clip may start a little early.
- Hardware encoders (NVENC, QSV, VideoToolbox) are not used; software x264/VP9 is slower but
  consistent everywhere.
- `dry_run` still runs `ffprobe` on the input (read-only) so that the command it prints is the real
  one.
- Only the first video and audio stream of a file are used.

## Tests

```bash
uv sync --extra test
uv run ruff check src tests && uv run mypy --strict src
uv run pytest tests
```

The tests generate real media with ffmpeg's `lavfi` sources (`testsrc2`, `sine`), run every tool,
and check the results with ffprobe: durations within tolerance, sizes under their targets, codecs,
GIF frame counts, odd dimensions, no-audio inputs, corrupt and truncated files, existing outputs.
The runner is tested against real processes (progress, timeout, cancel, killing a grandchild
process), and `tests/test_stdio_handshake.py` launches the installed server over stdio and calls
every tool through MCP, including cancelling a running job.

## Learn how it's built

[ffmpeg-mcp-the-hard-way](https://github.com/0xmortuex/ffmpeg-mcp-the-hard-way) builds a minimal version of this server in
seven modules.

## License

MIT
