"""Settings: the .env file, folders and tuning defaults.

The .env file sits in the project root (next to stream_clipper.py). Real
environment variables always win over it. CLIPPER_ENV can point elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("CLIPPER_ENV") or ROOT_DIR / ".env")


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines into os.environ (without overriding real env vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].strip()
        if key and not os.environ.get(key):
            os.environ[key] = val


def env(key: str) -> str | None:
    return os.environ.get(key) or None


def save_env(key: str, value: str) -> None:
    """Write/replace KEY=value in the .env file (used to store Twitch tokens)."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value


load_env()

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
DEFAULT_OUT = env("CLIPPER_OUT") or str(Path.home() / "Videos" / "StreamClips")
CACHE_DIR = Path(env("CLIPPER_CACHE") or Path.home() / ".cache" / "stream_clipper")
PROCESSED_FILE = CACHE_DIR / "processed_streams.json"

# ----------------------------------------------------------------------------
# Tuning defaults - edit these, or override with command-line flags
# ----------------------------------------------------------------------------
CLIP_LENGTH = 30        # seconds per clip
PEAK_POSITION = 0.65    # where the loud moment sits inside the clip (0.65 = build-up before it)
MARKER_POSITION = 0.85  # markers are pressed *after* the moment, so it sits near the end
MAX_CLIPS = 8           # auto-detected clips per stream (markers/--at are always kept)
MIN_GAP = 45            # seconds between two auto-detected clips
THRESHOLD = 3.0         # minimum highlight score; lower = more (weaker) clips
SKIP_START = 60         # ignore the "starting soon" part
SKIP_END = 30

AUDIO_SR = 8000         # analysis sample rate; plenty for loudness
SILENCE_DB = -50        # quieter seconds never count as highlights
HYPE_WORDS = ("lul", "lol", "lmao", "pog", "kekw", "omg", "wtf", "clip", "no way", "holy", "gg")

# libass style for burned-in captions (sizes are relative to a 288px-tall canvas)
CAPTION_STYLE = ("FontName=Arial,Fontsize=13,Bold=1,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
                 "Alignment=2,MarginV=70")
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".ts", ".mov", ".flv"}

# ----------------------------------------------------------------------------
# Twitch
# ----------------------------------------------------------------------------
TWITCH_REDIRECT = "http://localhost:3000"
TWITCH_LOGIN_PORT = 3000
