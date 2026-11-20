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
CLIP_LENGTH = 60        # seconds per clip
PEAK_POSITION = 0.65    # where the loud moment sits inside the clip (0.65 = build-up before it)
MARKER_POSITION = 0.85  # markers are pressed *after* the moment, so it sits near the end
MAX_CLIPS = 8           # auto-detected clips per stream (markers/--at are always kept)
MIN_GAP = None          # seconds between two auto clips; None = the clip length (no overlap)
THRESHOLD = 3.0         # minimum highlight score; lower = more (weaker) clips
SKIP_START = 60         # ignore the "starting soon" part
SKIP_END = 30

AUDIO_SR = 8000         # analysis sample rate; plenty for loudness
SILENCE_DB = -50        # quieter seconds never count as highlights
HYPE_WORDS = ("lul", "lol", "lmao", "pog", "kekw", "omg", "wtf", "clip", "no way", "holy", "gg")

# --remove-music: per-second "Music" probability from the AudioSet classifier
MUSIC_ON = 0.35         # a music stretch starts above this...
MUSIC_OFF = 0.20        # ...and lasts while it stays above this
MUSIC_MIN = 8           # seconds; shorter music blips are left alone
MUSIC_GAP = 10          # seconds; music stretches closer than this are merged
MUSIC_PAD = 2           # seconds added on both sides of a stretch
# ...and in those stretches, the separated vocals are kept only where they are:
VOICE_SIM = 0.65        # your voice: similarity to your voiceprint (0-1) at least this
KEEP_GAME_SPEECH = False  # also keep other clear speech (game characters)? Lets lyrics through
SPEECH_KEEP = 0.5       # with KEEP_GAME_SPEECH: "Speech" probability at least this...
SPEECH_MUSIC_MAX = 0.15 # ...and a "Music" probability below this
VOICE_QUIET_DB = -45    # vocals quieter than this (dBFS) are silence anyway

# --parts: the whole VOD in long parts for YouTube, cut at natural moments
PART_MIN = 30 * 60      # seconds; a part is between these two lengths...
PART_MAX = 60 * 60      # ...except the last one, which can be up to PART_MAX + PART_MIN / 2
BLACK_LEVEL = 20        # a pixel darker than this (0-255) counts as black...
BLACK_FRAC = 0.85       # ...and a frame with this share of black pixels is a fade / loading screen

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
