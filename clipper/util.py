"""Small shared helpers: logging, the error type, time formatting, subprocesses."""
from __future__ import annotations

import subprocess
from pathlib import Path


class ClipperError(Exception):
    """An expected failure with a user-facing message.

    The CLI prints it and exits 1 (OBS runs: the line ends up in obs.log).
    """


def log(msg: str) -> None:
    """Print a progress line (OBS runs redirect this into obs.log)."""
    print(f"[clipper] {msg}", flush=True)


def hms(sec: float) -> str:
    """3725 -> '1:02:05'"""
    s = int(round(sec))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def twitch_t(sec: float) -> str:
    """3725 -> '1h02m05s' (Twitch ?t= format, also used in clip file names)"""
    s = int(sec)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def parse_ts(text: str) -> float:
    """'1:02:05' / '62:05' / '3725' -> seconds"""
    sec = 0.0
    for part in text.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def run(cmd: list[str], **kw) -> str:
    """Run a command, return stdout, raise ClipperError with stderr on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise ClipperError(f"{Path(cmd[0]).name} failed:\n{r.stderr[-2000:]}")
    return r.stdout
