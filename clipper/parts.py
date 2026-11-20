"""--parts: the whole VOD as 45-60 min parts for YouTube, cut at natural moments.

Each cut is made somewhere between PART_MIN and PART_MAX after the previous one, at
the best-scoring keyframe in that window:
  - a black / loading screen (dark and flat): the best sign a mission, cutscene or
    scene just ended, so it's worth the most
  - a big picture change from the previous keyframe (a new scene)
  - quiet audio around it (you aren't in the middle of a sentence; the audio used is
    the music-free track with --remove-music)
  - never inside a highlight clip
  - a small bonus for longer parts, so fewer parts when nothing else decides
Only keyframes are looked at (Twitch puts one every ~2 s), so the cuts are lossless
copies of the video: fast, no quality loss, and exactly on a keyframe.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, DARK_LEVEL, PART_MAX, PART_MIN
from .models import Highlight, Source
from .util import ClipperError, hms, log, run, twitch_t

THUMB_W, THUMB_H = 64, 36
CHANGE_FULL = 40.0      # mean grey-level change between keyframes that counts as a new scene
QUIET_SPAN = 5          # seconds on each side of a cut that should be quiet
HIGHLIGHT_MARGIN = 5    # seconds around a highlight clip that are off limits


@dataclass
class Part:
    index: int
    start: float
    end: float
    why: str
    file: str = ""


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def scan_keyframes(path: Path, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Time and a tiny grey thumbnail (THUMB_H x THUMB_W) of every keyframe. Only the
    keyframes are decoded, so a long VOD takes a minute or two (cached)."""
    if cache and cache.exists():
        d = np.load(cache)
        return d["times"], d["thumbs"]
    log("scanning the video for scene changes and black screens...")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-skip_frame", "nokey", "-i", str(path),
                        "-an", "-vf", f"scale={THUMB_W}:{THUMB_H},format=gray,showinfo",
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True)
    if r.returncode:
        raise ClipperError(f"ffmpeg could not scan the video: {r.stderr.decode(errors='replace')[-300:]}")
    times = np.array([float(t) for t in re.findall(rb"pts_time:\s*([-\d.]+)", r.stderr)])
    size = THUMB_W * THUMB_H
    n = min(len(times), len(r.stdout) // size)
    thumbs = np.frombuffer(r.stdout[:n * size], np.uint8).reshape(n, THUMB_H, THUMB_W)
    times = times[:n]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, times=times, thumbs=thumbs)
    return times, thumbs


def cut_scores(times: np.ndarray, thumbs: np.ndarray, db: np.ndarray,
               highlights: list[Highlight]) -> tuple[np.ndarray, list[list[str]]]:
    """Score and reasons for cutting at each keyframe (without the length bonus)."""
    f = thumbs.astype(np.float32)
    dark = (f.mean(axis=(1, 2)) < DARK_LEVEL) & (f.std(axis=(1, 2)) < 12)
    change = np.zeros(len(f))
    change[1:] = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    change = np.minimum(change / CHANGE_FULL, 1.0)

    level = np.median(db) if len(db) else 0.0
    quiet = np.zeros(len(times))
    for k, t in enumerate(times):
        win = db[max(0, int(t) - QUIET_SPAN): int(t) + QUIET_SPAN + 1]
        if len(win):
            quiet[k] = np.clip((level - win.mean()) / 10, 0, 1)

    busy = np.zeros(len(times), bool)
    for h in highlights:
        busy |= (times >= h.start - HIGHLIGHT_MARGIN) & (times <= h.start + h.length + HIGHLIGHT_MARGIN)

    score = 3.0 * dark + 1.0 * change + 2.0 * quiet - 5.0 * busy
    reasons = []
    for k in range(len(times)):
        why = (["black/loading screen"] if dark[k] else []) + (["scene change"] if change[k] >= 0.6 else [])
        why += ["quiet"] if quiet[k] >= 0.5 else []
        reasons.append(why)
    return score, reasons


def choose_cuts(times: np.ndarray, score: np.ndarray, reasons: list[list[str]], duration: float,
                min_len: float = PART_MIN, max_len: float = PART_MAX) -> list[tuple[float, str]]:
    """Cut times (keyframe times) and why each was chosen. The last part may run up to
    max_len + min_len / 3, so it is never a short leftover."""
    tail = min_len / 3
    cuts, prev = [], 0.0
    while duration - prev > max_len + tail:
        lo, hi = prev + min_len, min(prev + max_len, duration - tail)
        idx = np.flatnonzero((times >= lo) & (times <= hi))
        if not len(idx):                      # no keyframe there (shouldn't happen): last one before hi
            before = np.flatnonzero(times <= hi)
            if not len(before) or times[before[-1]] <= prev:
                break
            k = before[-1]
            cuts.append((float(times[k]), "no keyframe in the window"))
            prev = float(times[k])
            continue
        bonus = 0.5 * (times[idx] - lo) / max(hi - lo, 1.0)
        k = idx[int(np.argmax(score[idx] + bonus))]
        why = ", ".join(reasons[k]) or "no clear break - best spot available"
        cuts.append((float(times[k]), why))
        prev = float(times[k])
    return cuts


# ----------------------------------------------------------------------------
# Cutting
# ----------------------------------------------------------------------------
def plan_parts(src: Source, duration: float, db: np.ndarray, highlights: list[Highlight],
               min_len: float, max_len: float) -> list[Part]:
    times, thumbs = scan_keyframes(src.path, CACHE_DIR / f"{src.key}_keyframes.npz")
    if len(times) < 2:
        raise ClipperError("found no keyframes in the video - can't split it into parts")
    score, reasons = cut_scores(times, thumbs, db, highlights)
    cuts = choose_cuts(times, score, reasons, duration, min_len, max_len)
    starts = [0.0] + [t for t, _ in cuts]
    ends = [t for t, _ in cuts] + [duration]
    whys = ["start of the stream"] + [w for _, w in cuts]
    return [Part(i, a, b, w) for i, (a, b, w) in enumerate(zip(starts, ends, whys), 1)]


def write_parts(src: Source, parts: list[Part], out_dir: Path) -> None:
    """Lossless cuts of the video (the audio is re-encoded only when it comes from the
    music-free track). Old part files in the folder are replaced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part_*.mp4"):
        old.unlink()
    for p in parts:
        name = f"part_{p.index:02d}_{twitch_t(p.start)}.mp4"
        log(f"writing {name}  ({hms(p.start)} - {hms(p.end)})")
        # +0.01 s: seeking lands on the last keyframe at or before the time, so this
        # stays on the cut's own keyframe even if the time was rounded down
        ss, length = f"{p.start + 0.01 if p.start else 0:.3f}", f"{p.end - p.start:.3f}"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", ss, "-i", str(src.path)]
        if src.audio:
            cmd += ["-ss", ss, "-i", str(src.audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy"]
        run(cmd + ["-t", length, "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                   str(out_dir / name)])
        p.file = name


def write_parts_report(src: Source, parts: list[Part], out_dir: Path) -> None:
    lines = [f"# {src.title} - full VOD in parts", "",
             "| Part | File | Start | Length | Cut here because |", "|---|---|---|---|---|"]
    for p in parts:
        when = hms(p.start)
        if src.url and "twitch.tv/videos" in src.url:
            when = f"[{when}]({src.url}?t={twitch_t(max(0.0, src.to_vod_time(p.start)))})"
        lines.append(f"| {p.index} | {p.file} | {when} | {hms(p.end - p.start)} | {p.why} |")
    (out_dir / "parts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
