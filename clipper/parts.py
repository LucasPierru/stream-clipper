"""--parts: the whole VOD as 30-60 min parts for YouTube, cut at natural moments.

Each cut is made somewhere between PART_MIN and PART_MAX after the previous one, at
the best-scoring keyframe in that window:
  - a fade to black / loading screen: games fade out between missions, when you sleep,
    on a time skip or a loading screen. The longer the black, the better. Found on a
    5 fps scan of the whole video (a fade lasts about a second, so keyframes alone,
    one every ~2 s, miss most of them), ignoring overlays such as the webcam.
  - never inside a cutscene: letterbox bars (black bands at the top and bottom) mark one
  - not in the few minutes before a long cutscene (driving to a mission, walking up to
    the character who starts it), and preferably not in the minutes after one (often
    the mission it opened): a long cutscene usually opens or closes a mission
  - quiet audio around it (you aren't in the middle of a sentence; the audio used is
    the music-free track with --remove-music)
  - a big picture change from the previous keyframe (a new scene), worth a little
  - never inside a highlight clip
  - a small bonus for longer parts, so fewer parts when nothing else decides
The cuts are lossless copies of the video, so they land exactly on a keyframe (Twitch
puts one every ~2 s): the one right after the middle of the black.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (BLACK_FRAC, BLACK_LEVEL, CACHE_DIR, PART_MAX, PART_MIN)
from .models import Highlight, Source
from .util import ClipperError, hms, log, run, twitch_t

THUMB_W, THUMB_H = 64, 36
SCAN_FPS = 5            # frames per second looked at for fades and cutscenes
CHANGE_FULL = 40.0      # mean grey-level change between keyframes that counts as a new scene
QUIET_SPAN = 5          # seconds on each side of a cut that should be quiet
HIGHLIGHT_MARGIN = 5    # seconds around a highlight clip that are off limits
CUTSCENE_SPAN = 3       # seconds on each side of a cut that must be free of letterbox bars
LONG_CUTSCENE = 30      # seconds; a cutscene this long usually opens or closes a mission
LEAD_UP = 180           # seconds before a long cutscene: heading into a mission, don't cut
AFTER_CUTSCENE = 300    # seconds after one: often the mission it opened (a mild penalty)


@dataclass
class Part:
    index: int
    start: float
    end: float
    why: str
    file: str = ""


@dataclass
class Frames:
    """The 5 fps scan: per frame, the share of near-black pixels, the mean grey level,
    and whether letterbox bars (a cutscene) are on screen."""
    fps: float
    dark: np.ndarray
    mean: np.ndarray
    letterbox: np.ndarray


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def scan_keyframes(path: Path, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Time and a tiny grey thumbnail (THUMB_H x THUMB_W) of every keyframe. Only the
    keyframes are decoded, so a long VOD takes a minute or two (cached)."""
    if cache and cache.exists():
        d = np.load(cache)
        return d["times"], d["thumbs"]
    log("finding the keyframes...")
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


def frame_features(thumbs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dark share, mean, letterbox) for (n, THUMB_H, THUMB_W) grey thumbnails.

    Letterbox: in most columns the outer row at the top and at the bottom is black and
    the next one nearly so (the bars are thin: ~1.5 rows of 36 for a 2:1 picture), while
    the picture between them isn't. Requiring it in only 60% of the columns lets an
    overlay (webcam, follower count) sit over a bar."""
    f = thumbs.astype(np.int16)
    dark = (f < BLACK_LEVEL).mean(axis=(1, 2))
    mean = f.mean(axis=(1, 2))
    bars = (f[:, 0] < 10) & (f[:, 1] < 24) & (f[:, -1] < 10) & (f[:, -2] < 24)
    inner = (f[:, 3:-3] >= 24).mean(axis=(1, 2))
    letterbox = (bars.mean(axis=1) >= 0.6) & (inner > 0.25)
    return dark.astype(np.float32), mean.astype(np.float32), letterbox


def scan_frames(path: Path, cache: Path | None = None) -> Frames:
    """Decode the whole video at SCAN_FPS as tiny grey thumbnails, streamed (cached).
    Non-reference frames are skipped by the decoder, which makes it ~25% faster;
    about 4 minutes for a 3 h 1080p60 VOD."""
    if cache and cache.exists():
        d = np.load(cache)
        return Frames(float(d["fps"]), d["dark"], d["mean"], d["letterbox"])
    log("scanning the video for fades to black and cutscenes (a few minutes)...")
    size = THUMB_W * THUMB_H
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-skip_frame", "noref", "-i", str(path), "-an",
                             "-vf", f"fps={SCAN_FPS},scale={THUMB_W}:{THUMB_H},format=gray",
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parts, rest = [], b""
    try:
        while True:
            buf = proc.stdout.read(size * 2000)
            if not buf:
                break
            buf = rest + buf
            n = len(buf) // size
            rest = buf[n * size:]
            if n:
                parts.append(frame_features(np.frombuffer(buf[:n * size], np.uint8).reshape(n, THUMB_H, THUMB_W)))
    finally:
        proc.stdout.close()
        err = proc.stderr.read()
        proc.stderr.close()
    if proc.wait() != 0 or not parts:
        raise ClipperError(f"ffmpeg could not scan the video: {err.decode(errors='replace')[-300:]}")
    fr = Frames(float(SCAN_FPS), *(np.concatenate(x) for x in zip(*parts)))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fps=fr.fps, dark=fr.dark, mean=fr.mean, letterbox=fr.letterbox)
    return fr


def fade_events(fr: Frames) -> list[tuple[float, float]]:
    """(start, end) in seconds of every fade to black: frames where at least BLACK_FRAC
    of the picture is near-black (overlays such as the webcam may stay), clearly darker
    than just before or after - so a dark night scene that stays dark doesn't count."""
    black = fr.dark >= BLACK_FRAC
    events, n = [], len(black)
    idx = np.flatnonzero(black)
    if not len(idx):
        return events
    breaks = np.flatnonzero(np.diff(idx) > 1)
    for a, b in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]):
        inside = fr.mean[a:b + 1].mean()
        before = np.median(fr.mean[max(0, a - 15):max(0, a - 2)]) if a >= 3 else 0.0
        after = np.median(fr.mean[b + 3:b + 16]) if b + 3 < n else 0.0
        if max(before, after) >= inside + 15:
            events.append((a / fr.fps, (b + 1) / fr.fps))
    return events


def long_cutscenes(fr: Frames) -> list[tuple[float, float]]:
    """(start, end) in seconds of letterboxed stretches of at least LONG_CUTSCENE
    (gaps up to 5 s, e.g. a shot without bars, are bridged)."""
    idx = np.flatnonzero(fr.letterbox)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > 5 * fr.fps)
    starts, ends = np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]
    return [(a / fr.fps, (b + 1) / fr.fps) for a, b in zip(starts, ends) if (b + 1 - a) / fr.fps >= LONG_CUTSCENE]


def cut_scores(times: np.ndarray, thumbs: np.ndarray, db: np.ndarray,
               highlights: list[Highlight], fr: Frames | None = None) -> tuple[np.ndarray, list[list[str]]]:
    """Score and reasons for cutting at each keyframe (without the length bonus)."""
    f = thumbs.astype(np.float32)
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

    fade = np.zeros(len(times))
    fade_why = [""] * len(times)
    cutscene = np.zeros(len(times), bool)
    story = np.zeros(len(times))
    if fr is not None and len(fr.dark):
        for a, b in long_cutscenes(fr):
            story[(times >= a - LEAD_UP) & (times < a - CUTSCENE_SPAN)] += 3.0
            story[(times > b + 10) & (times <= b + AFTER_CUTSCENE)] += 1.5
        lb = np.r_[0, np.cumsum(fr.letterbox)]

        def letterboxed(a: float, b: float) -> float:
            i, j = int(max(0, a) * fr.fps), int(min(len(fr.letterbox), max(0, b) * fr.fps))
            return (lb[j] - lb[i]) / (j - i) if j > i else 0.0

        def mid_cutscene(a: float, b: float) -> bool:
            """Bars both before a and after b: inside a cutscene, not where one starts or ends."""
            return min(letterboxed(a - CUTSCENE_SPAN, a), letterboxed(b, b + CUTSCENE_SPAN)) >= 0.5

        for k, t in enumerate(times):
            cutscene[k] = mid_cutscene(t, t)
        for a, b in fade_events(fr):
            length = b - a
            # a single black frame is a camera cut or the phone opening, not a break
            value = 4.0 if length >= 3 else 3.0 if length >= 1 else 1.5 if length >= 0.4 else 0.0
            if mid_cutscene(a, b):                     # a fade inside a cutscene
                value *= 0.25
            if not value:
                continue
            # the first keyframe near the end of the black (the middle of a short one):
            # the part starts on the new scene, not on seconds of black
            k = int(np.searchsorted(times, max((a + b) / 2, b - 1.0) - 0.1))
            if k >= len(times) or times[k] > b + 3:
                continue
            if value > fade[k]:
                fade[k] = value
                fade_why[k] = (f"black/loading screen ({length:.0f} s)" if length >= 3
                               else f"fade to black ({length:.1f} s)")

    score = fade + 0.5 * change + 2.0 * quiet - 5.0 * busy - 4.0 * cutscene - np.minimum(story, 3.0)
    reasons = []
    for k in range(len(times)):
        why = [fade_why[k]] if fade[k] >= 1 else []
        why += ["scene change"] if change[k] >= 0.6 else []
        why += ["quiet"] if quiet[k] >= 0.5 else []
        reasons.append(why)
    return score, reasons


def choose_cuts(times: np.ndarray, score: np.ndarray, reasons: list[list[str]], duration: float,
                min_len: float = PART_MIN, max_len: float = PART_MAX) -> list[tuple[float, str]]:
    """Cut times (keyframe times) and why each was chosen. The last part may run up to
    max_len + min_len / 2, so it is never a short leftover (at least min_len / 2)."""
    tail = min_len / 2
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
    if duration <= max_len + min_len / 2:
        return [Part(1, 0.0, duration, "start of the stream")]
    times, thumbs = scan_keyframes(src.path, CACHE_DIR / f"{src.key}_keyframes.npz")
    if len(times) < 2:
        raise ClipperError("found no keyframes in the video - can't split it into parts")
    fr = scan_frames(src.path, CACHE_DIR / f"{src.key}_frames.npz")
    score, reasons = cut_scores(times, thumbs, db, highlights, fr)
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
