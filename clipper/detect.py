"""Highlight detection: per-second signals (audio, chat) and picking the moments."""
from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import AUDIO_SR, CACHE_DIR, HYPE_WORDS, MARKER_POSITION, SILENCE_DB
from .models import Highlight, Source
from .util import ClipperError, hms, log, run


# ----------------------------------------------------------------------------
# Signal helpers
# ----------------------------------------------------------------------------
def rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Centred rolling window, edge-padded so the output has the same length."""
    w = max(1, min(w, len(x)))
    left = w // 2
    xp = np.pad(x, (left, w - 1 - left), mode="edge")
    return fn(sliding_window_view(xp, w), axis=-1)


def rolling_nanmedian(x: np.ndarray, w: int, fill: float) -> np.ndarray:
    """Centred rolling median that ignores NaN; windows with no data get `fill`."""
    w = max(1, min(w, len(x)))
    left = w // 2
    xp = np.pad(x, (left, w - 1 - left), mode="edge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        med = np.nanmedian(sliding_window_view(xp, w), axis=-1)
    return np.where(np.isnan(med), fill, med)


def robust_z(x: np.ndarray, floor: float) -> np.ndarray:
    """z-score using median/MAD, so a few huge spikes don't hide the others."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, floor)


# ----------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------
def loudness_per_second(path: Path, cache: Path) -> np.ndarray:
    """Loudness in dBFS for every second of the video (cached for re-runs)."""
    if cache.exists():
        return np.load(cache)
    log("analysing audio (one pass over the whole VOD)...")
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                             "-ar", str(AUDIO_SR), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    parts, done, next_log = [], 0, 1800
    while True:
        buf = proc.stdout.read(AUDIO_SR * 2 * 60)  # one minute at a time
        if not buf:
            break
        x = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32)
        n = len(x) // AUDIO_SR
        if n:
            parts.append(np.sqrt((x[: n * AUDIO_SR].reshape(n, AUDIO_SR) ** 2).mean(axis=1)))
        if len(x) > n * AUDIO_SR:
            parts.append(np.array([np.sqrt((x[n * AUDIO_SR:] ** 2).mean())]))
        done += len(x) / AUDIO_SR
        if done >= next_log:
            log(f"  {hms(done)} analysed")
            next_log += 1800
    proc.wait()
    if proc.returncode or not parts:
        raise ClipperError("could not read the audio track")
    db = 20 * np.log10(np.concatenate(parts) / 32768 + 1e-6)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, db)
    return db


def audio_score(db: np.ndarray) -> np.ndarray:
    """High where you get suddenly louder than the last few minutes (yelling, laughing)."""
    # Silent seconds (silent "starting soon" screen, muted mic, BRB, DMCA-muted VOD
    # parts) are left out of the baseline and replaced by it, so they neither drag the
    # baseline down (making normal talking right after look like a highlight) nor
    # turn sound coming back after silence into a "sudden loud moment".
    silent = db < SILENCE_DB
    voiced = db[~silent]
    fill = float(np.median(voiced)) if len(voiced) else float(SILENCE_DB)
    base = rolling_nanmedian(np.where(silent, np.nan, db), 300, fill)  # 5-min baseline
    ref = np.where(silent, base, db)
    level = robust_z(ref - base, floor=1.0)
    k = 5                                          # jump vs. the previous 5 s
    padded = np.concatenate([np.full(k, ref[0]), ref])
    cs = np.concatenate([[0.0], np.cumsum(padded)])
    prev_mean = (cs[k:k + len(db)] - cs[:len(db)]) / k
    jump = robust_z(ref - prev_mean, floor=1.0)
    score = np.clip(level, 0, None) + 0.5 * np.clip(jump, 0, None)
    score[silent] = 0                              # near-silence never counts
    return rolling(score, 3, np.mean)


# ----------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------
def chat_score(src: Source, n: int, args) -> np.ndarray | None:
    exe = shutil.which("TwitchDownloaderCLI")
    if not exe:
        log("--chat: TwitchDownloaderCLI not found on PATH, skipping chat")
        return None
    if not src.vod_id:
        log("--chat: no VOD id (use --vod-id for local files), skipping chat")
        return None
    out = CACHE_DIR / f"{src.vod_id}_chat.json"
    if not out.exists():
        log("downloading chat replay...")
        run([exe, "chatdownload", "--id", src.vod_id, "-o", str(out)])
    comments = json.loads(out.read_text(encoding="utf-8")).get("comments", [])
    if len(comments) < args.chat_min_messages:
        log(f"--chat: only {len(comments)} messages, not enough to be useful - skipping")
        return None
    counts = np.zeros(n)
    for c in comments:
        # chat reacts late; offsets are VOD time, convert to file time
        t = int(src.from_vod_time(c.get("content_offset_seconds", 0)) - args.chat_delay)
        if 0 <= t < n:
            body = ((c.get("message") or {}).get("body") or "").lower()
            counts[t] += 1 + sum(w in body for w in HYPE_WORDS)
    log(f"chat: {len(comments)} messages used")
    return np.clip(robust_z(rolling(counts, 10, np.sum), floor=1.0), 0, None)


# ----------------------------------------------------------------------------
# Picking
# ----------------------------------------------------------------------------
def pick(total: np.ndarray, parts: dict[str, np.ndarray], duration: float,
         forced: list[tuple[float, str]], args) -> list[Highlight]:
    """Forced moments (markers / --at) are always kept; then greedily take the
    highest-scoring seconds above --threshold, at least --min-gap apart."""
    L = min(args.clip_length, duration)
    latest_start = max(0.0, duration - L)
    picks: list[Highlight] = []

    for t, label in forced:
        if not 0 <= t <= duration:
            log(f"timestamp {hms(t)} is outside the video, ignored")
            continue
        start = min(max(0.0, t - MARKER_POSITION * L), latest_start)
        i = min(int(t), len(total) - 1)
        picks.append(Highlight(t, start, L, float(total[i]), [label]))

    auto = 0
    for t in np.argsort(total)[::-1]:
        if auto >= args.max_clips or total[t] < args.threshold:
            break
        if t < args.skip_start or t > duration - args.skip_end:
            continue
        if any(abs(t - h.peak) < args.min_gap for h in picks):
            continue
        reasons = [f"{name} (+{arr[t]:.1f})" for name, arr in parts.items() if arr[t] >= 1.0]
        start = min(max(0.0, t - args.peak_position * L), latest_start)
        picks.append(Highlight(float(t), start, L, float(total[t]), reasons or ["combined"]))
        auto += 1

    if auto == 0:
        top = np.sort(total)[::-1][:5]
        log(f"no auto highlight above --threshold {args.threshold} "
            f"(top scores: {', '.join(f'{s:.1f}' for s in top)}) - try a lower threshold")
    return sorted(picks, key=lambda h: h.peak)
