"""--remove-music: take music (in-game radio, songs) out of the audio, keeping the voice.

Twitch only mutes part of the copyrighted music in a VOD, and YouTube's Content ID
claims what's left. Nothing here can tell copyrighted music from other music, so all
music goes:

  1. Detection: an AudioSet classifier (AST, via transformers) gives a "Music"
     probability for overlapping 10 s windows -> one value per second -> stretches
     of music (hysteresis, short gaps merged, short blips dropped, a little padding).
  2. Separation, only on those stretches: Demucs splits the audio into vocals and the
     rest, and the stretch is replaced by the vocals (your voice; a song's singer can
     leak through), with short crossfades. Everything else is left untouched.

The result is a full-length AAC track in the cache that clips, captions, highlight
detection and VOD parts all use instead of the video's own audio. Both models run on
an NVIDIA GPU when torch has CUDA (minutes for a long VOD); the CPU works but is slow.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import (CACHE_DIR, MUSIC_GAP, MUSIC_MIN, MUSIC_OFF, MUSIC_ON, MUSIC_PAD)
from .util import ClipperError, hms, log

DETECT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"   # AudioSet tagger, has a "Music" class
SEPARATE_MODEL = "htdemucs"
DETECT_SR = 16000       # what AST expects
WINDOW = 10.0           # seconds per classifier window (AST's native input length)
HOP = 3.0               # seconds between windows
BATCH = 16
SEP_SR = 44100          # Demucs' rate; also the rate of the cleaned track
SEP_PIECE = 600         # seconds of audio handed to Demucs at once (memory)
FADE = 0.5              # crossfade between the original audio and the voice, seconds
INSTALL_HINT = ('--remove-music needs extra packages: pip install -e ".[music]" '
                "(for an NVIDIA GPU, install the CUDA build of torch first - see README)")


# ----------------------------------------------------------------------------
# Audio I/O
# ----------------------------------------------------------------------------
def read_pcm(path: Path, rate: int, channels: int, chunk_s: float = 60) -> Iterator[np.ndarray]:
    """The audio as float32 chunks of shape (samples, channels), in order, from one
    ffmpeg process (never the whole VOD in memory)."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", str(channels),
                             "-ar", str(rate), "-f", "f32le", "-"], stdout=subprocess.PIPE)
    size = int(rate * chunk_s) * channels * 4
    try:
        while True:
            buf = proc.stdout.read(size)
            if not buf:
                break
            n = len(buf) // (4 * channels) * 4 * channels
            yield np.frombuffer(buf[:n], np.float32).reshape(-1, channels)
    finally:
        proc.stdout.close()
        proc.wait()


def device() -> str:
    """"cuda" if torch can use an NVIDIA GPU, else "cpu"."""
    try:
        import torch
    except ImportError:
        raise ClipperError(INSTALL_HINT) from None
    if torch.cuda.is_available():
        log(f"music removal on the GPU ({torch.cuda.get_device_name(0)})")
        return "cuda"
    log("music removal on the CPU - this is slow (no CUDA build of torch / no NVIDIA GPU found)")
    return "cpu"


# ----------------------------------------------------------------------------
# 1. Where is there music?
# ----------------------------------------------------------------------------
def load_detector(dev: str):
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
    except ImportError:
        raise ClipperError(INSTALL_HINT) from None
    log("loading the music detector...")
    fe = ASTFeatureExtractor.from_pretrained(DETECT_MODEL)
    model = ASTForAudioClassification.from_pretrained(DETECT_MODEL).to(dev).eval()
    music = model.config.label2id["Music"]

    def predict(windows: list[np.ndarray]) -> np.ndarray:
        inputs = fe(windows, sampling_rate=DETECT_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(dev) for k, v in inputs.items()}).logits
        return torch.sigmoid(logits[:, music]).float().cpu().numpy()
    return predict


def window_starts(duration: float) -> np.ndarray:
    last = max(0.0, duration - WINDOW)
    starts = np.arange(0.0, last + 1e-6, HOP)
    if len(starts) == 0 or starts[-1] < last:
        starts = np.append(starts, last)
    return starts


def per_second(starts: np.ndarray, probs: np.ndarray, duration: float) -> np.ndarray:
    """Mean probability of the windows that cover each second."""
    n = int(np.ceil(duration))
    total, count = np.zeros(n), np.zeros(n)
    for s, p in zip(starts, probs):
        a, b = int(s), min(n, int(np.ceil(s + WINDOW)))
        total[a:b] += p
        count[a:b] += 1
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def music_probability(path: Path, duration: float, predict, cache: Path | None = None) -> np.ndarray:
    """Probability of music for every second of the audio (cached)."""
    if cache and cache.exists():
        return np.load(cache)
    log("looking for music (one pass over the whole VOD)...")
    win, hop = int(WINDOW * DETECT_SR), int(HOP * DETECT_SR)
    starts, probs, batch = [], [], []
    buf, buf_start, next_s, next_log = np.zeros(0, np.float32), 0, 0, 1800

    def flush():
        probs.extend(predict(batch))
        batch.clear()

    for chunk in read_pcm(path, DETECT_SR, 1):
        buf = np.concatenate([buf, chunk[:, 0]])
        while next_s + win <= buf_start + len(buf):
            a = next_s - buf_start
            batch.append(buf[a:a + win])
            starts.append(next_s / DETECT_SR)
            next_s += hop
            if len(batch) == BATCH:
                flush()
        drop = next_s - buf_start          # samples no window needs any more
        if drop > 0:
            buf, buf_start = buf[drop:], buf_start + drop
        if next_s / DETECT_SR >= next_log:
            log(f"  ... {hms(next_s / DETECT_SR)}")
            next_log += 1800
    total = (buf_start + len(buf)) / DETECT_SR
    if len(buf) and (not starts or starts[-1] + WINDOW < total):   # the tail
        batch.append(buf[-win:])
        starts.append(max(0.0, total - WINDOW))
    if batch:
        flush()
    prob = per_second(np.array(starts), np.array(probs), max(duration, total))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, prob)
    return prob


def music_spans(prob: np.ndarray, on: float = MUSIC_ON, off: float = MUSIC_OFF,
                min_len: float = MUSIC_MIN, gap: float = MUSIC_GAP,
                pad: float = MUSIC_PAD) -> list[tuple[float, float]]:
    """Stretches of music, in seconds. A stretch starts where the probability goes
    above `on` and lasts while it stays above `off`; stretches less than `gap` apart are
    merged (a song keeps playing while you talk over it), ones shorter than `min_len`
    are dropped, and each is padded by `pad` on both sides."""
    spans, start = [], None
    for t, p in enumerate(prob):
        if start is None and p >= on:
            start = t
            while start > 0 and prob[start - 1] >= off:    # include the rise
                start -= 1
        elif start is not None and p < off:
            spans.append([start, t])
            start = None
    if start is not None:
        spans.append([start, len(prob)])
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    end = float(len(prob))
    return [(max(0.0, a - pad), min(end, b + pad)) for a, b in merged if b - a >= min_len]


# ----------------------------------------------------------------------------
# 2. Keep only the voice in those stretches
# ----------------------------------------------------------------------------
def load_separator(dev: str):
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError:
        raise ClipperError(INSTALL_HINT) from None
    log("loading the voice separator (Demucs)...")
    model = get_model(SEPARATE_MODEL)
    model.eval()
    vocals = model.sources.index("vocals")

    def separate(mix: np.ndarray) -> np.ndarray:
        """(samples, 2) at SEP_SR -> the vocals, same shape."""
        out = []
        for a in range(0, len(mix), SEP_PIECE * SEP_SR):
            wav = torch.from_numpy(np.ascontiguousarray(mix[a:a + SEP_PIECE * SEP_SR].T))
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                sources = apply_model(model, ((wav - mean) / std)[None], device=dev,
                                      split=True, overlap=0.25, progress=False)[0]
            out.append((sources[vocals] * std + mean).T.cpu().numpy())
        return np.concatenate(out).astype(np.float32)
    return separate


def blend(mix: np.ndarray, voice: np.ndarray, fade: int) -> np.ndarray:
    """The voice in the middle, crossfading from/to the original over `fade` samples
    at both ends, so the cut-over isn't a click."""
    n = len(mix)
    w = np.ones(n, np.float32)
    f = min(fade, n // 2)
    if f:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        w[:f], w[n - f:] = ramp, ramp[::-1]
    w = w[:, None]
    return mix * (1 - w) + voice * w


def write_clean_track(path: Path, spans: list[tuple[float, float]], out: Path, separate) -> None:
    """Copy the audio to `out` (AAC), with every span replaced by its separated voice.
    One sequential pass; only one span is held in memory at a time."""
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SEP_SR), "-ac", "2",
                            "-i", "-", "-c:a", "aac", "-b:a", "192k", "-f", "mp4", str(tmp)],
                           stdin=subprocess.PIPE)
    ranges = [(int(a * SEP_SR), int(b * SEP_SR)) for a, b in spans]
    fade = int(FADE * SEP_SR)
    k, held, pos = 0, [], 0

    def finish_span():
        mix = np.concatenate(held)
        log(f"  separating the voice in {hms(ranges[k][0] / SEP_SR)}-{hms(ranges[k][0] / SEP_SR + len(mix) / SEP_SR)}")
        enc.stdin.write(blend(mix, separate(mix), fade).tobytes())
        held.clear()

    try:
        for chunk in read_pcm(path, SEP_SR, 2):
            i = 0
            while i < len(chunk):
                p = pos + i
                if k < len(ranges) and p >= ranges[k][0]:
                    take = min(len(chunk) - i, ranges[k][1] - p)
                    held.append(chunk[i:i + take])
                    i += take
                    if p + take >= ranges[k][1]:
                        finish_span()
                        k += 1
                else:
                    nxt = ranges[k][0] if k < len(ranges) else p + len(chunk)
                    take = min(len(chunk) - i, nxt - p)
                    enc.stdin.write(chunk[i:i + take].tobytes())
                    i += take
            pos += len(chunk)
        if held:                       # the audio ended inside a span
            finish_span()
        enc.stdin.close()
    except BaseException:
        enc.kill()                     # don't wait on an encoder whose input never ends
        enc.wait()
        tmp.unlink(missing_ok=True)
        raise
    if enc.wait() != 0:
        tmp.unlink(missing_ok=True)
        raise ClipperError("ffmpeg failed while writing the music-free audio")
    tmp.replace(out)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def remove_music(path: Path, key: str, duration: float) -> tuple[Path | None, list[tuple[float, float]]]:
    """The music-free audio track for this video (cached by `key`), and the stretches
    that were changed. (None, []) if there's no music."""
    track, info = CACHE_DIR / f"{key}_nomusic.m4a", CACHE_DIR / f"{key}_music.json"
    if track.exists() and info.exists():
        spans = [tuple(s) for s in json.loads(info.read_text(encoding="utf-8"))]
        log(f"music-free audio already made ({len(spans)} music stretches)")
        return track, spans
    dev = device()
    prob = music_probability(path, duration, load_detector(dev), CACHE_DIR / f"{key}_musicprob.npy")
    spans = music_spans(prob)
    total = sum(b - a for a, b in spans)
    if not spans:
        log("no music found - keeping the original audio")
        return None, []
    log(f"music in {len(spans)} stretches ({hms(total)} in all) - keeping only the voice there")
    write_clean_track(path, spans, track, load_separator(dev))
    info.write_text(json.dumps(spans), encoding="utf-8")
    return track, spans
