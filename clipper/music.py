"""--remove-music: take music (in-game radio, songs) out of the audio, keeping the voice.

Twitch only mutes part of the copyrighted music in a VOD, and YouTube's Content ID
claims what's left. Nothing here can tell copyrighted music from other music, so all
music goes:

  1. Detection: an AudioSet classifier (AST, via transformers) gives a "Music"
     probability for overlapping 10 s windows -> one value per second -> stretches
     of music (hysteresis, short gaps merged, short blips dropped, a little padding).
  2. Separation, only on those stretches: Demucs splits the audio into vocals and the
     rest. The vocals still hold the song's singer, so they are filtered too:
  3. Voice filter: every half second of the vocals is kept only if it is *your* voice
     (a speaker-verification model, WavLM, compares it with a voiceprint learned from
     the parts of the stream without music). Everything else - lyrics, a singer, a
     choir, and game characters talking over the music - is muted. Keeping "clear
     speech" too (KEEP_GAME_SPEECH) let lyrics through: once Demucs strips the
     instruments, a singer's voice alone often looks like plain speech to AST. Where you
     talk over a song, the singer under your voice stays.

The result is a full-length AAC track in the cache that clips, captions, highlight
detection and VOD parts all use instead of the video's own audio. The models run on
an NVIDIA GPU when torch has CUDA (minutes for a long VOD); the CPU works but is slow.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .config import (CACHE_DIR, KEEP_GAME_SPEECH, MUSIC_GAP, MUSIC_MIN, MUSIC_OFF, MUSIC_ON,
                     MUSIC_PAD, SPEECH_KEEP, SPEECH_MUSIC_MAX, VOICE_QUIET_DB, VOICE_SIM)
from .util import ClipperError, hms, log

DETECT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"   # AudioSet tagger, has "Music" and "Speech"
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"              # speaker verification (x-vectors)
SEPARATE_MODEL = "htdemucs"
DETECT_SR = 16000       # what AST and WavLM expect
WINDOW = 10.0           # seconds per music-detection window (AST's native input length)
HOP = 3.0               # seconds between windows
BATCH = 16
SEP_SR = 44100          # Demucs' rate; also the rate of the cleaned track
PIECE = 300             # seconds of a music stretch processed at once (memory)
CONTEXT = 5             # extra seconds on each side of a piece, so its edges sound right
FADE = 0.5              # crossfade between the original audio and the filtered voice, seconds
CELL = 0.5              # the voice filter decides every this many seconds
VOICE_WIN = 1.5         # seconds of audio per speaker comparison
TAG_WIN, TAG_HOP = 3.0, 1.5   # AST windows on the separated vocals
ENROLL_WIN = 2.0        # seconds per voiceprint sample
ENROLL_MAX = 600        # voiceprint samples taken from the stream
ENROLL_MIN = 20         # fewer than this: no voiceprint, only clear speech is kept
INSTALL_HINT = ('--remove-music needs extra packages: pip install -e ".[music]" '
                "(for an NVIDIA GPU, install the CUDA build of torch first - see README)")
VERSION = 3             # of the cached music-free track; older ones are made again

Tagger = Callable[[list], np.ndarray]       # windows (16 kHz mono) -> (n, 2): music, speech
Embedder = Callable[[list], np.ndarray]     # windows (16 kHz mono) -> (n, d), unit length


# ----------------------------------------------------------------------------
# Audio I/O
# ----------------------------------------------------------------------------
def read_pcm(path: Path, rate: int, channels: int, chunk_s: float = 60) -> Iterator[np.ndarray]:
    """The audio as float32 chunks of shape (samples, channels), in order, from one
    ffmpeg process (never the whole VOD in memory)."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", str(channels),
                             "-ar", str(rate), "-f", "f32le", "-"], stdout=subprocess.PIPE)
    size = int(rate * chunk_s) * channels * 4
    finished = False
    try:
        while True:
            buf = proc.stdout.read(size)
            if not buf:
                finished = True
                break
            n = len(buf) // (4 * channels) * 4 * channels
            yield np.frombuffer(buf[:n], np.float32).reshape(-1, channels)
    finally:
        if not finished:       # the reader stopped early: stop ffmpeg before it complains
            proc.kill()
        proc.stdout.close()
        proc.wait()


def to_16k_mono(x: np.ndarray, rate: int = SEP_SR) -> np.ndarray:
    """(samples, channels) at `rate` -> mono at DETECT_SR (linear interpolation: plenty
    for classifying speech, and no extra dependency)."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = int(len(mono) * DETECT_SR / rate)
    return np.interp(np.arange(n) * (rate / DETECT_SR), np.arange(len(mono)), mono).astype(np.float32)


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
# Models
# ----------------------------------------------------------------------------
def load_detector(dev: str) -> Tagger:
    """AST: "Music" and "Speech" probabilities for each window."""
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
    except ImportError:
        raise ClipperError(INSTALL_HINT) from None
    log("loading the music detector...")
    fe = ASTFeatureExtractor.from_pretrained(DETECT_MODEL)
    model = ASTForAudioClassification.from_pretrained(DETECT_MODEL).to(dev).eval()
    labels = [model.config.label2id["Music"], model.config.label2id["Speech"]]

    def tag(windows: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(windows), BATCH):
            inputs = fe(windows[i:i + BATCH], sampling_rate=DETECT_SR, return_tensors="pt")
            with torch.no_grad():
                logits = model(**{k: v.to(dev) for k, v in inputs.items()}).logits
            out.append(torch.sigmoid(logits[:, labels]).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)
    return tag


def load_speaker(dev: str) -> Embedder:
    """WavLM speaker-verification embeddings (unit length: the dot product is the cosine)."""
    try:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
    except ImportError:
        raise ClipperError(INSTALL_HINT) from None
    log("loading the voice recognizer...")
    fe = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
    model = WavLMForXVector.from_pretrained(SPEAKER_MODEL).to(dev).eval()

    def embed(windows: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(windows), 64):
            inputs = fe(windows[i:i + 64], sampling_rate=DETECT_SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                e = model(**{k: v.to(dev) for k, v in inputs.items()}).embeddings
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
    return embed


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
        wav = torch.from_numpy(np.ascontiguousarray(mix.T))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(model, ((wav - mean) / std)[None], device=dev,
                                  split=True, overlap=0.25, progress=False)[0]
        return (sources[vocals] * std + mean).T.cpu().numpy().astype(np.float32)
    return separate


# ----------------------------------------------------------------------------
# 1. Where is there music?
# ----------------------------------------------------------------------------
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


def music_probability(path: Path, duration: float, tag: Tagger, cache: Path | None = None) -> np.ndarray:
    """Probability of music for every second of the audio (cached)."""
    if cache and cache.exists():
        return np.load(cache)
    log("looking for music (one pass over the whole VOD)...")
    win, hop = int(WINDOW * DETECT_SR), int(HOP * DETECT_SR)
    starts, probs, batch = [], [], []
    buf, buf_start, next_s, next_log = np.zeros(0, np.float32), 0, 0, 1800

    def flush():
        probs.extend(tag(batch)[:, 0])
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
# 2. Your voice
# ----------------------------------------------------------------------------
def voiceprint(path: Path, spans: list[tuple[float, float]], db: np.ndarray, embed: Embedder,
               cache: Path | None = None) -> np.ndarray | None:
    """The streamer's voice as one embedding, learned from the loud parts of the stream
    without music (where the audio is mostly you). Game voices are in there too, so
    the centre is refined 5 times on the half of the samples closest to it - the
    streamer, who talks the most. None if the stream has too little of that."""
    if cache and cache.exists():
        v = np.load(cache)
        return v if v.size > 1 else None
    music = np.zeros(len(db) + 1, bool)
    for a, b in spans:
        music[max(0, int(a) - 3):int(b) + 3] = True
    w = int(ENROLL_WIN)
    loud = db >= np.median(db[db > -50]) if (db > -50).any() else np.zeros(len(db), bool)
    cand = [t for t in range(0, len(db) - w, w) if loud[t:t + w].all() and not music[t:t + w].any()]
    v = None
    if len(cand) >= ENROLL_MIN:
        picks = sorted(np.random.default_rng(0).choice(cand, min(ENROLL_MAX, len(cand)), replace=False).tolist())
        windows, k = [], 0
        buf, buf_start = np.zeros(0, np.float32), 0      # buf_start: sample index of buf[0]
        for chunk in read_pcm(path, DETECT_SR, 1):
            buf = np.concatenate([buf, chunk[:, 0]])
            while k < len(picks) and (picks[k] + w) * DETECT_SR <= buf_start + len(buf):
                a = picks[k] * DETECT_SR - buf_start
                windows.append(buf[a:a + w * DETECT_SR].copy())
                k += 1
            keep = w * DETECT_SR                          # a window may start in this tail
            buf_start += max(0, len(buf) - keep)
            buf = buf[-keep:]
            if k == len(picks):
                break
        if len(windows) >= ENROLL_MIN:
            e = embed(windows)
            c = e.mean(axis=0)
            for _ in range(5):
                sim = e @ (c / np.linalg.norm(c))
                c = e[sim >= np.quantile(sim, 0.5)].mean(axis=0)
            v = (c / np.linalg.norm(c)).astype(np.float32)
    if v is None:
        log("not enough of your voice outside the music to learn it - "
            + ("keeping only clear speech there" if KEEP_GAME_SPEECH else "the music stretches will be silent"))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, v if v is not None else np.zeros(1, np.float32))
    return v


# ----------------------------------------------------------------------------
# 3. Keep only your voice (and clear speech) in the separated vocals
# ----------------------------------------------------------------------------
def voice_gain(vocals: np.ndarray, embed: Embedder, tag: Tagger | None,
               voice: np.ndarray | None, keep_speech: bool = KEEP_GAME_SPEECH) -> tuple[np.ndarray, dict[str, float]]:
    """Per-sample gain (0-1) for separated vocals at SEP_SR, and the seconds kept as
    your voice / other speech and muted. Decided every CELL seconds: kept if it sounds
    like you (VOICE_SIM); with `keep_speech`, also if it sounds like speech with little
    music in it (game characters - but a singer often passes too). Everything else
    with sound in it is muted."""
    x = to_16k_mono(vocals)
    cell = int(CELL * DETECT_SR)
    ncell = max(1, int(np.ceil(len(x) / cell)))
    padded = np.pad(x, (0, ncell * cell - len(x)))
    level = 10 * np.log10((padded.reshape(ncell, cell) ** 2).mean(axis=1) + 1e-10)
    loud = level > VOICE_QUIET_DB
    centres = np.arange(ncell) * cell + cell // 2

    yours = np.zeros(ncell, bool)
    if voice is not None and loud.any():
        vw = int(VOICE_WIN * DETECT_SR)
        idx = np.flatnonzero(loud)
        starts = np.clip(centres[idx] - vw // 2, 0, max(0, len(x) - vw))
        yours[idx] = embed([x[a:a + vw] for a in starts]) @ voice >= VOICE_SIM

    speech = np.zeros(ncell, bool)
    rest = loud & ~yours
    if keep_speech and tag is not None and rest.any():
        tw, th = int(TAG_WIN * DETECT_SR), int(TAG_HOP * DETECT_SR)
        starts = list(range(0, max(1, len(x) - tw + 1), th))
        if starts[-1] + tw < len(x):
            starts.append(len(x) - tw)
        starts = np.array(starts)
        covers = (centres[None, :] >= starts[:, None]) & (centres[None, :] < starts[:, None] + tw)
        need = covers[:, rest].any(axis=1)
        total, count = np.zeros((ncell, 2)), np.zeros(ncell)
        if need.any():
            probs = tag([x[a:a + tw] for a in starts[need]])
            total += covers[need].T.astype(float) @ probs
            count += covers[need].sum(axis=0)
        mean = total / np.maximum(count, 1)[:, None]
        speech = rest & (count > 0) & (mean[:, 1] >= SPEECH_KEEP) & (mean[:, 0] < SPEECH_MUSIC_MAX)

    keep = yours | speech
    keep = keep | np.r_[keep[1:], False] | np.r_[False, keep[:-1]]   # don't clip word edges
    per = int(CELL * SEP_SR)
    gain = np.repeat(keep.astype(np.float32), per)
    gain = np.pad(gain, (0, max(0, len(vocals) - len(gain))))[:len(vocals)]
    smooth = int(0.05 * SEP_SR)                                       # 50 ms ramps, no clicks
    c = np.r_[0, np.cumsum(np.pad(gain, (smooth // 2, smooth - smooth // 2)), dtype=np.float64)]
    gain = ((c[smooth:] - c[:-smooth]) / smooth)[:len(vocals)].astype(np.float32)
    stats = {"your voice": yours.sum() * CELL, "other speech": speech.sum() * CELL,
             "muted": (loud & ~(yours | speech)).sum() * CELL}
    return gain, stats


# ----------------------------------------------------------------------------
# Writing the music-free track
# ----------------------------------------------------------------------------
def blend_weight(a: int, b: int, r0: int, r1: int) -> np.ndarray:
    """Weight of the processed audio for samples a..b of the span r0..r1: 1, ramping
    from/to the original over FADE at both ends of the span, so there's no click."""
    fade = max(1, int(FADE * SEP_SR))
    x = np.arange(a, b)
    w = np.clip((x - r0) / fade, 0, 1) * np.clip((r1 - x) / fade, 0, 1)
    return w.astype(np.float32)[:, None]


def write_clean_track(path: Path, spans: list[tuple[float, float]], out: Path,
                      process: Callable[[np.ndarray], np.ndarray]) -> None:
    """Copy the audio to `out` (AAC), with every span replaced by `process(span audio)`
    (stereo, SEP_SR). One sequential pass; a span is handled in pieces of PIECE
    seconds, each with CONTEXT seconds of audio on both sides, so memory stays small
    however long the music plays."""
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SEP_SR), "-ac", "2",
                            "-i", "-", "-c:a", "aac", "-b:a", "192k", "-f", "mp4", str(tmp)],
                           stdin=subprocess.PIPE)
    ranges = [(int(a * SEP_SR), int(b * SEP_SR)) for a, b in spans]
    piece, ctx = PIECE * SEP_SR, CONTEXT * SEP_SR
    st = {"k": 0, "held": None, "h0": 0, "done": 0}   # span index, its audio from h0, written up to

    def emit(a: int, b: int, r0: int, r1: int) -> None:
        held, h0 = st["held"], st["h0"]
        if b <= a:
            return
        s0, s1 = max(r0, a - ctx, h0), min(h0 + len(held), b + ctx)
        seg = held[s0 - h0:s1 - h0]
        rep = process(seg)[a - s0:b - s0]
        mix = held[a - h0:b - h0]
        w = blend_weight(a, b, r0, r1)
        enc.stdin.write((mix * (1 - w) + rep * w).astype(np.float32).tobytes())
        st["done"] = b

    def finish(r0: int, r1: int) -> None:
        while r1 - st["done"] > piece:
            emit(st["done"], st["done"] + piece, r0, r1)
        emit(st["done"], r1, r0, r1)
        st["held"] = None
        st["k"] += 1

    try:
        pos = 0
        for chunk in read_pcm(path, SEP_SR, 2):
            i = 0
            while i < len(chunk):
                p, k = pos + i, st["k"]
                if k < len(ranges) and p >= ranges[k][0]:
                    r0, r1 = ranges[k]
                    if st["held"] is None:
                        log(f"  keeping only the voice in {hms(r0 / SEP_SR)}-{hms(r1 / SEP_SR)}")
                        st.update(held=np.zeros((0, 2), np.float32), h0=p, done=p)
                    take = min(len(chunk) - i, r1 - p)
                    st["held"] = np.concatenate([st["held"], chunk[i:i + take]])
                    i += take
                    end = st["h0"] + len(st["held"])
                    while end - st["done"] >= piece + ctx:      # enough audio after the piece
                        emit(st["done"], st["done"] + piece, r0, r1)
                        drop = st["done"] - ctx - st["h0"]
                        if drop > 0:
                            st["held"], st["h0"] = st["held"][drop:], st["h0"] + drop
                    if end >= r1:
                        finish(r0, r1)
                else:
                    nxt = ranges[k][0] if k < len(ranges) else p + len(chunk)
                    take = min(len(chunk) - i, nxt - p)
                    enc.stdin.write(chunk[i:i + take].tobytes())
                    i += take
            pos += len(chunk)
        if st["held"] is not None:     # the audio ended inside a span
            finish(ranges[st["k"]][0], st["h0"] + len(st["held"]))
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
    from .detect import loudness_per_second

    track, info = CACHE_DIR / f"{key}_nomusic.m4a", CACHE_DIR / f"{key}_music.json"
    if track.exists() and info.exists():
        meta = json.loads(info.read_text(encoding="utf-8"))
        if isinstance(meta, dict) and meta.get("version") == VERSION:
            spans = [tuple(s) for s in meta["spans"]]
            log(f"music-free audio already made ({len(spans)} music stretches)")
            return track, spans
    dev = device()
    tag = load_detector(dev)
    prob = music_probability(path, duration, tag, CACHE_DIR / f"{key}_musicprob.npy")
    spans = music_spans(prob)
    if not spans:
        log("no music found - keeping the original audio")
        return None, []
    log(f"music in {len(spans)} stretches ({hms(sum(b - a for a, b in spans))} in all) - "
        "keeping only the voices there")
    db = loudness_per_second(path, CACHE_DIR / f"{key}_loudness.npy")
    embed = load_speaker(dev)
    voice = voiceprint(path, spans, db, embed, CACHE_DIR / f"{key}_voice.npy")
    separate = load_separator(dev)
    totals = {"your voice": 0.0, "other speech": 0.0, "muted": 0.0}

    def process(mix: np.ndarray) -> np.ndarray:
        vocals = separate(mix)
        gain, stats = voice_gain(vocals, embed, tag, voice)
        for name, s in stats.items():
            totals[name] += s
        return vocals * gain[:, None]

    write_clean_track(path, spans, track, process)
    (CACHE_DIR / f"{key}_nomusic_loudness.npy").unlink(missing_ok=True)   # was of an older track
    log("voice filter (about): " + ", ".join(f"{name} {hms(s)}" for name, s in totals.items()))
    info.write_text(json.dumps({"version": VERSION, "spans": spans}), encoding="utf-8")
    return track, spans
