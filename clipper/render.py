"""Cutting clips with ffmpeg (vertical layout) and optional Whisper captions."""
from __future__ import annotations

from pathlib import Path

from .config import CAPTION_STYLE
from .models import Highlight
from .util import ClipperError, log, run


# ----------------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------------
def video_filter(fmt: str, crop: str, srt: str | None) -> str:
    if fmt == "horizontal":
        chain = "[0:v]null[v0]"
    elif crop == "center":
        chain = "[0:v]crop=ih*9/16:ih,scale=1080:1920,setsar=1[v0]"
    else:  # gameplay full-width in the middle, blurred copy filling the background
        chain = ("[0:v]split=2[bg][fg];"
                 "[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                 "gblur=sigma=10,scale=1080:1920[bg2];"
                 "[fg]scale=1080:-2[fg2];"
                 "[bg2][fg2]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    if srt:
        return chain + f";[v0]subtitles={srt}:force_style='{CAPTION_STYLE}'[v]"
    return chain + ";[v0]null[v]"


def cut(src: Path, h: Highlight, out: Path, args, srt: str | None) -> None:
    """Encode one clip. Runs in the output folder so the .srt path needs no escaping."""
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{h.start:.2f}", "-i", str(src),
         "-t", f"{h.length:.2f}", "-filter_complex", video_filter(args.format, args.crop, srt),
         "-map", "[v]", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", out.name], cwd=out.parent)


# ----------------------------------------------------------------------------
# Captions
# ----------------------------------------------------------------------------
def load_whisper(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ClipperError("--captions needs faster-whisper:  pip install faster-whisper") from None
    log(f"loading Whisper model '{model_name}'...")
    return WhisperModel(model_name, device="auto", compute_type="default")


def transcribe(model, src: Path, h: Highlight, lang: str | None, tmp: Path):
    """Word-level transcript of one clip's audio (times relative to the clip)."""
    wav = tmp / "seg.wav"
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{h.start:.2f}", "-i", str(src),
         "-t", f"{h.length:.2f}", "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    segments, _ = model.transcribe(str(wav), language=lang, word_timestamps=True, vad_filter=True)
    return [w for s in segments for w in (s.words or [])]


def srt_time(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def write_srt(words, path: Path, per_line: int = 3) -> str:
    """Write words as short subtitle lines; return the plain transcript."""
    blocks, text = [], []
    for i in range(0, len(words), per_line):
        grp = words[i:i + per_line]
        line = " ".join(w.word.strip() for w in grp)
        text.append(line)
        blocks.append(f"{len(blocks) + 1}\n{srt_time(grp[0].start)} --> {srt_time(grp[-1].end)}\n{line}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return " ".join(text)
