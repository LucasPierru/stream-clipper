"""Cutting clips with ffmpeg (layouts: webcam split, fill, blur) and optional Whisper captions."""
from __future__ import annotations

from pathlib import Path

from .config import CAPTION_STYLE
from .facecam import Box
from .models import Highlight
from .util import ClipperError, log, run


# ----------------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------------
OUT_W, OUT_H = 1080, 1920
TOP_MIN, TOP_MAX = 0.25, 0.42   # webcam panel height, as a share of the clip height


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def split_geometry(src_w: int, src_h: int, cam: Box) -> tuple[Box, int, Box, int]:
    """Webcam crop + panel height on top, gameplay crop + panel height below.
    Both crops match their panel's aspect ratio, so nothing is stretched."""
    top_h = _even(min(max(OUT_W * cam.h / cam.w, TOP_MIN * OUT_H), TOP_MAX * OUT_H))
    bot_h = OUT_H - top_h
    # webcam: trim the overlay to the panel's aspect ratio, keeping it centred
    want = OUT_W / top_h
    cw, ch = (cam.h * want, cam.h) if cam.w / cam.h > want else (cam.w, cam.w / want)
    cam_crop = Box(int(cam.x + (cam.w - cw) / 2), int(cam.y + (cam.h - ch) / 2), _even(cw), _even(ch))
    # gameplay: full-height centre crop, nudged off the webcam if they overlap
    gw = min(src_w, src_h * OUT_W / bot_h)
    gh = gw * bot_h / OUT_W
    gx = (src_w - gw) / 2
    if cam.x < gx + gw and gx < cam.x + cam.w:
        if cam.x + cam.w / 2 < src_w / 2:
            gx = min(max(gx, cam.x + cam.w), src_w - gw)
        else:
            gx = max(min(gx, cam.x - gw), 0)
    game = Box(int(gx), int((src_h - gh) / 2), _even(gw), _even(gh))
    return cam_crop, top_h, game, bot_h


def video_filter(layout: str, src_w: int = 1920, src_h: int = 1080, cam: Box | None = None,
                 srt: str | None = None) -> str:
    """ffmpeg filter graph for one clip.

    split      webcam on top, gameplay crop below (needs `cam`)
    fill       gameplay cropped to fill the whole 9:16 frame
    blur       full 16:9 frame over a blurred, zoomed copy of itself
    horizontal unchanged 16:9
    """
    if layout == "horizontal":
        chain = "[0:v]null[v0]"
    elif layout == "split" and cam:
        c, top_h, g, bot_h = split_geometry(src_w, src_h, cam)
        chain = (f"[0:v]split=2[cam][game];"
                 f"[cam]crop={c.w}:{c.h}:{c.x}:{c.y},scale={OUT_W}:{top_h},setsar=1[top];"
                 f"[game]crop={g.w}:{g.h}:{g.x}:{g.y},scale={OUT_W}:{bot_h},setsar=1[bot];"
                 f"[top][bot]vstack=inputs=2[v0]")
    elif layout == "blur":
        chain = ("[0:v]split=2[bg][fg];"
                 "[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                 f"gblur=sigma=10,scale={OUT_W}:{OUT_H}[bg2];"
                 f"[fg]scale={OUT_W}:-2[fg2];"
                 "[bg2][fg2]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    else:  # fill
        chain = f"[0:v]crop='min(iw,ih*9/16)':ih,scale={OUT_W}:{OUT_H},setsar=1[v0]"
    if srt:
        return chain + f";[v0]subtitles={srt}:force_style='{CAPTION_STYLE}'[v]"
    return chain + ";[v0]null[v]"


def cut(src: Path, h: Highlight, out: Path, args, filter_graph: str) -> None:
    """Encode one clip. Runs in the output folder so the .srt path needs no escaping."""
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{h.start:.2f}", "-i", str(src),
         "-t", f"{h.length:.2f}", "-filter_complex", filter_graph,
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
