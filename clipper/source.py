"""Turn the user's input (local file, VOD URL, or channel name) into a video on disk."""
from __future__ import annotations

import datetime as dt
import os
import struct
from pathlib import Path

from .config import CACHE_DIR, VIDEO_EXTS
from .models import Source
from .util import ClipperError, log, run


def vod_url(vod_id: str | None) -> str | None:
    return f"https://www.twitch.tv/videos/{vod_id}" if vod_id else None


def resolve_source(args) -> Source:
    if args.source and Path(args.source).expanduser().is_file():
        src = local_source(Path(args.source), args.vod_id, args.date)
        if args.recording_start and not args.date:
            src.date = local_time(args.recording_start).date()  # not mtime = end of recording
        return src

    url = args.source
    if args.channel and not url:
        url = latest_archive_url(args.channel)
    if not url:
        raise ClipperError("give a VOD URL, a local video file, or --channel NAME")
    if "://" not in url:
        raise ClipperError(f"file not found: {url}")
    src = download(url, args.max_height)
    if args.date:
        src.date = args.date
    return src


def local_source(path: Path, vod_id: str | None, date: dt.date | None) -> Source:
    p = path.expanduser().resolve()
    return Source(p, date or dt.date.fromtimestamp(p.stat().st_mtime), p.stem,
                  vod_url(vod_id), vod_id)


def _yt_dlp():
    try:
        import yt_dlp
    except ImportError:
        raise ClipperError("yt-dlp is not installed:  pip install yt-dlp") from None
    return yt_dlp


def latest_archive_url(channel: str) -> str:
    yt_dlp = _yt_dlp()
    listing = f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"
    log(f"looking up the latest past broadcast of {channel}")
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "playlist_items": "1"}) as ydl:
        info = ydl.extract_info(listing, download=False)
    entries = [e for e in (info.get("entries") or []) if e]
    if not entries:
        raise ClipperError(f"no past broadcasts found for '{channel}' (is 'Store past broadcasts' on?)")
    e = entries[0]
    return e.get("url") or f"https://www.twitch.tv/videos/{str(e['id']).lstrip('v')}"


def download(url: str, max_height: int) -> Source:
    yt_dlp = _yt_dlp()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"best[height<={max_height}]/best",
        "outtmpl": str(CACHE_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
    }
    log(f"downloading {url}  (to {CACHE_DIR})")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        raise ClipperError(f"download failed: {e}") from None
    if not path.exists():
        found = [p for p in CACHE_DIR.glob(f"{info['id']}.*") if p.suffix in VIDEO_EXTS]
        if not found:
            raise ClipperError("download finished but the video file could not be found")
        path = found[0]
    if is_fragmented_mp4(path):
        defragment(path)

    if info.get("timestamp"):  # stream start time -> a stream past midnight keeps its start date
        date = dt.datetime.fromtimestamp(info["timestamp"]).date()
    elif info.get("upload_date"):
        date = dt.datetime.strptime(info["upload_date"], "%Y%m%d").date()
    else:
        date = dt.date.today()
    vod = str(info.get("id", "")).lstrip("v") or None
    return Source(path.resolve(), date, info.get("title") or path.stem,
                  vod_url(vod) or url, vod, downloaded=True)


def is_fragmented_mp4(path: Path) -> bool:
    """True for a fragmented MP4 (many moof/mdat chunks, no index), which Twitch VODs
    now download as. ffmpeg has to walk every chunk header each time it opens one, so
    a seek in a multi-hour VOD takes 10+ s instead of a fraction of a second."""
    try:
        with open(path, "rb") as f:
            for _ in range(64):  # the first top-level boxes are enough to tell
                head = f.read(8)
                if len(head) < 8:
                    return False
                size, kind = struct.unpack(">I4s", head)
                if kind == b"moof":
                    return True
                if kind == b"mdat":     # a plain MP4 reaches its media data before any fragment
                    return False
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0] - 8
                elif size == 0:
                    return False
                f.seek(size - 8, os.SEEK_CUR)
    except (OSError, struct.error):
        return False
    return False


def defragment(path: Path) -> None:
    """Rewrite a fragmented MP4 as a normal one with an index at the front (stream copy,
    no re-encoding), so frame grabs and clip cuts can seek instantly."""
    tmp = path.with_name(path.stem + ".defrag" + path.suffix)
    log(f"rewriting {path.name} with a seek index (one-time, no re-encoding)...")
    try:
        run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c", "copy",
             "-movflags", "+faststart", str(tmp)])
        os.replace(tmp, path)
    except (ClipperError, OSError) as e:
        tmp.unlink(missing_ok=True)
        log(f"could not rewrite the VOD, continuing with the slow original ({str(e).splitlines()[0]})")


def local_time(t: dt.datetime) -> dt.datetime:
    """Aware datetime in the PC's timezone (naive input is taken as local time)."""
    return t.astimezone()


# ----------------------------------------------------------------------------
# Finding the Twitch VOD of a stream that just ended (OBS-triggered runs)
# ----------------------------------------------------------------------------
VOD_MATCH_WINDOW = dt.timedelta(minutes=10)


def _created(v: dict) -> dt.datetime:
    return dt.datetime.fromisoformat(v["created_at"].replace("Z", "+00:00"))


def match_archive(channel: str, stream_start: dt.datetime | None) -> dict | None:
    """The archive VOD created closest to stream_start (within 10 min), or the
    newest one if the start time is unknown. A VOD is listed as soon as the
    stream starts, so this works right after the stream ends."""
    from . import twitch  # local import: only needed for OBS runs

    vids = twitch.recent_archives(twitch.user_id(channel), first=5)
    if stream_start is None:
        return vids[0] if vids else None
    ref = local_time(stream_start)
    near = [v for v in vids if abs(_created(v) - ref) < VOD_MATCH_WINDOW]
    return min(near, key=lambda v: abs(_created(v) - ref)) if near else None


def find_stream_vod(channel: str | None, stream_start: dt.datetime | None) -> dict:
    """VOD of the stream that just ended (no local recording). Raises if not found."""
    if not channel:
        raise ClipperError("set TWITCH_CHANNEL in .env so the VOD of the stream can be found")
    try:
        vod = match_archive(channel, stream_start)
    except ClipperError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ClipperError(f"could not look up the Twitch VOD ({e})") from None
    if not vod:
        raise ClipperError("no Twitch VOD for this stream - turn on 'Store past broadcasts'")
    return vod


def link_recording(src: Source, channel: str | None, recording_start: dt.datetime,
                   stream_start: dt.datetime | None) -> str | None:
    """Local recording: find the Twitch VOD of the same stream, and set
    src.vod_id / src.url / src.vod_offset so links, --markers and --chat line up.

    Returns the stream id (for processed_streams.json), or None if no VOD was
    found. Never raises: without a match the clips are still made, just without
    Twitch links.
    """
    rec = local_time(recording_start)
    if not channel:
        log("no TWITCH_CHANNEL set - clipping without Twitch links")
        return None
    try:
        vod = match_archive(channel, stream_start or rec)
    except Exception as e:  # noqa: BLE001
        log(f"could not look up the Twitch VOD ({e}) - clipping without Twitch links")
        return None
    if not vod:
        log("no Twitch VOD matches this recording - clipping without Twitch links")
        return None

    # VOD time 0 = when the stream started. OBS's own stream-start time is the most
    # precise; Twitch's created_at is the fallback.
    vod_zero = local_time(stream_start) if stream_start else _created(vod)
    src.vod_id = str(vod["id"])
    src.url = vod_url(src.vod_id)
    src.vod_offset = (vod_zero - rec).total_seconds()
    log(f"matched Twitch VOD {src.vod_id} (VOD starts {src.vod_offset:+.0f}s into the recording)")
    return vod.get("stream_id") or None


def probe_duration(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())
