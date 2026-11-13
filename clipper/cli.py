"""Command line: argument parsing and the entry point."""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys

from . import config as C
from .facecam import Box
from .pipeline import process
from .twitch import twitch_login
from .util import ClipperError, log

USAGE = """Turn a Twitch VOD into ready-to-post vertical highlight clips.

Examples:
  python stream_clipper.py https://www.twitch.tv/videos/2299999999
  python stream_clipper.py --channel mychannel            # latest past broadcast
  python stream_clipper.py my_vod.mp4 --dry-run            # only list the moments
  python stream_clipper.py --channel mychannel --captions --at 1:02:30
  python stream_clipper.py --twitch-login                  # one-time, only for --markers
  (OBS starts runs itself through obs/stream_clipper_obs.py - see docs/auto-clip-flow.md)
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="stream_clipper", description=USAGE,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", help="Twitch VOD URL or local video file")
    p.add_argument("--channel", default=C.env("TWITCH_CHANNEL"),
                   help="use the latest past broadcast of this channel (default: TWITCH_CHANNEL)")
    p.add_argument("--out", default=C.DEFAULT_OUT, help=f"clips root folder (default: {C.DEFAULT_OUT})")
    p.add_argument("--date", type=dt.date.fromisoformat, help="override folder date (YYYY-MM-DD)")
    p.add_argument("--vod-id", help="Twitch VOD id for a local file (enables --markers/--chat/links)")

    p.add_argument("--twitch-login", action="store_true",
                   help="one-time login so --markers can read your stream markers")

    r = p.add_argument_group("automatic mode (set by the OBS script obs/stream_clipper_obs.py)")
    r.add_argument("--stream-ended", action="store_true",
                   help="the stream just ended: find its VOD (Helix), wait --vod-delay, clip it")
    r.add_argument("--stream-start", type=dt.datetime.fromisoformat, metavar="ISO",
                   help="when the stream started; picks the right VOD / lines it up with a recording")
    r.add_argument("--recording-start", type=dt.datetime.fromisoformat, metavar="ISO",
                   help="when the local recording started; dates the folder and finds the Twitch VOD")
    r.add_argument("--vod-delay", type=int, default=120,
                   help="seconds to wait after the stream ends for Twitch to finalise the VOD")

    g = p.add_argument_group("highlight detection")
    g.add_argument("--max-clips", type=int, default=C.MAX_CLIPS, help="max auto-detected clips")
    g.add_argument("--clip-length", type=float, default=C.CLIP_LENGTH, help="seconds per clip")
    g.add_argument("--peak-position", type=float, default=C.PEAK_POSITION,
                   help="where the moment sits in the clip, 0-1")
    g.add_argument("--min-gap", type=float, default=C.MIN_GAP,
                   help="min seconds between clips (default: the clip length, so clips never overlap)")
    g.add_argument("--threshold", type=float, default=C.THRESHOLD, help="min score for auto clips")
    g.add_argument("--skip-start", type=float, default=C.SKIP_START)
    g.add_argument("--skip-end", type=float, default=C.SKIP_END)
    g.add_argument("--at", action="append", default=[], metavar="H:MM:SS",
                   help="force a clip at this VOD time (repeatable)")
    g.add_argument("--markers", action="store_true", help="use your Twitch stream markers")
    g.add_argument("--chat", action="store_true", help="use chat activity (off by default)")
    g.add_argument("--chat-delay", type=float, default=6, help="seconds chat lags behind the moment")
    g.add_argument("--chat-min-messages", type=int, default=50)
    g.add_argument("--chat-weight", type=float, default=1.0)
    g.add_argument("--audio-weight", type=float, default=1.0)

    o = p.add_argument_group("output")
    o.add_argument("--format", choices=["vertical", "horizontal"], default="vertical")
    o.add_argument("--layout", choices=["auto", "split", "fill", "blur"], default="auto",
                   help="vertical layout: auto (webcam on top if found, else fill), split "
                        "(always webcam on top), fill (gameplay fills the frame), blur "
                        "(whole 16:9 frame over a blurred copy)")
    o.add_argument("--cam", type=parse_box, metavar="X,Y,W,H",
                   help="webcam position in source-video pixels, instead of detecting it")
    o.add_argument("--captions", action="store_true", help="burn in captions (needs faster-whisper)")
    o.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large-v3")
    o.add_argument("--lang", default=None, help="caption language, e.g. en or fr (default: auto)")
    o.add_argument("--crf", type=int, default=20, help="video quality, lower = better/bigger")
    o.add_argument("--max-height", type=int, default=1080, help="max VOD resolution to download")
    o.add_argument("--dry-run", action="store_true", help="only list highlights, cut nothing")
    o.add_argument("--keep-vod", action="store_true",
                   help="keep the downloaded VOD in the cache (handy to re-run while tuning)")
    return p.parse_args(argv)


def parse_box(text: str) -> Box:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X,Y,W,H, e.g. 0,80,480,270") from None
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise argparse.ArgumentTypeError("X,Y must be >= 0 and W,H > 0")
    return Box(x, y, w, h)


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise ClipperError(f"{tool} not found on PATH - install ffmpeg first")


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        if args.twitch_login:
            twitch_login()
        else:
            check_ffmpeg()
            process(args)
    except ClipperError as e:
        log(f"error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log("stopped")
