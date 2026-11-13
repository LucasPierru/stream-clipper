"""process(): one VOD in, a folder of clips out. Used by OBS-triggered and manual runs."""
from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path

import numpy as np

from . import twitch
from .config import CACHE_DIR
from .detect import audio_score, chat_score, loudness_per_second, pick
from .models import Highlight, Source
from .output import clip_name, make_output_dir, write_report
from .render import cut, load_whisper, transcribe, write_srt
from .source import find_stream_vod, link_recording, probe_duration, resolve_source
from .state import ProcessedStreams
from .util import hms, log, parse_ts


def process(args) -> None:
    if args.stream_ended and not args.source:  # OBS: stream just ended, no recording
        args = ended_stream_args(args)
        if args is None:
            return
    src = resolve_source(args)
    if args.recording_start and not src.downloaded:  # local OBS recording of a stream
        stream_id = link_recording(src, args.channel, args.recording_start, args.stream_start)
        if stream_id and not args.dry_run:
            # mark it before the slow part, so a second run won't clip the stream again
            ProcessedStreams().add(stream_id)
    duration = probe_duration(src.path)
    log(f"source: {src.title}  ({hms(duration)}, stream date {src.date})")

    highlights = find_highlights(src, duration, args)
    if not highlights:
        log("nothing to cut")
        return
    for i, h in enumerate(highlights, 1):
        log(f"  #{i:02d}  {hms(h.peak)}  score {h.score:5.1f}  {', '.join(h.reasons)}")
    if args.dry_run:
        return

    out_dir = make_output_dir(args.out, src.date, src.key)
    log(f"writing {len(highlights)} clips to {out_dir}")
    render_clips(src, highlights, out_dir, args)
    write_report(out_dir, src, highlights)

    if args.delete_vod and src.downloaded:
        src.path.unlink(missing_ok=True)
        log("deleted downloaded VOD")
    log(f"done -> {out_dir}  (see highlights.md)")


def ended_stream_args(args):
    """Find the VOD of the stream that just ended, claim it in processed_streams.json
    (so it is never clipped twice), wait for Twitch to finalise it, and return
    args pointing at its URL. None if it was already clipped."""
    vod = find_stream_vod(args.channel, args.stream_start)
    log(f"stream ended - its VOD is {vod['url']}")
    stream_id = vod.get("stream_id")
    done = ProcessedStreams()
    if stream_id and stream_id in done:
        log("this stream was already clipped - nothing to do")
        return None
    if stream_id and not args.dry_run:
        done.add(stream_id)
    if args.vod_delay:
        log(f"waiting {args.vod_delay}s for Twitch to finalise the VOD...")
        time.sleep(args.vod_delay)
    run_args = copy.copy(args)
    run_args.source, run_args.channel = vod["url"], None
    return run_args


def find_highlights(src: Source, duration: float, args) -> list[Highlight]:
    db = loudness_per_second(src.path, CACHE_DIR / f"{src.key}_loudness.npy")
    parts: dict[str, np.ndarray] = {"loud moment": args.audio_weight * audio_score(db)}
    if args.chat:
        c = chat_score(src, len(db), args)
        if c is not None:
            parts["chat burst"] = args.chat_weight * c
    total = sum(parts.values())

    forced = [(parse_ts(t), f"manual {t}") for t in args.at]
    if args.markers:
        forced += [(src.from_vod_time(t), f"marker{': ' + d if d else ''}")
                   for t, d in twitch.markers(src.vod_id)]
    return pick(total, parts, duration, forced, args)


def render_clips(src: Source, highlights: list[Highlight], out_dir: Path, args) -> None:
    model = load_whisper(args.whisper_model) if args.captions else None
    with tempfile.TemporaryDirectory() as tmp:
        for i, h in enumerate(highlights, 1):
            name = clip_name(i, h)
            out = out_dir / f"{name}.mp4"
            srt = None
            if model:
                words = transcribe(model, src.path, h, args.lang, Path(tmp))
                if words:
                    h.caption = write_srt(words, out_dir / f"{name}.srt")
                    srt = f"{name}.srt"
            log(f"cutting {out.name}")
            cut(src.path, h, out, args, srt)
            h.file = out.name
