"""Where clips go (date folders) and the highlights.md / highlights.json report."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from .models import Highlight, Source
from .util import hms, twitch_t


def make_output_dir(root: str, date: dt.date, key: str) -> Path:
    """<root>/<date>/ ; a second stream the same day goes to <date>_2/, etc.
    Re-running the same VOD reuses its folder and replaces its old clips."""
    base = Path(root).expanduser() / date.isoformat()
    cand, i = base, 2
    while True:
        tag = cand / ".source"
        if not cand.exists() or not any(cand.iterdir()):
            cand.mkdir(parents=True, exist_ok=True)
            tag.write_text(key, encoding="utf-8")
            return cand
        if tag.exists() and tag.read_text(encoding="utf-8").strip() == key:
            for old in cand.glob("[0-9][0-9]_*"):
                if old.suffix in (".mp4", ".srt"):
                    old.unlink()
            return cand
        cand, i = Path(f"{base}_{i}"), i + 1


def clip_name(index: int, h: Highlight) -> str:
    """01_1h23m45s"""
    return f"{index:02d}_{twitch_t(h.peak)}"


def write_report(out_dir: Path, src: Source, highlights: list[Highlight]) -> None:
    lines = [f"# {src.title}", "", f"Stream date: {src.date}  ", f"Source: {src.url or src.path}  "]
    if src.url and src.downloaded is False:
        lines.append(f"Recording: {src.path}  ")
    if src.vod_offset:
        lines.append(f"The VOD starts {src.vod_offset:+.0f}s into the recording (links are corrected for it).  ")
    lines += ["", "| # | File | Time | Score | Why | Said |", "|---|---|---|---|---|---|"]
    for i, h in enumerate(highlights, 1):
        when = hms(h.peak)  # file time, same as the clip name and --at
        if src.url and "twitch.tv/videos" in src.url:
            when = f"[{when}]({src.url}?t={twitch_t(max(0.0, src.to_vod_time(h.start)))})"
        said = h.caption[:80].replace("|", "/")
        lines.append(f"| {i} | {h.file} | {when} | {h.score:.1f} | {', '.join(h.reasons)} | {said} |")
    (out_dir / "highlights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "highlights.json").write_text(json.dumps(
        {"title": src.title, "date": src.date.isoformat(), "source": src.url or str(src.path),
         "file": str(src.path), "vod_offset": src.vod_offset,
         "clips": [asdict(h) for h in highlights]}, indent=2, ensure_ascii=False), encoding="utf-8")
