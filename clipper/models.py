from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Source:
    """A video on disk, plus what we know about the stream it came from."""
    path: Path
    date: dt.date
    title: str
    url: str | None = None
    vod_id: str | None = None
    downloaded: bool = False  # True if we downloaded it (so --delete-vod may remove it)
    # Seconds into this file where the Twitch VOD starts (VOD time = file time - vod_offset).
    # 0 for downloaded VODs; set for local OBS recordings linked to their VOD.
    vod_offset: float = 0.0

    def to_vod_time(self, t: float) -> float:
        return t - self.vod_offset

    def from_vod_time(self, t: float) -> float:
        return t + self.vod_offset

    @property
    def key(self) -> str:
        """Stable id for caches and the output folder's .source tag."""
        return self.vod_id or f"{self.path.stem}_{self.path.stat().st_size}"


@dataclass
class Highlight:
    peak: float           # the moment itself, in VOD seconds
    start: float          # clip start, in VOD seconds
    length: float
    score: float
    reasons: list[str] = field(default_factory=list)
    file: str = ""        # filled in once cut
    caption: str = ""     # transcript, if --captions
