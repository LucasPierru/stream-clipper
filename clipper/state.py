"""processed_streams.json: Twitch stream ids that were already clipped.

Written by OBS-triggered runs as soon as they find the stream's VOD, so the same
stream is never clipped twice (e.g. pressing "Clip latest VOD now" afterwards).
Re-read from disk on every check, since each run is a separate process.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import PROCESSED_FILE


class ProcessedStreams:
    def __init__(self, path: Path = PROCESSED_FILE):
        self.path = path

    def _load(self) -> set[str]:
        try:
            return set(json.loads(self.path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return set()

    def __contains__(self, stream_id: str) -> bool:
        return stream_id in self._load()

    def add(self, stream_id: str) -> None:
        ids = self._load() | {stream_id}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(ids)), encoding="utf-8")
