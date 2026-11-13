"""Finding the VOD after a stream, linking local recordings - with a fake Helix API."""
import datetime as dt

import pytest

from clipper import pipeline, source, twitch
from clipper.models import Source
from clipper.state import ProcessedStreams
from clipper.util import ClipperError

START = dt.datetime(2026, 11, 12, 20, 0, tzinfo=dt.timezone.utc)


def archive(vod_id, stream_id, created, url=None):
    return {"id": vod_id, "stream_id": stream_id, "url": url or f"https://www.twitch.tv/videos/{vod_id}",
            "created_at": created.isoformat().replace("+00:00", "Z")}


@pytest.fixture
def fake_twitch(monkeypatch):
    vids = [archive("555", "S5", START + dt.timedelta(seconds=4)),
            archive("444", "S4", START - dt.timedelta(days=1))]
    monkeypatch.setattr(twitch, "user_id", lambda login: "U1")
    monkeypatch.setattr(twitch, "recent_archives", lambda uid, first=5: vids[:first])
    return vids


def test_match_archive_by_stream_start(fake_twitch):
    assert source.match_archive("me", START)["id"] == "555"
    assert source.match_archive("me", START - dt.timedelta(days=1, seconds=-30))["id"] == "444"
    assert source.match_archive("me", START + dt.timedelta(hours=3)) is None   # nothing within 10 min
    assert source.match_archive("me", None)["id"] == "555"                     # unknown -> newest


def test_find_stream_vod_errors(fake_twitch, monkeypatch):
    with pytest.raises(ClipperError, match="TWITCH_CHANNEL"):
        source.find_stream_vod(None, START)
    with pytest.raises(ClipperError, match="no Twitch VOD"):
        source.find_stream_vod("me", START + dt.timedelta(hours=3))

    def boom(uid, first=5):
        raise OSError("network down")
    monkeypatch.setattr(twitch, "recent_archives", boom)
    with pytest.raises(ClipperError, match="network down"):
        source.find_stream_vod("me", START)


def test_ended_stream_claims_waits_and_dedupes(fake_twitch, make_args, monkeypatch):
    slept = []
    monkeypatch.setattr(pipeline.time, "sleep", slept.append)
    args = make_args("--stream-ended", "--stream-start", START.isoformat(), "--channel", "me")
    run_args = pipeline.ended_stream_args(args)
    assert run_args.source == "https://www.twitch.tv/videos/555"
    assert run_args.channel is None and args.source is None     # original args untouched
    assert slept == [120]
    assert "S5" in ProcessedStreams()
    assert pipeline.ended_stream_args(args) is None             # second run: already clipped


def test_ended_stream_dry_run_does_not_claim(fake_twitch, make_args, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    args = make_args("--stream-ended", "--channel", "me", "--dry-run", "--vod-delay", "0")
    assert pipeline.ended_stream_args(args) is not None
    assert "S5" not in ProcessedStreams()


def test_link_recording_sets_offset(fake_twitch, tmp_path):
    src = Source(tmp_path / "rec.mkv", dt.date(2026, 11, 12), "rec")
    rec_start = START - dt.timedelta(seconds=3)
    assert source.link_recording(src, "me", rec_start, START) == "S5"
    assert (src.vod_id, src.vod_offset) == ("555", 3.0)
    assert src.from_vod_time(100) == 103 and src.to_vod_time(103) == 100

    src2 = Source(tmp_path / "rec.mkv", dt.date(2026, 11, 12), "rec")
    assert source.link_recording(src2, "me", rec_start, None) == "S5"   # falls back to created_at
    assert src2.vod_offset == 7.0


def test_link_recording_never_raises(tmp_path, monkeypatch):
    src = Source(tmp_path / "rec.mkv", dt.date(2026, 11, 12), "rec")
    assert source.link_recording(src, None, START, START) is None
    monkeypatch.setattr(twitch, "user_id", lambda login: (_ for _ in ()).throw(OSError("offline")))
    assert source.link_recording(src, "me", START, START) is None
    assert src.vod_id is None and src.vod_offset == 0.0


def test_missing_local_file_is_a_clean_error(make_args, tmp_path):
    with pytest.raises(ClipperError, match="file not found"):
        source.resolve_source(make_args(str(tmp_path / "nope.mp4")))
