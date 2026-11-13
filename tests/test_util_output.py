"""Time helpers, .env loading, output folders, reports and subtitles."""
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from clipper import config
from clipper.models import Highlight, Source
from clipper.output import clip_name, make_output_dir, write_report
from clipper.render import srt_time, video_filter, write_srt
from clipper.state import ProcessedStreams
from clipper.util import hms, parse_ts, twitch_t


@pytest.mark.parametrize("sec,h,t", [(0, "0:00:00", "0h00m00s"), (3725, "1:02:05", "1h02m05s")])
def test_time_formats(sec, h, t):
    assert hms(sec) == h
    assert twitch_t(sec) == t


@pytest.mark.parametrize("text,sec", [("1:02:05", 3725), ("62:05", 3725), ("90", 90), ("0:01.5", 1.5)])
def test_parse_ts(text, sec):
    assert parse_ts(text) == sec


def test_load_env_parsing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nA=1\nexport B="two words"\nC=3 # trailing\nD=\nKEEP=from-file\n',
                   encoding="utf-8")
    for k in "ABCD":
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KEEP", "from-env")
    config.load_env(env)
    import os
    assert (os.environ["A"], os.environ["B"], os.environ["C"]) == ("1", "two words", "3")
    assert os.environ["KEEP"] == "from-env"      # real environment wins
    assert config.env("D") is None


def test_save_env_replaces_and_appends(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TWITCH_TOKEN=old\nOTHER=x\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    config.save_env("TWITCH_TOKEN", "new")
    config.save_env("TWITCH_REFRESH_TOKEN", "r")
    assert env.read_text(encoding="utf-8") == "TWITCH_TOKEN=new\nOTHER=x\nTWITCH_REFRESH_TOKEN=r\n"


def test_output_dir_same_day_and_rerun(tmp_path):
    day = dt.date(2026, 11, 12)
    a = make_output_dir(str(tmp_path), day, "vod1")
    assert a.name == "2026-11-12"
    (a / "01_0h01m00s.mp4").write_text("old clip")
    (a / "notes.txt").write_text("mine")
    b = make_output_dir(str(tmp_path), day, "vod2")       # second stream that day
    assert b.name == "2026-11-12_2"
    again = make_output_dir(str(tmp_path), day, "vod1")   # re-run: same folder, old clips gone
    assert again == a
    assert not (a / "01_0h01m00s.mp4").exists()
    assert (a / "notes.txt").exists()


def test_report_links_use_vod_offset(tmp_path):
    src = Source(tmp_path / "rec.mkv", dt.date(2026, 11, 12), "My stream",
                 url="https://www.twitch.tv/videos/999", vod_id="999", vod_offset=5.0)
    h = Highlight(peak=105.0, start=80.0, length=30.0, score=4.2, reasons=["marker: boom"])
    h.file = clip_name(1, h) + ".mp4"
    write_report(tmp_path, src, [h])
    md = (tmp_path / "highlights.md").read_text(encoding="utf-8")
    assert "01_0h01m45s.mp4" in md
    assert "videos/999?t=0h01m15s" in md                  # 80 s in the file = 75 s in the VOD
    data = json.loads((tmp_path / "highlights.json").read_text(encoding="utf-8"))
    assert data["vod_offset"] == 5.0 and data["clips"][0]["reasons"] == ["marker: boom"]


def test_srt_grouping(tmp_path):
    words = [SimpleNamespace(word=f" w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(7)]
    text = write_srt(words, tmp_path / "c.srt", per_line=3)
    srt = (tmp_path / "c.srt").read_text(encoding="utf-8")
    assert text == "w0 w1 w2 w3 w4 w5 w6"
    assert srt.count("-->") == 3
    assert "00:00:00,000 --> 00:00:01,400" in srt
    assert srt_time(3725.25) == "01:02:05,250"


def test_video_filter_variants():
    assert "gblur" in video_filter("vertical", "blur", None)
    assert "crop=ih*9/16" in video_filter("vertical", "center", None)
    assert video_filter("horizontal", "blur", "01.srt").endswith("subtitles=01.srt:force_style='"
                                                                 + config.CAPTION_STYLE + "'[v]")


def test_processed_streams_shared_between_instances(tmp_path):
    path = tmp_path / "p.json"
    a, b = ProcessedStreams(path), ProcessedStreams(path)
    a.add("S1")
    assert "S1" in b                                       # re-read from disk
    path.write_text("not json")
    assert "S1" not in a                                   # corrupt file = empty, not a crash
