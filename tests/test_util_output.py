"""Time helpers, .env loading, output folders, reports and subtitles."""
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from clipper import config
from clipper.models import Highlight, Source
from clipper.output import clip_name, make_output_dir, write_report
from clipper.facecam import Box
from clipper.render import split_geometry, srt_time, video_filter, write_srt
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
    assert "gblur" in video_filter("blur")
    assert "crop='min(iw,ih*9/16)':ih" in video_filter("fill")
    assert video_filter("split") == video_filter("fill")       # split without a cam -> fill
    graph = video_filter("split", 1920, 1080, Box(0, 160, 480, 270))
    assert "vstack" in graph and "crop=480:268:0:160" in graph   # trimmed to the even panel height
    assert video_filter("horizontal", srt="01.srt").endswith(
        "subtitles=01.srt:force_style='" + config.CAPTION_STYLE + "'[v]")


@pytest.mark.parametrize("src,cam", [((1920, 1080), Box(0, 160, 480, 270)),
                                     ((852, 480), Box(0, 73, 204, 149)),
                                     ((1920, 1080), Box(1500, 700, 420, 380)),
                                     ((1280, 720), Box(40, 40, 600, 200))])
def test_split_geometry_fits_and_keeps_aspect(src, cam):
    src_w, src_h = src
    c, top_h, g, bot_h = split_geometry(src_w, src_h, cam)
    assert top_h + bot_h == 1920 and top_h % 2 == 0
    assert 0.25 * 1920 <= top_h <= 0.42 * 1920
    for box in (c, g):                                       # crops stay inside the video
        assert box.x >= 0 and box.y >= 0 and box.x + box.w <= src_w and box.y + box.h <= src_h
    assert cam.x <= c.x and c.x + c.w <= cam.x + cam.w + 1   # webcam crop stays in the overlay
    assert abs(c.w / c.h - 1080 / top_h) < 0.03              # no stretching
    assert abs(g.w / g.h - 1080 / bot_h) < 0.03
    if cam.x + cam.w <= src_w - g.w:                          # room to dodge: gameplay avoids the cam
        assert g.x >= cam.x + cam.w or g.x + g.w <= cam.x


def test_processed_streams_shared_between_instances(tmp_path):
    path = tmp_path / "p.json"
    a, b = ProcessedStreams(path), ProcessedStreams(path)
    a.add("S1")
    assert "S1" in b                                       # re-read from disk
    path.write_text("not json")
    assert "S1" not in a                                   # corrupt file = empty, not a crash


def test_is_fragmented_mp4(tmp_path):
    import struct
    from clipper.source import is_fragmented_mp4

    def boxes(*kinds):
        return b"".join(struct.pack(">I4s", 16, k) + b"\0" * 8 for k in kinds)

    plain, frag, back = tmp_path / "plain.mp4", tmp_path / "frag.mp4", tmp_path / "back.mp4"
    plain.write_bytes(boxes(b"ftyp", b"moov", b"mdat"))
    frag.write_bytes(boxes(b"ftyp", b"moov", b"moof", b"mdat"))
    back.write_bytes(boxes(b"ftyp", b"mdat", b"moov"))   # index at the end: still seekable
    assert not is_fragmented_mp4(plain)
    assert is_fragmented_mp4(frag)
    assert not is_fragmented_mp4(back)
    assert not is_fragmented_mp4(tmp_path / "missing.mp4")
