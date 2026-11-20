"""--parts: choosing natural cut points, and the lossless cutting on a generated video."""
import shutil
import subprocess

import numpy as np
import pytest

from clipper import cli
from clipper.models import Highlight
from clipper.parts import choose_cuts, cut_scores

MIN = 60


def grid(duration, step=2.0):
    times = np.arange(0, duration, step)
    return times, np.zeros(len(times)), [[] for _ in times]


def test_cuts_land_on_the_best_spot_in_each_window():
    times, score, reasons = grid(180 * MIN)
    for t, why in ((50 * MIN, "black/loading screen"), (101 * MIN, "scene change")):
        k = int(np.searchsorted(times, t))
        score[k], reasons[k] = 3.0, [why]
    cuts = choose_cuts(times, score, reasons, 180 * MIN, 45 * MIN, 60 * MIN)
    assert [round(t / MIN) for t, _ in cuts[:2]] == [50, 101]
    assert cuts[0][1] == "black/loading screen"
    lengths = np.diff([0] + [t for t, _ in cuts] + [180 * MIN]) / MIN
    assert all(45 <= x <= 60 for x in lengths[:-1]) and 15 <= lengths[-1] <= 75


def test_without_a_clear_break_parts_run_long():
    """Nothing scores: the length bonus makes each part as long as allowed."""
    times, score, reasons = grid(150 * MIN)
    cuts = choose_cuts(times, score, reasons, 150 * MIN, 45 * MIN, 60 * MIN)
    assert [round(t / MIN) for t, _ in cuts] == [60, 120]
    assert cuts[0][1] == "no clear break - best spot available"


def test_last_part_is_never_a_short_leftover():
    for total, n_cuts in ((70, 0), (75, 0), (80, 1), (130, 1), (140, 2)):
        times, score, reasons = grid(total * MIN)
        cuts = choose_cuts(times, score, reasons, total * MIN, 45 * MIN, 60 * MIN)
        assert len(cuts) == n_cuts, total
        bounds = [0] + [t for t, _ in cuts] + [total * MIN]
        lengths = np.diff(bounds) / MIN
        assert lengths.min() >= 15 and lengths.max() <= 75, (total, lengths)


def test_cut_scores():
    times = np.array([0.0, 2, 4, 6, 8, 10])
    thumbs = np.full((6, 36, 64), 120, np.uint8)
    thumbs[2] = 5                                      # black screen
    thumbs[4] = np.random.default_rng(0).integers(0, 255, (36, 64))   # new scene
    db = np.full(20, -20.0)
    db[8:14] = -45                                     # quiet around 10 s
    hl = [Highlight(peak=12, start=11, length=1, score=5)]       # off limits: 6-17 s
    score, reasons = cut_scores(times, thumbs, db, hl)
    assert "black/loading screen" in reasons[2]
    assert "scene change" in reasons[4]
    assert "quiet" in reasons[5]
    assert score[3] < 0 and score[5] < 1             # within 5 s of the highlight
    assert score[2] > score[0]


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not on PATH")


def duration_of(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                                check=True).stdout)


@needs_ffmpeg
def test_cli_parts_end_to_end(tmp_path):
    """150 s video, a black screen at 70-74 s, parts of 45-75 s: the first cut is in
    the black screen, and the parts add up to the whole video."""
    vod = tmp_path / "vod.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=150",
                    "-f", "lavfi", "-i", "sine=f=220:d=150", "-vf",
                    "drawbox=c=black:t=fill:enable='between(t,70,74)'", "-c:v", "libx264",
                    "-preset", "ultrafast", "-g", "60", "-c:a", "aac", "-shortest", str(vod)], check=True)
    cli.main([str(vod), "--out", str(tmp_path / "out"), "--date", "2026-11-20", "--parts",
              "--part-length", "0.75-1.25", "--max-clips", "0", "--skip-end", "10"])
    folder = tmp_path / "out" / "2026-11-20" / "parts"
    files = sorted(folder.glob("part_*.mp4"))
    assert len(files) == 2
    first = duration_of(files[0])
    assert 70 <= first <= 75
    assert abs(first + duration_of(files[1]) - 150) < 1
    md = (folder / "parts.md").read_text(encoding="utf-8")
    assert "black/loading screen" in md and files[1].name in md
    # a re-run replaces the old parts
    cli.main([str(vod), "--out", str(tmp_path / "out"), "--date", "2026-11-20", "--parts",
              "--part-length", "2-3", "--max-clips", "0"])
    assert [f.name for f in folder.glob("part_*.mp4")] == ["part_01_0h00m00s.mp4"]


def test_part_length_flag():
    assert cli.parse_args(["--part-length", "45-60"]).part_length == (2700, 3600)
    assert cli.parse_args([]).part_length == (2700, 3600)
    with pytest.raises(SystemExit):
        cli.parse_args(["--part-length", "60-45"])
