"""--parts: choosing natural cut points, and the lossless cutting on a generated video."""
import shutil
import subprocess

import numpy as np
import pytest

from clipper import cli
from clipper.models import Highlight
from clipper.parts import Frames, choose_cuts, cut_scores, fade_events, frame_features, long_cutscenes

MIN = 60


def grid(duration, step=2.0):
    times = np.arange(0, duration, step)
    return times, np.zeros(len(times)), [[] for _ in times]


def test_cuts_land_on_the_best_spot_in_each_window():
    times, score, reasons = grid(180 * MIN)
    for t, why in ((50 * MIN, "black/loading screen (5 s)"), (101 * MIN, "fade to black (1.2 s)")):
        k = int(np.searchsorted(times, t))
        score[k], reasons[k] = 3.0, [why]
    cuts = choose_cuts(times, score, reasons, 180 * MIN, 30 * MIN, 60 * MIN)
    assert [round(t / MIN) for t, _ in cuts[:2]] == [50, 101]
    assert cuts[0][1] == "black/loading screen (5 s)"
    lengths = np.diff([0] + [t for t, _ in cuts] + [180 * MIN]) / MIN
    assert all(30 <= x <= 60 for x in lengths[:-1]) and 15 <= lengths[-1] <= 75


def test_without_a_clear_break_parts_run_long():
    """Nothing scores: the length bonus makes each part as long as allowed."""
    times, score, reasons = grid(150 * MIN)
    cuts = choose_cuts(times, score, reasons, 150 * MIN, 30 * MIN, 60 * MIN)
    assert [round(t / MIN) for t, _ in cuts] == [60, 120]
    assert cuts[0][1] == "no clear break - best spot available"


def test_last_part_is_never_a_short_leftover():
    for total, n_cuts in ((70, 0), (75, 0), (80, 1), (130, 1), (140, 2)):
        times, score, reasons = grid(total * MIN)
        cuts = choose_cuts(times, score, reasons, total * MIN, 30 * MIN, 60 * MIN)
        assert len(cuts) == n_cuts, total
        bounds = [0] + [t for t, _ in cuts] + [total * MIN]
        lengths = np.diff(bounds) / MIN
        assert lengths.min() >= 15 and lengths.max() <= 75, (total, lengths)


def scan(duration, black=(), letterbox=(), fps=5.0):
    """A fake 5 fps scan: a bright picture, black during `black`, letterboxed during `letterbox`."""
    n = int(duration * fps)
    t = np.arange(n) / fps
    dark, mean, lb = np.zeros(n, np.float32), np.full(n, 100.0, np.float32), np.zeros(n, bool)
    for a, b in black:
        on = (t >= a) & (t < b)
        dark[on], mean[on] = 1.0, 2.0
    for a, b in letterbox:
        lb[(t >= a) & (t < b)] = True
    return Frames(fps, dark, mean, lb)


def test_fades_are_dips_to_black_not_dark_scenes():
    fr = scan(60, black=[(10, 11), (30, 30.2)])
    fr.dark[40 * 5:55 * 5], fr.mean[40 * 5:55 * 5] = 0.9, 8.0      # a long dark night scene
    fr.mean[35 * 5:40 * 5] = fr.mean[55 * 5:] = 12.0               # ...dim before and after too
    assert fade_events(fr) == [(10.0, 11.0), (30.0, 30.2)]


def test_frame_features_see_black_and_letterbox_bars():
    thumbs = np.full((3, 36, 64), 120, np.uint8)
    thumbs[1] = 3                                                  # black
    thumbs[2, :2], thumbs[2, -2:] = 2, 2                           # bars...
    thumbs[2, :14, :17] = 150                                      # ...with the webcam over the top one
    dark, mean, lb = frame_features(thumbs)
    assert dark[1] == 1.0 and dark[0] == 0.0
    assert list(lb) == [False, False, True]


def test_cut_scores():
    times = np.arange(0.0, 120, 2)
    thumbs = np.full((len(times), 36, 64), 120, np.uint8)
    thumbs[20] = np.random.default_rng(0).integers(0, 255, (36, 64))   # new scene at 40 s
    db = np.full(120, -20.0)
    db[95:106] = -45                                    # quiet around 100 s
    fr = scan(120, black=[(10, 12), (30, 30.2), (61, 61.6), (75, 76)], letterbox=[(66, 90)])
    hl = [Highlight(peak=112, start=110, length=5, score=5)]       # off limits: 105-120 s
    score, reasons = cut_scores(times, thumbs, db, hl, fr)
    at = lambda t: int(np.searchsorted(times, t))
    assert reasons[at(12)] == ["fade to black (2.0 s)"]           # first keyframe as the black ends
    assert score[at(12)] == pytest.approx(3.0)
    assert reasons[at(30)] == [] and reasons[at(32)] == []        # one black frame: the phone, a camera cut
    assert reasons[at(62)][0] == "fade to black (0.6 s)"           # just before a cutscene starts: fine
    assert score[at(76)] < 0                                       # a fade in the middle of a cutscene
    assert score[at(80)] == pytest.approx(-4.0)                    # mid-cutscene
    assert "scene change" in reasons[at(40)]
    assert "quiet" in reasons[at(100)]
    assert score[at(110)] < 0                                      # inside a highlight
    assert score[at(12)] > score[at(40)]


def test_no_cut_while_heading_into_a_long_cutscene():
    """A 60 s cutscene at 400-460 s (with a 3 s shot without bars): the 3 minutes before
    it are the lead-up to a mission, the minutes after it probably the mission."""
    times = np.arange(0.0, 1000, 2)
    thumbs = np.full((len(times), 36, 64), 120, np.uint8)
    fr = scan(1000, letterbox=[(400, 430), (433, 460)])
    assert long_cutscenes(fr) == [(400.0, 460.0)]
    score, _ = cut_scores(times, thumbs, np.full(1000, -20.0), [], fr)
    at = lambda t: score[int(np.searchsorted(times, t))]
    assert at(300) == pytest.approx(-3.0)        # walking up to the mission
    assert at(200) == 0                          # before that: free
    assert at(600) == pytest.approx(-1.5)        # just after: likely the mission
    assert at(900) == 0


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not on PATH")


def duration_of(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                                check=True).stdout)


@needs_ffmpeg
def test_cli_parts_end_to_end(tmp_path):
    """150 s video, a black screen at 70-74 s, parts of 45-75 s: the first cut is at
    the end of the black screen, and the parts add up to the whole video."""
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
    assert cli.parse_args([]).part_length == (1800, 3600)
    with pytest.raises(SystemExit):
        cli.parse_args(["--part-length", "60-45"])
