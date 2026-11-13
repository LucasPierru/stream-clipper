"""Full pipeline on a generated video: detection, cutting, layout, report. Needs ffmpeg."""
import json
import shutil
import subprocess

import pytest

from clipper import cli

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture(scope="module")
def vod(tmp_path_factory):
    """150 s test video, quiet tone with one loud burst at 0:01:40."""
    path = tmp_path_factory.mktemp("vod") / "test_vod.mp4"
    audio = "(0.03+0.75*between(t,100,103))*sin(2*PI*220*t)+0.02*(random(0)-0.5)"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=150",
                    "-f", "lavfi", "-i", f"aevalsrc='{audio}':s=44100:d=150",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
                   check=True)
    return path


def dims(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return tuple(int(x) for x in out.strip().split(","))


def test_vertical_clips_and_report(vod, tmp_path):
    cli.main([str(vod), "--out", str(tmp_path), "--date", "2026-11-12", "--at", "0:00:20",
              "--skip-end", "10"])
    folder = tmp_path / "2026-11-12"
    report = json.loads((folder / "highlights.json").read_text(encoding="utf-8"))
    clips = report["clips"]
    assert [c["reasons"][0].split()[0] for c in clips] == ["manual", "loud"]
    assert abs(clips[1]["peak"] - 101) <= 3
    for c in clips:
        assert dims(folder / c["file"]) == (1080, 1920)
    assert (folder / "highlights.md").exists() and (folder / ".source").exists()


def test_horizontal_and_dry_run(vod, tmp_path):
    cli.main([str(vod), "--out", str(tmp_path / "dry"), "--dry-run"])
    assert not (tmp_path / "dry").exists()
    cli.main([str(vod), "--out", str(tmp_path), "--date", "2026-11-12", "--format", "horizontal",
              "--max-clips", "1"])
    (clip,) = (tmp_path / "2026-11-12").glob("01_*.mp4")
    assert dims(clip) == (640, 360)


def test_split_layout_with_manual_cam(vod, tmp_path):
    cli.main([str(vod), "--out", str(tmp_path), "--date", "2026-11-12", "--max-clips", "1",
              "--cam", "0,40,200,112"])
    (clip,) = (tmp_path / "2026-11-12").glob("01_*.mp4")
    assert dims(clip) == (1080, 1920)
