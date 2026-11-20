"""--remove-music: music stretches, the streamed voice-only replacement, the full CLI.
The real models (AST, Demucs) are replaced by fakes; they need big downloads."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from clipper import cli, music
from clipper.music import music_spans, per_second, window_starts

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")


def test_music_spans_merge_drop_and_pad():
    prob = np.zeros(300)
    prob[50:100] = 0.9            # a song
    prob[100:105] = 0.25          # fading out, still above `off`
    prob[108:130] = 0.8           # back after a 3 s dip: same stretch
    prob[200:204] = 0.9           # a 4 s blip: ignored
    prob[290:] = 0.5              # music running off the end
    spans = music_spans(prob, on=0.35, off=0.2, min_len=8, gap=10, pad=2)
    assert spans == [(48, 132), (288, 300)]


def test_music_spans_hysteresis_includes_the_rise():
    prob = np.zeros(100)
    prob[20:25] = 0.25            # the song starting
    prob[25:60] = 0.9
    assert music_spans(prob, pad=0) == [(20, 60)]


def test_no_music_no_spans():
    assert music_spans(np.full(500, 0.1)) == []


def test_windows_cover_every_second():
    starts = window_starts(65)
    assert starts[0] == 0 and starts[-1] == 55          # last window ends at the end
    prob = per_second(starts, np.ones(len(starts)), 65)
    assert len(prob) == 65 and prob.min() == 1.0


def tone(path, seconds, loud):
    """Stereo 44.1 kHz test audio: a quiet tone, loud between `loud` seconds."""
    a, b = loud
    expr = f"(0.02+0.6*between(t,{a},{b}))*sin(2*PI*330*t)"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"aevalsrc='{expr}|{expr}':s=44100:d={seconds}", str(path)], check=True)


def rms_per_second(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    n = len(x) // 8000
    return np.sqrt((x[:n * 8000].reshape(n, 8000) ** 2).mean(axis=1))


def loud_windows(windows):
    """Fake detector: 'music' = loud."""
    return np.array([float(np.sqrt((w ** 2).mean()) > 0.1) for w in windows])


@needs_ffmpeg
def test_music_probability_streams_the_whole_file(tmp_path):
    src = tmp_path / "a.wav"
    tone(src, 120, (40, 80))
    prob = music.music_probability(src, 120, loud_windows)
    assert len(prob) == 120
    assert prob[45:75].min() > 0.9 and prob[:25].max() == 0 and prob[95:].max() == 0
    (a, b), = music_spans(prob, pad=0)
    assert 30 <= a <= 42 and 78 <= b <= 90


@needs_ffmpeg
def test_clean_track_replaces_only_the_spans(tmp_path):
    src, out = tmp_path / "a.wav", tmp_path / "clean.m4a"
    tone(src, 90, (30, 60))
    music.write_clean_track(src, [(30.0, 60.0)], out, lambda mix: np.zeros_like(mix))  # 'voice' = silence
    before, after = rms_per_second(src), rms_per_second(out)
    assert abs(len(after) - 90) <= 1
    assert after[31:59].max() < 0.005                    # the music is gone
    assert np.allclose(after[2:28], before[2:28], rtol=0.1)   # the rest is untouched
    assert np.allclose(after[62:88], before[62:88], rtol=0.1)


@needs_ffmpeg
def test_cli_remove_music_end_to_end(tmp_path, monkeypatch):
    """Video with a loud 'song' at 40-80 s: the clip's audio is silent there."""
    vod = tmp_path / "vod.mp4"
    expr = "(0.03+0.6*between(t,40,80))*sin(2*PI*330*t)"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=120",
                    "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d=120", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(vod)], check=True)
    monkeypatch.setattr(music, "device", lambda: "cpu")
    monkeypatch.setattr(music, "load_detector", lambda dev: loud_windows)
    monkeypatch.setattr(music, "load_separator", lambda dev: lambda mix: np.zeros_like(mix))
    cli.main([str(vod), "--out", str(tmp_path / "out"), "--date", "2026-11-20", "--remove-music",
              "--at", "0:01:00", "--clip-length", "20", "--max-clips", "0", "--format", "horizontal"])
    folder = tmp_path / "out" / "2026-11-20"
    report = json.loads((folder / "highlights.json").read_text(encoding="utf-8"))
    (span,) = report["music_removed"]
    assert span[0] <= 42 and span[1] >= 78
    assert "Music removed" in (folder / "highlights.md").read_text(encoding="utf-8")
    clip = folder / report["clips"][0]["file"]           # 47-67 s: inside the song
    assert rms_per_second(clip)[1:-1].max() < 0.005


def test_missing_packages_give_a_clear_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError(name)
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(music.ClipperError, match="music"):
        music.device()


@needs_ffmpeg
def test_a_failing_separator_raises_instead_of_hanging(tmp_path):
    src, out = tmp_path / "a.wav", tmp_path / "clean.m4a"
    tone(src, 30, (5, 15))

    def broken(mix):
        raise RuntimeError("out of GPU memory")
    with pytest.raises(RuntimeError, match="GPU memory"):
        music.write_clean_track(src, [(5.0, 15.0)], out, broken)
    assert not out.exists() and not list(tmp_path.glob("*.part*"))
