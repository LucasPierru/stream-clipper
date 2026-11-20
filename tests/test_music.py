"""--remove-music: music stretches, the voice filter, the streamed replacement, the CLI.
The real models (AST, WavLM, Demucs) are replaced by fakes; they need big downloads."""
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
    """Fake detector: 'music' = loud; never speech."""
    return np.array([[float(np.sqrt((w ** 2).mean()) > 0.1), 0.0] for w in windows]).reshape(-1, 2)


def pitch(w, rate=music.DETECT_SR):
    spec = np.abs(np.fft.rfft(w))
    return np.argmax(spec) * rate / len(w)


def by_pitch(windows):
    """Fake speaker model: 220 Hz is 'you', anything else someone else."""
    return np.array([[1.0, 0.0] if abs(pitch(w) - 220) < 30 else [0.0, 1.0] for w in windows])


def tag_by_pitch(windows):
    """Fake AST on the vocals: 660 Hz is a game character talking, 440 Hz a singer."""
    out = []
    for w in windows:
        f = pitch(w)
        out.append([0.0, 0.9] if abs(f - 660) < 30 else [0.6, 0.2] if abs(f - 440) < 30 else [0.0, 0.9])
    return np.array(out)


def voices(seconds_each=4):
    """You (220 Hz), a singer (440 Hz), silence, a game character talking (660 Hz)."""
    sr = music.SEP_SR
    t = np.arange(4 * seconds_each * sr) / sr
    freq = np.select([t < 4, t < 8, t < 12], [220, 440, 0], 660)
    x = (0.3 * np.sin(2 * np.pi * freq * t) * (freq > 0)).astype(np.float32)
    return np.stack([x, x], axis=1)


def sec(gain, a, b):
    return gain[int(a * music.SEP_SR):int(b * music.SEP_SR)]


def test_voice_filter_keeps_only_you():
    gain, stats = music.voice_gain(voices(), by_pitch, tag_by_pitch, np.array([1.0, 0.0]))
    assert sec(gain, 0.6, 3.4).min() > 0.99            # you
    assert sec(gain, 4.6, 7.4).max() < 0.01            # the singer
    assert sec(gain, 12.6, 15.4).max() < 0.01          # a game character, over music: muted too
    assert stats["your voice"] == pytest.approx(4, abs=1)
    assert stats["muted"] == pytest.approx(8, abs=1)


def test_voice_filter_can_keep_game_speech():
    gain, _ = music.voice_gain(voices(), by_pitch, tag_by_pitch, np.array([1.0, 0.0]), keep_speech=True)
    assert sec(gain, 0.6, 3.4).min() > 0.99
    assert sec(gain, 4.6, 7.4).max() < 0.01
    assert sec(gain, 12.6, 15.4).min() > 0.99


def test_without_a_voiceprint_the_music_stretches_go_silent():
    gain, _ = music.voice_gain(voices(), by_pitch, tag_by_pitch, None)
    assert gain.max() < 0.01


@needs_ffmpeg
def test_voiceprint_is_the_voice_heard_most(tmp_path):
    """You (220 Hz) talk most; a game voice (440 Hz) a little; music in 60-90 s is skipped."""
    src = tmp_path / "a.wav"
    expr = "0.3*sin(2*PI*if(lt(t,60),220,if(lt(t,90),330,440))*t)"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"aevalsrc='{expr}':s=16000:d=100", str(src)], check=True)
    db = np.full(100, -20.0)
    seen = []

    def embed(windows):
        seen.extend(pitch(w) for w in windows)
        return by_pitch(windows)
    v = music.voiceprint(src, [(60.0, 90.0)], db, embed)
    assert v @ np.array([1.0, 0.0]) > 0.99
    assert not any(abs(f - 330) < 30 for f in seen)          # never sampled the music


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
def test_a_long_span_is_processed_in_pieces(tmp_path, monkeypatch):
    """With 10 s pieces, a 40 s stretch goes through in pieces of at most 10 + 2 x 1 s,
    seamlessly: the whole stretch is at half level, the rest untouched."""
    monkeypatch.setattr(music, "PIECE", 10)
    monkeypatch.setattr(music, "CONTEXT", 1)
    src, out = tmp_path / "a.wav", tmp_path / "clean.m4a"
    tone(src, 70, (0, 70))
    sizes = []

    def half(mix):
        sizes.append(len(mix) / music.SEP_SR)
        return mix * 0.5
    music.write_clean_track(src, [(15.0, 55.0)], out, half)
    assert len(sizes) >= 4 and max(sizes) <= 12.01
    before, after = rms_per_second(src), rms_per_second(out)
    assert np.allclose(after[16:54], before[16:54] * 0.5, rtol=0.05)
    assert np.allclose(after[2:14], before[2:14], rtol=0.05)
    assert np.allclose(after[57:68], before[57:68], rtol=0.05)


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
    monkeypatch.setattr(music, "load_speaker", lambda dev: by_pitch)
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
