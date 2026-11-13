"""Highlight detection and picking, on synthetic per-second loudness."""
import numpy as np

from clipper.config import MARKER_POSITION
from clipper.detect import audio_score, pick, robust_z, rolling


def loudness(seconds=900, bursts=(), base=-30.0, seed=0):
    """Per-second dBFS with small noise and +25 dB bursts of 3 s."""
    rng = np.random.default_rng(seed)
    db = base + rng.normal(0, 1.0, seconds)
    for t in bursts:
        db[t:t + 3] += 25
    return db


def test_rolling_keeps_length_and_centres():
    x = np.arange(10, dtype=float)
    out = rolling(x, 3, np.mean)
    assert len(out) == len(x)
    assert out[5] == 5.0


def test_robust_z_ignores_outliers():
    x = np.array([0.0] * 99 + [100.0])
    z = robust_z(x, floor=1.0)
    assert z[-1] == 100.0          # MAD is 0 -> floor of 1 keeps it finite
    assert abs(z[0]) < 1e-9


def test_audio_score_peaks_at_bursts():
    bursts = (200, 500, 750)
    score = audio_score(loudness(bursts=bursts))
    top = sorted(np.argsort(score)[::-1][:30])
    for t in bursts:
        assert any(abs(i - t) <= 3 for i in top)
    assert score[100] < 3.0        # a quiet second is not a highlight


def test_silence_and_its_end_never_score():
    db = loudness(seconds=600)
    db[300:330] = -60              # 30 s of silence (muted mic, BRB, DMCA-muted VOD part)
    score = audio_score(db)
    assert score[300:330].max() < 1.0
    assert score[330:340].max() < 2.0   # sound coming back is not a "sudden loud moment"


def test_pick_finds_bursts_with_peak_position(make_args):
    args = make_args()
    db = loudness(bursts=(200, 500, 750))
    score = audio_score(db)
    picks = pick(score, {"loud moment": score}, 900.0, [], args)
    assert len(picks) == 3
    for h, t in zip(picks, (200, 500, 750)):
        assert abs(h.peak - t) <= 3
        assert h.length == args.clip_length
        assert abs(h.start - (h.peak - args.peak_position * h.length)) < 1e-6
        assert h.reasons[0].startswith("loud moment")


def test_pick_respects_max_clips_gap_and_skips(make_args):
    db = loudness(bursts=(30, 200, 220, 500, 750, 890))  # 30 = skip-start, 890 = skip-end
    score = audio_score(db)
    picks = pick(score, {"loud moment": score}, 900.0, [], make_args("--max-clips", "2"))
    assert len(picks) == 2
    peaks = [h.peak for h in picks]
    assert all(60 <= p <= 870 for p in peaks)
    assert abs(peaks[1] - peaks[0]) >= 45


def test_forced_clips_always_kept_and_block_nearby_auto(make_args):
    db = loudness(bursts=(500,))
    score = audio_score(db)
    picks = pick(score, {"loud moment": score}, 900.0, [(510.0, "marker: wow"), (5000.0, "outside")],
                 make_args())
    assert [h.reasons for h in picks] == [["marker: wow"]]   # burst at 500 is within --min-gap
    h = picks[0]
    assert abs(h.start - (510 - MARKER_POSITION * h.length)) < 1e-6


def test_pick_nothing_above_threshold(make_args):
    score = audio_score(loudness())
    assert pick(score, {"loud moment": score}, 900.0, [], make_args("--threshold", "50")) == []


def test_clip_clamped_to_video_end(make_args):
    score = np.zeros(100)
    picks = pick(score, {}, 100.0, [(99.0, "manual")], make_args())
    assert picks[0].start + picks[0].length <= 100.0
