"""Webcam detection: face clustering, the overlay rectangle, layout planning, --cam."""
import argparse

import numpy as np
import pytest

from clipper import cli, facecam, pipeline
from clipper.facecam import Box, cam_rect, face_cluster
from clipper.models import Highlight, Source


def test_face_cluster_keeps_the_face_that_does_not_move():
    faces = [(i, 70 + i % 3, 85, 26, 35) for i in range(30)]          # streamer: same spot
    faces += [(i, 300 + 7 * i, 50 + 3 * i, 40, 50) for i in range(10)]  # game characters
    assert face_cluster(faces, 40, 640, 360) == Box(71, 85, 26, 35)


def test_face_cluster_rejects_rare_or_huge_faces():
    assert face_cluster([(i, 70, 85, 26, 35) for i in range(8)], 40, 640, 360) is None  # 20%
    assert face_cluster([(i, 200, 50, 150, 200) for i in range(30)], 40, 640, 360) is None
    assert face_cluster([], 40, 640, 360) is None


def still_map(cam=(0, 55, 153, 112), person=(40, 70, 70, 97), seed=0):
    """Per-pixel variation: gameplay changes a lot, the room barely, the person a bit."""
    rng = np.random.default_rng(seed)
    s = rng.normal(65, 6, (360, 640))
    x, y, w, h = cam
    s[y:y + h, x:x + w] = rng.normal(18, 4, (h, w))
    px, py, pw, ph = person
    s[py:py + ph, px:px + pw] = rng.normal(45, 8, (ph, pw))
    return s


def test_cam_rect_finds_the_overlay_edges():
    box = cam_rect(still_map(), Box(70, 85, 26, 35))
    assert box is not None
    assert abs(box.x - 0) <= 2 and abs(box.y - 55) <= 2
    assert abs(box.w - 153) <= 3 and abs(box.h - 112) <= 3


def test_cam_rect_rejects_when_there_is_no_overlay():
    rng = np.random.default_rng(1)
    assert cam_rect(rng.normal(65, 6, (360, 640)), Box(300, 150, 26, 35)) is None


@pytest.mark.parametrize("text,ok", [("0,80,480,270", True), ("1,2,3", False), ("0,0,-5,10", False)])
def test_parse_box(text, ok):
    if ok:
        assert cli.parse_box(text) == Box(0, 80, 480, 270)
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_box(text)


def plan(monkeypatch, tmp_path, argv, cam=None, visible=(True, False)):
    monkeypatch.setattr(pipeline, "find_facecam", lambda path, duration: cam)
    seen = iter(visible)
    monkeypatch.setattr(pipeline, "cam_visible", lambda path, c, start, length: next(seen))
    monkeypatch.setattr(pipeline, "probe_duration", lambda path: 3600.0)
    src = Source(tmp_path / "v.mp4", None, "v")
    hl = [Highlight(100, 80, 30, 5), Highlight(900, 880, 30, 5)]
    return pipeline.plan_layouts(src, hl, cli.parse_args(argv))


def test_plan_layouts(monkeypatch, tmp_path):
    found = facecam.Facecam(Box(0, 146, 408, 298), Box(0, 55, 153, 112), np.zeros((112, 153)), 2.67)
    assert plan(monkeypatch, tmp_path, [], cam=None) == [("fill", None)] * 2
    assert plan(monkeypatch, tmp_path, [], cam=found) == [("split", found.box), ("fill", None)]
    assert plan(monkeypatch, tmp_path, ["--layout", "split"], cam=found) == [("split", found.box)] * 2
    assert plan(monkeypatch, tmp_path, ["--layout", "blur"], cam=found) == [("blur", None)] * 2
    assert plan(monkeypatch, tmp_path, ["--cam", "1,2,300,200"]) == [("split", Box(1, 2, 300, 200))] * 2
    assert plan(monkeypatch, tmp_path, ["--format", "horizontal"]) == [("horizontal", None)] * 2


def test_model_file_is_bundled():
    assert facecam.MODEL.is_file() and facecam.MODEL.stat().st_size > 200_000


def test_refine_box_snaps_to_exact_edges_and_shaves_the_blend():
    """Real overlay: x 0..119, y 60..149 in a 480x270 video. The coarse box from the
    640-px analysis is a few pixels too big; refining must land just inside the edges."""
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 255, (12, 270, 480)).astype(np.float32)        # gameplay
    frames[:, 60:150, :120] = 100 + rng.normal(0, 3, (12, 90, 120))         # still room
    frames[:, 60:150, 120] = (frames[:, 60:150, 119] + frames[:, 60:150, 121]) / 2  # blended edge
    coarse = Box(0, 57, 124, 96)                                             # 3-4 px too big
    box = facecam.refine_box(frames, coarse, (True, False, False, False), scale=0.75)
    assert box.x == 0                                  # on the frame border: untouched
    assert box.y == 61                                 # first still row 60, + 1 px inset
    assert box.x + box.w == 119                        # first changing column 120, - 1 px inset
    assert box.y + box.h == 149
    # every pixel inside the result is room, none is gameplay or blend
    assert frames[:, box.y:box.y + box.h, box.x:box.x + box.w].std(axis=0).max() < 20


def test_refine_edge_direction():
    profile = np.array([60.0] * 10 + [15.0] * 10)      # changing, then still
    assert facecam.refine_edge(profile, 12, 5, -1) == 10   # first still pixel
    assert facecam.refine_edge(profile[::-1].copy(), 8, 5, +1) == 10   # first changing pixel


# ----------------------------------------------------------------------------
# A chat panel attached to the webcam (translucent black box under or next to it)
# ----------------------------------------------------------------------------
def _blur(a, k):
    c = np.cumsum(np.cumsum(np.pad(a, k, mode="edge"), 0), 1)
    return (c[2 * k:, 2 * k:] - c[:-2 * k, 2 * k:] - c[2 * k:, :-2 * k] + c[:-2 * k, :-2 * k])[
        :a.shape[0], :a.shape[1]] / (4 * k * k)


CAM = (0, 55, 153, 112)     # x, y, w, h of the real webcam, analysis pixels
FACE = Box(70, 85, 26, 35)


def stream_frames(chat="below", opacity=0.65, text=True, desk=False, drift=0.0, seed=0, n=40):
    """Grey analysis frames: changing gameplay, a still room with a moving person, and
    optionally a translucent chat panel (dimmed gameplay + text lines) next to the cam."""
    rng = np.random.default_rng(seed)
    H, W = 360, 640
    x, y, w, h = CAM
    room = _blur(rng.uniform(40, 200, (H, W)), 4)[y:y + h, x:x + w]
    if desk:
        room[95:] = 60                                       # straight desk edge inside the cam
    yy, xx = np.mgrid[y:y + h, x:x + w]
    out = []
    for t in range(n):
        f = np.clip((_blur(rng.uniform(0, 255, (H, W)), 3) - 128) * 4 + rng.uniform(40, 200), 0, 255)
        cam = room * (1 + drift * (t / n - 0.5)) + rng.normal(0, 2, room.shape)
        dy, dx = rng.integers(-3, 4, 2)
        person = ((xx - 83 - dx) / 30) ** 2 + ((yy - 140 - dy) / 45) ** 2 < 1
        person |= ((xx - 83 - dx) / 14) ** 2 + ((yy - 102 - dy) / 18) ** 2 < 1
        cam[person] = 170 + rng.normal(0, 4, person.sum())
        f[y:y + h, x:x + w] = cam
        if chat:
            px0, py0, px1, py1 = (x, y + h, x + w, y + h + 140) if chat == "below" else (x + w, y, x + w + 120, y + h)
            panel = f[py0:py1, px0:px1] * (1 - opacity)
            for _ in range(rng.integers(0, 6) if text else 0):
                ty, tx = rng.integers(0, py1 - py0 - 6), rng.integers(0, 20)
                panel[ty:ty + 5, tx:tx + rng.integers(30, px1 - px0 - tx)] = 230
            f[py0:py1, px0:px1] = panel
        out.append(f)
    return np.array(out, np.float32)


def detect(frames):
    still = frames.std(axis=0)
    box = cam_rect(still, FACE)
    assert box is not None
    return box, facecam.trim_panels(np.median(frames, axis=0), still, box, FACE)


def assert_is_cam(box):
    x, y, w, h = CAM
    assert max(abs(box.x - x), abs(box.y - y), abs(box.x + box.w - x - w), abs(box.y + box.h - y - h)) <= 2, box


@pytest.mark.parametrize("kw", [
    dict(),                                                   # chat under the cam
    dict(chat="right"),                                       # chat next to it
    dict(opacity=0.4),
    dict(desk=True),                                          # cut at the panel, not the desk
    dict(opacity=0.85, text=False, drift=0.4),                # dark quiet panel, room light drifting
], ids=["below", "right", "light", "desk", "dark-quiet-drift"])
def test_chat_panel_is_cut_off_the_webcam(kw):
    raw, box = detect(stream_frames(**kw))
    assert_is_cam(box)


def test_chat_panel_was_taken_in_before():
    """The bug: the panel edge is a stronger step than the cam edge, so cam_rect alone
    returned webcam + chat."""
    raw, _ = detect(stream_frames())
    assert raw.h > CAM[3] + 100


@pytest.mark.parametrize("kw", [dict(chat=None), dict(chat=None, desk=True, drift=0.4)], ids=["plain", "desk-drift"])
def test_no_panel_nothing_is_cut(kw):
    raw, box = detect(stream_frames(**kw))
    assert box == raw
    assert_is_cam(box)


def test_refine_box_next_to_a_chat_panel():
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 255, (12, 270, 480)).astype(np.float32)
    frames[:, 150:250, :120] *= 0.35                                    # translucent chat
    frames[:, 60:150, :120] = 100 + rng.normal(0, 3, (12, 90, 120))     # still room
    box = facecam.refine_box(frames, Box(0, 57, 124, 92), (True, False, False, False), scale=0.75)
    assert box.y + box.h == 149                                         # last cam row, - 1 px inset
