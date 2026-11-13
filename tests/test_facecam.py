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
