"""Find the streamer's webcam overlay in the video, so clips can show it on top.

Two signals, over frames sampled across the whole stream:
  1. Faces (YuNet): the webcam is where a face shows up at the same spot in most
     frames. Faces in gameplay (characters, cutscenes) move around and are ignored.
  2. Stillness: the streamer's room barely changes for hours, while gameplay changes
     all the time. Around the face, the overlay's edges are where per-pixel variation
     steps from low (room) to high (gameplay).
Everything is analysed on frames scaled to ANALYSIS_W pixels wide, then scaled back.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .util import log, run

MODEL = Path(__file__).resolve().parent / "data" / "face_detection_yunet_2023mar.onnx"
ANALYSIS_W = 640
SAMPLES = 40            # frames sampled across the stream
MIN_SHARE = 0.4         # the face must be at the same spot in >= 40% of the frames
MAX_FACE_H = 0.3        # a bigger face = full-screen camera, not an overlay
STEP_K = 4              # pixels averaged on each side of a candidate edge
MATCH_DIFF = 30         # per-clip check: median grey-level difference vs the usual cam image


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def scaled(self, f: float) -> "Box":
        return Box(round(self.x * f), round(self.y * f), round(self.w * f), round(self.h * f))

    def __str__(self) -> str:
        return f"{self.w}x{self.h} at ({self.x},{self.y})"


@dataclass
class Facecam:
    box: Box                 # in source-video pixels
    small: Box               # the same, in analysis pixels
    reference: np.ndarray    # usual look of the cam area (median grey frame, analysis scale)
    scale: float             # source pixels per analysis pixel


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------
def video_size(path: Path) -> tuple[int, int]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height", "-of", "csv=p=0", str(path)])
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def grab_frames(path: Path, times, width: int = ANALYSIS_W) -> np.ndarray:
    """BGR frames (N, H, W, 3) at the given seconds; unreadable ones are skipped."""
    frames, size = [], None
    for t in times:
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
                            "-frames:v", "1", "-vf", f"scale={width}:-2",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        if r.returncode or not r.stdout:
            continue
        h = len(r.stdout) // (width * 3)
        if size is None:
            size = h
        if h == size:
            frames.append(np.frombuffer(r.stdout[: h * width * 3], np.uint8).reshape(h, width, 3))
    return np.stack(frames) if frames else np.zeros((0, 0, width, 3), np.uint8)


# ----------------------------------------------------------------------------
# Faces
# ----------------------------------------------------------------------------
def detect_faces(frames: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """(frame index, x, y, w, h) for every face found."""
    import cv2
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    h, w = frames.shape[1:3]
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), 0.7)
    out = []
    for i, f in enumerate(frames):
        _, res = det.detect(np.ascontiguousarray(f))
        for r in (res if res is not None else []):
            out.append((i, *(int(v) for v in r[:4])))
    return out


def face_cluster(faces, n_frames: int, width: int, height: int) -> Box | None:
    """The face that sits at the same spot in most frames (median box), or None."""
    if not faces or n_frames == 0:
        return None
    tol = 0.05 * width
    centres = np.array([(x + w / 2, y + h / 2) for _, x, y, w, h in faces])
    best: list[int] = []
    for c in centres:
        near = [j for j, d in enumerate(np.hypot(*(centres - c).T)) if d <= tol]
        if len({faces[j][0] for j in near}) > len({faces[j][0] for j in best}):
            best = near
    frames_hit = len({faces[j][0] for j in best})
    if frames_hit < max(3, MIN_SHARE * n_frames):
        return None
    x, y, w, h = (int(np.median([faces[j][k] for j in best])) for k in range(1, 5))
    if h > MAX_FACE_H * height:
        return None
    return Box(x, y, w, h)


# ----------------------------------------------------------------------------
# The overlay rectangle around the face
# ----------------------------------------------------------------------------
def _edge(profile: np.ndarray, lo: int, hi: int, outward: int, min_step: float) -> int | None:
    """Strongest still -> changing step between lo and hi (changing side = `outward`)."""
    k = STEP_K
    best, pos = min_step, None
    for i in range(max(lo, k), min(hi, len(profile) - k)):
        a, b = profile[i - k:i].mean(), profile[i:i + k].mean()
        step = (b - a) if outward > 0 else (a - b)
        if step > best:
            best, pos = step, i
    return pos


def cam_rect(still: np.ndarray, face: Box) -> Box | None:
    """Overlay rectangle around the face, from the per-pixel variation map `still`."""
    height, width = still.shape
    min_step = 0.25 * float(np.median(still))
    cx, cy = face.x + face.w // 2, face.y + face.h // 2
    reach = 8 * face.h
    rows = still[max(0, face.y - face.h // 2): face.y + 2 * face.h]
    col = np.median(rows, axis=0)
    right = _edge(col, cx + face.w, cx + reach, +1, min_step)
    left = _edge(col, cx - reach, cx - face.w, -1, min_step)
    x0, x1 = left or 0, right or width
    row = np.median(still[:, x0:x1], axis=1)
    bottom = _edge(row, cy + face.h, cy + reach, +1, min_step)
    top = _edge(row, cy - reach, cy - face.h, -1, min_step)
    y0, y1 = top or 0, bottom or height
    box = Box(x0, y0, x1 - x0, y1 - y0)
    # sanity: a real overlay holds the face, is clearly bigger than it, and isn't the whole screen
    if (box.w < 1.5 * face.w or box.h < 1.5 * face.h or box.w * box.h > 0.45 * width * height
            or not (x0 <= face.x and face.x + face.w <= x1 and y0 <= face.y and face.y + face.h <= y1)):
        return None
    return box


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def find_facecam(path: Path, duration: float) -> Facecam | None:
    """Look for a webcam overlay over the whole stream. None if there isn't a clear one."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        log("opencv not installed - no webcam detection (pip install opencv-python-headless)")
        return None
    margin = min(300.0, duration * 0.05)
    times = np.linspace(margin, max(margin, duration - margin), SAMPLES)
    frames = grab_frames(path, times)
    if len(frames) < 5:
        return None
    height, width = frames.shape[1:3]
    face = face_cluster(detect_faces(frames), len(frames), width, height)
    if not face:
        return None
    grey = frames.astype(np.float32).mean(axis=3)
    small = cam_rect(grey.std(axis=0), face)
    if not small:
        return None
    src_w, _ = video_size(path)
    scale = src_w / width
    ref = np.median(grey[:, small.y:small.y + small.h, small.x:small.x + small.w], axis=0)
    return Facecam(small.scaled(scale), small, ref, scale)


def cam_visible(path: Path, cam: Facecam, start: float, length: float) -> bool:
    """Is the webcam overlay on screen during this clip? (not a BRB / full-screen scene)"""
    times = [start + length * f for f in (0.2, 0.5, 0.8)]
    frames = grab_frames(path, times)
    s = cam.small
    for f in frames:
        area = f.astype(np.float32).mean(axis=2)[s.y:s.y + s.h, s.x:s.x + s.w]
        if area.shape == cam.reference.shape and np.median(np.abs(area - cam.reference)) < MATCH_DIFF:
            return True
    return False
