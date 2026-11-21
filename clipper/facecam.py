"""Find the streamer's webcam overlay in the video, so clips can show it on top.

Two signals, over frames sampled across the whole stream:
  1. Faces (YuNet): the webcam is where a face shows up at the same spot in most
     frames. Faces in gameplay (characters, cutscenes) move around and are ignored.
  2. Seams: the overlay's border is a straight line at the same place in every frame,
     where two unrelated pictures meet (webcam | gameplay), so the pixels on each side
     change independently of each other. Walking out from the face, the first such
     line on each side is the webcam's edge - also when a translucent chat panel is
     attached to it, whose own outer edge has the same gameplay on both sides.
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
SEAM_MIN = 8.0          # an overlay edge: a straight line whose grey-level jump stands out this much
LOCAL = 6               # ...from the lines within this many pixels
SEAM_GAP = 2            # sides of a line compared this many pixels away (skips blended pixels)
SEAM_CORR = 0.5         # across the webcam's edge the two sides change independently (corr < this)
SEAM_MIN_STD = 5.0      # ...which only means something if the far side changes at all
MATCH_DIFF = 30         # per-clip check: median grey-level difference vs the usual cam image
BORDER_SHOW = 3.0       # ...or on half the cam's sides, the picture breaks >= 3x more across the line than beside it


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
    # Some ffprobe builds print the stream more than once, so read only the first line.
    w, h = out.split()[0].split(",")[:2]
    return int(w), int(h)


def grab_frames(path: Path, times, width: int = ANALYSIS_W, grey: bool = False) -> np.ndarray:
    """Frames at the given seconds: BGR (N, H, W, 3), or grey (N, H, W) if `grey`.
    Unreadable ones are skipped."""
    ch, fmt = (1, "gray") if grey else (3, "bgr24")
    frames, size = [], None
    for t in times:
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
                            "-frames:v", "1", "-vf", f"scale={width}:-2",
                            "-f", "rawvideo", "-pix_fmt", fmt, "-"], capture_output=True)
        if r.returncode or not r.stdout:
            continue
        h = len(r.stdout) // (width * ch)
        if size is None:
            size = h
        if h == size:
            frame = np.frombuffer(r.stdout[: h * width * ch], np.uint8)
            frames.append(frame.reshape(h, width) if grey else frame.reshape(h, width, 3))
    if not frames:
        return np.zeros((0, 0, width) if grey else (0, 0, width, 3), np.uint8)
    return np.stack(frames)


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
def seam_profile(grey: np.ndarray) -> np.ndarray:
    """grey (N, rows, cols): typical grey-level jump between column i and i+1 (median
    over the rows, then over the frames). The overlay's border is a straight line where
    two unrelated pictures meet, so it shows up in every frame; edges in the game or in
    the room move around (or are faint) and mostly don't."""
    d = np.abs(np.diff(grey, axis=2))
    return np.median(np.median(d, axis=1), axis=0)


def _strength(seam: np.ndarray, i: int) -> float:
    """How much line i stands out from its surroundings."""
    return float(seam[i] - np.median(seam[max(0, i - LOCAL): i + LOCAL + 1]))


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel correlation over the frames (axis 0)."""
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    return (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0) + 1e-6)


def overlay_edge(grey: np.ndarray, start: int, stop: int) -> int | None:
    """grey (N, rows, cols), rows spanning the webcam. Walk from column `start` towards
    `stop` and return the first overlay edge as a box coordinate (exclusive end walking
    right, first webcam column walking left), or None if there isn't one.

    An edge is a straight line where the two sides change independently over the
    stream: webcam on one side, gameplay (or a translucent panel over gameplay) on the
    other. A desk or shelf edge in the room is a line too, but both sides stay still or
    change together with the light; the outer edge of a translucent chat panel is a
    line too, but it's the same gameplay on both sides, just dimmed. So walking out from
    the face, the first such line is the webcam's own edge, never a panel attached to it."""
    n = grey.shape[2]
    seam = seam_profile(grey)
    step = 1 if stop > start else -1
    for i in range(start, stop, step):
        if not (1 <= i < len(seam) - 1) or seam[i] < seam[i - 1] or seam[i] <= seam[i + 1]:
            continue
        if _strength(seam, i) < SEAM_MIN:
            continue
        inside, outside = (i - SEAM_GAP, i + 1 + SEAM_GAP) if step > 0 else (i + 1 + SEAM_GAP, i - SEAM_GAP)
        if not (0 <= inside < n and 0 <= outside < n):
            continue
        # each side's own detail: minus its brightness along the line in that frame, so a
        # room lit by the monitor doesn't seem to change with the gameplay next to it
        a, b = grey[:, :, inside], grey[:, :, outside]
        a = a - np.median(a, axis=1, keepdims=True)
        b = b - np.median(b, axis=1, keepdims=True)
        if np.median(b.std(axis=0)) < SEAM_MIN_STD:
            continue                        # nothing changes beyond it: still the room
        if np.median(_corr(a, b)) < SEAM_CORR:
            return i + 1
    return None


def cam_rect(grey: np.ndarray, face: Box) -> Box | None:
    """Overlay rectangle around the face, from grey analysis frames (N, H, W) in which
    the webcam is on screen."""
    _, height, width = grey.shape
    cx, cy = face.x + face.w // 2, face.y + face.h // 2
    reach = 8 * face.h
    band = grey[:, max(0, face.y - face.h // 2): face.y + 3 * face.h // 2]
    x1 = overlay_edge(band, face.x + face.w, min(width - 1, cx + reach)) or width
    x0 = overlay_edge(band, face.x - 1, max(0, cx - reach)) or 0
    cols = grey[:, :, x0 + (x1 - x0) // 10: x1 - (x1 - x0) // 10].transpose(0, 2, 1)
    y1 = overlay_edge(cols, face.y + face.h, min(height - 1, cy + reach)) or height
    y0 = overlay_edge(cols, face.y - 1, max(0, cy - reach)) or 0
    box = Box(x0, y0, x1 - x0, y1 - y0)
    # sanity: a real overlay holds the face, is clearly bigger than it, and isn't the whole screen
    if (box.w < 1.5 * face.w or box.h < 1.5 * face.h or box.w * box.h > 0.45 * width * height
            or not (x0 <= face.x and face.x + face.w <= x1 and y0 <= face.y and face.y + face.h <= y1)):
        return None
    return box


def refine_edge(seam: np.ndarray, guess: int, radius: int, outward: int) -> int:
    """Exact position of the overlay edge near box coordinate `guess`, at full
    resolution (`seam` from seam_profile). Returns the exclusive end (outward=+1) or the
    first webcam pixel (outward=-1). Pixels on the edge can be a blend of webcam and
    gameplay: every line around the peak that still jumps noticeably counts as edge,
    and the result stops before all of them."""
    lo, hi = max(0, guess - 1 - radius), min(len(seam), guess + radius)
    if hi <= lo:
        return guess
    i = lo + int(np.argmax(seam[lo:hi]))
    base = float(np.median(seam[lo:hi]))
    thr = base + 0.25 * (float(seam[i]) - base)
    a = b = i
    while a - 1 >= lo and seam[a - 1] > thr:
        a -= 1
    while b + 1 < hi and seam[b + 1] > thr:
        b += 1
    return a + 1 if outward > 0 else b + 1


def refine_box(grey: np.ndarray, coarse: Box, at_border: tuple[bool, bool, bool, bool],
               scale: float) -> Box:
    """Snap each edge of `coarse` (source pixels) to the real overlay edge, measured on
    full-resolution grey frames, then shave `inset` pixels so the edge pixel - a blend
    of webcam and gameplay - is never included. Edges on the frame border stay put."""
    _, src_h, src_w = grey.shape
    r = int(np.ceil(2 * scale)) + 3
    inset = max(1, src_h // 540)
    x0, y0, x1, y1 = coarse.x, coarse.y, coarse.x + coarse.w, coarse.y + coarse.h
    ys = slice(y0 + coarse.h // 10, y1 - coarse.h // 10)   # inner span, away from corners
    xs = slice(x0 + coarse.w // 10, x1 - coarse.w // 10)
    left, top, right, bottom = at_border

    def cols(c: int) -> tuple[np.ndarray, int]:
        a, b = max(0, c - r - 2), min(src_w, c + r + 2)
        return seam_profile(grey[:, ys, a:b]), a

    def rows(c: int) -> tuple[np.ndarray, int]:
        a, b = max(0, c - r - 2), min(src_h, c + r + 2)
        return seam_profile(grey[:, a:b, xs].transpose(0, 2, 1)), a

    nx0, ny0, nx1, ny1 = 0 if left else x0, 0 if top else y0, src_w if right else x1, src_h if bottom else y1
    if not left:
        prof, a = cols(x0)
        nx0 = a + refine_edge(prof, x0 - a, r, -1) + inset
    if not right:
        prof, a = cols(x1)
        nx1 = a + refine_edge(prof, x1 - a, r, +1) - inset
    if not top:
        prof, a = rows(y0)
        ny0 = a + refine_edge(prof, y0 - a, r, -1) + inset
    if not bottom:
        prof, a = rows(y1)
        ny1 = a + refine_edge(prof, y1 - a, r, +1) - inset
    return Box(nx0, ny0, nx1 - nx0, ny1 - ny0)


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
    faces = detect_faces(frames)
    face = face_cluster(faces, len(frames), width, height)
    if not face:
        return None
    grey = frames.astype(np.float32).mean(axis=3)
    # only the frames where the webcam is on screen (not BRB or full-screen scenes)
    fx, fy = face.x + face.w / 2, face.y + face.h / 2
    on = sorted({i for i, x, y, w, h in faces if np.hypot(x + w / 2 - fx, y + h / 2 - fy) <= 0.05 * width})
    if len(on) >= 5:
        grey = grey[on]
    small = cam_rect(grey, face)
    if not small:
        log(f"a face was found at {face} (analysis pixels), but no webcam overlay edges around it")
        return None
    src_w, _ = video_size(path)
    scale = src_w / width
    ref = np.median(grey[:, small.y:small.y + small.h, small.x:small.x + small.w], axis=0)
    # the 640-px analysis is only accurate to a few source pixels: re-measure the edges
    # on full-resolution frames so no gameplay shows at the panel's edges
    at_border = (small.x == 0, small.y == 0, small.x + small.w >= width, small.y + small.h >= height)
    full = grab_frames(path, times[::2], width=src_w, grey=True).astype(np.float32)
    box = small.scaled(scale)
    if len(full) >= 5:
        box = refine_box(full, box, at_border, scale)
    return Facecam(box, small, ref, scale)


def border_score(grey: np.ndarray, box: Box) -> list[float]:
    """For one grey frame: how sharply the picture breaks along each side of `box` that
    isn't on the frame border - the jump across the line, compared with the same
    distance just inside and just outside it. ~1 where the picture simply continues,
    well above where the overlay's edge is."""
    height, width = grey.shape
    g, d = SEAM_GAP, 2 * SEAM_GAP + 1
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    rs, cs = slice(y0 + box.h // 10, y1 - box.h // 10), slice(x0 + box.w // 10, x1 - box.w // 10)
    sides = []   # (inside line, outside line), each a 1-D array of pixels along the side
    if x1 + g + d < width:
        sides.append([grey[rs, x1 - 1 - g - d], grey[rs, x1 - 1 - g], grey[rs, x1 + g], grey[rs, x1 + g + d]])
    if x0 - 1 - g - d >= 0:
        sides.append([grey[rs, x0 + g + d], grey[rs, x0 + g], grey[rs, x0 - 1 - g], grey[rs, x0 - 1 - g - d]])
    if y1 + g + d < height:
        sides.append([grey[y1 - 1 - g - d, cs], grey[y1 - 1 - g, cs], grey[y1 + g, cs], grey[y1 + g + d, cs]])
    if y0 - 1 - g - d >= 0:
        sides.append([grey[y0 + g + d, cs], grey[y0 + g, cs], grey[y0 - 1 - g, cs], grey[y0 - 1 - g - d, cs]])
    out = []
    for in2, in1, out1, out2 in sides:
        across = np.median(np.abs(in1 - out1))
        beside = min(np.median(np.abs(in2 - in1)), np.median(np.abs(out1 - out2)))
        out.append(float(across / (beside + 2)))
    return out


def cam_visible(path: Path, cam: Facecam, start: float, length: float) -> bool:
    """Is the webcam overlay on screen during this clip? (not a BRB / full-screen scene)
    Yes if, on one of 3 frames, the cam area looks like usual, or the overlay's border
    is there - the latter also when the room looks different (e.g. the sun came up
    during a night stream, or a light was switched on)."""
    times = [start + length * f for f in (0.2, 0.5, 0.8)]
    frames = grab_frames(path, times)
    s = cam.small
    for f in frames:
        grey = f.astype(np.float32).mean(axis=2)
        area = grey[s.y:s.y + s.h, s.x:s.x + s.w]
        if area.shape == cam.reference.shape and np.median(np.abs(area - cam.reference)) < MATCH_DIFF:
            return True
        scores = border_score(grey, s)
        if scores and sum(v >= BORDER_SHOW for v in scores) * 2 >= len(scores):
            return True
    return False
