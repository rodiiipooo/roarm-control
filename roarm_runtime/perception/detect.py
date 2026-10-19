"""Perception: object detection + depth projection + hand/grip analysis +
head visual servoing + reachability estimation.

Design notes
------------
- Object detection: ultralytics YOLOv8 on hardware; a color-blob fallback
  (works with the SimCamera and with any distinctly colored object).
- Hand analysis: MediaPipe hand landmarks on hardware; a depth-blob fallback
  provides position only (grip classification needs landmarks).
- All 3D points are produced in the *camera* frame, then transformed to the
  robot *base* frame with extrinsics from config/extrinsics.py.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cameras import Frame


# ---------------------------------------------------------------- detections

@dataclass
class Detection:
    label: str
    uv: tuple[float, float]          # pixel centroid
    bbox: tuple[int, int, int, int]
    confidence: float
    xyz_cam: Optional[np.ndarray] = None   # 3D in camera frame (m)
    xyz_base: Optional[np.ndarray] = None  # 3D in robot base frame (m)


@dataclass
class HandObservation:
    present: bool
    xyz_base: Optional[np.ndarray] = None
    grip: str = "unknown"            # "pinch" | "open_palm" | "fist" | "unknown"
    grip_confidence: float = 0.0
    palm_normal: Optional[np.ndarray] = None   # approach vector hint


def uv_depth_to_xyz(u: float, v: float, depth_mm: np.ndarray, cam, win: int = 5) -> Optional[np.ndarray]:
    """Median depth in a small window → metric 3D point in camera frame."""
    h, w = depth_mm.shape
    u, v = int(u), int(v)
    x0, x1 = max(0, u - win), min(w, u + win + 1)
    y0, y1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size < 4:
        return None
    z = float(np.median(valid)) / 1000.0
    X = (u - cam.cx) * z / cam.fx
    Y = (v - cam.cy) * z / cam.fy
    return np.array([X, Y, z])


class ColorBlobDetector:
    """HSV color blob detector — reliable fallback and sim-compatible."""

    def __init__(self, hsv_lo=(100, 120, 100), hsv_hi=(130, 255, 255), min_area=300):
        self.lo, self.hi, self.min_area = hsv_lo, hsv_hi, min_area

    def detect(self, frame: Frame, cam) -> list[Detection]:
        import cv2
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(self.lo), np.array(self.hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            u, v = x + w / 2, y + h / 2
            out.append(Detection("object", (u, v), (x, y, w, h),
                                 confidence=min(1.0, area / 4000.0),
                                 xyz_cam=uv_depth_to_xyz(u, v, frame.depth, cam)))
        return out


class YoloDetector:
    """YOLOv8 object detection (ultralytics) — used on real hardware."""

    def __init__(self, weights: str = "yolov8n.pt", target_classes: tuple[str, ...] | None = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.target_classes = target_classes  # None → any class

    def detect(self, frame: Frame, cam) -> list[Detection]:
        res = self.model.predict(frame.rgb[:, :, ::-1], verbose=False)[0]
        out = []
        for b in res.boxes:
            cls = res.names[int(b.cls)]
            if self.target_classes and cls not in self.target_classes:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            out.append(Detection(cls, (u, v), (x1, y1, x2 - x1, y2 - y1),
                                 float(b.conf),
                                 uv_depth_to_xyz(u, v, frame.depth, cam)))
        return out


class HandAnalyzer:
    """MediaPipe hand landmarks → position + grip classification.

    Grip classification from landmark geometry:
      pinch     — thumb tip ↔ index tip distance small, other fingers folded
      open_palm — all fingers extended, wide spread
      fist      — all folded
    Depth at the wrist landmark gives the hand's 3D position.
    """

    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self._hands = None
        if not simulate:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=1, min_detection_confidence=0.6,
                min_tracking_confidence=0.5)

    def analyze(self, frame: Frame, cam) -> HandObservation:
        if self.simulate or self._hands is None:
            return self._sim_hand(frame, cam)
        res = self._hands.process(frame.rgb)
        if not res.multi_hand_landmarks:
            return HandObservation(present=False)
        lm = res.multi_hand_landmarks[0].landmark
        h, w = frame.depth.shape
        pts = np.array([[l.x * w, l.y * h] for l in lm])
        wrist = pts[0]
        xyz = uv_depth_to_xyz(wrist[0], wrist[1], frame.depth, cam)
        grip, conf, normal = self._classify(pts)
        return HandObservation(present=True, xyz_base=None, grip=grip,
                               grip_confidence=conf, palm_normal=normal,
                               **{}) if False else HandObservation(
            present=True, grip=grip, grip_confidence=conf, palm_normal=normal,
            xyz_base=None)  # xyz_base filled by tracker after extrinsics

    def _classify(self, pts: np.ndarray):
        def d(a, b): return float(np.linalg.norm(pts[a] - pts[b]))
        wrist = 0
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]
        ext = [d(t, wrist) > d(m, wrist) * 1.35 for t, m in zip(tips[1:], mcps[1:])]
        pinch_dist = d(4, 8) / (d(0, 9) + 1e-6)
        if pinch_dist < 0.35 and sum(ext[1:]) <= 1:
            return "pinch", 0.8, None
        if all(ext):
            # palm normal from wrist→index_mcp × wrist→pinky_mcp (2D approx)
            v1 = pts[5] - pts[0]; v2 = pts[17] - pts[0]
            n = np.array([v1[0] * v2[1] - v1[1] * v2[0], 0, 1.0])
            n = n / (np.linalg.norm(n) + 1e-9)
            return "open_palm", 0.85, n
        if sum(ext) == 0:
            return "fist", 0.7, None
        return "unknown", 0.3, None

    def _sim_hand(self, frame: Frame, cam) -> HandObservation:
        # sim camera draws the hand as a gray-brown blob; find it by color
        import cv2
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array((5, 20, 60)), np.array((30, 90, 200)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return HandObservation(present=False)
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return HandObservation(present=False)
        M = cv2.moments(c)
        u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
        xyz = uv_depth_to_xyz(u, v, frame.depth, cam)
        # alternate grip modes slowly in sim so both handover styles get exercised
        grip = "pinch" if int(time.time() / 8) % 2 == 0 else "open_palm"
        return HandObservation(present=True, grip=grip, grip_confidence=0.7,
                               xyz_base=None) if xyz is None else HandObservation(
            present=True, grip=grip, grip_confidence=0.7, xyz_base=None)
