"""Astra Pro camera access.

Real backend: Orbbec OpenNI2 SDK (already installed on your MINISFORUM) for
depth + RGB. Sim backend: synthetic frames so the whole stack runs without
hardware — same interface.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray            # HxWx3 uint8
    depth: np.ndarray          # HxW uint16, millimetres (0 = invalid)
    timestamp: float


class SimCamera:
    """Synthetic RGB-D source. For the head role the scene is anchored in the
    BASE frame and re-rendered through the live pan/tilt head pose (fed in via
    set_head_pose), so visual servoing actually converges. The arm role is a
    fixed workspace view."""

    def __init__(self, role: str = "head", width: int = 640, height: int = 480,
                 fps: int = 30, hfov_deg: float = 58.0):
        import math
        self.role, self.w, self.h, self.fps = role, width, height, fps
        self.hfov = math.radians(hfov_deg)
        self.fx = self.w / (2 * math.tan(self.hfov / 2))
        self.fy = self.fx
        self.cx, self.cy = self.w / 2, self.h / 2
        # scene anchored in BASE frame (metres): object on the table in front
        # of the robot, hand nearby
        self._obj_base = np.array([0.30, 0.05, 0.03])
        self._hand_base = np.array([0.28, -0.06, 0.14])
        self._t0 = time.time()
        self._running = False
        self._T_base_cam = np.eye(4)   # head pose, updated via set_head_pose
        self._arm_q: Optional[np.ndarray] = None  # live joint state, set_arm_q

    def set_head_pose(self, T_base_cam: np.ndarray) -> None:
        self._T_base_cam = np.asarray(T_base_cam, dtype=float)

    def set_arm_q(self, q) -> None:
        self._arm_q = np.asarray(q, dtype=float)

    def _base_to_cam(self, pt_base: np.ndarray) -> np.ndarray:
        T = np.linalg.inv(self._T_base_cam)
        return (T @ np.append(pt_base, 1.0))[:3]

    def start(self) -> bool:
        self._running = True
        return True

    def stop(self) -> None:
        self._running = False

    def grab(self) -> Optional[Frame]:
        if not self._running:
            return None
        t = time.time() - self._t0
        # scene drifts slowly in the BASE frame so head tracking has work to do
        obj_b = self._obj_base + np.array([0.03 * np.sin(0.4 * t),
                                           0.04 * np.sin(0.23 * t + 1.0), 0.0])
        hand_b = self._hand_base + np.array([0.02 * np.sin(0.9 * t),
                                             0.02 * np.sin(0.5 * t), 0.01 * np.sin(0.7 * t)])
        rgb = np.full((self.h, self.w, 3), (24, 28, 34), dtype=np.uint8)
        depth = np.full((self.h, self.w), 2000, dtype=np.uint16)  # background 2 m

        # transform base-frame scene points into this camera's live frame
        obj = self._base_to_cam(obj_b)
        hand = self._base_to_cam(hand_b)

        import cv2

        def draw_sphere(center_cam, radius_m, color):
            if center_cam[2] <= 0.05:
                return
            u = self.fx * center_cam[0] / center_cam[2] + self.cx
            v = self.fy * center_cam[1] / center_cam[2] + self.cy
            r_px = self.fx * radius_m / center_cam[2]
            cv2.circle(rgb, (int(u), int(v)), int(r_px), color, -1)
            yy, xx = np.mgrid[0:self.h, 0:self.w]
            mask = (xx - u) ** 2 + (yy - v) ** 2 <= r_px ** 2
            depth[mask] = int(center_cam[2] * 1000)

        def project(pt_cam):
            if pt_cam[2] <= 0.05:
                return None
            return (self.fx * pt_cam[0] / pt_cam[2] + self.cx,
                    self.fy * pt_cam[1] / pt_cam[2] + self.cy,
                    pt_cam[2])

        # the arm itself, rendered from the live joint state so both cameras
        # see it move (link chain via FK, in the base frame)
        if self._arm_q is not None:
            from ..kinematics.arm_model import link_points
            pts = link_points(self._arm_q)
            chain = ["base", "shoulder", "elbow", "wrist", "tcp"]
            uv = {k: project(self._base_to_cam(pts[k])) for k in chain}
            for a, b in zip(chain[:-1], chain[1:]):
                if uv[a] and uv[b]:
                    cv2.line(rgb, (int(uv[a][0]), int(uv[a][1])),
                             (int(uv[b][0]), int(uv[b][1])), (185, 195, 205), 7)
            for k in ("shoulder", "elbow", "wrist"):
                if uv[k]:
                    cv2.circle(rgb, (int(uv[k][0]), int(uv[k][1])), 6, (90, 200, 190), -1)
            if uv["tcp"]:
                cv2.circle(rgb, (int(uv["tcp"][0]), int(uv["tcp"][1])), 5, (230, 170, 60), -1)

        draw_sphere(obj, 0.035, (40, 160, 255))          # blue object
        draw_sphere(hand, 0.085, (140, 120, 110))        # hand blob
        return Frame(rgb=rgb, depth=depth, timestamp=time.time())


class AstraProCamera:
    """Real Orbbec Astra Pro via OpenNI2 (openni2 / primesense bindings).

    Depth comes straight off the sensor; RGB either via OpenNI color stream
    or the Astra's separate UVC webcam interface (/dev/videoN with OpenCV).
    """

    def __init__(self, role: str = "head", uri: str | None = None,
                 width: int = 640, height: int = 480):
        self.role, self.uri = role, uri
        self.w, self.h = width, height
        self.fx = 570.3     # Astra Pro approx intrinsics @640x480 — calibrate!
        self.fy = 570.3
        self.cx, self.cy = width / 2, height / 2
        self._dev = None
        self._depth_stream = None
        self._color_cap = None

    def start(self) -> bool:
        from openni import openni2  # on the MINISFORUM
        import cv2
        openni2.initialize()
        self._dev = openni2.Device.open_any() if self.uri is None else openni2.Device.open_file(self.uri)
        self._depth_stream = self._dev.create_depth_stream()
        self._depth_stream.start()
        # Astra Pro RGB is a separate UVC device — pick by role order
        idx = 0 if self.role == "head" else 2
        self._color_cap = cv2.VideoCapture(idx)
        self._color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self._color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        return True

    def stop(self) -> None:
        if self._depth_stream:
            self._depth_stream.stop()
        if self._color_cap:
            self._color_cap.release()

    def grab(self) -> Optional[Frame]:
        import cv2
        from openni import openni2
        if self._depth_stream is None:
            return None
        buf = self._depth_stream.read_frame().get_buffer_as_uint16()
        depth = np.frombuffer(buf, dtype=np.uint16).reshape(self.h, self.w)
        ok, rgb = (False, None)
        if self._color_cap:
            ok, bgr = self._color_cap.read()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else np.zeros((self.h, self.w, 3), np.uint8)
        return Frame(rgb=rgb, depth=depth, timestamp=time.time())


def make_camera(role: str, simulate: bool):
    return SimCamera(role=role) if simulate else AstraProCamera(role=role)
