"""Perception coordinator: runs both cameras, fuses detections into base
frame, drives the pan/tilt head to track the target, and answers the two
questions the policy engine cares about:

  1. Where is the object / hand right now (base frame)?
  2. Is it within reach?
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .cameras import make_camera, Frame
from .detect import (ColorBlobDetector, YoloDetector, HandAnalyzer,
                     Detection, HandObservation)
from .extrinsics import load_extrinsics, cam_to_base, head_pose
from ..capabilities import CapabilityProfile, DEFAULT_PROFILE


@dataclass
class WorldState:
    timestamp: float = 0.0
    object_xyz: Optional[list[float]] = None      # base frame
    object_label: Optional[str] = None
    object_confidence: float = 0.0
    hand_present: bool = False
    hand_xyz: Optional[list[float]] = None
    hand_grip: str = "unknown"
    hand_grip_confidence: float = 0.0
    reachable: bool = False
    reach_reason: str = "no target"
    head: dict = field(default_factory=lambda: {"pan": 0.0, "tilt": 0.0})
    fps_head: float = 0.0

    def to_dict(self):
        return asdict(self)


class PerceptionCoordinator:
    def __init__(self, simulate: bool = True,
                 profile: CapabilityProfile = DEFAULT_PROFILE,
                 use_yolo: bool = False):
        self.simulate = simulate
        self.p = profile
        self.head_cam = make_camera("head", simulate)
        self.arm_cam = make_camera("arm", simulate)
        if simulate or not use_yolo:
            self.detector = ColorBlobDetector()
        else:
            self.detector = YoloDetector()
        self.hand = HandAnalyzer(simulate=simulate)
        self.extrinsics = load_extrinsics()
        self.head = None            # HeadDriver, injected by runtime
        self.arm = None             # ArmDriver, injected by runtime (sim scene)
        self._obj_hold = None       # (xyz, label, conf, expiry) loss hysteresis
        self.world = WorldState()
        self.tracking_enabled = True
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frames: dict[str, Optional[Frame]] = {"head": None, "arm": None}
        self.stream_warnings: dict[str, str] = {}   # watchdog → /api/status

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        # fail fast: a camera that can't start must crash startup with a
        # clear message instead of serving a black feed forever
        try:
            if not self.head_cam.start():
                raise RuntimeError("head camera start() returned False")
            if not self.arm_cam.start():
                raise RuntimeError("arm camera start() returned False")
        except Exception as e:
            raise RuntimeError(
                f"camera startup failed: {type(e).__name__}: {e} — "
                f"check USB cables and that no other app holds the device") from e
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._diagnose_boot()

    def frames_ok(self) -> dict[str, bool]:
        """True per role once a real frame has been captured and encoded."""
        ok = {}
        for role in ("head", "arm"):
            f = self._frames.get(role)
            ok[role] = f is not None and f.rgb.size > 0
        return ok

    def _diagnose_boot(self) -> None:
        """Log the mounted extrinsics at startup so a stale placeholder
        config is visible in the log instead of silently misplacing points."""
        import logging
        log = logging.getLogger("roarm.perception")
        for role in ("head", "arm"):
            T = self.extrinsics[role]
            pos = T[:3, 3].round(3).tolist()
            view = T[:3, 2].round(3).tolist()
            log.info("%s cam mount: pos=%s view_dir=%s", role, pos, view)
        log.info("extrinsics source: config/extrinsics.json (placeholders "
                 "until ChArUco hand-eye calibration is run)")

    def stop(self) -> None:
        self._running = False
        self.head_cam.stop()
        self.arm_cam.stop()

    def latest_jpegs(self) -> dict[str, Optional[bytes]]:
        import cv2
        out = {}
        for role in ("head", "arm"):
            f = self._frames.get(role)
            if f is None:
                out[role] = None
                continue
            bgr = cv2.cvtColor(self._annotate(role, f), cv2.COLOR_RGB2BGR)
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            out[role] = jpg.tobytes() if ok else None
        return out

    def _annotate(self, role: str, f: Frame) -> np.ndarray:
        import cv2
        img = f.rgb.copy()
        if role == "head" and self.world.object_xyz is not None:
            # overlay reachability + grip state (bottom-left, clear of UI chips)
            txt = f"reach: {'YES' if self.world.reachable else 'NO'} ({self.world.reach_reason})"
            cv2.putText(img, txt, (8, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (60, 220, 120) if self.world.reachable else (230, 120, 60), 1)
            if self.world.hand_present:
                cv2.putText(img, f"hand: {self.world.hand_grip}", (8, img.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 80), 1)
        return img

    # ------------------------------------------------------------- main loop
    def _loop(self) -> None:
        last_t, frames_n, fps = time.time(), 0, 0.0
        while self._running:
            # 0. feed the live head pose into the sim camera so the scene
            #    renders through the actual pan/tilt orientation
            head_st = self.head.state() if self.head else {"pan": 0.0, "tilt": 0.0}
            mount = self.extrinsics["head"]
            T_head_live = head_pose(mount, head_st["pan"], head_st["tilt"])
            if self.simulate:
                if hasattr(self.head_cam, "set_head_pose"):
                    self.head_cam.set_head_pose(T_head_live)
                if hasattr(self.arm_cam, "set_head_pose"):
                    self.arm_cam.set_head_pose(self.extrinsics["arm"])  # fixed mount
                # feed live joint state so the sim renders the arm itself
                if self.arm is not None:
                    try:
                        q_now = self.arm.get_state().q
                        self.head_cam.set_arm_q(q_now)
                        self.arm_cam.set_arm_q(q_now)
                    except Exception:
                        pass

            hf = self.head_cam.grab()
            af = self.arm_cam.grab()
            if hf is None:
                time.sleep(0.01)
                continue
            self._frames["head"] = hf
            if af is not None:
                self._frames["arm"] = af

            # 1. object detection on head cam
            try:
                dets = self.detector.detect(hf, self.head_cam)
            except Exception as e:
                self._dbg = getattr(self, "_dbg", {"n": 0, "det": 0, "err": None})
                self._dbg["err"] = f"{type(e).__name__}: {e}"
                dets = []
            obj = max(dets, key=lambda d: d.confidence, default=None)
            self._dbg = getattr(self, "_dbg", {"n": 0, "det": 0, "err": None})
            self._dbg["n"] += 1
            self._dbg["det"] += 1 if obj else 0

            # 2. hand analysis
            hand = self.hand.analyze(hf, self.head_cam)

            # 3. camera frame → base frame, using the LIVE head pose
            obj_base = cam_to_base(obj.xyz_cam, T_head_live) if obj else None
            hand_base = None
            if hand.present:
                # HandAnalyzer gives 2D + depth → xyz_cam (sim: blob; hardware:
                # wrist landmark depth). Lift to base frame here.
                import cv2
                hsv = cv2.cvtColor(hf.rgb, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, np.array((5, 20, 60)), np.array((30, 90, 200)))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        from .detect import uv_depth_to_xyz
                        hu, hv = M["m10"] / M["m00"], M["m01"] / M["m00"]
                        hand_base = cam_to_base(uv_depth_to_xyz(hu, hv, hf.depth, self.head_cam),
                                                T_head_live)

            # 3b. target-loss hysteresis: keep the last fix for a short window
            #     so world state doesn't flicker between tracker updates
            now = time.time()
            if obj_base is not None:
                self._obj_hold = (obj_base, obj.label, obj.confidence, now + 0.6)
            elif self._obj_hold is not None and now < self._obj_hold[3]:
                obj_base = self._obj_hold[0]
            else:
                self._obj_hold = None
            obj_label = obj.label if obj else (self._obj_hold[1] if self._obj_hold else None)
            obj_conf = obj.confidence if obj else (self._obj_hold[2] if self._obj_hold else 0.0)

            # 4. head tracking (visual servoing): keep object centred when seen,
            #    else run a slow search scan so the workspace gets covered.
            #    pan > 0 turns left, so a right-of-centre blob needs pan MINUS.
            if self.head is not None:
                if self.tracking_enabled and obj is not None:
                    u, v = obj.uv
                    ex = (u - self.head_cam.cx) / self.head_cam.cx
                    ey = (v - self.head_cam.cy) / self.head_cam.cy
                    self.head.set(head_st["pan"] - 0.45 * ex,
                                  head_st["tilt"] + 0.45 * ey)
                elif obj is None and obj_base is None:
                    # search: the workspace sits ~13° of servo-tilt below the
                    # 35° mount pitch, slightly left (pan ~+8°) of centre
                    t = time.time()
                    scan_tilt = 0.20 + 0.25 * math.sin(0.6 * t)    # total 33°..61° down
                    scan_pan = 0.15 + 0.50 * math.sin(0.28 * t)    # ±29° around +9°
                    self.head.set(scan_pan, scan_tilt)

            # 5. reachability: workspace gate + IK feasibility probe
            reachable, reason = False, "no target"
            if obj_base is not None:
                ok, why = self.p.in_workspace(*obj_base)
                reachable, reason = (True, "in workspace") if ok else (False, why)

            with self._lock:
                self.world = WorldState(
                    timestamp=time.time(),
                    object_xyz=obj_base.tolist() if obj_base is not None else None,
                    object_label=obj_label,
                    object_confidence=obj_conf,
                    hand_present=hand.present,
                    hand_xyz=hand_base.tolist() if isinstance(hand_base, np.ndarray) else None,
                    hand_grip=hand.grip,
                    hand_grip_confidence=hand.grip_confidence,
                    reachable=reachable,
                    reach_reason=reason,
                    head=self.head.state() if self.head else {},
                    fps_head=fps,
                )
            frames_n += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = frames_n / (now - last_t)
                frames_n, last_t = 0, now
            time.sleep(0.005)

    def get_world(self) -> WorldState:
        with self._lock:
            return self.world
