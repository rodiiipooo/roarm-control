"""Handover state machine — grip-aware object handoff between robot and human.

Sequences (both directions handled):

  grasp_first=True (robot gives object to human):
    PERCEIVE → GRASP_OBJECT → LIFT → WAIT_HAND → APPROACH_HAND
    (grip-aware) → RELEASE → RETREAT

  grasp_first=False (human gives object to robot / robot takes from hand):
    WAIT_HAND → CLASSIFY_GRIP → APPROACH (grip-aware) → GRASP → RETREAT

Grip-adaptive approach (from HandAnalyzer classification):
  pinch     — human holds object between fingers: approach from ABOVE along
              the hand's vertical axis, gripper vertical, descend and close.
  open_palm — object resting in palm: approach from the SIDE/front at palm
              height, wrist pitched to keep gripper jaws parallel to palm
              plane, close gently.
  unknown   — conservative: hover at approach distance, do not close.

Safety: every waypoint goes through the capability profile; motion slows to
handover_max_speed whenever the hand is within person_min_clearance of the
planned path; any E-stop or cancel aborts immediately.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from ..capabilities import CapabilityProfile, DEFAULT_PROFILE
from ..hal.drivers import ArmDriver
from ..kinematics.ik import ik_solve, trapezoid_trajectory, validate_trajectory
from ..perception.coordinator import PerceptionCoordinator


class Phase(Enum):
    PERCEIVE = auto()
    GRASP_OBJECT = auto()
    LIFT = auto()
    WAIT_HAND = auto()
    CLASSIFY_GRIP = auto()
    APPROACH = auto()
    RELEASE = auto()
    GRASP = auto()
    RETREAT = auto()
    DONE = auto()
    ABORTED = auto()


GRIPPER_OPEN = 1.85
GRIPPER_CLOSED = 0.25
GRIPPER_GENTLE = 0.55     # partial close for delicate handover


class HandoverStateMachine:
    def __init__(self, arm: ArmDriver, perc: PerceptionCoordinator,
                 profile: CapabilityProfile = DEFAULT_PROFILE,
                 cancel: threading.Event | None = None):
        self.arm, self.perc, self.p = arm, perc, profile
        self.cancel = cancel or threading.Event()
        self.phase = Phase.PERCEIVE
        self.note = ""

    # ---------------------------------------------------------------- run
    def run(self, grasp_first: bool, speed_scale: float) -> "ExecutionReport":
        from .executor import ExecutionReport
        ss = min(speed_scale, self.p.handover_max_speed_m_s / self.p.tcp_max_speed_m_s)
        try:
            if grasp_first:
                steps = [self._perceive_object, self._grasp_object,
                         self._lift, self._wait_hand, self._classify_grip,
                         lambda ss: self._approach(give=True), self._release,
                         self._retreat]
            else:
                steps = [self._wait_hand, self._classify_grip,
                         lambda ss: self._approach(give=False), self._grasp_from_hand,
                         self._retreat]
            for step in steps:
                if self.cancel.is_set() or self.arm.estopped:
                    self.phase = Phase.ABORTED
                    return ExecutionReport(False, f"aborted in {self.phase.name}", "handover")
                err = step(ss)
                if err:
                    self.phase = Phase.ABORTED
                    return ExecutionReport(False, f"{self.phase.name}: {err}", "handover")
            self.phase = Phase.DONE
            return ExecutionReport(True, "handover complete", "handover")
        except Exception as e:
            self.phase = Phase.ABORTED
            return ExecutionReport(False, f"handover fault: {e}", "handover")

    # ------------------------------------------------------------- phases
    def _perceive_object(self, ss) -> Optional[str]:
        self.phase = Phase.PERCEIVE
        deadline = time.time() + 5.0
        while time.time() < deadline:
            w = self.perc.get_world()
            if w.object_xyz is not None and w.reachable:
                self._obj = np.array(w.object_xyz)
                return None
            if w.object_xyz is not None:
                return f"object seen but not reachable: {w.reach_reason}"
            time.sleep(0.1)
        return "no object perceived within 5 s"

    def _grasp_object(self, ss) -> Optional[str]:
        self.phase = Phase.GRASP_OBJECT
        pre = self._obj + np.array([0.0, 0.0, self.p.handover_approach_dist_m])
        err = self._goto(pre, ss)
        if err: return err
        err = self._set_gripper(GRIPPER_OPEN)
        if err: return err
        err = self._goto(self._obj, ss * 0.6)          # slow final descent
        if err: return err
        return self._set_gripper(GRIPPER_CLOSED)

    def _lift(self, ss) -> Optional[str]:
        self.phase = Phase.LIFT
        lift = self._obj + np.array([0.0, 0.0, 0.12])
        return self._goto(lift, ss)

    def _wait_hand(self, ss) -> Optional[str]:
        self.phase = Phase.WAIT_HAND
        deadline = time.time() + 15.0
        while time.time() < deadline:
            w = self.perc.get_world()
            if w.hand_present and w.hand_xyz is not None:
                self._hand = np.array(w.hand_xyz)
                return None
            if self.cancel.is_set():
                return "cancelled while waiting for hand"
            time.sleep(0.1)
        return "no hand detected within 15 s"

    def _classify_grip(self, ss) -> Optional[str]:
        self.phase = Phase.CLASSIFY_GRIP
        # settle: sample grip classification over ~1 s, take the majority
        votes = {}
        t0 = time.time()
        while time.time() - t0 < 1.0:
            w = self.perc.get_world()
            if w.hand_present:
                votes[w.hand_grip] = votes.get(w.hand_grip, 0) + 1
                if w.hand_xyz is not None:
                    self._hand = np.array(w.hand_xyz)
            time.sleep(0.1)
        self._grip = max(votes, key=votes.get) if votes else "unknown"
        return None

    def _approach(self, give: bool) -> "Optional[str]":
        self.phase = Phase.APPROACH
        hand = self._hand
        ok, why = self.p.in_workspace(*hand)
        if not ok:
            return f"hand position rejected: {why}"

        if self._grip == "pinch":
            # come from above, gripper vertical, stop approach_dist above object/hand
            target = hand + np.array([0.0, 0.0, self.p.handover_approach_dist_m])
            pre = target + np.array([0.0, 0.0, 0.08])
        elif self._grip == "open_palm":
            # come from the front at palm height, wrist level with palm plane
            target = hand + np.array([-self.p.handover_approach_dist_m, 0.0, 0.03])
            pre = target + np.array([-0.08, 0.0, 0.02])
        else:
            # unknown grip → hover at a safe standoff, do not close/release
            target = hand + np.array([-self.p.person_min_clearance_m, 0.0, 0.10])
            pre = target + np.array([-0.06, 0.0, 0.06])

        err = self._goto(pre, min(0.4, self.p.handover_max_speed_m_s / self.p.tcp_max_speed_m_s) * 2)
        if err: return err
        err = self._goto(target, self.p.handover_max_speed_m_s / self.p.tcp_max_speed_m_s)
        if err: return err
        if self._grip == "unknown":
            self._hover_only = True
        else:
            self._hover_only = False
        return None

    def _release(self, ss) -> Optional[str]:
        self.phase = Phase.RELEASE
        if getattr(self, "_hover_only", False):
            time.sleep(1.0)                # hold position, let the human take it
        return self._set_gripper(GRIPPER_OPEN)

    def _grasp_from_hand(self, ss) -> Optional[str]:
        self.phase = Phase.GRASP
        if getattr(self, "_hover_only", False):
            return "grip style unclear — refusing to close near a hand"
        return self._set_gripper(GRIPPER_GENTLE)

    def _retreat(self, ss) -> Optional[str]:
        self.phase = Phase.RETREAT
        q = self.arm.get_state().q
        home = np.array([0.0, -0.6, 1.4, 0.8, q[4], q[5]])
        return self._goto_q(home, ss)

    # ------------------------------------------------------------ motion io
    def _goto(self, xyz: np.ndarray, speed_scale: float) -> Optional[str]:
        q0 = self.arm.get_state().q
        q, msg = ik_solve(np.asarray(xyz, dtype=float), q0, self.p)
        if q is None:
            return f"IK failed: {msg}"
        return self._goto_q(q, speed_scale)

    def _goto_q(self, q_target: np.ndarray, speed_scale: float) -> Optional[str]:
        q0 = self.arm.get_state().q
        qs = trapezoid_trajectory(q0, q_target, self.p, speed_scale)
        ok, why = validate_trajectory(qs, self.p)
        if not ok:
            return f"trajectory rejected: {why}"
        dt = 1.0 / self.p.bus_rate_hz
        for q in qs:
            if self.cancel.is_set() or self.arm.estopped:
                return "aborted mid-motion"
            self.arm.command_joints(q, speed_scale)
            time.sleep(dt)
        return None

    def _set_gripper(self, pos: float) -> Optional[str]:
        q = self.arm.get_state().q.copy()
        q[5] = pos
        self.arm.command_joints(q, 0.8)
        time.sleep(0.4)
        return None
