"""Policy executor: turns a validated Policy into constraint-checked motion.

Runs as the single writer to the arm driver — all motion goes through here,
so capability gating and E-stop semantics live in exactly one place.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .schemas import (Policy, ReachPolicy, HandoverPolicy, TrackPolicy,
                      WavePolicy, HomePolicy, GripperPolicy, IdlePolicy)
from ..capabilities import CapabilityProfile, DEFAULT_PROFILE
from ..hal.drivers import ArmDriver, HeadDriver
from ..kinematics.ik import ik_solve, trapezoid_trajectory, validate_trajectory
from ..kinematics.arm_model import forward_kinematics
from ..perception.coordinator import PerceptionCoordinator


@dataclass
class ExecutionReport:
    ok: bool
    detail: str
    policy_kind: str = ""


class Executor:
    HOME_Q = np.array([0.0, 0.9, 1.1, 0.7, 0.0, 0.2])

    def __init__(self, arm: ArmDriver, head: HeadDriver,
                 perception: PerceptionCoordinator,
                 profile: CapabilityProfile = DEFAULT_PROFILE):
        self.arm, self.head, self.perc, self.p = arm, head, perception, profile
        self._busy = threading.Lock()
        self._cancel = threading.Event()
        self.last_report: Optional[ExecutionReport] = None

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------- dispatch
    def execute(self, policy: Policy) -> ExecutionReport:
        if self.arm.estopped:
            rep = ExecutionReport(False, "E-stop active — clear it first", policy.kind)
            self.last_report = rep
            return rep
        if not self._busy.acquire(blocking=False):
            return ExecutionReport(False, "executor busy with another policy", policy.kind)
        self._cancel.clear()
        try:
            handler = {
                "reach": self._do_reach, "handover": self._do_handover,
                "track": self._do_track, "wave": self._do_wave,
                "home": self._do_home, "gripper": self._do_gripper,
                "idle": self._do_idle,
            }[policy.kind]
            rep = handler(policy)
        except Exception as e:
            rep = ExecutionReport(False, f"executor fault: {e}", policy.kind)
        finally:
            self._busy.release()
        self.last_report = rep
        return rep

    # ------------------------------------------------------------- policies
    def _do_idle(self, pol: IdlePolicy) -> ExecutionReport:
        return ExecutionReport(True, pol.reason or "idle", pol.kind)

    def _do_home(self, pol: HomePolicy) -> ExecutionReport:
        return self._move_joints(self.HOME_Q, 0.5, "homed")

    def _do_gripper(self, pol: GripperPolicy) -> ExecutionReport:
        q = self.arm.get_state().q.copy()
        q[5] = self.p.joints[5].max_rad * 0.95 if pol.action == "open" else 0.25
        return self._move_joints(q, 0.8, f"gripper {pol.action}")

    def _do_wave(self, pol: WavePolicy) -> ExecutionReport:
        q_home = self.HOME_Q.copy()
        q_hi = np.array([0.0, 1.0, 1.2, 0.5, 0.0, 0.2])
        rep = self._move_joints(q_hi, pol.speed_scale, "raise")
        if not rep.ok:
            return rep
        for _ in range(pol.cycles):
            for roll in (0.5, -0.5):
                q = q_hi.copy(); q[4] = roll
                rep = self._move_joints(q, pol.speed_scale, "wave")
                if not rep.ok:
                    return rep
        return self._move_joints(q_home, pol.speed_scale, "wave done")

    def _do_track(self, pol: TrackPolicy) -> ExecutionReport:
        self.perc.tracking_enabled = True
        t0 = time.time()
        while time.time() - t0 < pol.duration_s:
            if self._cancel.is_set():
                return ExecutionReport(False, "track cancelled", pol.kind)
            time.sleep(0.1)
        return ExecutionReport(True, f"tracked for {pol.duration_s:.0f}s", pol.kind)

    def _do_reach(self, pol: ReachPolicy) -> ExecutionReport:
        target = self._resolve_target(pol)
        if target is None:
            return ExecutionReport(False, "no valid target in perception", pol.kind)
        return self._reach_xyz(target, pol.speed_scale, pol.kind)

    def _do_handover(self, pol: HandoverPolicy) -> ExecutionReport:
        from .handover import HandoverStateMachine
        sm = HandoverStateMachine(self.arm, self.perc, self.p, self._cancel)
        return sm.run(grasp_first=pol.grasp_first, speed_scale=pol.speed_scale)

    # ------------------------------------------------------------- helpers
    def _resolve_target(self, pol: ReachPolicy) -> Optional[np.ndarray]:
        w = self.perc.get_world()
        if pol.target == "point" and pol.xyz:
            return np.array(pol.xyz)
        if pol.target == "object" and w.object_xyz:
            return np.array(w.object_xyz)
        if pol.target == "hand" and w.hand_present and w.hand_xyz:
            return np.array(w.hand_xyz)
        return None

    def _reach_xyz(self, xyz: np.ndarray, speed_scale: float, kind: str) -> ExecutionReport:
        ok, why = self.p.in_workspace(*xyz)
        if not ok:
            return ExecutionReport(False, f"capability profile rejected target: {why}", kind)
        q0 = self.arm.get_state().q
        q_sol, msg = ik_solve(xyz, q0, self.p)
        if q_sol is None:
            return ExecutionReport(False, f"IK failed: {msg}", kind)
        return self._move_joints(q_sol, speed_scale, f"reached [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")

    def _move_joints(self, q_target: np.ndarray, speed_scale: float, label: str) -> ExecutionReport:
        q0 = self.arm.get_state().q
        qs = trapezoid_trajectory(q0, q_target, self.p, speed_scale)
        ok, why = validate_trajectory(qs, self.p)
        if not ok:
            return ExecutionReport(False, f"trajectory rejected: {why}")
        dt = 1.0 / self.p.bus_rate_hz
        for q in qs:
            if self._cancel.is_set():
                return ExecutionReport(False, f"{label} cancelled")
            if self.arm.estopped:
                return ExecutionReport(False, "E-stop during motion")
            self.arm.command_joints(q, speed_scale)
            time.sleep(dt)
        return ExecutionReport(True, label)
