"""Runtime orchestrator — owns the hardware drivers, perception coordinator,
planner, and executor; exposes one clean facade to the API layer."""
from __future__ import annotations

import numpy as np

from .capabilities import DEFAULT_PROFILE, CapabilityProfile
from .config import RuntimeCfg, load as load_cfg
from .hal.drivers import SimArmDriver, SerialArmDriver, HeadDriver, ArmDriver
from .perception.coordinator import PerceptionCoordinator
from .policy.planner import KimiPlanner
from .policy.executor import Executor, ExecutionReport
from .policy.schemas import Policy


class RobotRuntime:
    def __init__(self, cfg: RuntimeCfg | None = None,
                 profile: CapabilityProfile = DEFAULT_PROFILE):
        self.cfg = cfg or load_cfg()
        self.p = profile
        if self.cfg.simulate:
            self.arm: ArmDriver = SimArmDriver(profile)
        elif self.cfg.r2s.enabled:
            # drive the arm through the user's proven roarm_follow stack (ZMQ)
            from .hal.r2s_bridge import ZmqArmPublisher
            self.arm = ZmqArmPublisher(self.cfg.r2s.joint_endpoint, profile)
        else:
            self.arm = SerialArmDriver(self.cfg.serial.arm_port,
                                       self.cfg.serial.baud, profile)
        self.head = HeadDriver(simulate=self.cfg.simulate)
        self.perc = PerceptionCoordinator(simulate=self.cfg.simulate,
                                          profile=profile,
                                          use_yolo=self.cfg.use_yolo)
        self.perc.head = self.head
        self.perc.arm = self.arm    # lets the sim scene render the live arm
        self.planner = KimiPlanner(profile=profile)
        self.executor = Executor(self.arm, self.head, self.perc, profile)
        self.started = False

    def start(self) -> None:
        self.arm.connect()
        self.arm.enable_torque(True)
        self.perc.start()
        self.started = True

    def stop(self) -> None:
        self.perc.stop()
        self.arm.enable_torque(False)
        self.arm.disconnect()
        self.started = False

    # -------------------------------------------------------------- facade
    def prompt(self, text: str) -> dict:
        world = self.perc.get_world()
        policy, note = self.planner.plan(text, world)
        rep = self.executor.execute(policy)
        return {"policy": policy.model_dump(), "planner_note": note,
                "execution": {"ok": rep.ok, "detail": rep.detail}}

    def teleop_joint(self, idx: int, delta_rad: float, speed_scale: float = 0.4) -> dict:
        q = self.arm.get_state().q.copy()
        q[idx] = self.p.clamp_joint(idx, q[idx] + float(delta_rad))
        self.arm.command_joints(q, speed_scale)
        return {"q": self.arm.get_state().q.tolist()}

    def teleop_cartesian(self, dx=0.0, dy=0.0, dz=0.0, speed_scale: float = 0.4) -> dict:
        from .kinematics.ik import ik_solve
        st = self.arm.get_state()
        tgt = st.tcp_xyz + np.array([dx, dy, dz])
        ok, why = self.p.in_workspace(*tgt)
        if not ok:
            return {"error": f"workspace guard: {why}"}
        q, msg = ik_solve(tgt, st.q, self.p)
        if q is None:
            return {"error": msg}
        q[5] = st.q[5]
        self.arm.command_joints(q, speed_scale)
        return {"tcp": tgt.tolist()}

    def estop(self) -> None:
        self.executor.cancel()
        self.arm.estop()

    def clear_estop(self) -> None:
        self.arm.clear_estop()

    def status(self) -> dict:
        st = self.arm.get_state()
        w = self.perc.get_world()
        return {
            "driver": self.arm.name,
            "simulate": self.cfg.simulate,
            "estopped": self.arm.estopped,
            "busy": self.executor.busy,
            "q": [round(float(v), 4) for v in st.q],
            "tcp_xyz": [round(float(v), 4) for v in st.tcp_xyz],
            "moving": st.moving,
            "torques": st.torques_enabled,
            "world": w.to_dict(),
            "handover_phase": getattr(self.executor, "last_report", None) and self.executor.last_report.detail,
            "stream_warnings": dict(self.perc.stream_warnings),
            "cameras_ok": self.perc.frames_ok(),
        }
