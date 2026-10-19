"""r2s bridge: drive the real RoArm-M3 through the user's proven stack instead
of touching the serial bus ourselves.

Architecture on the live rig (all on the MINISFORUM):

    THIS PROGRAM  --JointMessage v1 (ZMQ PUB)-->  tcp://127.0.0.1:5555
                                                       |
                                   scripts/roarm_follow.py (SUB)
                                   - joint-limit clamp + trapezoid/Ruckig smoothing
                                   - motion gate (R2S_ROARM_ENABLE_MOTION=1)
                                   - dead-port / pose-mismatch watchdogs
                                                       |
                                                 RoArm-M3 on COM3

So this program never opens the serial port: it publishes desired joint
angles and your existing follower owns the hardware, exactly like your
Astra teleop does. Handover execution can likewise delegate to your guarded
scripts/palm_grasp.py control socket (ZMQ REP JSON).

Message contract (from scripts/isaac_bridge_core.py):
    JointMessage(t: float, joint_names: [str], q: [float], gripper: float,
                 source: str, schema_version: 1, frame: ..., extra: {...})
    M3 native names: R_base R_shoulder R_elbow (+ wrist/roll via extra) R_gripper
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1
M3_NAMES = ["R_base", "R_shoulder", "R_elbow", "R_wrist_pitch", "R_wrist_roll", "R_gripper"]


@dataclass
class JointMessage:
    t: float
    joint_names: list
    q: list
    gripper: float = 0.0
    source: str = "roarm_prompt_gui"
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "t": self.t, "joint_names": self.joint_names, "q": [float(v) for v in self.q],
            "gripper": float(self.gripper), "source": self.source,
            "schema_version": self.schema_version, "extra": self.extra,
        })


class ZmqArmPublisher:
    """Publishes 6-DOF joint targets for roarm_follow. Keeps an ArmDriver-
    compatible interface so the Executor can target it transparently."""

    name = "r2s-zmq"

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555",
                 profile=None):
        from ..capabilities import DEFAULT_PROFILE
        self.p = profile or DEFAULT_PROFILE
        self.endpoint = endpoint
        self._ctx = None
        self._sock = None
        self._q = np.zeros(self.p.n_joints())
        self._estopped = False
        self._connected = False

    # ---- ArmDriver interface ----
    def connect(self) -> bool:
        import zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.connect(self.endpoint)   # follower binds; we connect
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._sock:
            self._sock.close(0)
        if self._ctx:
            self._ctx.term()
        self._connected = False

    def enable_torque(self, on: bool) -> None:
        pass   # follower owns torque

    @property
    def estopped(self) -> bool:
        return self._estopped

    def estop(self) -> None:
        # stop streaming new targets; follower's watchdog holds the arm
        self._estopped = True

    def clear_estop(self) -> None:
        self._estopped = False

    def get_state(self):
        from .drivers import ArmState
        from ..kinematics.arm_model import forward_kinematics
        tcp = forward_kinematics(self._q)[0:3, 3]
        return ArmState(q=self._q.copy(), dq=np.zeros(self.p.n_joints()),
                        tcp_xyz=tcp, moving=False, updated_at=time.time())

    def command_joints(self, q_target: np.ndarray, speed_scale: float = 1.0) -> None:
        if self._estopped:
            raise RuntimeError("E-stop active")
        q = np.asarray(q_target, dtype=float).copy()
        for i in range(self.p.n_joints()):
            q[i] = self.p.clamp_joint(i, q[i])
        self._q = q
        msg = JointMessage(
            t=time.time(),
            joint_names=M3_NAMES[:5],
            q=[float(v) for v in q[:5]],
            gripper=float(q[5]),
            extra={"speed_scale": float(speed_scale)},
        )
        if self._sock:
            self._sock.send_string(msg.to_json())


class PalmGraspClient:
    """Delegates handover execution to scripts/palm_grasp.py over its ZMQ REP
    control socket — the guarded, extrinsic-gated path you already trust."""

    def __init__(self, ctl_endpoint: str = "tcp://127.0.0.1:5560"):
        self.ctl = ctl_endpoint
        self._ctx = None
        self._sock = None

    def connect(self) -> bool:
        import zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, 2000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.ctl)
        return True

    def _cmd(self, cmd: str, **kw) -> dict:
        if not self._sock:
            self.connect()
        self._sock.send_string(json.dumps({"cmd": cmd, **kw}))
        return json.loads(self._sock.recv_string())

    def status(self) -> dict:   return self._cmd("status")
    def arm(self, motion: bool) -> dict: return self._cmd("arm", motion=motion)
    def execute(self) -> dict:  return self._cmd("execute")
    def abort(self) -> dict:    return self._cmd("abort")
    def keepalive(self) -> dict: return self._cmd("keepalive")
