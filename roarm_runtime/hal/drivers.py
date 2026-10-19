"""Hardware abstraction layer: one ArmDriver interface, two backends.

- SimArmDriver   : kinematic simulation, used when no servos are present
                   (and by the sandbox preview of the web GUI).
- SerialArmDriver: Waveshare RoArm-M3 over ST3215 serial bus (real hardware,
                   requires `pip install waveshare-roarm` or the raw serial
                   JSON protocol — see RealDriver notes).
- HeadDriver     : Hiwonder pan/tilt head (2 bus servos), same pattern.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..capabilities import CapabilityProfile, DEFAULT_PROFILE


@dataclass
class ArmState:
    q: np.ndarray                 # joint positions (rad), len 6
    dq: np.ndarray                # joint velocities
    tcp_xyz: np.ndarray           # tool center point in base frame
    moving: bool = False
    torques_enabled: bool = True
    last_error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


class ArmDriver:
    """Interface both backends implement."""

    name = "abstract"

    def connect(self) -> bool: raise NotImplementedError
    def disconnect(self) -> None: raise NotImplementedError
    def enable_torque(self, on: bool) -> None: raise NotImplementedError
    def get_state(self) -> ArmState: raise NotImplementedError
    def command_joints(self, q_target: np.ndarray, speed_scale: float = 1.0) -> None:
        raise NotImplementedError
    def estop(self) -> None: raise NotImplementedError
    def clear_estop(self) -> None: raise NotImplementedError
    @property
    def estopped(self) -> bool: raise NotImplementedError


class SimArmDriver(ArmDriver):
    """First-order joint simulation toward commanded targets."""

    name = "sim"

    def __init__(self, profile: CapabilityProfile = DEFAULT_PROFILE):
        self.p = profile
        self._q = np.zeros(self.p.n_joints())
        self._q[1] = 0.9          # shoulder raised
        self._q[2] = 1.1          # elbow bent — a sane home pose
        self._q_target = self._q.copy()
        self._dq = np.zeros(self.p.n_joints())
        self._lock = threading.Lock()
        self._estopped = False
        self._torque = True
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def enable_torque(self, on: bool) -> None:
        self._torque = on

    @property
    def estopped(self) -> bool:
        return self._estopped

    def estop(self) -> None:
        with self._lock:
            self._estopped = True
            self._q_target = self._q.copy()
            self._dq[:] = 0.0

    def clear_estop(self) -> None:
        with self._lock:
            self._estopped = False

    def get_state(self) -> ArmState:
        self._step()
        with self._lock:
            from ..kinematics.arm_model import forward_kinematics
            tcp = forward_kinematics(self._q)[0:3, 3]
            return ArmState(
                q=self._q.copy(), dq=self._dq.copy(), tcp_xyz=tcp,
                moving=bool(np.any(np.abs(self._dq) > 1e-3)),
                torques_enabled=self._torque,
                updated_at=time.time(),
            )

    def command_joints(self, q_target: np.ndarray, speed_scale: float = 1.0) -> None:
        with self._lock:
            if self._estopped:
                raise RuntimeError("E-stop active")
            q = np.asarray(q_target, dtype=float).copy()
            for i in range(self.p.n_joints()):
                q[i] = self.p.clamp_joint(i, q[i])
            self._q_target = q
            self._speed_scale = float(np.clip(speed_scale, 0.05, 1.0))

    def _step(self) -> None:
        now = time.time()
        dt = min(now - getattr(self, "_last", now), 0.1)
        self._last = now
        if dt <= 0:
            return
        with self._lock:
            err = self._q_target - self._q
            vmax = np.array([j.max_vel_rad_s for j in self.p.joints]) * getattr(self, "_speed_scale", 1.0)
            step = np.clip(err, -vmax * dt, vmax * dt)
            self._dq = step / dt
            self._q = self._q + step


class SerialArmDriver(ArmDriver):
    """RoArm-M3 via Waveshare serial JSON protocol (/dev/ttyUSB* or /dev/ttyACM*).

    Runs the same joint-level interface on top of the Waveshare `Roarm` class
    when available, else raw serial JSON commands (T:102 joint angle ctrl).
    """

    name = "serial"

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200,
                 profile: CapabilityProfile = DEFAULT_PROFILE):
        self.port, self.baud, self.p = port, baud, profile
        self._ser = None
        self._estopped = False
        self._q = np.zeros(self.p.n_joints())
        self._lock = threading.Lock()

    def connect(self) -> bool:
        import serial  # pyserial, on the MINISFORUM
        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        return self._ser.is_open

    def disconnect(self) -> None:
        if self._ser:
            self._ser.close()

    def enable_torque(self, on: bool) -> None:
        self._send({"T": 210, "cmd": 1 if on else 0})   # torque lock cmd

    @property
    def estopped(self) -> bool:
        return self._estopped

    def estop(self) -> None:
        self._estopped = True
        self._send({"T": 0})                            # immediate stop cmd
        self.enable_torque(False)

    def clear_estop(self) -> None:
        self._estopped = False
        self.enable_torque(True)

    def get_state(self) -> ArmState:
        # RoArm-M3 reports joint feedback with T:105 — parse here on hardware.
        from ..kinematics.arm_model import forward_kinematics
        with self._lock:
            tcp = forward_kinematics(self._q)[0:3, 3]
            return ArmState(q=self._q.copy(), dq=np.zeros(6), tcp_xyz=tcp,
                            updated_at=time.time())

    def command_joints(self, q_target: np.ndarray, speed_scale: float = 1.0) -> None:
        if self._estopped:
            raise RuntimeError("E-stop active")
        q = [self.p.clamp_joint(i, float(v)) for i, v in enumerate(q_target)]
        with self._lock:
            self._q = np.array(q)
        deg = [round(math.degrees(v), 2) for v in q]
        self._send({"T": 102, "base": deg[0], "shoulder": deg[1], "elbow": deg[2],
                    "hand": deg[3], "spd": int(100 * speed_scale), "acc": 10})

    def _send(self, obj: dict) -> None:
        if self._ser:
            import json
            self._ser.write((json.dumps(obj) + "\n").encode())


class HeadDriver:
    """Hiwonder-based pan/tilt head (2 bus servos). Sim fallback included."""

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self.pan = 0.0
        self.tilt = 0.0
        self._pan_target = 0.0
        self._tilt_target = 0.0
        self._last = time.time()
        self.pan_limit = (-2.4, 2.4)
        self.tilt_limit = (-1.1, 1.1)
        self.slew = 3.0  # rad/s

    def set(self, pan: float, tilt: float) -> None:
        self._pan_target = float(np.clip(pan, *self.pan_limit))
        self._tilt_target = float(np.clip(tilt, *self.tilt_limit))

    def step(self) -> None:
        now = time.time()
        dt = min(now - self._last, 0.1)
        self._last = now
        for attr, tgt in (("pan", self._pan_target), ("tilt", self._tilt_target)):
            cur = getattr(self, attr)
            err = tgt - cur
            setattr(self, attr, cur + float(np.clip(err, -self.slew * dt, self.slew * dt)))

    def state(self) -> dict:
        self.step()
        return {"pan": self.pan, "tilt": self.tilt,
                "pan_deg": math.degrees(self.pan), "tilt_deg": math.degrees(self.tilt)}
