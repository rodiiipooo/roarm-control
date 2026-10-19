"""RoArm-M3 5+1 hardware capability profile.

Every trajectory the policy engine generates is validated against these
limits BEFORE it reaches the bus. Values are conservative defaults for the
Waveshare RoArm-M3 (ST3215 serial bus servos) — tune to your unit.

Joint order matches the RoArm-M3 SDK convention:
  0 base, 1 shoulder, 2 elbow, 3 wrist_pitch, 4 wrist_roll, 5 gripper
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class JointLimit:
    name: str
    min_rad: float          # hard joint limit (rad)
    max_rad: float
    max_vel_rad_s: float    # conservative continuous velocity
    max_acc_rad_s2: float   # acceleration limit for trapezoid profiles
    torque_limit: float     # 0..1 fraction of servo stall torque budget


@dataclass(frozen=True)
class CapabilityProfile:
    """Preconfigured hardware capacity constraints. Immutable at runtime."""

    joints: tuple[JointLimit, ...] = (
        JointLimit("base",        -math.pi,        math.pi,        1.2, 4.0, 0.70),
        JointLimit("shoulder",    -1.60,           1.60,           1.0, 3.5, 0.65),
        JointLimit("elbow",       -1.70,           1.70,           1.0, 3.5, 0.65),
        JointLimit("wrist_pitch", -1.65,           1.65,           1.4, 5.0, 0.60),
        JointLimit("wrist_roll",  -math.pi,        math.pi,        1.8, 6.0, 0.50),
        JointLimit("gripper",     -0.20,           1.90,           2.5, 8.0, 0.55),
    )

    # Cartesian workspace (metres, base frame). RoArm-M3 reach ≈ 0.45 m.
    reach_radius_m: float = 0.45
    min_reach_radius_m: float = 0.06      # singularity guard around base axis
    z_min_m: float = -0.05                # below table guard
    z_max_m: float = 0.50

    # TCP (tool center point) speed caps in Cartesian space
    tcp_max_speed_m_s: float = 0.25
    tcp_max_acc_m_s2: float = 0.8

    # Interaction / handover safety
    handover_max_speed_m_s: float = 0.12  # slower when a human hand is in the loop
    handover_approach_dist_m: float = 0.06
    person_min_clearance_m: float = 0.10  # never plan TCP closer than this to a detected hand during transit
    gripper_hold_torque: float = 0.40

    # Servo bus
    bus_rate_hz: float = 50.0
    telemetry_rate_hz: float = 20.0

    def n_joints(self) -> int:
        return len(self.joints)

    def clamp_joint(self, idx: int, q: float) -> float:
        j = self.joints[idx]
        return max(j.min_rad, min(j.max_rad, q))

    def in_workspace(self, x: float, y: float, z: float) -> tuple[bool, str]:
        r = math.hypot(x, y)
        if r > self.reach_radius_m:
            return False, f"radial reach {r:.3f} m > {self.reach_radius_m} m"
        if r < self.min_reach_radius_m:
            return False, f"too close to base axis ({r:.3f} m)"
        if not (self.z_min_m <= z <= self.z_max_m):
            return False, f"z={z:.3f} m outside [{self.z_min_m}, {self.z_max_m}]"
        return True, "ok"


DEFAULT_PROFILE = CapabilityProfile()
