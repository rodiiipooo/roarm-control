"""RoArm-M3 kinematic chain (5 DOF arm + gripper).

Denavit–Hartenberg-flavoured model tuned to RoArm-M3 link lengths.
Lengths (m) are the published approximations; refine with a calibration
routine against the real unit for millimetre-class work.
"""
from __future__ import annotations

import math

import numpy as np

# Link lengths (m)
L_BASE = 0.065     # base to shoulder axis
L_UPPER = 0.205    # shoulder to elbow
L_FORE = 0.205     # elbow to wrist
L_WRIST = 0.105    # wrist to tool centre point


def _rot_x(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
def _rot_y(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,-s,0],[0,1,0,0],[s,0,c,0],[0,0,0,1]])
def _rot_z(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
def _trans(x, y, z):
    m = np.eye(4); m[0, 3], m[1, 3], m[2, 3] = x, y, z; return m


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """q = [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper] → 4x4 TCP pose.

    Shoulder/elbow/wrist angles are measured from the VERTICAL (servo zero =
    arm pointing straight up, matching the RoArm-M3's physical zero), so each
    pitch joint rotates about Y by (pi/2 - q).
    """
    base, shoulder, elbow, wrist_pitch, wrist_roll = q[:5]
    T = _rot_z(base) @ _trans(0, 0, L_BASE)
    T = T @ _rot_y(math.pi / 2 - shoulder) @ _trans(L_UPPER, 0, 0)
    T = T @ _rot_y(-elbow) @ _trans(L_FORE, 0, 0)
    T = T @ _rot_y(-wrist_pitch) @ _trans(L_WRIST, 0, 0)
    T = T @ _rot_z(wrist_roll)
    return T


def link_points(q: np.ndarray) -> dict[str, np.ndarray]:
    """Joint positions (base frame) along the chain — for visualisation."""
    base, shoulder, elbow, wrist_pitch, _ = q[:5]
    pts = {"base": np.zeros(3)}
    T = _rot_z(base) @ _trans(0, 0, L_BASE)
    pts["shoulder"] = T[:3, 3].copy()
    T = T @ _rot_y(math.pi / 2 - shoulder) @ _trans(L_UPPER, 0, 0)
    pts["elbow"] = T[:3, 3].copy()
    T = T @ _rot_y(-elbow) @ _trans(L_FORE, 0, 0)
    pts["wrist"] = T[:3, 3].copy()
    T = T @ _rot_y(-wrist_pitch) @ _trans(L_WRIST, 0, 0)
    pts["tcp"] = T[:3, 3].copy()
    return pts


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """Numerical 6x5 Jacobian (position + z-axes) — adequate at 50 Hz control rates."""
    eps = 1e-6
    T0 = forward_kinematics(q)
    p0 = T0[0:3, 3]
    J = np.zeros((6, 5))
    for i in range(5):
        dq = q.copy(); dq[i] += eps
        p1 = forward_kinematics(dq)[0:3, 3]
        J[0:3, i] = (p1 - p0) / eps
    # angular part from finite-diff of rotation matrices
    for i in range(5):
        dq = q.copy(); dq[i] += eps
        R1 = forward_kinematics(dq)[0:3, 0:3]
        dR = (R1 - T0[0:3, 0:3]) / eps
        J[3, i] = dR[2, 1] * 0.5 - dR[1, 2] * 0.5
        J[4, i] = dR[0, 2] * 0.5 - dR[2, 0] * 0.5
        J[5, i] = dR[1, 0] * 0.5 - dR[0, 1] * 0.5
    return J
