"""Damped least-squares IK with joint-limit weighting + trajectory generation."""
from __future__ import annotations

import numpy as np

from .arm_model import forward_kinematics, geometric_jacobian
from ..capabilities import CapabilityProfile, DEFAULT_PROFILE


def ik_solve(target_xyz: np.ndarray, q_seed: np.ndarray,
             profile: CapabilityProfile = DEFAULT_PROFILE,
             max_iter: int = 200, tol: float = 1e-4,
             _restarts: int = 3) -> tuple[np.ndarray | None, str]:
    """Position IK. Retries with progressively straightened seeds when the
    initial seed sits at a joint limit (e.g. elbow), which stalls DLS."""
    seeds = [q_seed.copy().astype(float)]
    for k in range(_restarts):
        s = np.zeros_like(q_seed, dtype=float)
        s[1] = 0.6 + 0.2 * k     # shoulder raised
        s[2] = 0.9 - 0.2 * k     # elbow away from limits
        s[3] = 0.5
        seeds.append(s)
    best, best_err = None, float("inf")
    for q0 in seeds:
        q, err = _dls(target_xyz, q0, profile, max_iter, tol)
        if err < best_err:
            best, best_err = q, err
        if err < tol:
            return q, "ok"
    if best is not None and best_err < 5e-3:
        return best, "ok (approx)"
    return None, "IK did not converge — target likely outside dexterous workspace"


def _dls(target_xyz: np.ndarray, q: np.ndarray, profile: CapabilityProfile,
         max_iter: int, tol: float) -> tuple[np.ndarray, float]:
    lam = 0.08
    for _ in range(max_iter):
        p = forward_kinematics(q)[0:3, 3]
        err = target_xyz - p
        errn = float(np.linalg.norm(err))
        if errn < tol:
            return q, errn
        J = geometric_jacobian(q)[0:3, :]                       # 3x5
        # joint-limit damping: pull away from limits
        W = np.ones(5)
        for i, jl in enumerate(profile.joints[:5]):
            span = jl.max_rad - jl.min_rad
            edge = min(q[i] - jl.min_rad, jl.max_rad - q[i]) / span
            W[i] = 0.2 + 0.8 * np.clip(edge * 4, 0.0, 1.0)
        Jw = J @ np.diag(W)
        dq = np.diag(W) @ Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2 * np.eye(3), err)
        dq = np.clip(dq, -0.15, 0.15)
        q[:5] += dq
        for i, jl in enumerate(profile.joints[:5]):
            q[i] = np.clip(q[i], jl.min_rad + 1e-3, jl.max_rad - 1e-3)
    return q, float(np.linalg.norm(forward_kinematics(q)[0:3, 3] - target_xyz))


def trapezoid_trajectory(q0: np.ndarray, q1: np.ndarray,
                         profile: CapabilityProfile = DEFAULT_PROFILE,
                         speed_scale: float = 1.0,
                         rate_hz: float | None = None) -> np.ndarray:
    """Synchronised trapezoidal joint trajectory. Returns (N, 6) array of setpoints."""
    rate = rate_hz or profile.bus_rate_hz
    speed_scale = float(np.clip(speed_scale, 0.05, 1.0))
    vmax = np.array([j.max_vel_rad_s for j in profile.joints]) * speed_scale
    amax = np.array([j.max_acc_rad_s2 for j in profile.joints]) * speed_scale
    dist = np.abs(q1 - q0)
    # per-joint trapezoid time, then synchronise to the slowest
    t_joint = np.where(dist > 1e-6,
                       np.where(dist < vmax**2 / amax,
                                2 * np.sqrt(dist / amax),
                                dist / vmax + vmax / amax),
                       0.0)
    T = max(float(t_joint.max()), 1.0 / rate)
    n = max(int(T * rate) + 1, 2)
    t = np.linspace(0.0, T, n)
    qs = np.zeros((n, len(q0)))
    for j in range(len(q0)):
        d = q1[j] - q0[j]
        if abs(d) < 1e-9:
            qs[:, j] = q0[j]
            continue
        v = min(vmax[j], abs(d) / (0.5 * T))
        a = min(amax[j], v / (0.5 * T) * 2)
        ta = v / a
        td = T - ta
        s = np.where(t < ta, 0.5 * a * t**2,
                     np.where(t < td, 0.5 * a * ta**2 + v * (t - ta),
                              0.5 * a * ta**2 + v * (td - ta) + v * (t - td) - 0.5 * a * (t - td)**2))
        s = np.clip(s / (s[-1] if s[-1] > 1e-9 else 1.0), 0, 1)
        qs[:, j] = q0[j] + d * s
    return qs


def validate_trajectory(qs: np.ndarray, profile: CapabilityProfile = DEFAULT_PROFILE,
                        rate_hz: float | None = None) -> tuple[bool, str]:
    """Capability gate: joint limits, velocity, acceleration."""
    rate = rate_hz or profile.bus_rate_hz
    for i, jl in enumerate(profile.joints):
        col = qs[:, i]
        if col.min() < jl.min_rad - 1e-3 or col.max() > jl.max_rad + 1e-3:
            return False, f"{jl.name} exceeds joint limits"
        if len(col) > 1:
            vel = np.diff(col) * rate
            if np.abs(vel).max() > jl.max_vel_rad_s * 1.05:
                return False, f"{jl.name} exceeds velocity limit"
            if len(vel) > 1 and np.abs(np.diff(vel) * rate).max() > jl.max_acc_rad_s2 * 1.10:
                return False, f"{jl.name} exceeds acceleration limit"
    return True, "ok"
