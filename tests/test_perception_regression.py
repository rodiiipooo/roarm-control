"""Regression tests for the dark-feed incident (2026-10-20).

Every bug that made the camera feed invisible or the tracker unstable is
locked in here. Run from the roarm_control dir:

    python -m pytest tests/ -q        (or: python tests/test_perception_regression.py)

If any of these fail after an edit, the incident is recurring — do not ship.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roarm_runtime.perception.extrinsics import (default_extrinsics, head_pose,
                                                 cam_to_base)
from roarm_runtime.perception.cameras import SimCamera
from roarm_runtime.kinematics.arm_model import link_points
from roarm_runtime.hal.drivers import HeadDriver

OBJ_B = np.array([0.30, 0.05, 0.03])   # must match SimCamera's scene anchor


def _project(cam, T, pt):
    c = (np.linalg.inv(T) @ np.append(pt, 1.0))[:3]
    if c[2] <= 0.05:
        return None
    return cam.fx * c[0] / c[2] + cam.cx, cam.fy * c[1] / c[2] + cam.cy


def test_pan_is_horizontal_not_roll():
    """Regression: pan used to rotate about the camera's view axis (a roll),
    which made the tracker oscillate to ±137°. Pan must move image content
    horizontally; tilt must move it vertically."""
    ext = default_extrinsics()
    cam = SimCamera(role="head")
    T0 = head_pose(ext["head"], 0.0, 0.5)
    Tp = head_pose(ext["head"], 0.2, 0.5)
    Tt = head_pose(ext["head"], 0.0, 0.7)
    u0, v0 = _project(cam, T0, OBJ_B)
    up, vp = _project(cam, Tp, OBJ_B)
    ut, vt = _project(cam, Tt, OBJ_B)
    assert abs(up - u0) > 5 * abs(vp - v0), "pan must move mostly horizontally"
    assert abs(vt - v0) > 5 * abs(ut - u0), "tilt must move mostly vertically"


def test_closed_loop_tracker_converges():
    """The exact servo loop from the coordinator: pan -= g·ex, tilt += g·ey.
    Must converge to <20 px error and stay within head limits."""
    ext = default_extrinsics()
    cam = SimCamera(role="head")
    head = HeadDriver(simulate=True)
    head.set(0.0, 0.4)
    uv = None
    # simulate the live 20 Hz loop — the driver's slew limit means
    # convergence needs wall-clock time, so step real time per iteration
    for _ in range(240):                       # ~12 s at 20 Hz
        st = head.state()
        T = head_pose(ext["head"], st["pan"], st["tilt"])
        uv = _project(cam, T, OBJ_B)
        assert uv is not None, "tracker drove the object out of view"
        ex = (uv[0] - cam.cx) / cam.cx
        ey = (uv[1] - cam.cy) / cam.cy
        head.set(st["pan"] - 0.45 * ex, st["tilt"] + 0.45 * ey)
        time.sleep(1 / 20)
    assert abs(uv[0] - cam.cx) < 20 and abs(uv[1] - cam.cy) < 20
    assert abs(st["pan"]) <= 2.4 and abs(st["tilt"]) <= 1.1


def test_scan_pose_sees_workspace():
    """The search scan must actually cover the workspace (it used to look
    behind the robot, which is why detection flickered 'no target')."""
    ext = default_extrinsics()
    cam = SimCamera(role="head")
    found = False
    for t in np.linspace(0, 25, 400):
        tilt = 0.20 + 0.25 * math.sin(0.6 * t)
        pan = 0.15 + 0.50 * math.sin(0.28 * t)
        if _project(cam, head_pose(ext["head"], pan, tilt), OBJ_B):
            found = True
            break
    assert found, "search scan never brings the workspace into view"


def test_cam_to_base_round_trip():
    ext = default_extrinsics()
    T = head_pose(ext["head"], 0.14, 0.22)
    c = (np.linalg.inv(T) @ np.append(OBJ_B, 1.0))[:3]
    back = cam_to_base(c, T)
    assert np.allclose(back, OBJ_B, atol=1e-9)


def test_arm_camera_sees_workspace():
    """The arm feed was black because the camera mount looked at empty floor.
    Wrist, TCP, object and hand must all project inside the frame."""
    ext = default_extrinsics()
    cam = SimCamera(role="arm")
    pts = link_points(np.array([0.0, 0.9, 1.1, 0.7, 0.0, 0.2]))
    targets = [pts["wrist"], pts["tcp"], OBJ_B, np.array([0.28, -0.06, 0.14])]
    for p in targets:
        uv = _project(cam, ext["arm"], p)
        assert uv is not None, f"point {p} behind arm camera"
        assert 30 < uv[0] < 610 and 30 < uv[1] < 450, f"point {p} off-frame: {uv}"


def test_sim_renders_content_both_roles():
    """Dark-feed canary: both sim cameras must emit frames with real content
    (mean brightness well above the background, not a black rectangle)."""
    ext = default_extrinsics()
    q = np.array([0.0, 0.9, 1.1, 0.7, 0.0, 0.2])
    for role, T in (("head", head_pose(ext["head"], 0.14, 0.22)),
                    ("arm", ext["arm"])):
        cam = SimCamera(role=role)
        cam.set_head_pose(T)
        cam.set_arm_q(q)
        cam.start()
        f = cam.grab()
        assert f is not None and f.rgb.size > 0
        assert f.rgb.max() > 100, f"{role} frame has no scene content"
        assert f.rgb.mean() > 25, f"{role} frame is near-black"
        assert (f.depth > 0).any()


def test_world_state_hysteresis(tmp_path=None):
    """World state must not flicker 'no target' between tracker updates:
    a fresh detection holds for ~0.6 s even if the next frames miss."""
    from roarm_runtime.perception.coordinator import PerceptionCoordinator
    pc = PerceptionCoordinator(simulate=True)
    from roarm_runtime.hal.drivers import HeadDriver, SimArmDriver
    pc.head = HeadDriver(simulate=True)
    pc.arm = SimArmDriver()
    pc.arm.connect()
    pc.start()
    try:
        t0 = time.time()
        locked = dark = 0
        while time.time() - t0 < 6.0:
            w = pc.get_world()
            if w.object_xyz is not None:
                locked += 1
            else:
                dark += 1
            time.sleep(0.05)
        assert locked > 0, "never detected the object at all"
        assert dark / max(locked + dark, 1) < 0.3, \
            f"world state dark {dark}/{locked+dark} samples — hysteresis broken"
    finally:
        pc.stop()


def test_startup_refuses_blind(tmp_path=None):
    """If a camera cannot start, coordinator.start() must raise — never
    serve a black feed silently."""
    from roarm_runtime.perception.coordinator import PerceptionCoordinator
    pc = PerceptionCoordinator(simulate=True)

    class DeadCam:
        def start(self): return False
        def stop(self): pass
        def grab(self): return None
    pc.arm_cam = DeadCam()
    try:
        pc.start()
    except RuntimeError as e:
        assert "camera" in str(e).lower()
    else:
        raise AssertionError("blind startup was not refused")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
