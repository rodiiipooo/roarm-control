"""Camera→base extrinsics and the perception coordinator.

Extrinsic convention: T_base_cam is the 4x4 pose of the camera frame in the
robot base frame. Camera frame is optical: z forward, x right, y down.

Defaults below are PLACEHOLDERS — measure yours (or run a small hand-eye
calibration against a known target) and edit config/extrinsics.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Camera pose in base frame. Base frame: x forward (workspace), y left,
    z up. Camera frame is optical: z out of the lens, x image-right,
    y image-down. At zero angles the camera looks level toward +x with the
    image upright. yaw > 0 turns LEFT (toward +y), pitch > 0 tilts DOWN
    toward the workspace, roll > 0 rolls the image clockwise."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # camera axes in base frame at zero angles:
    # view (z) = +x, image-right (x) = -y, image-down (y) = -z
    R0 = np.array([[0.0, 0.0, 1.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])   # pitch down
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])   # yaw left
    Rzroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ R0 @ Rzroll
    T[:3, 3] = [x, y, z]
    return T


def default_extrinsics() -> dict[str, np.ndarray]:
    # Head camera sits on the pan/tilt head above and behind the base,
    # pitched down toward the workspace. Arm camera is low, facing up at the arm.
    return {
        "head": _pose(-0.05, 0.0, 0.42, pitch=math.radians(35)),   # tilted down at workspace
        # low camera on the table beyond the workspace, looking back and
        # slightly up at the arm so wrist/object/hand all stay in frame
        "arm":  _pose(0.50, -0.25, 0.03, pitch=math.radians(-15), yaw=math.radians(135)),
    }


def load_extrinsics(path: str | Path | None = None) -> dict[str, np.ndarray]:
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "extrinsics.json"
    if p.exists():
        raw = json.loads(p.read_text())
        return {k: np.array(v, dtype=float).reshape(4, 4) for k, v in raw.items()}
    return default_extrinsics()


def cam_to_base(xyz_cam: np.ndarray | None, T_base_cam: np.ndarray) -> np.ndarray | None:
    if xyz_cam is None:
        return None
    pt = T_base_cam @ np.append(xyz_cam, 1.0)
    return pt[:3]


def head_pose(mount_T: np.ndarray, pan: float, tilt: float) -> np.ndarray:
    """Live head-camera pose. A real pan/tilt head pans about the WORLD
    vertical axis through the mount point, then tilts about the panned
    camera's local x (image-right) axis. Positive tilt pitches the camera
    DOWN toward the workspace, matching the Hiwonder servo convention.
    pan/tilt in radians."""
    cp, sp = math.cos(pan), math.sin(pan)
    ct, st = math.cos(tilt), math.sin(tilt)
    Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])   # world-frame pan
    Rx = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])  # local tilt, + = down
    T = mount_T.copy()
    T[:3, :3] = Rz @ mount_T[:3, :3] @ Rx
    return T
