"""System configuration — edit for your rig.

Camera identification: with the USB3→USB4 adapter on the MINISFORUM,
/dev/videoN indices can shift across reboots. Prefer stable IDs:
  ls -l /dev/v4l/by-id/    →  use those paths for head_video/arm_video.
Serial numbers are even better (Astra Pro exposes one via OpenNI).
"""
from dataclasses import dataclass, field


@dataclass
class CameraCfg:
    role: str
    uri: str | None = None          # OpenNI device URI / serial (None = auto)
    video_path: str | None = None   # e.g. /dev/v4l/by-id/usb-Orbbec_...-video-index0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class SerialCfg:
    arm_port: str = "COM3"                  # RoArm-M3 bus servo adapter (COM3 on your rig)
    head_port: str | None = None            # Hiwonder bus servos (often on the arm adapter or its own)
    baud: int = 115200


@dataclass
class R2sCfg:
    """How to reach the user's existing r2s stack on the MINISFORUM."""
    enabled: bool = True                    # False → talk to serial/cameras directly
    joint_endpoint: str = "tcp://127.0.0.1:5555"   # roarm_follow SUB (bind); we connect+PUB
    palm_grasp_ctl: str = "tcp://127.0.0.1:5560"   # palm_grasp.py ZMQ REP control socket
    frame_hub: str = "http://127.0.0.1:8766"       # ui_frames FrameHub (camera previews)
    workspace_cam_serial: str = "17110210225"      # head pan/tilt Astra
    underarm_cam_serial: str = "17122121839"       # under-arm Astra


@dataclass
class RuntimeCfg:
    simulate: bool = True               # False on the MINISFORUM with hardware
    use_yolo: bool = False              # True once ultralytics is installed
    cameras: list[CameraCfg] = field(default_factory=lambda: [
        CameraCfg(role="head"),         # through USB3→USB4 adapter
        CameraCfg(role="arm"),          # direct USB
    ])
    serial: SerialCfg = field(default_factory=SerialCfg)
    r2s: R2sCfg = field(default_factory=R2sCfg)
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def load() -> RuntimeCfg:
    import json, os
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "config" / "runtime.json"
    if p.exists():
        raw = json.loads(p.read_text())
        cams = [CameraCfg(**c) for c in raw.get("cameras", [])] or None
        cfg = RuntimeCfg(
            simulate=raw.get("simulate", True),
            use_yolo=raw.get("use_yolo", False),
            serial=SerialCfg(**raw.get("serial", {})),
            r2s=R2sCfg(**raw.get("r2s", {})),
            api_host=raw.get("api_host", "0.0.0.0"),
            api_port=raw.get("api_port", 8000),
        )
        if cams:
            cfg.cameras = cams
        return cfg
    return RuntimeCfg()
