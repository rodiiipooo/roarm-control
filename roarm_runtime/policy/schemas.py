"""Policy schemas — the contract between the LLM planner and the executor.

The LLM may ONLY return one of these policies with these parameters; the
executor validates everything against the CapabilityProfile before motion.
"""
from __future__ import annotations

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


class ReachPolicy(BaseModel):
    kind: Literal["reach"] = "reach"
    target: Literal["object", "hand", "point"] = "object"
    xyz: Optional[list[float]] = None      # for target="point", base frame metres
    speed_scale: float = Field(0.6, ge=0.05, le=1.0)


class HandoverPolicy(BaseModel):
    kind: Literal["handover"] = "handover"
    grasp_first: bool = True               # grasp the object, then present it
    speed_scale: float = Field(0.4, ge=0.05, le=0.6)


class TrackPolicy(BaseModel):
    kind: Literal["track"] = "track"
    duration_s: float = Field(10.0, gt=0, le=120)
    follow_with_arm: bool = False


class WavePolicy(BaseModel):
    kind: Literal["wave"] = "wave"
    cycles: int = Field(2, ge=1, le=5)
    speed_scale: float = Field(0.7, ge=0.05, le=1.0)


class HomePolicy(BaseModel):
    kind: Literal["home"] = "home"


class GripperPolicy(BaseModel):
    kind: Literal["gripper"] = "gripper"
    action: Literal["open", "close"] = "close"


class IdlePolicy(BaseModel):
    kind: Literal["idle"] = "idle"
    reason: str = ""


Policy = Union[ReachPolicy, HandoverPolicy, TrackPolicy, WavePolicy,
               HomePolicy, GripperPolicy, IdlePolicy]

POLICY_JSON_SCHEMA_HINT = """\
Respond with EXACTLY ONE JSON object, no prose, matching one of:
{"kind":"reach","target":"object"|"hand"|"point","xyz":[x,y,z]?,"speed_scale":0..1}
{"kind":"handover","grasp_first":bool,"speed_scale":0..0.6}
{"kind":"track","duration_s":1..120,"follow_with_arm":bool}
{"kind":"wave","cycles":1..5,"speed_scale":0..1}
{"kind":"home"}
{"kind":"gripper","action":"open"|"close"}
{"kind":"idle","reason":"..."}
Rules: positions are metres in the robot base frame (x forward, y left, z up).
Choose "idle" when the request is unsafe, impossible, or unclear."""
