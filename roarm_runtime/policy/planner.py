"""LLM planner: user prompt + world state → validated Policy (Kimi / Moonshot).

Falls back to a deterministic rule parser when no API key is set, so the
system is fully usable offline for the core commands.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from .schemas import (Policy, ReachPolicy, HandoverPolicy, TrackPolicy,
                      WavePolicy, HomePolicy, GripperPolicy, IdlePolicy,
                      POLICY_JSON_SCHEMA_HINT)
from ..perception.coordinator import WorldState
from ..capabilities import CapabilityProfile, DEFAULT_PROFILE

MODEL = "kimi-k2-0711-preview"   # or moonshot-v1-8k; set what your key has


def _world_brief(w: WorldState) -> str:
    bits = []
    if w.object_xyz:
        bits.append(f"object '{w.object_label}' at [{', '.join(f'{v:.3f}' for v in w.object_xyz)}] m, "
                    f"reachable={w.reachable} ({w.reach_reason})")
    else:
        bits.append("no object detected")
    if w.hand_present:
        bits.append(f"hand present, grip={w.hand_grip}")
    else:
        bits.append("no hand detected")
    return "; ".join(bits)


class KimiPlanner:
    def __init__(self, api_key: Optional[str] = None,
                 profile: CapabilityProfile = DEFAULT_PROFILE):
        self.key = api_key or os.environ.get("MOONSHOT_API_KEY")
        self.p = profile

    def plan(self, prompt: str, world: WorldState) -> tuple[Policy, str]:
        if not self.key:
            return self._rule_parse(prompt, world)
        try:
            return self._llm_parse(prompt, world)
        except Exception as e:            # LLM failure must never stop the robot
            pol, note = self._rule_parse(prompt, world)
            return pol, f"LLM unavailable ({e}); fell back to rules: {note}"

    # ------------------------------------------------------------------ LLM
    def _llm_parse(self, prompt: str, world: WorldState) -> tuple[Policy, str]:
        from openai import OpenAI          # moonshot is OpenAI-compatible
        client = OpenAI(api_key=self.key, base_url="https://api.moonshot.cn/v1")
        sys = ("You are a robot motion planner for a RoArm-M3 5+1 arm with two "
               "depth cameras (head + arm). Convert the user's request into a "
               "movement policy. The robot has hard limits: reach radius "
               f"{self.p.reach_radius_m} m, z range [{self.p.z_min_m}, {self.p.z_max_m}] m. "
               "Never plan outside them.\n" + POLICY_JSON_SCHEMA_HINT)
        user = f"World state: {_world_brief(world)}\nUser request: {prompt}"
        r = client.chat.completions.create(
            model=MODEL, temperature=0.1,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}])
        text = r.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return IdlePolicy(reason="LLM returned no JSON"), "planner error"
        return self._validate(json.loads(m.group(0)), world), "LLM policy"

    # ---------------------------------------------------------------- rules
    def _rule_parse(self, prompt: str, world: WorldState) -> tuple[Policy, str]:
        t = prompt.lower()
        if any(k in t for k in ("hand over", "handover", "give me", "pass me", "hand it",
                                "take the", "take it", "take this", "from my hand",
                                "from the hand", "snatch", "pick from hand")):
            # grasp_first=False when the human is holding it out to the robot
            give_words = ("give me", "pass me", "hand it", "hand over", "handover")
            grasp_first = any(k in t for k in give_words)
            return self._validate(HandoverPolicy(grasp_first=grasp_first).model_dump(), world), "rule: handover"
        if any(k in t for k in ("reach", "grab", "pick", "get", "touch", "grasp")):
            tgt = "hand" if "hand" in t else "object"
            return self._validate(ReachPolicy(target=tgt).model_dump(), world), f"rule: reach {tgt}"
        if any(k in t for k in ("track", "follow", "watch")):
            return TrackPolicy(), "rule: track"
        if "wave" in t or "hello" in t or "hi" == t.strip():
            return WavePolicy(), "rule: wave"
        if "home" in t or "reset" in t or "rest" in t:
            return HomePolicy(), "rule: home"
        if "open" in t and "grip" in t or "release" in t or "let go" in t:
            return GripperPolicy(action="open"), "rule: gripper open"
        if "close" in t and "grip" in t or "squeeze" in t:
            return GripperPolicy(action="close"), "rule: gripper close"
        return IdlePolicy(reason="no matching rule"), "no rule matched"

    # ------------------------------------------------------------- validate
    def _validate(self, raw: dict, world: WorldState) -> Policy:
        kind = raw.get("kind", "idle")
        cls = {"reach": ReachPolicy, "handover": HandoverPolicy, "track": TrackPolicy,
               "wave": WavePolicy, "home": HomePolicy, "gripper": GripperPolicy,
               "idle": IdlePolicy}.get(kind, IdlePolicy)
        try:
            pol = cls(**{k: v for k, v in raw.items() if k in cls.model_fields})
        except Exception:
            return IdlePolicy(reason="policy failed schema validation")
        # capability gating at the policy level
        if isinstance(pol, ReachPolicy) and pol.target == "point" and pol.xyz:
            ok, why = self.p.in_workspace(*pol.xyz)
            if not ok:
                return IdlePolicy(reason=f"target rejected by capability profile: {why}")
        # reach for the object requires a perceived object (reach-for-hand and
        # handover key off the hand instead, and handover runs its own wait loop)
        if isinstance(pol, ReachPolicy) and pol.target == "object" and world.object_xyz is None:
            return IdlePolicy(reason="no object perceived — cannot reach it")
        return pol
