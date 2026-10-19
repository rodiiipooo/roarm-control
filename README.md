# RoArm Control — prompt-driven robot runtime + web GUI

Control your **RoArm-M3 5+1** arm and **dual Astra Pro** cameras with natural-language
prompts that compile into **capability-checked movement policies**, plus a full manual
teleop web GUI.

Built to sit **on top of your existing `r2s` stack** — it never touches the serial bus
or cameras directly; it drives your proven components through their existing interfaces.

---

## What it does

- **Prompt → policy**: type "take the object from my hand" / "reach for the object" /
  "wave hello". A planner (Kimi/Moonshot API, with an offline rule-based fallback)
  converts it into a constrained movement policy.
- **Capability gate**: every policy and every generated trajectory is validated against a
  hardware profile (`roarm_runtime/capabilities.py`) — joint limits, velocity, acceleration,
  workspace envelope, and slow handover speeds — **before any motion**.
- **Perception**: dual Astra Pro (head pan/tilt + under-arm), object detection with
  depth→3D, hand detection with grip classification (pinch / open-palm / fist),
  head visual servoing to track the target, and a reachability answer for the planner.
- **Handover**: grip-aware state machine — approaches from above for a pinch grip,
  from the side for an open palm, and refuses to close if the grip is unclear.
- **Web GUI**: live camera feeds, prompt console, full joint/Cartesian/head teleop,
  world-state telemetry, and a big E-Stop. Responsive (desktop + mobile).

---

## Architecture (live rig)

```
┌──────────────────────────────────────────────────────────────┐
│  THIS PROGRAM  (roarm_runtime, on the MINISFORUM)            │
│                                                              │
│  Web GUI (browser) ──HTTP──> FastAPI :8000                   │
│                                 │                            │
│            PerceptionCoordinator (2× Astra, head tracking)   │
│                                 │                            │
│            KimiPlanner ──> Executor ──> CapabilityProfile    │
│                                 │  (validates everything)    │
└─────────────────────────────────┼────────────────────────────┘
                                  │ JointMessage v1 (ZMQ PUB)
                                  ▼  tcp://127.0.0.1:5555
┌──────────────────────────────────────────────────────────────┐
│  YOUR EXISTING r2s STACK                                     │
│  scripts/roarm_follow.py  (SUB) — clamp, Ruckig smoothing,   │
│      motion gate, watchdogs  ──>  RoArm-M3 on COM3           │
│  scripts/palm_grasp.py    (guarded handover, ctl socket)     │
│  ui_frames.py             (FrameHub camera previews)         │
└──────────────────────────────────────────────────────────────┘
```

The arm is driven by **your `roarm_follow.py`**, which already does joint clamping,
trapezoid/Ruckig smoothing, and the motion gate. This program just publishes joint
targets — exactly like your Astra teleop does.

---

## Quick start — SIMULATION (no hardware, safe to try anywhere)

```bash
cd roarm_control
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the runtime (simulated arm + cameras, fully working perception/policies)
# ALWAYS use the launcher — it guarantees a single healthy instance:
# kills stale processes, checks the port, and verifies both cameras deliver
# frames before reporting OK. Running `python -m roarm_runtime.api.server`
# directly risks the multi-instance port race that caused dark feeds.
bash start_runtime.sh          # start (safe to re-run)
bash start_runtime.sh stop     # stop
```

Then open the GUI. Two options:

- **Preview the prebuilt GUI**: serve `app/dist/` and append the runtime address —
  ```
  cd app/dist && python -m http.server 8899
  # open http://127.0.0.1:8899/?api=http://127.0.0.1:8000
  ```
- **Or set it at build time**: `VITE_ROBOT_API=http://127.0.0.1:8000 npm run build`

Try in the prompt box: `reach for the object`, `take the object from my hand`, `wave hello`.

---

## Going LIVE on the MINISFORUM (your real hardware)

1. **Start your r2s follower** (owns COM3, applies all safety gates):
   ```powershell
   set R2S_ROARM_ENABLE_MOTION=1
   python scripts/roarm_follow.py --model m3 --right-port COM3 --endpoint tcp://127.0.0.1:5555 --enable-motion
   ```

2. **Point this runtime at the real stack** — edit `config/runtime.json`:
   ```json
   {
     "simulate": false,
     "use_yolo": true,
     "r2s": {
       "enabled": true,
       "joint_endpoint": "tcp://127.0.0.1:5555",
       "palm_grasp_ctl": "tcp://127.0.0.1:5560",
       "workspace_cam_serial": "17110210225",
       "underarm_cam_serial": "17122121839"
     }
   }
   ```

3. **Set the planner brain** (Kimi):
   ```bash
   export MOONSHOT_API_KEY=sk-...        # Windows: set MOONSHOT_API_KEY=sk-...
   ```
   Without a key it falls back to the offline rule parser (core commands still work).

4. **Run** (the launcher fails loudly if a camera is unplugged or the port is taken — a dark feed can never be silent again):
   ```bash
   bash start_runtime.sh
   ```

> **Safety**: keep `roarm_follow.py`'s motion gate on. This runtime *also* clamps and
> validates, but your follower is the last line of defence. Start with the arm
> unobstructed and the E-Stop button visible.

---

## Configuration

| File | What it holds |
|------|----------------|
| `roarm_runtime/capabilities.py` | Joint limits, velocities, workspace, handover speeds — **the hardware constraints**. Tune to your unit. |
| `config/runtime.json` | sim vs live, camera serials, r2s endpoints |
| `config/extrinsics.json` | camera→base transforms (run a hand-eye calibration, then fill these in) |

The camera serials default to your rig (head `17110210225`, under-arm `17122121839`),
matching `configs/astra_cameras.json`.

---

## What plugs into your existing code

| Your component | How this uses it |
|---|---|
| `roarm_follow.py` | arm driver (ZMQ JointMessage :5555) |
| `palm_grasp.py` | handover execution (ZMQ ctl :5560) via `PalmGraspClient` |
| `ui_frames.py` FrameHub | camera previews (can be shown in the GUI) |
| `r2s.perception.astra` | recommended real camera driver (swap into `perception/cameras.py`) |
| `voice.py` cues | reuse `is_out_of_reach()` for the reachability → voice path |

**To use your proven Astra driver instead of the minimal one here**, set
`use_r2s_cameras: true` and the runtime will import `r2s.perception.astra.AstraPro`
(depth registration, filters, stream recovery — already battle-tested on your rig).

---

## Regression tests

Every bug from the 2026-10-20 dark-feed incident is locked in by a test.
Run after ANY edit to perception/extrinsics/server code:

```bash
python tests/test_perception_regression.py    # or: python -m pytest tests/ -q
```

Covers: pan/tilt axis semantics, closed-loop tracker convergence, search-scan
workspace coverage, cam→base round-trip, arm-camera framing, non-black frames
in both roles, world-state hysteresis, and blind-startup refusal.

## Honest limitations

- **Extrinsics are placeholders** — run a ChArUco hand-eye calibration (you have
  `charuco_calibrate.py`) and fill `config/extrinsics.json`, or reach accuracy will be off.
- **Sim kinematics are approximate** — the RoArm-M3 link lengths are close but not
  calibrated; refine `arm_model.py` against the real unit for precision work.
- The sandbox preview ran in **simulation**; the real-hardware paths are built to match
  your documented interfaces but should be exercised gently on first live run.
