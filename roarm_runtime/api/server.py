"""FastAPI + WebSocket server bridging the robot runtime to the web GUI.

Endpoints:
  GET  /api/status                 full robot + world snapshot
  POST /api/prompt                 {text} → plan + execute
  POST /api/teleop/joint           {joint, delta, speed}
  POST /api/teleop/cartesian       {dx, dy, dz, speed}
  POST /api/teleop/head            {pan, tilt}  (absolute, rad)
  POST /api/estop   POST /api/estop/clear
  POST /api/cancel
  GET  /api/camera/{role}.mjpg     MJPEG stream (head|arm)
  WS   /ws                         status push at telemetry rate

Run:  python -m roarm_runtime.api.server   (from the roarm_control dir)
"""
from __future__ import annotations

import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from ..runtime import RobotRuntime


class CameraWentDarkError(RuntimeError):
    """Raised by the watchdog when a camera that previously delivered frames
    stops delivering (cable, driver crash, UVC contention). Surfaces as a
    clear warning instead of a silently black feed."""

runtime = RobotRuntime()
app = FastAPI(title="RoArm Control API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class PromptIn(BaseModel):
    text: str

class JointIn(BaseModel):
    joint: int
    delta: float
    speed: float = 0.4

class CartIn(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    speed: float = 0.4

class HeadIn(BaseModel):
    pan: float
    tilt: float


@app.exception_handler(RuntimeError)
def _runtime_error(_, exc: RuntimeError):
    # any unhandled motion/safety refusal surfaces as 409, not a bare 500
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.on_event("startup")
def _startup():
    if not runtime.started:
        runtime.start()
    # a dark feed must never be silent: after startup, wait for real frames
    deadline = time.time() + 5.0
    while time.time() < deadline:
        jpgs = runtime.perc.latest_jpegs()
        if jpgs.get("head") and jpgs.get("arm"):
            break
        time.sleep(0.2)
    missing = [r for r, j in runtime.perc.latest_jpegs().items() if not j]
    if missing:
        raise RuntimeError(
            f"camera(s) {missing} produced no frames within 5 s of startup — "
            f"check cables/UVC devices; refusing to serve a blind feed")
    _start_watchdog()


@app.on_event("shutdown")
def _shutdown():
    _watchdog_stop.set()
    runtime.stop()


# ------------------------------------------------------------ watchdog
# If a camera that has been streaming goes dark for >8 s, log it loudly and
# flag it in /api/status so the GUI can show why the feed froze.
import logging, threading

log = logging.getLogger("roarm.watchdog")
_watchdog_stop = threading.Event()


def _start_watchdog():
    def _run():
        dark_since: dict[str, float | None] = {"head": None, "arm": None}
        ever_lit: dict[str, bool] = {"head": False, "arm": False}
        while not _watchdog_stop.is_set():
            frames = runtime.perc._frames
            now = time.time()
            for role in ("head", "arm"):
                f = frames.get(role)
                if f is not None:
                    ever_lit[role] = True
                    dark_since[role] = None
                    runtime.perc.stream_warnings.pop(role, None)
                elif ever_lit[role]:
                    if dark_since[role] is None:
                        dark_since[role] = now
                    elif now - dark_since[role] > 8.0:
                        msg = (f"{role} camera stopped delivering frames — "
                               f"check cable / other apps holding the device")
                        if runtime.perc.stream_warnings.get(role) != msg:
                            log.error("WATCHDOG: %s", msg)
                        runtime.perc.stream_warnings[role] = msg
            _watchdog_stop.wait(2.0)

    threading.Thread(target=_run, daemon=True).start()

@app.get("/api/status")
def status():
    return runtime.status()

@app.get("/api/debug")
def debug():
    return {"perc_dbg": getattr(runtime.perc, "_dbg", None)}

@app.post("/api/prompt")
def prompt(inp: PromptIn):
    try:
        return runtime.prompt(inp.text)
    except RuntimeError as e:                       # e.g. E-stop active
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/api/teleop/joint")
def teleop_joint(inp: JointIn):
    try:
        return runtime.teleop_joint(inp.joint, inp.delta, inp.speed)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/api/teleop/cartesian")
def teleop_cart(inp: CartIn):
    try:
        return runtime.teleop_cartesian(inp.dx, inp.dy, inp.dz, inp.speed)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/api/teleop/head")
def teleop_head(inp: HeadIn):
    runtime.head.set(inp.pan, inp.tilt)
    return runtime.head.state()

@app.post("/api/estop")
def estop():
    runtime.estop()
    return {"estopped": True}

@app.post("/api/estop/clear")
def estop_clear():
    runtime.clear_estop()
    return {"estopped": False}

@app.post("/api/cancel")
def cancel():
    runtime.executor.cancel()
    return {"cancelled": True}

@app.get("/api/camera/{role}.mjpg")
def camera(role: str):
    if role not in ("head", "arm"):
        return {"error": "role must be head|arm"}

    def gen():
        while True:
            jpg = runtime.perc.latest_jpegs().get(role)
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1 / 20)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(runtime.status()))
            await asyncio.sleep(1.0 / runtime.p.telemetry_rate_hz)
    except WebSocketDisconnect:
        pass


def main():
    # hard guard against the multi-process port race that caused dark feeds:
    # if the port is already taken, fail immediately with a clear message
    # instead of letting uvicorn limp or stack instances.
    import socket, sys
    host, port = runtime.cfg.api_host, runtime.cfg.api_port
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host if host != "0.0.0.0" else "", port))
    except OSError:
        probe.close()
        sys.exit(f"FATAL: port {port} is already in use — another runtime (or "
                 f"service) is running. Kill it first: "
                 f"`pkill -f roarm_runtime.api.server`. Refusing to start.")
    probe.close()
    uvicorn.run("roarm_runtime.api.server:app", host=host,
                port=port, log_level="info")


if __name__ == "__main__":
    main()
