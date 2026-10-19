#!/usr/bin/env bash
# RoArm runtime launcher — guarantees a SINGLE healthy instance.
#
#   ./start_runtime.sh          start (killing any stale instances first)
#   ./start_runtime.sh stop     stop all instances
#
# Exits non-zero with a clear message if the port is still taken by a
# foreign service, or if the cameras produce no frames after startup.
set -u
cd "$(dirname "$0")"
PORT=$(python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("config/runtime.json")
print(json.loads(p.read_text()).get("api_port", 8000) if p.exists() else 8000)
EOF
)

# list runtime PIDs owned by the python interpreter only — never this script
# or its parent shell (their command lines also match the pattern)
our_pids() {
  ps -eo pid,comm,args | awk '$2 ~ /python/ && $0 ~ /roarm_runtime\.api\.server/ {print $1}'
}

case "${1:-start}" in
  stop)
    P=$(our_pids)
    [ -n "$P" ] && { kill $P 2>/dev/null; echo "stopped: $P"; } || echo "no instance was running."
    exit 0 ;;
esac

echo "== roarm runtime launcher =="

# 1. kill our own stale instances (the exact cause of the dark-feed bug)
PIDS=$(our_pids)
if [ -n "$PIDS" ]; then
  echo "killing stale runtime instance(s): $PIDS"
  kill $PIDS 2>/dev/null; sleep 2
  PIDS=$(our_pids)
  [ -n "$PIDS" ] && { kill -9 $PIDS 2>/dev/null; sleep 1; }
fi

# 2. is the port taken by something that ISN'T us? (e.g. Gradio, FrameHub)
#    probe with a real bind (REUSEADDR so TIME_WAIT after a restart is fine);
#    connect_ex alone can't distinguish a live listener from a stale socket
if python3 -c "
import socket,sys
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try: s.bind(('',$PORT))
except OSError: sys.exit(0)
sys.exit(1)"; then
  echo "FATAL: port $PORT is held by a foreign process:"
  (ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep ":$PORT " || true
  echo "Free the port or change api_port in config/runtime.json."
  exit 1
fi

# 3. start, detached, with a log
nohup python3 -m roarm_runtime.api.server > runtime.log 2>&1 &
NEWPID=$!
echo "started pid $NEWPID (log: runtime.log)"

# 4. verify: API answers AND both cameras deliver frames within 8 s
python3 - "$PORT" <<'EOF'
import json, sys, time, urllib.request
port = sys.argv[1]
deadline = time.time() + 8.0
while time.time() < deadline:
    try:
        d = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/status", timeout=2))
        cams = d.get("cameras_ok", {})
        if all(cams.get(r) for r in ("head", "arm")):
            w = d["world"]
            print(f"OK  driver={d['driver']}  cameras={cams}  "
                  f"reachable={w['reachable']} ({w['reach_reason']})")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(0.5)
print("FAIL: runtime started but cameras produced no frames within 8 s.")
print("Last log lines:")
print(open("runtime.log").read()[-2000:])
sys.exit(1)
EOF
