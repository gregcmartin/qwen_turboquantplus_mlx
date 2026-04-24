#!/usr/bin/env bash
# launch-opencode.sh — start TurboQuantPlus server, wait for readiness,
# then launch opencode pointed at it. Cleans up the server on exit.
#
# Usage:
#   ./launch-opencode.sh              # start server + opencode
#   ./launch-opencode.sh --init-config  # also write ~/.opencode.json for this model
#   HOST=0.0.0.0 PORT=9000 ./launch-opencode.sh   # custom bind/port
#
# Any args after flags handled here are forwarded to opencode.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
ENDPOINT="http://$HOST:$PORT/v1"
MODEL_PATH="$DIR/models/Qwen3.6-27B-MLX-mxfp8"
MODEL_ID="$MODEL_PATH"  # mlx_lm.server keys models by absolute path
OPENCODE_MODEL="local.$MODEL_ID"

init_config=0
args=()
for arg in "$@"; do
  case "$arg" in
    --init-config) init_config=1 ;;
    *) args+=("$arg") ;;
  esac
done

if [ ! -d "$MODEL_PATH" ]; then
  cat >&2 <<EOF
Model not found at $MODEL_PATH.

Run the conversion first:
  mkdir -p models
  source .venv/bin/activate
  python -m mlx_lm convert \\
    --hf-path Qwen/Qwen3.6-27B \\
    --mlx-path ./models/Qwen3.6-27B-MLX-mxfp8 \\
    -q --q-mode mxfp8
EOF
  exit 1
fi

if [ "$init_config" = 1 ]; then
  cfg="$HOME/.opencode.json"
  if [ -e "$cfg" ]; then
    echo "Refusing to overwrite existing $cfg — merge this snippet yourself:"
  else
    cat > "$cfg" <<JSON
{
  "agents": {
    "coder": { "model": "$OPENCODE_MODEL" },
    "task":  { "model": "$OPENCODE_MODEL" },
    "title": { "model": "$OPENCODE_MODEL" }
  }
}
JSON
    echo "Wrote $cfg"
  fi
fi

server_pid=""
started_server=0
cleanup() {
  if [ "$started_server" = 1 ]; then
    echo "Stopping TurboQuantPlus server on port $PORT ..."
    # mlx_lm.server can fork a worker, so kill by port pattern to catch both.
    pkill -TERM -f "mlx_lm.*(server|run_server).*--port $PORT\\b" 2>/dev/null || true
    if [ -n "$server_pid" ]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    # Give it a moment, then SIGKILL any stragglers on the port.
    sleep 1
    lingering=$(lsof -ti ":$PORT" 2>/dev/null || true)
    if [ -n "$lingering" ]; then
      echo "Force-killing lingering listeners on :$PORT ($lingering)"
      echo "$lingering" | xargs kill -KILL 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

if curl -sf "$ENDPOINT/models" >/dev/null 2>&1; then
  echo "Reusing existing server at $ENDPOINT"
else
  log="$DIR/.opencode-server.log"
  echo "Starting TurboQuantPlus server on $HOST:$PORT (log: $log) ..."
  "$DIR/run.sh" --serve --host "$HOST" --port "$PORT" > "$log" 2>&1 &
  server_pid=$!
  started_server=1

  for i in $(seq 1 120); do
    if curl -sf "$ENDPOINT/models" >/dev/null 2>&1; then
      echo "Server ready after ${i}s"
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server process died before becoming ready. Last log lines:" >&2
      tail -30 "$log" >&2
      exit 1
    fi
    sleep 1
  done

  if ! curl -sf "$ENDPOINT/models" >/dev/null 2>&1; then
    echo "Server did not become ready in 120s. See $log." >&2
    exit 1
  fi
fi

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode not found on PATH." >&2
  echo "Install opencode, then make sure ~/.opencode.json references:" >&2
  echo "  $OPENCODE_MODEL" >&2
  echo "(or re-run with --init-config to have this script write it)." >&2
  exit 127
fi

echo "Launching opencode → $ENDPOINT (model: $OPENCODE_MODEL)"
LOCAL_ENDPOINT="$ENDPOINT" opencode "${args[@]}"
