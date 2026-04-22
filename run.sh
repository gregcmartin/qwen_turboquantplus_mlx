#!/bin/bash
# TurboQuantPlus launcher
# Usage:
#   ./run.sh                     # interactive chat
#   ./run.sh -p "Hello world"    # single prompt
#   ./run.sh --benchmark --chat  # chat with token/s stats

DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/.venv/bin/activate"
exec python3 "$DIR/turboquant.py" "$@"
