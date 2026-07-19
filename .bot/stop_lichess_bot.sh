#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LICHESS_HOME="${LICHESS_HOME:-"$SCRIPT_DIR"}"
RUN_ID="${1:-${RUN_ID:-}}"

if [ -z "$RUN_ID" ]; then
  echo "Usage: bash $SCRIPT_DIR/stop_lichess_bot.sh <run-id>"
  echo "Known runs:"
  find "$LICHESS_HOME/runs" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" 2>/dev/null | sort || true
  exit 1
fi

RUN_DIR="$LICHESS_HOME/runs/$RUN_ID"
PID_FILE="$RUN_DIR/pid"

if [ ! -f "$PID_FILE" ]; then
  echo "pid file not found: $PID_FILE"
  exit 1
fi

PID="$(cat "$PID_FILE")"
echo "stopping lichess bot run_id=$RUN_ID pid=$PID"
kill -TERM "$PID" 2>/dev/null || true
sleep 5

if ps -p "$PID" >/dev/null 2>&1; then
  echo "process still alive, sending SIGKILL"
  kill -KILL "$PID" 2>/dev/null || true
fi

if ps -p "$PID" >/dev/null 2>&1; then
  echo "process still alive: $PID"
  exit 1
fi

echo "stopped"
