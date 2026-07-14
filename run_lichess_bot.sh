#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LICHESS_HOME="${LICHESS_HOME:-"$ROOT_DIR/data/lichess"}"
LICHESS_BOT_DIR="${LICHESS_BOT_DIR:-"$LICHESS_HOME/lichess-bot"}"
RUN_ID="${RUN_ID:-lichess_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="$LICHESS_HOME/runs/$RUN_ID"

MODEL="${MODEL:-models/champion.pth}"
DEVICE="${DEVICE:-cuda}"
SEARCH_TYPE="${SEARCH_TYPE:-only-mcts}"
MCTS_SIMS="${MCTS_SIMS:-0}"
MCTS_MIN_SIMS="${MCTS_MIN_SIMS:-0}"
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-64}"
MOVETIME_MS="${MOVETIME_MS:-1000}"
MOVE_OVERHEAD_MS="${MOVE_OVERHEAD_MS:-75}"
MIN_MOVETIME_MS="${MIN_MOVETIME_MS:-50}"
MAX_MOVETIME_MS="${MAX_MOVETIME_MS:-5000}"
TIME_DIVISOR="${TIME_DIVISOR:-30}"
INCREMENT_FRACTION="${INCREMENT_FRACTION:-0.75}"
C_PUCT="${C_PUCT:-0.5}"
C_PUCT_BASE="${C_PUCT_BASE:-19652}"
C_PUCT_FACTOR="${C_PUCT_FACTOR:-1.0}"
FPU_REDUCTION="${FPU_REDUCTION:-0.15}"
VIRTUAL_LOSS="${VIRTUAL_LOSS:-0.0}"
MCTS_TIME_FRACTION="${MCTS_TIME_FRACTION:-1.0}"
MATE_PLIES="${MATE_PLIES:-0}"
MATE_TOPK="${MATE_TOPK:-4}"
MATE_NODES="${MATE_NODES:-20000}"
ROOT_TOPN="${ROOT_TOPN:-5}"
LOG_SEARCH="${LOG_SEARCH:-false}"

CHALLENGE_CONCURRENCY="${CHALLENGE_CONCURRENCY:-1}"
CHALLENGE_VARIANTS="${CHALLENGE_VARIANTS:-standard}"
CHALLENGE_TIME_CONTROLS="${CHALLENGE_TIME_CONTROLS:-blitz,rapid,classical}"
CHALLENGE_MODES="${CHALLENGE_MODES:-casual}"
CHALLENGE_ONLY_BOT="${CHALLENGE_ONLY_BOT:-true}"
ALLOW_MATCHMAKING="${ALLOW_MATCHMAKING:-false}"

LICHESS_URL="${LICHESS_URL:-https://lichess.org}"
ENGINE_PYTHON="${ENGINE_PYTHON:-$(command -v python || command -v python3)}"
BOT_PYTHON="$LICHESS_BOT_DIR/venv/bin/python"
DEFAULT_CONFIG="$LICHESS_BOT_DIR/config.yml.default"
CONFIG_PATH="$RUN_DIR/config.yml"
ENGINE_WRAPPER="$RUN_DIR/chessai_uci.sh"
INFO_LOG="$RUN_DIR/info.log"

if [ -z "${LICHESS_TOKEN:-}" ]; then
  echo "LICHESS_TOKEN is required."
  echo "Example: export LICHESS_TOKEN=lip_..."
  exit 1
fi

if [ ! -x "$BOT_PYTHON" ] || [ ! -f "$DEFAULT_CONFIG" ]; then
  echo "lichess-bot is not installed in $LICHESS_BOT_DIR"
  echo "Run: bash setup_lichess_bot.sh"
  exit 1
fi

mkdir -p "$RUN_DIR"

if [[ "$MODEL" = /* ]]; then
  MODEL_ABS="$MODEL"
else
  MODEL_ABS="$ROOT_DIR/$MODEL"
fi

cat > "$ENGINE_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
exec "$ENGINE_PYTHON" "$ROOT_DIR/src/uci_engine.py"
EOF
chmod +x "$ENGINE_WRAPPER"

export LICHESS_URL
export ENGINE_DIR="$RUN_DIR"
export ENGINE_NAME="$(basename "$ENGINE_WRAPPER")"
export ENGINE_WORKING_DIR="$ROOT_DIR"
export MODEL_ABS
export DEVICE
export SEARCH_TYPE
export MCTS_SIMS
export MCTS_MIN_SIMS
export MCTS_BATCH_SIZE
export MOVETIME_MS
export MOVE_OVERHEAD_MS
export MIN_MOVETIME_MS
export MAX_MOVETIME_MS
export TIME_DIVISOR
export INCREMENT_FRACTION
export C_PUCT
export C_PUCT_BASE
export C_PUCT_FACTOR
export FPU_REDUCTION
export VIRTUAL_LOSS
export MCTS_TIME_FRACTION
export MATE_PLIES
export MATE_TOPK
export MATE_NODES
export ROOT_TOPN
export LOG_SEARCH
export CHALLENGE_CONCURRENCY
export CHALLENGE_VARIANTS
export CHALLENGE_TIME_CONTROLS
export CHALLENGE_MODES
export CHALLENGE_ONLY_BOT
export ALLOW_MATCHMAKING
export PGN_DIRECTORY="$RUN_DIR/pgn"

"$BOT_PYTHON" - "$DEFAULT_CONFIG" "$CONFIG_PATH" <<'PY'
import os
import sys

import yaml


def env_int(name):
    return int(float(os.environ[name]))


def env_float(name):
    return float(os.environ[name])


def env_bool(name):
    return str(os.environ[name]).strip().lower() in ("1", "true", "yes", "on")


def csv_list(name):
    return [item.strip() for item in os.environ[name].split(",") if item.strip()]


default_config, output_config = sys.argv[1], sys.argv[2]
with open(default_config, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config["token"] = os.environ["LICHESS_TOKEN"]
config["url"] = os.environ["LICHESS_URL"]

engine = config.setdefault("engine", {})
engine["dir"] = os.environ["ENGINE_DIR"]
engine["name"] = os.environ["ENGINE_NAME"]
engine["working_dir"] = os.environ["ENGINE_WORKING_DIR"]
engine["protocol"] = "uci"
engine["ponder"] = False
engine["silence_stderr"] = False
engine["uci_options"] = {
    "ModelPath": os.environ["MODEL_ABS"],
    "Device": os.environ["DEVICE"],
    "SearchType": os.environ["SEARCH_TYPE"],
    "MCTSSims": env_int("MCTS_SIMS"),
    "MCTSMinSims": env_int("MCTS_MIN_SIMS"),
    "MCTSBatchSize": env_int("MCTS_BATCH_SIZE"),
    "MoveTimeMS": env_int("MOVETIME_MS"),
    "MoveOverheadMS": env_int("MOVE_OVERHEAD_MS"),
    "MinMoveTimeMS": env_int("MIN_MOVETIME_MS"),
    "MaxMoveTimeMS": env_int("MAX_MOVETIME_MS"),
    "TimeDivisor": env_float("TIME_DIVISOR"),
    "IncrementFraction": env_float("INCREMENT_FRACTION"),
    "CPuct": env_float("C_PUCT"),
    "CPuctBase": env_float("C_PUCT_BASE"),
    "CPuctFactor": env_float("C_PUCT_FACTOR"),
    "FPUReduction": env_float("FPU_REDUCTION"),
    "VirtualLoss": env_float("VIRTUAL_LOSS"),
    "MCTSTimeFraction": env_float("MCTS_TIME_FRACTION"),
    "MatePlies": env_int("MATE_PLIES"),
    "MateTopK": env_int("MATE_TOPK"),
    "MateNodes": env_int("MATE_NODES"),
    "RootTopN": env_int("ROOT_TOPN"),
    "LogSearch": env_bool("LOG_SEARCH"),
}

challenge = config.get("challenge")
if isinstance(challenge, dict):
    if "concurrency" in challenge:
        challenge["concurrency"] = env_int("CHALLENGE_CONCURRENCY")
    if "variants" in challenge:
        challenge["variants"] = csv_list("CHALLENGE_VARIANTS")
    if "time_controls" in challenge:
        challenge["time_controls"] = csv_list("CHALLENGE_TIME_CONTROLS")
    if "modes" in challenge:
        challenge["modes"] = csv_list("CHALLENGE_MODES")
    if "accept_bot" in challenge:
        challenge["accept_bot"] = True
    if "only_bot" in challenge:
        challenge["only_bot"] = env_bool("CHALLENGE_ONLY_BOT")

matchmaking = config.get("matchmaking")
if isinstance(matchmaking, dict) and "allow_matchmaking" in matchmaking:
    matchmaking["allow_matchmaking"] = env_bool("ALLOW_MATCHMAKING")

config["pgn_directory"] = os.environ["PGN_DIRECTORY"]
config["pgn_file_grouping"] = "all"

with open(output_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

echo "lichess bot run id: $RUN_ID"
echo "lichess bot directory: $LICHESS_BOT_DIR"
echo "run directory: $RUN_DIR"
echo "config: $CONFIG_PATH"
echo "engine wrapper: $ENGINE_WRAPPER"
echo "model: $MODEL_ABS"
echo "device: $DEVICE"
echo "search_type: $SEARCH_TYPE"
echo "mcts_sims: $MCTS_SIMS"

if [ "${UPGRADE_BOT:-0}" = "1" ]; then
  echo "upgrading lichess account to BOT account"
  cd "$LICHESS_BOT_DIR"
  exec "$BOT_PYTHON" lichess-bot.py -u --config "$CONFIG_PATH"
fi

if [ "${FOREGROUND:-0}" = "1" ]; then
  echo "starting lichess bot in foreground"
  cd "$LICHESS_BOT_DIR"
  exec "$BOT_PYTHON" lichess-bot.py --config "$CONFIG_PATH"
fi

echo "starting lichess bot in background"
(
  cd "$LICHESS_BOT_DIR"
  nohup "$BOT_PYTHON" lichess-bot.py --config "$CONFIG_PATH" > "$INFO_LOG" 2>&1 &
  echo "$!" > "$RUN_DIR/pid"
)

echo "pid=$(cat "$RUN_DIR/pid")"
echo "log=$INFO_LOG"
echo "tail -f $INFO_LOG"
