#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LICHESS_HOME="${LICHESS_HOME:-"$ROOT_DIR/data/lichess"}"
LICHESS_BOT_DIR="${LICHESS_BOT_DIR:-"$LICHESS_HOME/lichess-bot"}"
LICHESS_BOT_REPO="${LICHESS_BOT_REPO:-https://github.com/lichess-bot-devs/lichess-bot.git}"

echo "lichess-bot setup start"
echo "home=$LICHESS_HOME"
echo "repo=$LICHESS_BOT_REPO"
echo "dir=$LICHESS_BOT_DIR"

mkdir -p "$LICHESS_HOME"

if [ ! -d "$LICHESS_BOT_DIR/.git" ]; then
  git clone "$LICHESS_BOT_REPO" "$LICHESS_BOT_DIR"
else
  git -C "$LICHESS_BOT_DIR" pull --ff-only
fi

cd "$LICHESS_BOT_DIR"

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "lichess-bot setup done"
echo "next: source $ROOT_DIR/.venv/bin/activate"
echo "next: read -rsp \"LICHESS_TOKEN: \" LICHESS_TOKEN && echo && export LICHESS_TOKEN"
echo "next: bash $ROOT_DIR/run_lichess_bot.sh"
