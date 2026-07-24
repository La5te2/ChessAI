#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PGN="${PGN:-data/games.pgn}"
MIN_FENS="${MIN_FENS:-50000}"
OUTPUT="${OUTPUT:-data/openings.gen.bin}"
UCI="${UCI:-models/stockfish/stockfish}"

MAX_ABS_CP="${MAX_ABS_CP:-80}"
BOOK_PLIES="${BOOK_PLIES:-8}"
UCI_DEPTH="${UCI_DEPTH:-10}"
UCI_MOVETIME_MS="${UCI_MOVETIME_MS:-0}"
UCI_THREADS="${UCI_THREADS:-4}"
UCI_HASH_MB="${UCI_HASH_MB:-512}"
LOG_EVERY="${LOG_EVERY:-1000}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: bash run_opening.sh [pgn] [min_fens] [output_bin]"
  echo "example: bash run_opening.sh data/games.pgn 50000 data/openings.gen.bin"
  exit 0
fi

if [[ $# -ge 1 ]]; then
  PGN="$1"
fi
if [[ $# -ge 2 ]]; then
  MIN_FENS="$2"
fi
if [[ $# -ge 3 ]]; then
  OUTPUT="$3"
fi

echo "opening generation start"
echo "opening source: pgn=${PGN}"
echo "opening output: output=${OUTPUT} min_fens=${MIN_FENS} book_plies=${BOOK_PLIES}"
echo "opening engine: uci=${UCI} uci_depth=${UCI_DEPTH} uci_movetime_ms=${UCI_MOVETIME_MS} uci_threads=${UCI_THREADS} uci_hash_mb=${UCI_HASH_MB}"
echo "opening filter: max_abs_cp=${MAX_ABS_CP} log_every=${LOG_EVERY}"

python scripts/opening_book.py \
  --pgn "${PGN}" \
  --uci "${UCI}" \
  --output "${OUTPUT}" \
  --max-abs-cp "${MAX_ABS_CP}" \
  --book-plies "${BOOK_PLIES}" \
  --min-fens "${MIN_FENS}" \
  --uci-depth "${UCI_DEPTH}" \
  --uci-movetime-ms "${UCI_MOVETIME_MS}" \
  --uci-threads "${UCI_THREADS}" \
  --uci-hash-mb "${UCI_HASH_MB}" \
  --log-every "${LOG_EVERY}"
