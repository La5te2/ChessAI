#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--foreground" ]]; then
  RUN_ID="${RUN_ID:-reinforce_$(date +%Y%m%d_%H%M%S)_$$}"
  DATA_RUN_DIR="data/runs/${RUN_ID}"
  mkdir -p "${DATA_RUN_DIR}"
  nohup bash "$0" --foreground "${RUN_ID}" > "${DATA_RUN_DIR}/info.log" 2>&1 < /dev/null &
  PID="$!"
  echo "${PID}" > "${DATA_RUN_DIR}/pid"
  echo "reinforce launched"
  echo "run_id=${RUN_ID}"
  echo "pid=${PID}"
  echo "log=${DATA_RUN_DIR}/info.log"
  echo "tail -f ${DATA_RUN_DIR}/info.log"
  exit 0
fi

RUN_ID="${2:?missing run id}"

MODEL="models/chessnet.pth"
SUPERVISED_DATA="data/games.h5"
UCI="models/stockfish/stockfish"
DEVICE="cuda"

ITERATIONS=10
GAMES_PER_ITER=500
PARALLEL=10
MAX_PLIES=150

OPENING_BOOK=""
BOOK_PLIES=8
MAX_BOOK_POSITIONS=50000

SAMPLE_TEMPERATURE=0.3
SAMPLE_TOPK=4
SHARP_GAP_CP=100
SHARP_TEMPERATURE=0.15
SHARP_TOPK=1

DELTA_WEIGHT=0.35
REGRET_WEIGHT=0.70
REGRET_SCALE_CP=250
BLUNDER_CP=200
BLUNDER_WEIGHT=0.60
REWARD_CLIP=2.0

UCI_DEPTH=12
UCI_MOVETIME_MS=0
UCI_MULTIPV=4
UCI_THREADS=1
UCI_HASH_MB=512

PPO_EPOCHS=25
TRAIN_MAX_STEPS=2500
BATCH_SIZE=256
TRAIN_WORKERS=4
LR=0.00005
SUPERVISED_WEIGHT=0.35
KL_WEIGHT=0.10
ENTROPY_WEIGHT=0.005
CRITIC_TARGET="teacher"

EVAL_GAMES=100
EVAL_SIMS=64
EVAL_WORKERS=10
EVAL_MAX_PLIES=150
EVAL_OPENING_BOOK="data/openings.gen.bin"
EVAL_MOVETIME_MS=1000
EVAL_UCI_DEPTH=12
EVAL_UCI_MULTIPV=6
EVAL_MIN_NET_WINS=5

LOG_EVERY=50
SEED=2026

echo "reinforce foreground start"
echo "run_id=${RUN_ID}"
echo "model=${MODEL}"
echo "rollout: games_per_iter=${GAMES_PER_ITER} parallel=${PARALLEL} max_plies=${MAX_PLIES} opening=startpos"
echo "sampling: temperature=${SAMPLE_TEMPERATURE} topk=${SAMPLE_TOPK} sharp_gap_cp=${SHARP_GAP_CP}"
echo "reward: delta_weight=${DELTA_WEIGHT} regret_weight=${REGRET_WEIGHT} regret_scale_cp=${REGRET_SCALE_CP} blunder_cp=${BLUNDER_CP} blunder_weight=${BLUNDER_WEIGHT}"
echo "teacher: uci=${UCI} depth=${UCI_DEPTH} multipv=${UCI_MULTIPV} threads=${UCI_THREADS}"
echo "train: ppo_epochs=${PPO_EPOCHS} train_max_steps=${TRAIN_MAX_STEPS} batch_size=${BATCH_SIZE} critic_target=${CRITIC_TARGET}"
echo "eval: games=${EVAL_GAMES} sims=${EVAL_SIMS} movetime_ms=${EVAL_MOVETIME_MS} opening_book=${EVAL_OPENING_BOOK}"

exec python src/reinforce.py \
  --run-id "${RUN_ID}" \
  --model "${MODEL}" \
  --supervised-data "${SUPERVISED_DATA}" \
  --uci "${UCI}" \
  --device "${DEVICE}" \
  --iterations "${ITERATIONS}" \
  --games-per-iter "${GAMES_PER_ITER}" \
  --parallel "${PARALLEL}" \
  --max-plies "${MAX_PLIES}" \
  --opening-book "${OPENING_BOOK}" \
  --book-plies "${BOOK_PLIES}" \
  --max-book-positions "${MAX_BOOK_POSITIONS}" \
  --sample-temperature "${SAMPLE_TEMPERATURE}" \
  --sample-topk "${SAMPLE_TOPK}" \
  --sharp-gap-cp "${SHARP_GAP_CP}" \
  --sharp-temperature "${SHARP_TEMPERATURE}" \
  --sharp-topk "${SHARP_TOPK}" \
  --delta-weight "${DELTA_WEIGHT}" \
  --regret-weight "${REGRET_WEIGHT}" \
  --regret-scale-cp "${REGRET_SCALE_CP}" \
  --blunder-cp "${BLUNDER_CP}" \
  --blunder-weight "${BLUNDER_WEIGHT}" \
  --reward-clip "${REWARD_CLIP}" \
  --uci-depth "${UCI_DEPTH}" \
  --uci-movetime-ms "${UCI_MOVETIME_MS}" \
  --uci-multipv "${UCI_MULTIPV}" \
  --uci-threads "${UCI_THREADS}" \
  --uci-hash-mb "${UCI_HASH_MB}" \
  --ppo-epochs "${PPO_EPOCHS}" \
  --train-max-steps "${TRAIN_MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-workers "${TRAIN_WORKERS}" \
  --lr "${LR}" \
  --supervised-weight "${SUPERVISED_WEIGHT}" \
  --kl-weight "${KL_WEIGHT}" \
  --entropy-weight "${ENTROPY_WEIGHT}" \
  --critic-target "${CRITIC_TARGET}" \
  --eval-games "${EVAL_GAMES}" \
  --eval-sims "${EVAL_SIMS}" \
  --eval-workers "${EVAL_WORKERS}" \
  --eval-max-plies "${EVAL_MAX_PLIES}" \
  --eval-opening-book "${EVAL_OPENING_BOOK}" \
  --eval-movetime-ms "${EVAL_MOVETIME_MS}" \
  --eval-uci-depth "${EVAL_UCI_DEPTH}" \
  --eval-uci-multipv "${EVAL_UCI_MULTIPV}" \
  --eval-min-net-wins "${EVAL_MIN_NET_WINS}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}"
